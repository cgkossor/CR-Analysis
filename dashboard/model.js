/* Pure model evaluation, shared by the dashboard and the parity test.
 *
 * This mirrors pipeline/design/matrix.py term-for-term. It is kept in its own
 * file with no DOM dependency so a test can load it under Node and check that
 * the browser reproduces Python's predictions exactly -- a silent divergence
 * here would make every number in the formulator tool wrong while the page
 * continued to look perfectly healthy.
 */
(function (root) {
  "use strict";

  /* A term name is "<comp>[*<comp>...][:v|:v^2]", or the bare intercept, or a
   * process-only term "v" / "v^2". Matches pipeline.design.matrix.term_names(). */
  function evalTerm(name, comp, v) {
    var parts = String(name).split(":");
    var head = parts[0];
    var power = 0;
    if (parts.length > 1) {
      power = parts[1] === "v^2" ? 2 : 1;
    } else if (head === "v" || head === "v^2") {
      power = head === "v^2" ? 2 : 1;
      head = "intercept";
    }
    var base = 1;
    if (head !== "intercept") {
      var factors = head.split("*");
      for (var i = 0; i < factors.length; i++) {
        var squared = /\^2$/.test(factors[i]);
        var key = factors[i].replace(/\^2$/, "");
        var val = comp[key];
        if (val === undefined) return NaN;
        base *= squared ? val * val : val;
      }
    }
    return base * Math.pow(v, power);
  }

  function predictResponse(surface, comp, v) {
    var total = 0;
    for (var i = 0; i < surface.coefficients.length; i++) {
      var c = surface.coefficients[i];
      total += c.estimate * evalTerm(c.name, comp, v);
    }
    return total;
  }

  function codeProcess(log10visc, levels) {
    var lo = Math.min.apply(null, levels);
    var hi = Math.max.apply(null, levels);
    if (hi === lo) return 0;
    return 2 * (log10visc - lo) / (hi - lo) - 1;
  }

  function weibull(t, fInf, td, beta) {
    if (t <= 0) return 0;
    return fInf * (1 - Math.exp(-Math.pow(t / td, beta)));
  }

  /* Predict a release profile. Monotonicity and [0,100] bounds are enforced
   * here rather than warned about: AC4 makes a violation a bug, not a caveat. */
  function predictProfile(surfaces, levels, apiWt, hpmcWt, lacWt, log10visc, times) {
    var total = apiWt + hpmcWt + lacWt;
    var comp = { api: apiWt / total, hpmc: hpmcWt / total, lactose: lacWt / total };
    var v = codeProcess(log10visc, levels);
    var params = {};
    Object.keys(surfaces).forEach(function (k) {
      params[k] = predictResponse(surfaces[k], comp, v);
    });
    var td = Math.pow(10, params.log10_td);
    var running = 0;
    var curve = times.map(function (t) {
      var y = Math.max(0, Math.min(100, weibull(t, params.weibull_f_inf, td, params.weibull_beta)));
      if (y < running) y = running; else running = y;
      return y;
    });
    return { params: params, td: td, curve: curve };
  }

  /* Time to 50% of the DOSE released -- the same quantity the pipeline reports
   * as t50, and the one a formulator means.
   *
   * Not the Weibull median. td*(ln 2)^(1/beta) is the time to half of F_inf,
   * which only coincides with half the dose when F_inf is 100. Matrix
   * formulations that never fully release have F_inf below 100, and the two
   * diverge by tens of percent exactly there -- on the slow formulations the
   * inverse design is most often asked about.
   *
   * Returns null when the formulation never reaches the level at all, so a
   * censored target cannot masquerade as a finite time. */
  function timeToPercent(td, beta, fInf, pct) {
    if (!(fInf > pct)) return null;
    return td * Math.pow(-Math.log(1 - pct / fInf), 1 / beta);
  }

  function t50From(td, beta, fInf) {
    return timeToPercent(td, beta, fInf === undefined ? 100 : fInf, 50);
  }

  /* ------------------------------------------- target profile -> design */

  /* How far a predicted curve sits from a target table. Each target row is
   * {t, pct, band}. The score is the worst-case deviation in units of that
   * row's band, so 1.0 means "just on the edge of the band somewhere".
   *
   * Two readings are returned. "feasible" asks whether the prediction itself
   * sits inside every band. "plausible" first forgives each deviation by the
   * cross-validated error (G6): the model cannot tell a formulation that misses
   * by less than its own error from one that hits. Mirrors
   * pipeline/optimize/target_profile.py. */
  function scoreAgainstTarget(pred, targets, cvRmse) {
    var worst = 0, worstPlausible = 0, cv = cvRmse > 0 ? cvRmse : 0;
    for (var i = 0; i < targets.length; i++) {
      var band = targets[i].band > 0 ? targets[i].band : 1e-9;
      var dev = Math.abs(pred[i] - targets[i].pct);
      if (!isFinite(dev)) return null;
      worst = Math.max(worst, dev / band);
      worstPlausible = Math.max(worstPlausible, Math.max(dev - cv, 0) / band);
    }
    return {
      score: worst, plausibleScore: worstPlausible,
      feasible: worst <= 1 + 1e-12, plausible: worstPlausible <= 1 + 1e-12
    };
  }

  /* FDA f2 over the target time points. Informational only: with four to six
   * points it is coarser than the band test, which is what decides. */
  function f2(reference, test) {
    var n = reference.length, ss = 0;
    if (!n) return null;
    for (var i = 0; i < n; i++) ss += Math.pow(reference[i] - test[i], 2);
    return 50 * Math.log(100 / Math.sqrt(1 + ss / n)) / Math.LN10;
  }

  /* Every formulation on a grid inside the tested region, scored against the
   * target. Drug load is searched too unless opts.lockApi pins it, which is
   * what makes the answers differ in composition and grade rather than by a
   * few percent around one fixed load. */
  function searchTargetProfile(surfaces, levels, grades, targets, opts) {
    var step = opts.step || 1.25, out = [];
    var times = targets.map(function (r) { return r.t; });
    var ref = targets.map(function (r) { return r.pct; });
    var aLo = opts.apiRange.lo, aHi = opts.apiRange.hi;
    if (opts.lockApi) {
      aLo = Math.max(aLo, opts.lockApi.value - opts.lockApi.tol);
      aHi = Math.min(aHi, opts.lockApi.value + opts.lockApi.tol);
    }
    for (var a = aLo; a <= aHi + 1e-9; a += step) {
      for (var h = opts.hpmcRange.lo; h <= opts.hpmcRange.hi + 1e-9; h += step) {
        var l = 100 - a - h;
        if (l < opts.lacRange.lo - 1e-9 || l > opts.lacRange.hi + 1e-9) continue;
        if (opts.inHull && !opts.inHull(a, h)) continue;
        for (var gi = 0; gi < grades.length; gi++) {
          var g = grades[gi];
          var pred = predictProfile(surfaces, levels, a, h, l, g.lv, times).curve;
          var s = scoreAgainstTarget(pred, targets, opts.cvRmse);
          if (!s) continue;
          out.push({
            api: a, hpmc: h, lac: l, grade: g.grade, lv: g.lv, pred: pred,
            score: s.score, plausibleScore: s.plausibleScore,
            feasible: s.feasible, plausible: s.plausible, f2: f2(ref, pred)
          });
        }
      }
    }
    out.sort(function (x, y) { return x.score - y.score; });
    return out;
  }

  /* The best candidate per grade and per drug-load third. Top-N by score
   * returns near-copies of one formulation; this returns the distinct ways of
   * reaching the target, which is the choice a formulator actually has. */
  function diverseShortlist(results, apiRange, accept) {
    var span = (apiRange.hi - apiRange.lo) || 1, best = {};
    results.forEach(function (r) {
      if (!accept(r)) return;
      var third = Math.min(2, Math.floor(3 * (r.api - apiRange.lo) / span));
      var key = r.grade + "|" + third;
      if (!best[key] || r.score < best[key].score) {
        best[key] = r;
        r.loadThird = ["low", "mid", "high"][third];
      }
    });
    return Object.keys(best).map(function (k) { return best[k]; })
      .sort(function (x, y) { return x.score - y.score; });
  }

  var api = {
    evalTerm: evalTerm,
    predictResponse: predictResponse,
    codeProcess: codeProcess,
    weibull: weibull,
    predictProfile: predictProfile,
    timeToPercent: timeToPercent,
    t50From: t50From,
    scoreAgainstTarget: scoreAgainstTarget,
    f2: f2,
    searchTargetProfile: searchTargetProfile,
    diverseShortlist: diverseShortlist
  };

  if (typeof module !== "undefined" && module.exports) { module.exports = api; }
  root.CRModel = api;
})(typeof window !== "undefined" ? window : globalThis);
