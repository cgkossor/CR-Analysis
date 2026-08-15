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

  var api = {
    evalTerm: evalTerm,
    predictResponse: predictResponse,
    codeProcess: codeProcess,
    weibull: weibull,
    predictProfile: predictProfile,
    timeToPercent: timeToPercent,
    t50From: t50From
  };

  if (typeof module !== "undefined" && module.exports) { module.exports = api; }
  root.CRModel = api;
})(typeof window !== "undefined" ? window : globalThis);
