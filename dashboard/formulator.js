/* Inverse design and equivalence guidance.
 *
 * The inverse problem a formulator has is not "pick three components that sum to
 * 100". Dose fixes the drug load, so the question is: given that load, which
 * grade and how much polymer? Two degrees of freedom, lactose as the balance,
 * nothing to reconcile by hand.
 *
 * Candidates are ranked on the named-response models -- t50, % released -- since
 * that is what the goals are written in. The predicted curve still comes from the
 * Weibull fit, because three parameters draw a curve and a single number does not.
 */
(function (root) {
  "use strict";

  var GRADE_COLOUR = { K100LV: "#3b7dd8", K4M: "#d9822b", K100M: "#8e5bb5" };
  var FALLBACK = ["#3b7dd8", "#d9822b", "#8e5bb5", "#3aa17e", "#c0504d"];

  function colourFor(g, i) {
    return GRADE_COLOUR[g] || FALLBACK[(i || 0) % FALLBACK.length];
  }

  /* ------------------------------------------------ named-response models */

  function evalTerm(name, comp, v) {
    var parts = String(name).split(":");
    var head = parts[0], power = 0;
    if (parts.length > 1) power = parts[1] === "v^2" ? 2 : 1;
    else if (head === "v" || head === "v^2") {
      power = head === "v^2" ? 2 : 1; head = "intercept";
    }
    var base = 1;
    if (head !== "intercept") {
      var factors = head.split("*");
      for (var i = 0; i < factors.length; i++) {
        var sq = /\^2$/.test(factors[i]);
        var val = comp[factors[i].replace(/\^2$/, "")];
        if (val === undefined) return NaN;
        base *= sq ? val * val : val;
      }
    }
    return base * Math.pow(v, power);
  }

  function predictResponse(resp, comp, v) {
    var total = 0;
    for (var i = 0; i < resp.coefficients.length; i++) {
      var c = resp.coefficients[i];
      total += c.estimate * evalTerm(c.name, comp, v);
    }
    return total;
  }

  /* --------------------------------------------------------- desirability */

  /* Derringer-Suich. Each response maps to 0-1, then a weighted geometric mean:
   * missing any one goal entirely gives zero rather than averaging into
   * respectability. */
  function individual(value, goal, span) {
    if (!isFinite(value)) return 0;
    if (goal.direction === "target") {
      var tol = goal.tolerance || Math.max(Math.abs(goal.target) * 0.5, 1e-6);
      var dev = Math.abs(value - goal.target) / tol;
      return dev >= 1 ? 0 : 1 - dev;
    }
    if (span.hi <= span.lo) return 0.5;
    var t = (value - span.lo) / (span.hi - span.lo);
    t = Math.max(0, Math.min(1, t));
    return goal.direction === "maximise" ? t : 1 - t;
  }

  function overall(parts, weights) {
    var wsum = 0, acc = 0, anyZero = false;
    for (var i = 0; i < parts.length; i++) {
      if (parts[i] <= 0) anyZero = true;
      acc += weights[i] * Math.log(Math.max(parts[i], 1e-12));
      wsum += weights[i];
    }
    if (anyZero) return 0;
    return wsum > 0 ? Math.exp(acc / wsum) : 0;
  }

  /* ---------------------------------------------------------------- hull */

  function convexHull(points) {
    var p = points.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    if (p.length < 3) return p;
    function half(src) {
      var out = [];
      for (var i = 0; i < src.length; i++) {
        while (out.length >= 2) {
          var o = out[out.length - 2], q = out[out.length - 1];
          if ((q[0] - o[0]) * (src[i][1] - o[1]) - (q[1] - o[1]) * (src[i][0] - o[0]) <= 0) {
            out.pop();
          } else break;
        }
        out.push(src[i]);
      }
      out.pop();
      return out;
    }
    return half(p).concat(half(p.reverse()));
  }

  function makeInHull(D) {
    var seen = {}, pts = [];
    D.design_points.forEach(function (p) {
      if (seen[p.case]) return;
      seen[p.case] = 1;
      pts.push([p.api_wt, p.hpmc_wt]);
    });
    var hull = convexHull(pts);
    return function (api, hpmc) {
      for (var i = 0; i < hull.length; i++) {
        var a = hull[i], b = hull[(i + 1) % hull.length];
        var cross = (b[0] - a[0]) * (hpmc - a[1]) - (b[1] - a[1]) * (api - a[0]);
        if (cross < -1e-9) return false;
      }
      return true;
    };
  }

  /* ------------------------------------------------------------- render */

  function render(D, api) {
    var $ = api.$, el = api.el, esc = api.esc, fmt = api.fmt,
        Chart = api.Chart, table = api.table, extent = api.extent;
    var inHull = makeInHull(D);
    var M = window.CRModel;

    var grades = [];
    D.design_points.forEach(function (p) {
      if (!grades.some(function (g) { return g.grade === p.grade; })) {
        grades.push({ grade: p.grade, lv: p.log10_visc, v: p.v_coded,
                      cp: p.viscosity_cp });
      }
    });
    grades.sort(function (a, b) { return a.cp - b.cp; });

    var levels = D.design_points.map(function (p) { return p.log10_visc; });
    var respByKey = {};
    D.doe.responses.forEach(function (r) { respByKey[r.key] = r; });

    /* Ranges observed in the design; nothing is proposed outside them. */
    var apiRange = extentOf("api_wt"), hpmcRange = extentOf("hpmc_wt"),
        lacRange = extentOf("lactose_wt");
    function extentOf(field) {
      var vals = D.design_points.map(function (p) { return p[field]; });
      return { lo: Math.min.apply(null, vals), hi: Math.max.apply(null, vals) };
    }

    /* Response spans, needed to normalise maximise/minimise desirabilities. */
    var spans = {};
    Object.keys(respByKey).forEach(function (k) {
      var vals = [];
      grades.forEach(function (g) {
        for (var a = apiRange.lo; a <= apiRange.hi; a += 5) {
          for (var h = hpmcRange.lo; h <= hpmcRange.hi; h += 5) {
            var l = 100 - a - h;
            if (l < lacRange.lo || l > lacRange.hi || !inHull(a, h)) continue;
            vals.push(predictResponse(respByKey[k],
              { api: a / 100, hpmc: h / 100, lactose: l / 100 }, g.v));
          }
        }
      });
      vals = vals.filter(isFinite);
      spans[k] = { lo: Math.min.apply(null, vals), hi: Math.max.apply(null, vals) };
    });

    var goalSelect = $("iv-goal");
    if (!goalSelect.options.length) {
      D.goals.forEach(function (g) {
        goalSelect.appendChild(el("option", { value: g.key }, g.label));
      });
      goalSelect.addEventListener("change", function () { syncPrompts(); draw(); });
    }

    function currentGoal() {
      for (var i = 0; i < D.goals.length; i++) {
        if (D.goals[i].key === goalSelect.value) return D.goals[i];
      }
      return D.goals[0];
    }

    function syncPrompts() {
      var goal = currentGoal();
      var host = $("iv-prompts");
      host.innerHTML = "";
      var defaults = { "Target t50 (h)": 8, "Target % at 4 h": 40,
                       "Target % at 12 h": 80 };
      goal.prompts.forEach(function (label, i) {
        var lab = el("label", null, label + " ");
        var input = el("input", {
          type: "number", id: "iv-prompt-" + i, step: "0.5",
          value: String(defaults[label] !== undefined ? defaults[label] : 50)
        });
        input.addEventListener("input", draw);
        lab.appendChild(input);
        host.appendChild(lab);
      });

      var wrap = $("iv-weights");
      wrap.innerHTML = "";
      var det = el("details");
      det.appendChild(el("summary", null,
        "What this goal optimises (" + goal.weights.length + " responses)"));
      var body = el("div", { class: "tablewrap" });
      body.appendChild(table([
        { key: "r", label: "response" }, { key: "d", label: "direction" },
        { key: "w", label: "weight", num: true }, { key: "n", label: "why" }
      ], goal.weights.map(function (w) {
        return {
          r: (respByKey[w.response] || {}).label || w.response,
          d: w.direction + (w.target !== null && w.target !== undefined
              ? " " + w.target + (w.tolerance ? " ±" + w.tolerance : "") : ""),
          w: w.weight, n: w.note
        };
      })));
      det.appendChild(body);
      det.appendChild(el("p", { class: "hint" },
        "These weights are a judgement, not a measurement. They are shown so you " +
        "can see what the ranking rests on; two candidates within about 0.05 of " +
        "each other are not separated by anything the data can support."));
      wrap.appendChild(det);
    }

    function targetsFor(goal) {
      var out = {};
      goal.prompts.forEach(function (label, i) {
        var node = $("iv-prompt-" + i);
        out[i] = node ? Number(node.value) : NaN;
      });
      return out;
    }

    function search() {
      var goal = currentGoal();
      var prompts = targetsFor(goal);
      var apiFixed = Number($("iv-api").value);
      var apiTol = Number($("iv-api-tol").value);
      var results = [];

      for (var a = apiFixed - apiTol; a <= apiFixed + apiTol + 1e-9; a += 1.25) {
        if (a < apiRange.lo - 1e-9 || a > apiRange.hi + 1e-9) continue;
        for (var h = hpmcRange.lo; h <= hpmcRange.hi + 1e-9; h += 1.25) {
          var l = 100 - a - h;
          if (l < lacRange.lo - 1e-9 || l > lacRange.hi + 1e-9) continue;
          if (!inHull(a, h)) continue;
          var comp = { api: a / 100, hpmc: h / 100, lactose: l / 100 };

          for (var gi = 0; gi < grades.length; gi++) {
            var g = grades[gi];
            var parts = [], weights = [], detail = {};
            var ok = true;
            for (var wi = 0; wi < goal.weights.length; wi++) {
              var w = goal.weights[wi];
              var resp = respByKey[w.response];
              if (!resp) { ok = false; break; }
              var value = predictResponse(resp, comp, g.v);
              detail[w.response] = value;
              var effective = w;
              if (w.target === null || w.target === undefined) {
                var idx = goal.prompts.length ? 0 : -1;
                if (w.direction === "target" && idx >= 0) {
                  var pi = goal.weights.filter(function (x) {
                    return x.direction === "target";
                  }).indexOf(w);
                  effective = {
                    direction: "target", target: prompts[pi],
                    tolerance: Math.max(Math.abs(prompts[pi]) * 0.35, 1)
                  };
                }
              }
              parts.push(individual(value, effective, spans[w.response]));
              weights.push(w.weight);
            }
            if (!ok) continue;
            var score = overall(parts, weights);
            if (score > 0.01) {
              results.push({ api: a, hpmc: h, lac: l, grade: g.grade,
                             lv: g.lv, score: score, detail: detail });
            }
          }
        }
      }
      results.sort(function (x, y) { return y.score - x.score; });
      return results;
    }

    function draw() {
      var results = search();
      var out = $("iv-results");
      out.innerHTML = "";

      if (!results.length) {
        out.innerHTML = '<div class="callout warn">No formulation inside the tested ' +
          "design space meets this goal at that drug load. Widen the tolerance, " +
          "relax the target, or accept that this target is outside what the design " +
          "can support — the tool will not extrapolate to reach it.</div>";
        drawSweetSpot([]);
        return;
      }

      var best = results[0].score;
      var tied = results.filter(function (r) { return r.score >= best - 0.05; });
      var shown = results.slice(0, 12);

      var head = el("div", { class: "callout ok" });
      head.innerHTML = "<b>" + tied.length + " near-equivalent option" +
        (tied.length === 1 ? "" : "s") + "</b> within 0.05 desirability of the best. " +
        "The data does not separate them, so choose on drug load, cost or " +
        "compressibility. Every row carries the same cross-validated error: <b>" +
        fmt(D.validation.profile_rmse_pct, 2) + "% released</b>.";
      out.appendChild(head);

      var cols = [
        { key: "d", label: "desirability", num: true },
        { key: "grade", label: "grade" },
        { key: "hpmc", label: "HPMC%", num: true },
        { key: "api", label: "API%", num: true },
        { key: "lac", label: "lactose%", num: true }
      ];
      var goal = currentGoal();
      goal.weights.forEach(function (w) {
        cols.push({ key: w.response, label: (respByKey[w.response] || {}).label ||
                    w.response, num: true });
      });
      cols.push({ key: "src", label: "nearest measured", html: true });

      var tw = el("div", { class: "tablewrap" });
      tw.appendChild(table(cols, shown.map(function (r) {
        var row = {
          d: fmt(r.score, 3), grade: r.grade, hpmc: fmt(r.hpmc, 1),
          api: fmt(r.api, 1), lac: fmt(r.lac, 1),
          src: nearestLink(r.api, r.hpmc, r.grade)
        };
        goal.weights.forEach(function (w) {
          row[w.response] = fmt(r.detail[w.response], 1);
        });
        return row;
      })));
      out.appendChild(tw);

      drawProfiles(shown.slice(0, 4));
      drawSweetSpot(results);
    }

    function nearestLink(a, h, grade) {
      var best = null, bd = Infinity;
      D.design_points.forEach(function (p) {
        if (p.grade !== grade) return;
        var d = Math.pow(p.api_wt - a, 2) + Math.pow(p.hpmc_wt - h, 2);
        if (d < bd) { bd = d; best = p; }
      });
      if (!best) return "—";
      return '<button class="src-link" onclick="CR_goToSource(' + best.case +
        ",'" + best.grade + "')\">case " + best.case + " / " + best.grade + "</button>";
    }

    function drawProfiles(top) {
      var host = $("iv-profiles");
      host.innerHTML = "";
      if (!top.length) return;
      var times = [];
      for (var t = D.meta.plot_min_time_h; t <= D.meta.plot_max_time_h; t += 0.25) {
        times.push(t);
      }
      var ch = new Chart(560, 320);
      ch.scales([D.meta.plot_min_time_h, D.meta.plot_max_time_h],
                [D.meta.plot_min_release_pct, D.meta.plot_max_release_pct])
        .axes("time (h)", "% released");
      ch.hline(D.meta.censoring_pct, "#b4541f", "5 4");
      var series = [];
      top.forEach(function (r, i) {
        var pred = M.predictProfile(D.surfaces, levels, r.api, r.hpmc, r.lac, r.lv, times);
        ch.line(times, pred.curve, colourFor(r.grade, i), { width: 2 });
        series.push({ xs: times, ys: pred.curve, colour: colourFor(r.grade, i),
                      path: ch.lastPath, baseWidth: 2, baseOpacity: 1,
                      label: r.grade + "  HPMC " + r.hpmc.toFixed(1) + "%" });
      });
      ch.interactive(series).mount(host);
      var lg = el("div", { class: "legend" });
      lg.innerHTML = top.map(function (r, i) {
        return '<span><i style="background:' + colourFor(r.grade, i) + '"></i>' +
          esc(r.grade) + " · HPMC " + r.hpmc.toFixed(1) + "%</span>";
      }).join("") + "<span>Curves from the Weibull fit; ranking from the named-" +
        "response models.</span>";
      host.appendChild(lg);
    }

    function drawSweetSpot(results) {
      var host = $("iv-sweetspot");
      host.innerHTML = "";
      if (!results.length) return;
      var byGrade = {};
      results.forEach(function (r) {
        (byGrade[r.grade] = byGrade[r.grade] || []).push(r);
      });

      var ch = new Chart(560, 330, { l: 56, r: 16, t: 16, b: 44 });
      ch.scales([apiRange.lo - 2, apiRange.hi + 2], [hpmcRange.lo - 2, hpmcRange.hi + 2])
        .axes("API (wt%)", "HPMC (wt%)");

      var step = 1.25;
      grades.forEach(function (g, gi) {
        (byGrade[g.grade] || []).forEach(function (r) {
          if (r.score < 0.4) return;
          ch.rect(r.api - step / 2, r.hpmc - step / 2,
                  r.api + step / 2, r.hpmc + step / 2,
                  colourFor(g.grade, gi), 0.12 + 0.5 * r.score);
        });
      });
      D.design_points.forEach(function (p) {
        ch.dots([p.api_wt], [p.hpmc_wt], "#1c2330", 3.2,
                ["case " + p.case + " / " + p.grade]);
      });
      var top = results[0];
      ch.dots([top.api], [top.hpmc], "#ffffff", 6,
              ["best: " + top.grade + " HPMC " + top.hpmc.toFixed(1) + "%"]);
      ch.mount(host);

      var lg = el("div", { class: "legend" });
      lg.innerHTML = grades.map(function (g, i) {
        return '<span><i style="background:' + colourFor(g.grade, i) + '"></i>' +
          esc(g.grade) + "</span>";
      }).join("") +
        "<span>Shading is desirability — darker meets the goal better.</span>" +
        '<span><i style="background:#1c2330"></i>measured formulation</span>' +
        "<span>Blank = outside the tested region; nothing is proposed there.</span>";
      host.appendChild(lg);
    }

    ["iv-api", "iv-api-tol"].forEach(function (id) {
      var node = $(id);
      if (node) node.addEventListener("input", draw);
    });

    syncPrompts();
    draw();
  }

  root.CRFormulator = { render: render };
})(typeof window !== "undefined" ? window : globalThis);
