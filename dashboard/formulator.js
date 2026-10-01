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

    function nearestPoint(a, h, grade) {
      var best = null, bd = Infinity;
      D.design_points.forEach(function (p) {
        if (p.grade !== grade) return;
        var d = Math.pow(p.api_wt - a, 2) + Math.pow(p.hpmc_wt - h, 2);
        if (d < bd) { bd = d; best = p; }
      });
      return best;
    }

    function nearestLink(a, h, grade) {
      var best = nearestPoint(a, h, grade);
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

    renderTarget(D, api, {
      grades: grades, levels: levels, inHull: inHull, nearestLink: nearestLink,
      nearestPoint: nearestPoint,
      apiRange: apiRange, hpmcRange: hpmcRange, lacRange: lacRange
    });
  }

  /* ------------------------------------------- target profile -> design */

  /* Listeners are bound once and call through this, so a re-render with new
   * data (another API) redraws against that data rather than a stale closure. */
  var TP = { rows: [], draw: function () {} };

  function defaultBand(t) { return t <= 2 ? 5 : 10; }

  function interpAt(xs, ys, x) {
    var pts = [];
    for (var i = 0; i < xs.length; i++) {
      if (ys[i] !== null && ys[i] !== undefined && isFinite(ys[i])) pts.push([xs[i], ys[i]]);
    }
    if (!pts.length) return NaN;
    if (x <= pts[0][0]) return pts[0][1];
    for (var j = 1; j < pts.length; j++) {
      if (x <= pts[j][0]) {
        var f = (x - pts[j - 1][0]) / (pts[j][0] - pts[j - 1][0] || 1);
        return pts[j - 1][1] + f * (pts[j][1] - pts[j - 1][1]);
      }
    }
    return pts[pts.length - 1][1];
  }

  function renderTarget(D, api, ctx) {
    var $ = api.$, el = api.el, esc = api.esc, fmt = api.fmt,
        Chart = api.Chart, table = api.table;
    var M = window.CRModel;
    if (!$("tp-table")) return;
    var cv = D.validation.profile_rmse_pct;
    /* Targets stop where the data stop: past the last analysed time the model
     * would be extrapolating in time, which nothing here is allowed to do. */
    var tMax = Math.min(D.meta.plot_max_time_h, Math.max.apply(null, D.grid_h));
    var defaultTimes = [1, 2, 4, 8, 12, 24].filter(function (t) { return t <= tMax + 1e-9; });

    /* Measured profiles as starting points: a real formulation is the
     * quickest honest target, and editing it shows what moves. */
    var preset = $("tp-preset");
    preset.innerHTML = "";
    preset.appendChild(el("option", { value: "" }, "custom (keep table)"));
    D.profiles.forEach(function (p, i) {
      preset.appendChild(el("option", { value: String(i) },
        "measured: case " + p.case + " / " + p.grade));
    });

    /* Only times the measured mean actually reaches. A replicate that stopped
     * early leaves the mean blank after that point; carrying the last value
     * forward would invent a plateau and make it the target. */
    function rowsFromProfile(p) {
      if (!p) return [];
      var last = -Infinity;
      D.grid_h.forEach(function (t, i) {
        var v = p.mean_pct[i];
        if (v !== null && v !== undefined && isFinite(v)) last = Math.max(last, t);
      });
      return defaultTimes.filter(function (t) { return t <= last + 1e-9; }).map(function (t) {
        var v = interpAt(D.grid_h, p.mean_pct, t);
        return { t: t, pct: Math.round(v * 10) / 10, band: defaultBand(t) };
      });
    }
    var start = Math.floor(D.profiles.length / 2);
    preset.value = D.profiles.length ? String(start) : "";
    TP.rows = rowsFromProfile(D.profiles[start]);
    if (!TP.rows.length) {
      TP.rows = defaultTimes.map(function (t) {
        return { t: t, pct: Math.min(100, Math.round(100 * t / (tMax || 24))), band: defaultBand(t) };
      });
    }

    function renderTable() {
      var host = $("tp-table");
      host.innerHTML = "";
      var t = el("table");
      var head = el("tr");
      ["time (h)", "target % released", "± band", ""].forEach(function (h) {
        head.appendChild(el("th", null, h));
      });
      var thead = el("thead"); thead.appendChild(head); t.appendChild(thead);
      var tb = el("tbody");
      TP.rows.forEach(function (r, i) {
        var tr = el("tr");
        ["t", "pct", "band"].forEach(function (k) {
          var td = el("td");
          var input = el("input", { type: "number", step: k === "t" ? "0.25" : "0.5",
            min: "0", value: String(r[k]), "data-row": String(i), "data-key": k });
          input.addEventListener("input", function () {
            TP.rows[i][k] = Number(input.value);
            preset.value = "";
            TP.draw();
          });
          td.appendChild(input);
          tr.appendChild(td);
        });
        var del = el("td");
        if (TP.rows.length > 2) {
          var b = el("button", { type: "button", title: "remove this time point" }, "×");
          b.addEventListener("click", function () {
            TP.rows.splice(i, 1); renderTable(); TP.draw();
          });
          del.appendChild(b);
        }
        tr.appendChild(del);
        tb.appendChild(tr);
      });
      t.appendChild(tb);
      host.appendChild(t);
    }
    TP.renderTable = renderTable;

    function validRows() {
      return TP.rows.filter(function (r) {
        return isFinite(r.t) && r.t > 0 && r.t <= tMax && isFinite(r.pct) && isFinite(r.band);
      }).slice().sort(function (a, b) { return a.t - b.t; });
    }

    TP.draw = function () {
      var targets = validRows();
      var lock = $("tp-lock").checked
        ? { value: Number($("tp-lock-api").value), tol: Number($("tp-lock-tol").value) } : null;
      if (lock && !(isFinite(lock.value) && isFinite(lock.tol) && lock.tol >= 0)) {
        $("tp-summary").innerHTML = '<div class="callout warn">Enter a drug load and a ' +
          "tolerance to lock to, or untick the lock.</div>";
        return;
      }
      var results = targets.length ? M.searchTargetProfile(D.surfaces, ctx.levels, ctx.grades,
        targets, { apiRange: ctx.apiRange, hpmcRange: ctx.hpmcRange, lacRange: ctx.lacRange,
          inHull: ctx.inHull, cvRmse: cv, lockApi: lock }) : [];
      var feasible = results.filter(function (r) { return r.feasible; });
      var plausible = results.filter(function (r) { return r.plausible; });
      var accept = feasible.length
        ? function (r) { return r.feasible; } : function (r) { return r.plausible; };
      var shortlist = M.diverseShortlist(results, ctx.apiRange, accept);
      drawSummary(targets, results, feasible, plausible);
      drawMap(results, shortlist);
      drawShortlist(targets, shortlist, feasible.length > 0);
      drawCurves(targets, shortlist.slice(0, 4));
    };

    function drawSummary(targets, results, feasible, plausible) {
      var host = $("tp-summary");
      host.innerHTML = "";
      var box;
      if (!targets.length) {
        box = el("div", { class: "callout warn" });
        box.textContent = "Enter at least one time point inside 0 to " + tMax + " h.";
      } else if (feasible.length) {
        var byGrade = {}, lo = Infinity, hi = -Infinity;
        feasible.forEach(function (r) {
          byGrade[r.grade] = (byGrade[r.grade] || 0) + 1;
          lo = Math.min(lo, r.api); hi = Math.max(hi, r.api);
        });
        box = el("div", { class: "callout ok" });
        box.innerHTML = "<b>Reachable.</b> " + feasible.length + " grid formulations hit every " +
          "band, across " + Object.keys(byGrade).length + " grade" +
          (Object.keys(byGrade).length === 1 ? "" : "s") + " (" +
          Object.keys(byGrade).map(function (g) { return esc(g) + ": " + byGrade[g]; }).join(", ") +
          ") and drug loads from " + fmt(lo, 1) + " to " + fmt(hi, 1) + " wt%.";
      } else if (plausible.length) {
        box = el("div", { class: "callout warn" });
        box.innerHTML = "<b>Only within model error.</b> No prediction sits inside every band, " +
          "but " + plausible.length + " do once the cross-validated error of " + fmt(cv, 2) +
          "% is allowed. Treat these as formulations worth making and measuring, not as " +
          "ones the model endorses.";
      } else {
        box = el("div", { class: "callout danger" });
        var best = results[0];
        box.innerHTML = "<b>Not reachable in the tested design space.</b> " +
          (best ? "The closest formulation misses a band by " + fmt(best.score, 2) +
            "× its width. " : "") +
          "Widen the bands, relax a time point, or accept that this profile needs a " +
          "composition or grade outside what was tested. The tool will not extrapolate.";
      }
      host.appendChild(box);
      var note = el("p", { class: "hint" });
      note.textContent = "Every prediction carries a cross-validated error of " + fmt(cv, 2) +
        "% released (leave-one-formulation-out). A band narrower than that asks for more " +
        "precision than the model has.";
      host.appendChild(note);
    }

    function drawMap(results, shortlist) {
      var host = $("tp-map");
      host.innerHTML = "";
      var step = 1.25;
      ctx.grades.forEach(function (g, gi) {
        var wrap = el("div", { class: "chart" });
        var ch = new Chart(330, 270, { l: 48, r: 10, t: 10, b: 40 });
        ch.scales([ctx.apiRange.lo - 2, ctx.apiRange.hi + 2],
                  [ctx.hpmcRange.lo - 2, ctx.hpmcRange.hi + 2])
          .axes("API (wt%)", "HPMC (wt%)");
        var colour = colourFor(g.grade, gi);
        results.forEach(function (r) {
          if (r.grade !== g.grade || !r.plausible) return;
          var op = r.feasible ? 0.25 + 0.55 * Math.max(0, 1 - r.score) : 0.1;
          ch.rect(r.api - step / 2, r.hpmc - step / 2, r.api + step / 2, r.hpmc + step / 2,
                  colour, op);
        });
        var seen = {};
        D.design_points.forEach(function (p) {
          if (seen[p.case]) return;
          seen[p.case] = 1;
          ch.dots([p.api_wt], [p.hpmc_wt], "#1c2330", 2.8, ["case " + p.case]);
        });
        shortlist.forEach(function (r) {
          if (r.grade !== g.grade) return;
          ch.dots([r.api], [r.hpmc], "#ffffff", 5.5,
                  [r.grade + " · API " + r.api.toFixed(1) + "% · HPMC " + r.hpmc.toFixed(1) + "%"]);
        });
        ch.mount(wrap);
        wrap.insertBefore(el("div", { class: "panel-title" },
          g.grade + " (" + Math.round(g.cp || Math.pow(10, g.lv)).toLocaleString() + " cP)"),
          wrap.firstChild);
        host.appendChild(wrap);
      });
      var lg = el("div", { class: "legend" });
      lg.innerHTML = '<span><i style="background:#1c2330"></i>measured composition</span>' +
        '<span><i style="background:#fff;border:1px solid #1c2330"></i>shortlisted candidate</span>';
      host.appendChild(lg);
    }

    function drawShortlist(targets, shortlist, strict) {
      var host = $("tp-results");
      host.innerHTML = "";
      if (!shortlist.length) return;
      var tablet = Number($("tp-tablet").value);
      var cols = [
        { key: "grade", label: "grade" }, { key: "load", label: "drug load" },
        { key: "api", label: "API%", num: true }, { key: "hpmc", label: "HPMC%", num: true },
        { key: "lac", label: "lactose%", num: true }
      ];
      if (tablet > 0) cols.push({ key: "mg", label: "API mg/tablet", num: true });
      targets.forEach(function (r, i) {
        cols.push({ key: "p" + i, label: fmt(r.t, 2).replace(/\.?0+$/, "") + " h", num: true });
      });
      cols.push({ key: "score", label: "worst miss (× band)", num: true });
      cols.push({ key: "f2", label: "f2", num: true });
      /* Shear proxy (Manuscripts Q3) of the nearest measured formulation, so
       * among candidates that all meet the profile the one least likely to
       * change with hydrodynamics can be picked. Only with disintegration data. */
      var shear = D.manuscript && D.manuscript.shear &&
        Object.keys(D.manuscript.shear.by_formulation).length
        ? D.manuscript.shear.by_formulation : null;
      if (shear) cols.push({ key: "shear", label: "shear sensitivity (proxy)" });
      cols.push({ key: "src", label: "nearest measured", html: true });
      var tw = el("div", { class: "tablewrap" });
      tw.appendChild(table(cols, shortlist.map(function (r) {
        var row = {
          grade: r.grade, load: r.loadThird, api: fmt(r.api, 1), hpmc: fmt(r.hpmc, 1),
          lac: fmt(r.lac, 1), mg: fmt(tablet * r.api / 100, 0), score: fmt(r.score, 2),
          f2: fmt(r.f2, 0), src: ctx.nearestLink(r.api, r.hpmc, r.grade)
        };
        r.pred.forEach(function (v, i) { row["p" + i] = fmt(v, 1); });
        if (shear) {
          var near = ctx.nearestPoint(r.api, r.hpmc, r.grade);
          var s = near ? shear[near.case + "|" + near.grade] : null;
          row.shear = s ? s.tier : "—";
        }
        return row;
      })));
      host.appendChild(tw);
      if (shear) {
        host.appendChild(el("p", { class: "hint" }, "Shear sensitivity is a " +
          D.manuscript.shear.label + ", taken from the nearest measured formulation " +
          "(low / mid / high third). See Manuscripts, Q3."));
      }
      host.appendChild(el("p", { class: "hint" }, strict
        ? "All rows sit inside every band. Rows within about 0.2 of each other in worst miss " +
          "are not separated by anything the data can support, so choose on drug load, " +
          "grade availability or cost."
        : "These rows reach the bands only within the cross-validated error."));
    }

    function drawCurves(targets, top) {
      var host = $("tp-profiles");
      host.innerHTML = "";
      if (!targets.length) return;
      var times = [];
      for (var t = D.meta.plot_min_time_h; t <= tMax + 1e-9; t += 0.25) times.push(t);
      var ch = new Chart(620, 330);
      ch.scales([D.meta.plot_min_time_h, tMax],
                [D.meta.plot_min_release_pct, D.meta.plot_max_release_pct])
        .axes("time (h)", "% released");
      var tx = targets.map(function (r) { return r.t; });
      ch.band(tx, targets.map(function (r) { return r.pct - r.band; }),
              targets.map(function (r) { return r.pct + r.band; }), "#2f6fd0", 0.14);
      var series = [];
      top.forEach(function (r, i) {
        var pred = M.predictProfile(D.surfaces, ctx.levels, r.api, r.hpmc, r.lac, r.lv, times);
        ch.line(times, pred.curve, colourFor(r.grade, i), { width: 2 });
        series.push({ xs: times, ys: pred.curve, colour: colourFor(r.grade, i),
                      path: ch.lastPath, baseWidth: 2, baseOpacity: 1,
                      label: r.grade + " API " + r.api.toFixed(1) + "% HPMC " + r.hpmc.toFixed(1) + "%" });
      });
      ch.dots(tx, targets.map(function (r) { return r.pct; }), "#1c2330", 3.6,
              targets.map(function (r) { return "target " + r.pct + "% ± " + r.band + " at " + r.t + " h"; }));
      ch.interactive(series).mount(host);
      var lg = el("div", { class: "legend" });
      lg.innerHTML = '<span><i style="background:#2f6fd0;opacity:.35"></i>target band</span>' +
        top.map(function (r, i) {
          return '<span><i style="background:' + colourFor(r.grade, i) + '"></i>' +
            esc(r.grade) + " · API " + r.api.toFixed(1) + "% · HPMC " + r.hpmc.toFixed(1) + "%</span>";
        }).join("");
      host.appendChild(lg);
    }

    if (!TP.bound) {
      TP.bound = true;
      $("tp-preset").addEventListener("change", function () {
        var v = $("tp-preset").value;
        if (v === "") return;
        TP.rows = TP.fromProfile(Number(v));
        TP.renderTable(); TP.draw();
      });
      $("tp-add").addEventListener("click", function () {
        var last = TP.rows[TP.rows.length - 1] || { t: 0, pct: 0 };
        var t = Math.min(last.t + 2, TP.tMax);
        TP.rows.push({ t: t, pct: Math.min(last.pct + 10, 100), band: defaultBand(t) });
        TP.renderTable(); TP.draw();
      });
      $("tp-reset").addEventListener("click", function () {
        TP.rows.forEach(function (r) { r.band = defaultBand(r.t); });
        TP.renderTable(); TP.draw();
      });
      ["tp-lock", "tp-lock-api", "tp-lock-tol", "tp-tablet"].forEach(function (id) {
        $(id).addEventListener("input", function () { TP.draw(); });
        $(id).addEventListener("change", function () { TP.draw(); });
      });
    }
    TP.fromProfile = function (i) { return rowsFromProfile(D.profiles[i]); };
    TP.tMax = tMax;

    renderTable();
    TP.draw();
  }

  root.CRFormulator = { render: render };
})(typeof window !== "undefined" ? window : globalThis);
