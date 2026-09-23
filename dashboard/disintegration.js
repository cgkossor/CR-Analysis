/* Disintegration tab: how long each tablet takes to break apart, and how that
 * lines up with its dissolution profile.
 *
 * Reads D.disintegration, written by pipeline/export/data_js.py when the
 * workbook has a Disintegration sheet. Without one the tab says how to add it.
 * The full DoE model for disintegration time lives in the DoE tab, so this tab
 * stays a summary.
 */
(function (root) {
  "use strict";

  function isNum(v) { return typeof v === "number" && isFinite(v); }

  function render(D, api) {
    var $ = api.$, el = api.el, esc = api.esc, fmt = api.fmt,
        Chart = api.Chart, table = api.table, svgEl = api.svgEl;
    var host = $("dt-body");
    host.innerHTML = "";
    var X = D.disintegration;

    if (!X) {
      var none = el("div", { class: "callout" });
      none.innerHTML = "This workbook has no <b>Disintegration</b> sheet, so there is " +
        "nothing to show here. Add the sheet (layout in README.md) and run the " +
        "pipeline again.";
      host.appendChild(none);
      return;
    }

    if (X.synthetic) {
      host.appendChild(el("div", { class: "callout warn" },
        "Synthetic disintegration data. These numbers were generated for testing, " +
        "not measured."));
    }

    var s = X.summary;
    var cards = [
      ["Formulations tested", s.formulations],
      ["Tablets tested", s.replicates_run],
      ["Still intact at test end", s.censored_formulations + " formulations"],
      ["Replicate agreement (ICC)", isNum(s.icc) ? fmt(s.icc, 2) : "n/a"],
      ["Rank correlation with dissolution Td",
        isNum(s.spearman_td) ? fmt(s.spearman_td, 2) : "n/a"]
    ];
    var wrap = el("div", { class: "cards" });
    cards.forEach(function (c) {
      var d = el("div", { class: "card" });
      d.appendChild(el("small", null, c[0]));
      d.appendChild(el("b", null, c[1]));
      wrap.appendChild(d);
    });
    host.appendChild(wrap);
    host.appendChild(el("p", { class: "hint" },
      "Replicate agreement runs from 0 to 1; above 0.9 means the tablets of one " +
      "formulation agree closely. Rank correlation runs from -1 to 1; near 1 means " +
      "formulations that dissolve slowly also disintegrate slowly." +
      (s.censored_replicates > 0
        ? " Tablets still intact when the test stopped are shown as open circles and " +
          "left out of the models."
        : "")));

    host.appendChild(el("h3", null, "Measured disintegration time by case"));
    host.appendChild(el("p", { class: "hint" },
      "Raw results, before any model. Each bar is the mean of that formulation's " +
      "tablets with a line for \u00b11 standard deviation; each dot is one tablet. " +
      "Hover a bar or dot for its value."));
    host.appendChild(bars(X, api));

    host.appendChild(el("h3", null, "Disintegration against dissolution"));
    host.appendChild(scatter(X, api, D));

    /* --- per-formulation table ------------------------------------------ */
    host.appendChild(el("h3", null, "Each formulation"));
    var t = el("div", { class: "tablewrap" });
    t.appendChild(table([
      { key: "cs", label: "case", num: true }, { key: "grade", label: "grade" },
      { key: "api", label: "API %", num: true }, { key: "hpmc", label: "HPMC %", num: true },
      { key: "dt", label: "disintegration (h)", num: true },
      { key: "ci", label: "95% range (h)", num: true },
      { key: "n", label: "tablets", num: true }, { key: "cv", label: "spread %", num: true },
      { key: "td", label: "dissolution Td (h)", num: true },
      { key: "lag", label: "disintegration / Td", num: true }
    ], X.points.map(function (p) {
      return {
        cs: p.case, grade: p.grade, api: fmt(p.api_wt, 1), hpmc: fmt(p.hpmc_wt, 1),
        dt: p.censored ? "> " + fmt(X.test_end_h, 1) : fmt(p.dt_h, 2),
        ci: p.censored || !isNum(p.ci_lo_h) ? "" : fmt(p.ci_lo_h, 2) + " to " + fmt(p.ci_hi_h, 2),
        n: p.n, cv: isNum(p.cv) ? fmt(p.cv * 100, 0) : "",
        td: fmt(p.td_h, 2), lag: p.censored ? "" : fmt(p.lag, 2)
      };
    })));
    host.appendChild(t);

    /* --- grade comparison ------------------------------------------------ */
    if (X.grade_ratios.length) {
      host.appendChild(el("h3", null, "Grade comparison at the same composition"));
      var ul = el("ul");
      X.grade_ratios.forEach(function (g) {
        ul.appendChild(el("li", null,
          g.a + " takes " + fmt(g.ratio, 2) + " times as long as " + g.b +
          " (95% range " + fmt(g.lo, 2) + " to " + fmt(g.hi, 2) + ")."));
      });
      host.appendChild(ul);
    }

    /* --- link to dissolution measures ----------------------------------- */
    host.appendChild(el("h3", null, "How disintegration tracks each dissolution measure"));
    host.appendChild(el("p", { class: "hint" },
      "Rank correlation across formulations. Positive: slower dissolution by this " +
      "measure goes with slower disintegration. Negative is expected for the " +
      "% released measures, where a slower tablet has released less."));
    var c = el("div", { class: "tablewrap" });
    c.appendChild(table([
      { key: "label", label: "dissolution measure" }, { key: "n", label: "formulations", num: true },
      { key: "rho", label: "rank correlation", num: true },
      { key: "ci", label: "95% range", num: true }
    ], X.correlations.map(function (r) {
      return {
        label: r.label, n: r.n, rho: fmt(r.spearman, 2),
        ci: isNum(r.lo) ? fmt(r.lo, 2) + " to " + fmt(r.hi, 2) : ""
      };
    })));
    host.appendChild(c);

    /* --- which predicts best --------------------------------------------- */
    if (X.prediction.length) {
      host.appendChild(el("h3", null, "What predicts disintegration best"));
      host.appendChild(el("p", { class: "hint" },
        "Each approach predicts every formulation with that formulation left out. " +
        "Q² of 1 is perfect; 0 is no better than the average."));
      var pw = el("div", { class: "tablewrap" });
      pw.appendChild(table([
        { key: "label", label: "predicted from" }, { key: "n", label: "formulations", num: true },
        { key: "q2", label: "Q²", num: true }
      ], X.prediction.map(function (x) {
        return { label: x.label, n: x.n, q2: fmt(x.q2, 2) };
      })));
      host.appendChild(pw);
    }

    if (X.doe_keys.length) {
      var more = el("p");
      var btn = el("button", { type: "button", class: "src-link" },
        "Open the full DoE model for disintegration time");
      btn.addEventListener("click", function () {
        var tab = root.document.querySelector('#tabs button[data-target="doe"]');
        if (tab) tab.click();
        var sel = $("doe-response");
        if (sel) {
          sel.value = X.doe_keys[0];
          sel.dispatchEvent(new root.Event("change"));
        }
      });
      more.appendChild(btn);
      host.appendChild(more);
    }
    void esc; void svgEl; void Chart;
  }

  var GRADE_COLOURS = { K100LV: "#3b7dd8", K4M: "#d9822b", K100M: "#8e5bb5" };
  var SPARE_COLOURS = ["#2a9d8f", "#c0392b", "#6c757d"];
  function gradeColour(g, i) { return GRADE_COLOURS[g] || SPARE_COLOURS[i % SPARE_COLOURS.length]; }

  function mean(v) { return v.reduce(function (a, b) { return a + b; }, 0) / v.length; }
  function sd(v) {
    if (v.length < 2) return NaN;
    var m = mean(v);
    return Math.sqrt(v.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / (v.length - 1));
  }

  /* Grouped bars from the raw tablet times: case on x, one bar per grade. */
  function bars(X, api) {
    var el = api.el, Chart = api.Chart, svgEl = api.svgEl, fmt = api.fmt;
    var box = el("div", { class: "chart" });
    var reps = X.replicates.filter(function (r) { return isNum(r.dt_h); });
    if (!reps.length) return box;

    var cases = [];
    reps.forEach(function (r) { if (cases.indexOf(r.case) < 0) cases.push(r.case); });
    cases.sort(function (a, b) { return a - b; });
    var grades = X.grade_order.filter(function (g) {
      return reps.some(function (r) { return r.grade === g; });
    });

    /* Minutes read better than fractions of an hour for fast tablets. */
    var top = Math.max.apply(null, reps.map(function (r) { return r.dt_h; }));
    var inMin = top < 2;
    var k = inMin ? 60 : 1, unit = inMin ? "min" : "h";

    var groups = {};
    reps.forEach(function (r) {
      var key = r.case + "|" + r.grade;
      (groups[key] = groups[key] || []).push(r);
    });
    var ymax = 0;
    Object.keys(groups).forEach(function (key) {
      var v = groups[key].map(function (r) { return r.dt_h * k; });
      var s = sd(v);
      ymax = Math.max(ymax, Math.max.apply(null, v), mean(v) + (isNum(s) ? s : 0));
    });
    ymax *= 1.08;

    var width = Math.max(560, 90 + cases.length * (grades.length * 16 + 14));
    var ch = new Chart(width, 330, { l: 56, r: 14, t: 30, b: 42 }).scales([0, cases.length], [0, ymax]);
    ch.axes("case", "disintegration time (" + unit + ")", []);

    var slot = (ch.px(1) - ch.px(0));
    var barW = Math.min(18, (slot * 0.8) / grades.length);
    cases.forEach(function (c, ci) {
      var label = svgEl("text", {
        x: ch.px(ci + 0.5), y: ch.h - ch.pad.b + 15, "text-anchor": "middle",
        "font-size": 10, fill: "#5d6879"
      });
      label.textContent = c;
      ch.svg.appendChild(label);

      grades.forEach(function (g, gi) {
        var rows = groups[c + "|" + g];
        if (!rows) return;
        var v = rows.map(function (r) { return r.dt_h * k; });
        var m = mean(v), s = sd(v);
        var colour = gradeColour(g, gi);
        var x0 = ch.px(ci + 0.5) - (grades.length * barW) / 2 + gi * barW;
        var bar = svgEl("rect", {
          x: x0 + 1, y: ch.py(m), width: barW - 2, height: ch.py(0) - ch.py(m),
          fill: colour, opacity: 0.75
        });
        var t = svgEl("title");
        t.textContent = "case " + c + " / " + g + ": mean " + fmt(m, 2) + " " + unit +
          (isNum(s) ? " (SD " + fmt(s, 2) + ")" : "") + ", " + v.length + " tablets";
        bar.appendChild(t);
        ch.svg.appendChild(bar);

        var xc = x0 + barW / 2;
        if (isNum(s)) {
          [[m - s, m + s]].forEach(function (lh) {
            ch.svg.appendChild(svgEl("line", {
              x1: xc, x2: xc, y1: ch.py(Math.max(lh[0], 0)), y2: ch.py(lh[1]),
              stroke: "#1c2330", "stroke-width": 1
            }));
            ch.svg.appendChild(svgEl("line", {
              x1: xc - 3, x2: xc + 3, y1: ch.py(lh[1]), y2: ch.py(lh[1]),
              stroke: "#1c2330", "stroke-width": 1
            }));
          });
        }
        rows.forEach(function (r, ri) {
          var dot = svgEl("circle", {
            cx: xc + (ri - (rows.length - 1) / 2) * 2.2, cy: ch.py(r.dt_h * k), r: 2.4,
            fill: r.censored ? "#fff" : "#1c2330", stroke: "#1c2330", "stroke-width": 0.8
          });
          var dt = svgEl("title");
          dt.textContent = "case " + c + " / " + g + ", tablet " + r.replicate + ": " +
            fmt(r.dt_h * k, 2) + " " + unit + (r.censored ? " (still intact)" : "");
          dot.appendChild(dt);
          ch.svg.appendChild(dot);
        });
      });
    });

    grades.forEach(function (g, gi) {
      var lx = ch.pad.l + 10 + gi * 90;
      ch.svg.appendChild(svgEl("rect", {
        x: lx, y: 10, width: 10, height: 10, fill: gradeColour(g, gi), opacity: 0.75
      }));
      var lt = svgEl("text", { x: lx + 14, y: 19, "font-size": 11, fill: "#3d4757" });
      lt.textContent = g;
      ch.svg.appendChild(lt);
    });

    box.appendChild(ch.svg);
    return box;
  }

  /* Disintegration time against the dissolution time scale, by grade. */
  function scatter(X, api, D) {
    var el = api.el, Chart = api.Chart, svgEl = api.svgEl, fmt = api.fmt;
    var box = el("div", { class: "chart" });
    var pts = X.points.filter(function (p) { return isNum(p.td_h) && isNum(p.dt_h); });
    if (!pts.length) return box;
    var xs = pts.map(function (p) { return p.td_h; });
    var ys = pts.map(function (p) { return p.dt_h; });
    var xmax = Math.max.apply(null, xs) * 1.08, ymax = Math.max.apply(null, ys) * 1.08;
    var ch = new Chart(560, 320).scales([0, xmax], [0, ymax]);
    ch.axes("dissolution time scale Td (h)", "disintegration time (h)");
    X.grade_order.forEach(function (g, i) {
      var colour = gradeColour(g, i);
      pts.filter(function (p) { return p.grade === g; }).forEach(function (p) {
        var c = svgEl("circle", {
          cx: ch.px(p.td_h), cy: ch.py(p.dt_h), r: 4.5,
          fill: p.censored ? "none" : colour, stroke: colour, "stroke-width": 1.5
        });
        var tip = svgEl("title");
        tip.textContent = "case " + p.case + " / " + p.grade + ": disintegration " +
          (p.censored ? "> " + fmt(X.test_end_h, 1) : fmt(p.dt_h, 2)) + " h, Td " +
          fmt(p.td_h, 2) + " h";
        c.appendChild(tip);
        ch.svg.appendChild(c);
      });
      var lx = 70 + i * 90;
      ch.svg.appendChild(svgEl("circle", { cx: lx, cy: 20, r: 4.5, fill: colour }));
      var lt = svgEl("text", { x: lx + 8, y: 24, "font-size": 11, fill: "#3d4757" });
      lt.textContent = g;
      ch.svg.appendChild(lt);
    });
    box.appendChild(ch.svg);
    void D;
    return box;
  }

  root.CRDisintegration = { render: render };
})(typeof window !== "undefined" ? window : globalThis);
