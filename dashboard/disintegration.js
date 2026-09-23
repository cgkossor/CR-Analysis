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
    var colours = { K100LV: "#3b7dd8", K4M: "#d9822b", K100M: "#8e5bb5" };
    var spare = ["#2a9d8f", "#c0392b", "#6c757d"];
    X.grade_order.forEach(function (g, i) {
      var colour = colours[g] || spare[i % spare.length];
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
