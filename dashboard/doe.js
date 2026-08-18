/* DoE Analysis tab.
 *
 * Plots lead, tables support. Each section states what this data shows before it
 * shows how — a reader should be able to take the finding without decoding a
 * coefficient table, and then check the working if they want to.
 *
 * Kept separate from app.js because it is a self-contained view; app.js provides
 * the chart primitives and calls renderDoe().
 */
(function (root) {
  "use strict";

  var GRADE_COLOUR = { K100LV: "#3b7dd8", K4M: "#d9822b", K100M: "#8e5bb5" };
  var FALLBACK = ["#3b7dd8", "#d9822b", "#8e5bb5", "#3aa17e", "#c0504d"];

  /* Perceptually ordered ramp. A rainbow would invent boundaries that are not in
   * the data — bands appear where the hue changes fastest, not where the response
   * does. */
  var RAMP = [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37]
  ];

  function rampColour(t) {
    if (!isFinite(t)) return null;
    var x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    var i = Math.floor(x), f = x - i;
    var a = RAMP[i], b = RAMP[Math.min(i + 1, RAMP.length - 1)];
    return "rgb(" + Math.round(a[0] + f * (b[0] - a[0])) + "," +
      Math.round(a[1] + f * (b[1] - a[1])) + "," +
      Math.round(a[2] + f * (b[2] - a[2])) + ")";
  }

  function colourFor(grade, i) {
    return GRADE_COLOUR[grade] || FALLBACK[(i || 0) % FALLBACK.length];
  }

  function renderDoe(D, api) {
    var $ = api.$, el = api.el, esc = api.esc, fmt = api.fmt,
        Chart = api.Chart, table = api.table, extent = api.extent,
        withTip = api.withTip;
    var doe = D.doe;
    if (!doe || !doe.responses.length) {
      $("doe-model").textContent = "No DoE analysis available.";
      return;
    }

    var select = $("doe-response");
    if (!select.options.length) {
      doe.responses.forEach(function (r) {
        select.appendChild(el("option", { value: r.key }, r.label));
      });
      select.value = "pct_12h";
      if (!select.value) select.value = doe.responses[0].key;
      select.addEventListener("change", function () { draw(); });
    }

    var notes = doe.method_notes || {};
    var noteMap = {
      "m-contour": "contour", "m-interaction": "interaction", "m-cox": "cox_trace",
      "m-pareto": "pareto", "m-halfnormal": "half_normal",
      "m-anova": "anova", "m-summary": "model_summary"
    };
    Object.keys(noteMap).forEach(function (id) {
      var node = $(id);
      if (node) node.textContent = notes[noteMap[id]] || "";
    });

    function current() {
      for (var i = 0; i < doe.responses.length; i++) {
        if (doe.responses[i].key === select.value) return doe.responses[i];
      }
      return doe.responses[0];
    }

    function drawContours(r) {
      var host = $("doe-contour");
      host.innerHTML = "";
      var lo = r.scale[0], hi = r.scale[1];
      var span = (hi - lo) || 1;

      r.grids.forEach(function (g) {
        var ch = new Chart(340, 285, { l: 46, r: 12, t: 22, b: 40 });
        var ax = g.api_axis, hx = g.hpmc_axis;
        ch.scales([ax[0], ax[ax.length - 1]], [hx[0], hx[hx.length - 1]])
          .axes("API (wt%)", "HPMC (wt%)");

        var dw = (ax[1] - ax[0]) || 1, dh = (hx[1] - hx[0]) || 1;
        for (var i = 0; i < g.z.length; i++) {
          for (var j = 0; j < g.z[i].length; j++) {
            var v = g.z[i][j];
            if (v === null || !isFinite(v)) continue;   /* outside the tested hull */
            ch.rect(ax[j], hx[i], ax[j] + dw, hx[i] + dh,
                    rampColour((v - lo) / span), 1);
          }
        }
        g.points.forEach(function (p) {
          ch.dots([p[0]], [p[1]], "#ffffff", 3.4, [p[0].toFixed(1) + " / " + p[1].toFixed(1)]);
        });
        ch.mount(host.appendChild(el("div", { style: "display:inline-block" })));
        var cap = el("div", { class: "legend" });
        cap.innerHTML = "<b>" + esc(g.grade) + "</b> &middot; " +
          Math.round(g.viscosity_cp).toLocaleString() + " cP";
        host.lastChild.appendChild(cap);
      });

      var scale = el("div", { class: "legend" });
      var swatches = "";
      for (var k = 0; k <= 10; k++) {
        swatches += '<i style="width:16px;background:' + rampColour(k / 10) + '"></i>';
      }
      scale.innerHTML = "<span>" + fmt(lo, 1) + " " + swatches + " " + fmt(hi, 1) +
        " " + esc(r.units) + "</span>" +
        "<span>White points are formulations actually run. Blank area is outside " +
        "the tested region — no prediction is offered there.</span>";
      host.appendChild(scale);
    }

    function drawInteraction(r) {
      var ip = r.interaction;
      var ch = new Chart(560, 320);
      var all = [];
      ip.series.forEach(function (s) { all = all.concat(s.y); });
      ch.scales([ip.x[0], ip.x[ip.x.length - 1]], extent(all))
        .axes(ip.factor.toUpperCase() + " (wt%)", r.label + " (" + r.units + ")");
      var series = [];
      ip.series.forEach(function (s, i) {
        ch.line(ip.x, s.y, colourFor(s.grade, i), { width: 2.2 });
        series.push({ xs: ip.x, ys: s.y, colour: colourFor(s.grade, i),
                      path: ch.lastPath, baseWidth: 2.2, baseOpacity: 1,
                      label: s.grade });
      });
      ch.interactive(series).mount($("doe-interaction"));
      var lg = el("div", { class: "legend" });
      lg.innerHTML = ip.series.map(function (s, i) {
        return '<span><i style="background:' + colourFor(s.grade, i) + '"></i>' +
          esc(s.grade) + "</span>";
      }).join("") + (ip.parallel
        ? "<span>Lines are near parallel: the levers act independently.</span>"
        : "<span>Lines fan apart: the levers interact.</span>");
      $("doe-interaction").appendChild(lg);
    }

    function drawTraces(r) {
      var ch = new Chart(560, 320);
      var all = [];
      r.traces.forEach(function (t) { all = all.concat(t.y); });
      var xs = [];
      r.traces.forEach(function (t) { xs = xs.concat(t.x); });
      ch.scales(extent(xs, 0.02), extent(all))
        .axes("component (wt%)", r.label + " (" + r.units + ")");
      var series = [];
      r.traces.forEach(function (t, i) {
        ch.line(t.x, t.y, FALLBACK[i % FALLBACK.length], { width: 2.2 });
        series.push({ xs: t.x, ys: t.y, colour: FALLBACK[i % FALLBACK.length],
                      path: ch.lastPath, baseWidth: 2.2, baseOpacity: 1,
                      label: t.label });
      });
      ch.interactive(series).mount($("doe-traces"));
      var lg = el("div", { class: "legend" });
      lg.innerHTML = r.traces.map(function (t, i) {
        return '<span><i style="background:' + FALLBACK[i % FALLBACK.length] + '"></i>' +
          esc(t.label) + "</span>";
      }).join("");
      $("doe-traces").appendChild(lg);
    }

    function drawPareto(r) {
      var e = r.effects;
      var n = e.terms.length;
      var ch = new Chart(420, Math.max(220, 20 * n + 60), { l: 118, r: 14, t: 14, b: 40 });
      var maxT = Math.max.apply(null, e.abs_t.concat([e.bonferroni || 0]));
      ch.scales([0, maxT * 1.08], [-0.5, n - 0.5]).axes("|standardised effect|", "");
      for (var i = 0; i < n; i++) {
        var y = n - 1 - i;
        ch.rect(0, y - 0.36, e.abs_t[i], y + 0.36,
                e.significant[i] ? "#2f6fd0" : "#b9c2d0", 1);
        var lab = api.svgEl("text", {
          x: ch.pad.l - 6, y: ch.py(y) + 3.5, "text-anchor": "end",
          "font-size": 9.5, fill: "#3d4757"
        });
        lab.textContent = e.terms[i];
        ch.svg.appendChild(lab);
      }
      if (isFinite(e.t_critical)) ch.vline(e.t_critical, "#c0504d", "5 3");
      if (isFinite(e.bonferroni)) ch.vline(e.bonferroni, "#7d1f1c", "2 3");
      ch.mount($("doe-pareto"));
      var lg = el("div", { class: "legend" });
      lg.innerHTML =
        '<span><i style="background:#c0504d"></i>p = 0.05 (|t| ' + fmt(e.t_critical, 2) + ")</span>" +
        '<span><i style="background:#7d1f1c"></i>Bonferroni (|t| ' + fmt(e.bonferroni, 2) + ")</span>";
      $("doe-pareto").appendChild(lg);
    }

    function drawHalfNormal(r) {
      var e = r.effects;
      var n = Math.min(e.terms.length, e.quantiles.length);
      var ch = new Chart(400, 320);
      var q = e.quantiles.slice(0, n);
      ch.scales([0, Math.max.apply(null, q) * 1.08], extent(e.abs_t.slice(0, n)))
        .axes("half-normal quantile", "|standardised effect|");
      var sorted = e.terms.map(function (t, i) {
        return { term: t, abs_t: e.abs_t[i], sig: e.significant[i] };
      }).sort(function (a, b) { return a.abs_t - b.abs_t; });
      var qs = q.slice().sort(function (a, b) { return a - b; });
      for (var i = 0; i < n; i++) {
        ch.dots([qs[i]], [sorted[i].abs_t],
                sorted[i].sig ? "#c0504d" : "#5d6879", 3.4,
                [sorted[i].term + "  |t| " + sorted[i].abs_t.toFixed(2)]);
      }
      ch.mount($("doe-halfnormal"));
      var lg = el("div", { class: "legend" });
      lg.innerHTML = "<span>Points on a line through the origin are inert; " +
        "points that leave it are real effects. Needs no error estimate.</span>";
      $("doe-halfnormal").appendChild(lg);
    }

    function drawAnova(r) {
      var a = r.anova;
      var rows = [{
        source: "<b>Model</b>", df: a.model.df, ss: fmt(a.model.adj_ss, 2),
        ms: fmt(a.model.adj_ms, 2), f: fmt(a.model.f, 2),
        p: a.model.p !== null && a.model.p < 0.0001 ? "&lt;0.0001" : fmt(a.model.p, 4)
      }];
      a.rows.forEach(function (row) {
        rows.push({
          source: (row.group ? "&nbsp;&nbsp;<b>" + esc(row.source) + "</b>"
                             : "&nbsp;&nbsp;&nbsp;&nbsp;" + esc(row.source)),
          df: row.df, ss: fmt(row.adj_ss, 2), ms: fmt(row.adj_ms, 2),
          f: fmt(row.f, 2),
          p: row.p !== null && row.p < 0.0001 ? "&lt;0.0001" : fmt(row.p, 4)
        });
      });
      rows.push({ source: "<b>Residual</b>", df: a.residual.df,
                  ss: fmt(a.residual.ss, 2), ms: "", f: "", p: "" });
      rows.push({ source: "<b>Total</b>", df: a.total.df,
                  ss: fmt(a.total.ss, 2), ms: "", f: "", p: "" });
      $("doe-anova").innerHTML = "";
      $("doe-anova").appendChild(table([
        { key: "source", label: "Source", html: true },
        { key: "df", label: "DF", num: true },
        { key: "ss", label: "Adj SS", num: true },
        { key: "ms", label: "Adj MS", num: true },
        { key: "f", label: "F", num: true },
        { key: "p", label: "P", num: true, html: true }
      ], rows));
    }

    function draw() {
      var r = current();
      $("doe-model").innerHTML = "model <code>" + esc(r.model) + "</code> · " +
        r.terms.length + " terms · " + r.n_obs + " design points";

      var cards = [
        ["R2", fmt(r.summary.r2, 4)],
        ["adjusted R2", fmt(r.summary.adj_r2, 4)],
        ["predicted R2 (PRESS)", fmt(r.summary.pred_r2, 4)],
        ["S", fmt(r.summary.s, 4) + " " + r.units],
        ["residual df", r.summary.residual_df]
      ];
      var wrap = $("doe-summary");
      wrap.innerHTML = "";
      cards.forEach(function (c) {
        var d = el("div", { class: "card" });
        var lab = el("small");
        lab.innerHTML = withTip(c[0]);
        d.appendChild(lab);
        d.appendChild(el("b", null, c[1]));
        wrap.appendChild(d);
      });

      var nb = $("doe-notes");
      nb.innerHTML = "";
      (r.notes || []).concat(r.anova.notes || []).forEach(function (n) {
        var c = el("div", { class: "callout warn" });
        c.textContent = n;
        nb.appendChild(c);
      });

      $("doe-take-contour").innerHTML = esc(r.takeaways.contour);
      $("doe-take-interaction").innerHTML = esc(r.takeaways.interaction);
      $("doe-take-traces").innerHTML = esc(r.takeaways.traces);
      $("doe-take-effects").innerHTML = esc(r.takeaways.effects);
      $("doe-take-anova").innerHTML = esc(r.takeaways.anova)
        .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

      drawContours(r);
      drawInteraction(r);
      drawTraces(r);
      drawPareto(r);
      drawHalfNormal(r);
      drawAnova(r);

      var un = $("doe-unavailable");
      un.innerHTML = "";
      if (doe.unavailable && doe.unavailable.length) {
        var c = el("div", { class: "callout warn" });
        c.innerHTML = "<b>Not modelled:</b> " + doe.unavailable.map(function (u) {
          return esc(u.label) + " — " + esc(u.reason);
        }).join("<br>");
        un.appendChild(c);
      }
    }

    draw();
  }

  root.CRDoe = { render: renderDoe };
})(typeof window !== "undefined" ? window : globalThis);
