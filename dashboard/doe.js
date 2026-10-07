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
      "m-contour": "contour",
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

    /* Takeaways are written once for the Markdown reports, where **x** is
     * bold. Escape first, then turn only that markup into <b>. */
    function prose(text) {
      return esc(text || "").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
    }

    /* Ternary contours, one triangle per grade. The fitted model fills the
     * tested region; each measured blend is a dot filled with its measured
     * value on the same colour scale, so a dot that stands out from its
     * surroundings is one the model does not fit. */
    function drawTernary(r) {
      var host = $("doe-contour");
      host.innerHTML = "";
      var T = r.ternary;
      if (!T) return;
      var vals = [];
      T.grades.forEach(function (g) {
        g.value.forEach(function (v) { if (v !== null && isFinite(v)) vals.push(v); });
      });
      T.points.forEach(function (p) { if (isFinite(p.value)) vals.push(p.value); });
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      var span = (hi - lo) || 1;
      /* Wide side margins: the corner labels sit outside the triangle. */
      var W = 340, padX = 62, s = W - 2 * padX, H = Math.round(s * 0.87) + 60;
      function X(x) { return padX + x * s; }
      function Y(y) { return H - 34 - y * s; }
      var NS = "http://www.w3.org/2000/svg";
      var row = el("div", { class: "trace-row" });

      T.grades.forEach(function (g) {
        var svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", "0 0 " + W + " " + H);
        svg.setAttribute("width", W); svg.setAttribute("height", H);
        T.triangles.forEach(function (t) {
          var v = (g.value[t[0]] + g.value[t[1]] + g.value[t[2]]) / 3;
          if (!isFinite(v)) return;
          var d = t.map(function (k, i) {
            return (i ? "L" : "M") + X(T.xy[k][0]).toFixed(1) + " " + Y(T.xy[k][1]).toFixed(1);
          }).join(" ") + "Z";
          var path = api.svgEl("path", { d: d, fill: rampColour((v - lo) / span),
                                         stroke: rampColour((v - lo) / span), "stroke-width": 0.5 });
          svg.appendChild(path);
        });
        T.ticks.forEach(function (t) {
          svg.appendChild(api.svgEl("line", {
            x1: X(t.ends[0][0]), y1: Y(t.ends[0][1]), x2: X(t.ends[1][0]), y2: Y(t.ends[1][1]),
            stroke: "#ffffff", "stroke-width": 0.5, opacity: 0.55
          }));
          var e = t.component === "hpmc" ? t.ends[1] : t.ends[0];
          var cx = 0.5, cy = Math.sqrt(3) / 6, dx = e[0] - cx, dy = e[1] - cy;
          var n = Math.sqrt(dx * dx + dy * dy) || 1;
          var lab = api.svgEl("text", { x: X(e[0] + dx / n * 0.05), y: Y(e[1] + dy / n * 0.05) + 3,
            "font-size": 8, "text-anchor": "middle", fill: "#5d6879" });
          lab.textContent = trim(t.value);
          svg.appendChild(lab);
        });
        var C = [[0, 0], [1, 0], [0.5, Math.sqrt(3) / 2]];
        svg.appendChild(api.svgEl("path", {
          d: "M" + C.map(function (c) { return X(c[0]) + " " + Y(c[1]); }).join("L") + "Z",
          fill: "none", stroke: "#1c2330", "stroke-width": 1
        }));
        T.corners.forEach(function (c, i) {
          var t = api.svgEl("text", {
            x: X(C[i][0]) + (i === 0 ? -4 : i === 1 ? 4 : 0),
            y: Y(C[i][1]) + (i === 2 ? -8 : 16),
            "font-size": 10, "font-weight": 600, fill: "#1c2330",
            "text-anchor": i === 0 ? "end" : i === 1 ? "start" : "middle"
          });
          t.textContent = c.label + " " + trim(c.max_wt) + "%";
          svg.appendChild(t);
        });
        T.points.filter(function (p) { return p.grade === g.grade; }).forEach(function (p) {
          var dot = api.svgEl("circle", { cx: X(p.x), cy: Y(p.y), r: 5,
            fill: rampColour((p.value - lo) / span), stroke: "#1c2330", "stroke-width": 1 });
          var tt = api.svgEl("title", {});
          tt.textContent = "measured " + fmt(p.value, 1) + " " + r.units;
          dot.appendChild(tt);
          svg.appendChild(dot);
        });
        var cell = el("div", { class: "chart" });
        cell.appendChild(el("div", { class: "panel-title" }, g.grade));
        cell.appendChild(svg);
        row.appendChild(cell);
      });
      host.appendChild(row);

      var scale = el("div", { class: "legend" });
      var swatches = "";
      for (var k = 0; k <= 10; k++) {
        swatches += '<i style="width:16px;background:' + rampColour(k / 10) + '"></i>';
      }
      scale.innerHTML = "<span>" + fmt(lo, 1) + " " + swatches + " " + fmt(hi, 1) +
        " " + esc(r.units) + "</span>" +
        "<span>Background: fitted model. Dots: measured blends, filled with their " +
        "measured value. Corners: each component at its highest possible level, the " +
        "others at their lowest tested. Blank: outside the tested region.</span>";
      host.appendChild(scale);
    }

    function trim(v) {
      return Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : String(Number(v.toPrecision(3)));
    }

    /* Predicted vs actual, by grade: the standard check of the model. */
    function drawFit(r) {
      var host = $("doe-fit");
      host.innerHTML = "";
      if (!r.fit || !r.fit.pairs.length) return;
      var P = r.fit.pairs;
      var all = [];
      P.forEach(function (p) { all.push(p.measured, p.predicted); });
      var dom = extent(all, 0.05);
      var ch = new Chart(420, 380, { l: 56, r: 16, t: 16, b: 44 });
      ch.scales(dom, dom).axes("predicted " + r.label.toLowerCase(), "measured");
      ch.line(dom, dom, "#5d6879", { dash: "5 4", width: 1 });
      var grades = [];
      P.forEach(function (p) { if (grades.indexOf(p.grade) < 0) grades.push(p.grade); });
      var order = ["K100LV", "K4M", "K100M"];
      grades.sort(function (a, b) {
        var ia = order.indexOf(a), ib = order.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || (a < b ? -1 : 1);
      });
      grades.forEach(function (g, i) {
        var pts = P.filter(function (p) { return p.grade === g; });
        ch.dots(pts.map(function (p) { return p.predicted; }),
                pts.map(function (p) { return p.measured; }), colourFor(g, i), 4,
                pts.map(function (p) {
                  return "case " + p.case + " / " + g + ": measured " + fmt(p.measured, 1) +
                    ", predicted " + fmt(p.predicted, 1);
                }));
      });
      ch.mount(host);
      var lg = el("div", { class: "legend" });
      lg.innerHTML = grades.map(function (g, i) {
        return '<span><i style="background:' + colourFor(g, i) + '"></i>' + esc(g) + "</span>";
      }).join("") + "<span>dashed: 1:1, a perfect prediction</span>";
      host.appendChild(lg);
      $("doe-take-fit").innerHTML = "R² " + fmt(r.fit.r2, 2) + " describes how closely the " +
        "model follows these points; predicted R² " + fmt(r.fit.pred_r2, 2) + " (leave-one-out) " +
        "how well it would predict a formulation it has not seen. " +
        (r.fit.pred_r2 !== null && r.fit.pred_r2 < 0.5
          ? "<b>Below 0.5: do not read this response's contours or traces as predictions.</b>"
          : "");
    }

    /* Piepel traces, per grade, inside the tested region only. Supporting
     * view: which component moves the response, along one direction. */
    function drawPiepel(r) {
      var host = $("doe-piepel");
      host.innerHTML = "";
      if (!r.piepel || !r.piepel.length) return;
      var all = [], xs = [];
      r.piepel.forEach(function (p) {
        p.traces.forEach(function (t) {
          t.y.forEach(function (v, i) { if (v !== null && isFinite(v)) { all.push(v); xs.push(t.x[i]); } });
        });
      });
      var yd = extent(all), xd = extent(xs, 0.02);
      var names = { api: "API", hpmc: "HPMC", lactose: "Lactose" };
      var row = el("div", { class: "trace-row" });
      r.piepel.forEach(function (p) {
        var ch = new Chart(330, 260, { l: 52, r: 10, t: 22, b: 40 });
        ch.scales(xd, yd).axes("component (wt%)", r.label + " (" + r.units + ")");
        p.traces.forEach(function (t, i) {
          ch.line(t.x, t.y.map(function (v) { return v === null ? NaN : v; }),
                  FALLBACK[i % FALLBACK.length], { width: 2 });
        });
        var title = api.svgEl("text", { x: 56, y: 14, "font-size": 12, "font-weight": 600,
                                        fill: "#1c2330" });
        title.textContent = p.grade;
        ch.svg.appendChild(title);
        var cell = el("div", { class: "chart" });
        ch.mount(cell);
        row.appendChild(cell);
      });
      host.appendChild(row);
      var lg = el("div", { class: "legend" });
      lg.innerHTML = r.piepel[0].traces.map(function (t, i) {
        return '<span><i style="background:' + FALLBACK[i % FALLBACK.length] + '"></i>' +
          names[t.component] + " rising</span>";
      }).join("") + "<span>model prediction from the average tested blend; not measured data</span>";
      host.appendChild(lg);
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

      $("doe-take-contour").innerHTML = prose(r.takeaways.contour);
      $("doe-take-effects").innerHTML = prose(r.takeaways.effects);
      $("doe-take-anova").innerHTML = prose(r.takeaways.anova)
        .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

      drawTernary(r);
      drawFit(r);
      drawPiepel(r);
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
