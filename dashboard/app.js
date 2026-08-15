/* CR Matrix Tablet workbench.
 *
 * No external libraries, no fetch(), no network of any kind. Charts are SVG
 * built by hand, so the page works when index.html is double-clicked straight
 * off the filesystem (G7). Data arrives via window.CR_DATA from data.js.
 */
(function () {
  "use strict";

  var D = window.CR_DATA;
  var GRADE_COLOUR = { K100LV: "#3b7dd8", K4M: "#d9822b", K100M: "#8e5bb5" };
  var FALLBACK = ["#3b7dd8", "#d9822b", "#8e5bb5", "#3aa17e", "#c0504d", "#7a8494"];

  function colourFor(grade, i) {
    return GRADE_COLOUR[grade] || FALLBACK[(i || 0) % FALLBACK.length];
  }
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    if (attrs) { Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); }); }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }
  /* A missing number renders as an em dash, never as "nan" or "NaN%": a
   * prediction shown beside the literal text "nan% error" looks like a broken
   * page, and worse, invites the reader to ignore the error estimate entirely
   * (G6 requires one to be present and legible). */
  function fmt(v, dp) {
    if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) return "—";
    if (typeof v !== "number") return String(v);
    return v.toFixed(dp === undefined ? 3 : dp);
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* ---------------------------------------------------------------- SVG */
  var NS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function Chart(width, height, pad) {
    this.w = width; this.h = height;
    this.pad = pad || { l: 52, r: 14, t: 14, b: 40 };
    this.svg = svgEl("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height });
  }
  Chart.prototype.scales = function (xd, yd) {
    var p = this.pad;
    this.xd = xd; this.yd = yd;
    this.px = function (x) {
      return p.l + (x - xd[0]) / (xd[1] - xd[0] || 1) * (this.w - p.l - p.r);
    };
    this.py = function (y) {
      return this.h - p.b - (y - yd[0]) / (yd[1] - yd[0] || 1) * (this.h - p.t - p.b);
    };
    return this;
  };
  Chart.prototype.axes = function (xlabel, ylabel, xticks, yticks) {
    var p = this.pad, self = this, i;
    var xs = xticks || niceTicks(this.xd[0], this.xd[1], 6);
    var ys = yticks || niceTicks(this.yd[0], this.yd[1], 5);
    for (i = 0; i < ys.length; i++) {
      var y = this.py(ys[i]);
      this.svg.appendChild(svgEl("line", {
        x1: p.l, x2: this.w - p.r, y1: y, y2: y, stroke: "#e6eaf1", "stroke-width": 1
      }));
      var tl = svgEl("text", { x: p.l - 7, y: y + 3.5, "text-anchor": "end", "font-size": 10, fill: "#5d6879" });
      tl.textContent = trimNum(ys[i]);
      this.svg.appendChild(tl);
    }
    for (i = 0; i < xs.length; i++) {
      var x = this.px(xs[i]);
      var t = svgEl("text", { x: x, y: this.h - p.b + 15, "text-anchor": "middle", "font-size": 10, fill: "#5d6879" });
      t.textContent = trimNum(xs[i]);
      this.svg.appendChild(t);
    }
    this.svg.appendChild(svgEl("line", {
      x1: p.l, x2: this.w - p.r, y1: this.h - p.b, y2: this.h - p.b, stroke: "#b9c2d0"
    }));
    this.svg.appendChild(svgEl("line", {
      x1: p.l, x2: p.l, y1: p.t, y2: this.h - p.b, stroke: "#b9c2d0"
    }));
    var xt = svgEl("text", { x: (p.l + this.w - p.r) / 2, y: this.h - 6, "text-anchor": "middle", "font-size": 11, fill: "#3d4757" });
    xt.textContent = xlabel; this.svg.appendChild(xt);
    var yt = svgEl("text", {
      x: 12, y: (p.t + this.h - p.b) / 2, "text-anchor": "middle", "font-size": 11, fill: "#3d4757",
      transform: "rotate(-90 12 " + ((p.t + this.h - p.b) / 2) + ")"
    });
    yt.textContent = ylabel; this.svg.appendChild(yt);
    void self;
    return this;
  };
  Chart.prototype.line = function (xs, ys, colour, opts) {
    opts = opts || {};
    var d = "", i, started = false;
    for (i = 0; i < xs.length; i++) {
      if (ys[i] === null || ys[i] === undefined || !isFinite(ys[i])) continue;
      d += (started ? "L" : "M") + this.px(xs[i]).toFixed(2) + " " + this.py(ys[i]).toFixed(2) + " ";
      started = true;
    }
    if (!started) return this;
    var path = svgEl("path", {
      d: d, fill: "none", stroke: colour, "stroke-width": opts.width || 1.7,
      "stroke-linejoin": "round", "stroke-linecap": "round"
    });
    if (opts.dash) path.setAttribute("stroke-dasharray", opts.dash);
    if (opts.opacity) path.setAttribute("opacity", opts.opacity);
    if (opts.title) { var tt = svgEl("title", {}); tt.textContent = opts.title; path.appendChild(tt); }
    this.svg.appendChild(path);
    return this;
  };
  Chart.prototype.dots = function (xs, ys, colour, r, titles) {
    for (var i = 0; i < xs.length; i++) {
      if (!isFinite(ys[i])) continue;
      var c = svgEl("circle", {
        cx: this.px(xs[i]).toFixed(2), cy: this.py(ys[i]).toFixed(2), r: r || 2.6,
        fill: colour, stroke: "#fff", "stroke-width": 0.7
      });
      if (titles && titles[i]) { var t = svgEl("title", {}); t.textContent = titles[i]; c.appendChild(t); }
      this.svg.appendChild(c);
    }
    return this;
  };
  Chart.prototype.hline = function (y, colour, dash) {
    this.svg.appendChild(svgEl("line", {
      x1: this.pad.l, x2: this.w - this.pad.r, y1: this.py(y), y2: this.py(y),
      stroke: colour, "stroke-width": 1, "stroke-dasharray": dash || "4 3"
    }));
    return this;
  };
  Chart.prototype.vline = function (x, colour, dash) {
    this.svg.appendChild(svgEl("line", {
      x1: this.px(x), x2: this.px(x), y1: this.pad.t, y2: this.h - this.pad.b,
      stroke: colour, "stroke-width": 1, "stroke-dasharray": dash || "4 3"
    }));
    return this;
  };
  Chart.prototype.rect = function (x0, y0, x1, y1, fill, opacity) {
    this.svg.appendChild(svgEl("rect", {
      x: Math.min(this.px(x0), this.px(x1)), y: Math.min(this.py(y0), this.py(y1)),
      width: Math.abs(this.px(x1) - this.px(x0)), height: Math.abs(this.py(y1) - this.py(y0)),
      fill: fill, opacity: opacity === undefined ? 0.25 : opacity
    }));
    return this;
  };
  Chart.prototype.mount = function (node) {
    node.innerHTML = "";
    node.appendChild(this.svg);
    return this;
  };

  function niceTicks(lo, hi, count) {
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
    var span = hi - lo;
    var step = Math.pow(10, Math.floor(Math.log(span / count) / Math.LN10));
    var err = span / count / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + step * 0.5; v += step) out.push(Math.round(v / step) * step);
    return out;
  }
  function trimNum(v) {
    if (Math.abs(v) >= 1000) return String(Math.round(v));
    if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
    return String(Number(v.toPrecision(3)));
  }
  function extent(values, padFrac) {
    var lo = Infinity, hi = -Infinity;
    values.forEach(function (v) { if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } });
    if (!isFinite(lo)) return [0, 1];
    var pad = (hi - lo) * (padFrac === undefined ? 0.06 : padFrac);
    if (pad === 0) pad = Math.abs(hi) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }

  /* ------------------------------------------------- model evaluation */
  /* Delegated to model.js so the browser and the parity test share one
   * implementation. See dashboard/model.js. */
  var M = window.CRModel;

  function viscLevels() {
    return D.design_points.map(function (p) { return p.log10_visc; });
  }
  function codeProcess(log10visc) { return M.codeProcess(log10visc, viscLevels()); }
  function predictProfile(apiWt, hpmcWt, lacWt, log10visc, times) {
    return M.predictProfile(D.surfaces, viscLevels(), apiWt, hpmcWt, lacWt, log10visc, times);
  }

  function inHull(apiWt, hpmcWt) {
    /* Convex hull membership of the tested compositions, by cross-product sign. */
    var pts = [];
    var seen = {};
    D.design_points.forEach(function (p) {
      var k = p.case;
      if (seen[k]) return;
      seen[k] = 1;
      pts.push([p.api_wt, p.hpmc_wt]);
    });
    var hull = convexHull(pts);
    for (var i = 0; i < hull.length; i++) {
      var a = hull[i], b = hull[(i + 1) % hull.length];
      var cross = (b[0] - a[0]) * (hpmcWt - a[1]) - (b[1] - a[1]) * (apiWt - a[0]);
      if (cross < -1e-9) return false;
    }
    return true;
  }
  function convexHull(points) {
    var p = points.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    if (p.length < 3) return p;
    function half(src) {
      var out = [];
      for (var i = 0; i < src.length; i++) {
        while (out.length >= 2) {
          var o = out[out.length - 2], q = out[out.length - 1];
          if ((q[0] - o[0]) * (src[i][1] - o[1]) - (q[1] - o[1]) * (src[i][0] - o[0]) <= 0) out.pop();
          else break;
        }
        out.push(src[i]);
      }
      out.pop();
      return out;
    }
    return half(p).concat(half(p.reverse()));
  }

  function gradeOptions(select) {
    var seen = {};
    D.design_points.forEach(function (p) { seen[p.grade] = p.log10_visc; });
    Object.keys(seen).sort(function (a, b) { return seen[a] - seen[b]; }).forEach(function (g) {
      var o = el("option", { value: g }, g + " (" + Math.round(Math.pow(10, seen[g])).toLocaleString() + " cP)");
      o.dataset.lv = seen[g];
      select.appendChild(o);
    });
  }

  function table(columns, rows) {
    var t = el("table");
    var thead = el("thead"), tr = el("tr");
    columns.forEach(function (c) {
      tr.appendChild(el("th", c.num ? { class: "num" } : null, c.label));
    });
    thead.appendChild(tr); t.appendChild(thead);
    var tb = el("tbody");
    rows.forEach(function (r) {
      var row = el("tr");
      columns.forEach(function (c) {
        var td = el("td", c.num ? { class: "num" } : null);
        var v = r[c.key];
        if (c.html) { td.innerHTML = v; } else { td.textContent = v === null || v === undefined ? "—" : v; }
        row.appendChild(td);
      });
      tb.appendChild(row);
    });
    t.appendChild(tb);
    return t;
  }

  function profileFor(cs, grade) {
    for (var i = 0; i < D.profiles.length; i++) {
      if (D.profiles[i].case === cs && D.profiles[i].grade === grade) return D.profiles[i];
    }
    return null;
  }

  /* ------------------------------------------------------------ tabs */
  var TABS = [
    ["explorer", "Database Explorer"],
    ["metrics", "Analysis & Metrics"],
    ["design", "Design Diagnostics"],
    ["surfaces", "Response Surfaces"],
    ["equivalence", "Equivalence Sets"],
    ["stress", "Design Stress Test"],
    ["formulator", "Formulator Tool"],
    ["guidelines", "Guidelines & Limitations"]
  ];

  function buildTabs() {
    var nav = $("tabs");
    TABS.forEach(function (t, i) {
      var b = el("button", { role: "tab", "data-target": t[0], "aria-selected": i === 0 ? "true" : "false" }, t[1]);
      b.addEventListener("click", function () { showTab(t[0]); });
      nav.appendChild(b);
    });
  }
  function showTab(name) {
    Array.prototype.forEach.call(document.querySelectorAll(".panel"), function (p) {
      p.hidden = p.getAttribute("data-tab") !== name;
    });
    Array.prototype.forEach.call(document.querySelectorAll("#tabs button"), function (b) {
      b.setAttribute("aria-selected", b.getAttribute("data-target") === name ? "true" : "false");
    });
  }

  function goToSource(cs, grade) {
    showTab("explorer");
    $("ex-grade").value = grade;
    renderExplorer();
    var target = document.getElementById("row-" + cs + "-" + grade);
    if (target) {
      target.scrollIntoView({ block: "center" });
      target.style.background = "#fff3cd";
      setTimeout(function () { target.style.background = ""; }, 2200);
    }
  }
  window.CR_goToSource = goToSource;

  function sourceLink(cs, grade, label) {
    return '<button class="src-link" onclick="CR_goToSource(' + cs + ',\'' + grade + '\')">' +
      esc(label || ("case " + cs + " / " + grade)) + "</button>";
  }

  /* ------------------------------------------------------- explorer */
  function renderExplorer() {
    var grade = $("ex-grade").value;
    var cens = $("ex-censoring").value;
    var showReps = $("ex-replicates").checked;
    var rows = D.profiles.filter(function (p) {
      return (!grade || p.grade === grade) && (!cens || p.censoring === cens);
    });

    var ch = new Chart(560, 340);
    ch.scales([0, D.meta.plot_max_time_h], [0, 105]).axes("time (h)", "% released");
    ch.hline(D.meta.censoring_pct, "#b4541f", "5 4");
    rows.forEach(function (p, i) {
      if (showReps) {
        p.replicates.forEach(function (r) {
          ch.line(D.grid_h, r.pct, colourFor(p.grade, i), { width: 0.8, opacity: 0.45 });
        });
      }
      ch.line(D.grid_h, p.mean_pct, colourFor(p.grade, i), {
        width: 1.6, title: "case " + p.case + " / " + p.grade
      });
    });
    ch.mount($("ex-chart"));

    var data = rows.map(function (p) {
      return {
        cs: p.case, grade: p.grade,
        api: fmt(p.api_wt, 1), hpmc: fmt(p.hpmc_wt, 1), lac: fmt(p.lactose_wt, 1),
        td: p.log10_td === null ? "—" : fmt(Math.pow(10, p.log10_td), 2),
        peak: fmt(p.mean_pct[p.mean_pct.length - 1], 1),
        cens: '<span class="pill ' + p.censoring + '">' + p.censoring + "</span>",
        ident: p.asymptote_identified ? "yes" : '<span class="pill flag">no</span>'
      };
    });
    var t = table([
      { key: "cs", label: "case", num: true }, { key: "grade", label: "grade" },
      { key: "api", label: "API%", num: true }, { key: "hpmc", label: "HPMC%", num: true },
      { key: "lac", label: "lactose%", num: true },
      { key: "td", label: "Td (h)", num: true }, { key: "peak", label: "% at 24 h", num: true },
      { key: "cens", label: "censoring", html: true },
      { key: "ident", label: "asymptote identified", html: true }
    ], data);
    Array.prototype.forEach.call(t.querySelectorAll("tbody tr"), function (tr, i) {
      tr.id = "row-" + data[i].cs + "-" + data[i].grade;
    });
    $("ex-table").innerHTML = "";
    $("ex-table").appendChild(t);
  }

  /* -------------------------------------------------------- metrics */
  function renderMetrics() {
    var q = D.quality;
    var cards = [
      ["formulations", q.n_ids], ["replicates each", q.n_replicates],
      ["timepoints", q.timepoints_h.length],
      ["never reach " + D.meta.censoring_pct + "%", q.fully_censored_ids.length],
      ["partially censored", q.partially_censored_ids.length],
      ["values > 100%", q.values_above_100],
      ["monotonicity steps", q.monotonicity_violations],
      ["min Peppas points", q.peppas_min_points]
    ];
    var wrap = $("metrics-quality"); wrap.innerHTML = "";
    cards.forEach(function (c) {
      var d = el("div", { class: "card" });
      d.appendChild(el("small", null, c[0]));
      d.appendChild(el("b", null, c[1]));
      wrap.appendChild(d);
    });

    var rows = D.design_points.map(function (p) {
      return {
        cs: p.case, grade: p.grade,
        td: fmt(Math.pow(10, p.log10_td_mean), 2),
        tdsd: fmt(p.log10_td_sd, 3),
        beta: fmt(p.weibull_beta_mean, 3),
        finf: fmt(p.weibull_f_inf_mean, 1),
        t50: p.censoring_status === "full" ? "—" : fmt(p.t50_mean, 2),
        mdt: fmt(p.mdt_h_mean, 2),
        cens: '<span class="pill ' + p.censoring_status + '">' + p.censoring_status + "</span>",
        src: sourceLink(p.case, p.grade, "view")
      };
    });
    $("metrics-table").innerHTML = "";
    $("metrics-table").appendChild(table([
      { key: "cs", label: "case", num: true }, { key: "grade", label: "grade" },
      { key: "td", label: "Td (h)", num: true },
      { key: "tdsd", label: "SD log₁₀Td (replicate)", num: true },
      { key: "beta", label: "β", num: true }, { key: "finf", label: "F∞ %", num: true },
      { key: "t50", label: "t50 (h)", num: true }, { key: "mdt", label: "MDT (h)", num: true },
      { key: "cens", label: "censoring", html: true },
      { key: "src", label: "source", html: true }
    ], rows));
  }

  /* --------------------------------------------------------- design */
  function renderDesign() {
    var m = D.quality.mixture;
    $("design-mixture").innerHTML =
      "<b>A1 — mixture constraint.</b> " + esc(m.verdict) +
      "<br>Component totals deviate by at most " + m.max_deviation +
      "; composition matrix rank <b>" + m.rank + "</b>, singular values [" +
      m.singular_values.map(function (v) { return fmt(v, 3); }).join(", ") + "].";

    var rows = Object.keys(D.design_models).filter(function (k) { return k.indexOf("singular") !== 0; })
      .map(function (k) {
        var d = D.design_models[k];
        return {
          model: k, terms: d.terms, rank: d.rank,
          est: d.estimable ? "yes" : '<span class="pill flag">no</span>',
          cond: fmt(d.condition_number, 1), dcrit: fmt(d.d_criterion, 4),
          geff: fmt(d.g_efficiency, 1), lev: fmt(d.max_leverage, 3)
        };
      });
    $("design-models").innerHTML = "";
    $("design-models").appendChild(table([
      { key: "model", label: "model" }, { key: "terms", label: "terms", num: true },
      { key: "rank", label: "rank", num: true }, { key: "est", label: "estimable", html: true },
      { key: "cond", label: "cond(X)", num: true }, { key: "dcrit", label: "D-criterion", num: true },
      { key: "geff", label: "G-eff %", num: true }, { key: "lev", label: "max leverage", num: true }
    ], rows));

    var sing = D.design_models["singular_intercept_plus_three_components"];
    var alias = $("design-alias"); alias.innerHTML = "";
    if (sing) {
      var c = el("div", { class: "callout danger" });
      c.innerHTML = "<b>The singular model.</b> An intercept alongside all three components lists " +
        sing.terms + " terms but has rank <b>" + sing.rank + "</b> — " + (sing.terms - sing.rank) +
        " exact linear dependencies. Because the components sum to a constant, the intercept is " +
        "exactly the scaled sum of the linear terms, and every squared term is a combination of " +
        "linear and cross terms. This is why the Scheffé form carries no squared terms.";
      alias.appendChild(c);
    }
    var quad = D.design_models["scheffe_quadratic"];
    if (quad && quad.aliases.length) {
      var rows2 = quad.aliases.map(function (a) {
        var loads = Object.keys(a.loads_on).map(function (k) {
          return k + " " + (a.loads_on[k] > 0 ? "+" : "") + fmt(a.loads_on[k], 3);
        }).join(", ");
        return { term: a.term, kind: a.exact ? "exact" : "partial", loads: loads };
      });
      var w = el("div", { class: "tablewrap" });
      w.appendChild(table([
        { key: "term", label: "term outside the fitted model" },
        { key: "kind", label: "alias" }, { key: "loads", label: "biases these coefficients" }
      ], rows2));
      alias.appendChild(w);
    }

    $("design-lof").innerHTML = "<b>Lack of fit.</b> " + esc(D.lack_of_fit.description);

    var fdsKey = D.design_models["scheffe_linear"] ? "scheffe_linear" : "scheffe_quadratic";
    var fds = D.design_models[fdsKey];
    if (fds && fds.fds.length) {
      var ch = new Chart(560, 300);
      var xs = fds.fds.map(function (_, i) { return i / (fds.fds.length - 1); });
      ch.scales([0, 1], extent(fds.fds, 0.08)).axes("fraction of design space", "scaled prediction variance");
      ch.line(xs, fds.fds, "#3aa17e", { width: 2 });
      ch.mount($("design-fds"));
    }
  }

  /* ------------------------------------------------------- surfaces */
  function renderSurfaces() {
    var name = $("sf-response").value;
    var s = D.surfaces[name];
    if (!s) return;

    var cards = [
      ["R²", fmt(s.r2, 4)], ["adjusted R²", fmt(s.adj_r2, 4)],
      ["predicted R² (PRESS)", fmt(s.pred_r2, 4)], ["RMSE", fmt(s.rmse, 4)],
      ["adequate precision", fmt(s.adequate_precision, 1)], ["residual df", s.df_residual]
    ];
    var wrap = $("sf-summary"); wrap.innerHTML = "";
    cards.forEach(function (c) {
      var d = el("div", { class: "card" });
      d.appendChild(el("small", null, c[0]));
      d.appendChild(el("b", null, c[1]));
      wrap.appendChild(d);
    });
    if (s.r2_gap_flagged || s.adequate_precision_flagged) {
      var f = el("div", { class: "card" });
      f.innerHTML = "<small>flags</small><b>" +
        (s.r2_gap_flagged ? '<span class="pill flag">adj–pred gap</span> ' : "") +
        (s.adequate_precision_flagged ? '<span class="pill flag">low precision</span>' : "") + "</b>";
      wrap.appendChild(f);
    }

    var bc = s.box_cox;
    $("sf-boxcox").innerHTML = "<b>Box–Cox.</b> λ = " + fmt(bc.lambda, 3) +
      (bc.ci_low !== null ? " (95% CI " + fmt(bc.ci_low, 3) + " … " + fmt(bc.ci_high, 3) + ")" : "") +
      " — " + esc(bc.recommendation) + (bc.note ? "<br><small>" + esc(bc.note) + "</small>" : "") +
      "<br><small>Model reduction: " + esc(s.rationale) + "</small>";

    $("sf-sequential").innerHTML = "";
    $("sf-sequential").appendChild(table([
      { key: "model", label: "sequential model" }, { key: "terms", label: "terms", num: true },
      { key: "added", label: "added", num: true }, { key: "ss", label: "sum of squares", num: true },
      { key: "f", label: "F", num: true }, { key: "p", label: "p", num: true }
    ], s.sequential.map(function (r) {
      return { model: r.model, terms: r.terms, added: r.added, ss: fmt(r.ss, 4), f: fmt(r.f, 2), p: fmt(r.p, 4) };
    })));

    $("sf-coefficients").innerHTML = "";
    $("sf-coefficients").appendChild(table([
      { key: "name", label: "term" }, { key: "est", label: "estimate", num: true },
      { key: "se", label: "std error", num: true }, { key: "t", label: "t", num: true },
      { key: "p", label: "p", num: true }, { key: "ci", label: "95% CI" }
    ], s.coefficients.slice().sort(function (a, b) { return a.p - b.p; }).map(function (c) {
      return {
        name: c.name, est: fmt(c.estimate, 4), se: fmt(c.std_error, 4),
        t: fmt(c.t, 2), p: c.p < 0.0001 ? "<0.0001" : fmt(c.p, 4),
        ci: fmt(c.ci_low, 3) + " … " + fmt(c.ci_high, 3)
      };
    })));

    var ch = new Chart(430, 300);
    ch.scales(extent(s.fitted), extent(s.studentised)).axes("fitted", "externally studentised residual");
    ch.hline(0, "#b9c2d0", "3 3");
    ch.dots(s.fitted, s.studentised, "#2f6fd0", 3.2, D.design_points.map(function (p) {
      return "case " + p.case + " / " + p.grade;
    }));
    ch.mount($("sf-residuals"));

    var ch2 = new Chart(430, 300);
    var idx = s.leverage.map(function (_, i) { return i + 1; });
    ch2.scales([0, idx.length + 1], extent(s.cooks_distance.concat([0]))).axes("design point", "Cook's distance");
    ch2.dots(idx, s.cooks_distance, "#8e5bb5", 3.2, D.design_points.map(function (p) {
      return "case " + p.case + " / " + p.grade;
    }));
    ch2.mount($("sf-leverage"));

    var lev = D.levers;
    var first = lev[0], last = lev[lev.length - 1];
    var loss = 1 - (last.fold_change_td - 1) / Math.max(first.fold_change_td - 1, 1e-9);
    var html = '<div class="callout ok"><b>The two levers are not additive.</b> Adding 10 wt% ' +
      "HPMC in place of lactose multiplies Td by ×" + fmt(first.fold_change_td, 2) + " at " +
      first.grade + ", but only ×" + fmt(last.fold_change_td, 2) + " at " + last.grade +
      " — about " + Math.round(loss * 100) + "% less effect. Reaching for both levers at once " +
      "buys less than the sum of their separate effects.</div>";
    var t = table([
      { key: "grade", label: "grade" }, { key: "visc", label: "viscosity (cP)", num: true },
      { key: "d", label: "Δ log₁₀(Td)", num: true }, { key: "f", label: "Td multiplier", num: true }
    ], lev.map(function (r) {
      return {
        grade: r.grade, visc: Math.round(r.viscosity_cp).toLocaleString(),
        d: fmt(r.delta_log10_td, 4), f: "×" + fmt(r.fold_change_td, 2)
      };
    }));
    $("sf-levers").innerHTML = html;
    var tw = el("div", { class: "tablewrap" }); tw.appendChild(t);
    $("sf-levers").appendChild(tw);
  }

  /* ---------------------------------------------------- equivalence */
  function renderEquivalence() {
    var s = D.equivalence.summary;
    $("eq-headline").innerHTML = "<b>Headline.</b> " + esc(s.demonstration) +
      "<br><br>Targets: " + s.n_targets + " · with a cross-grade equivalent: <b>" +
      s.cross_grade_count + "</b> · largest set: " + s.max_set_size +
      " · median set: " + fmt(s.median_set_size, 0) +
      (s.isolated.length ? " · <b>no equivalent at all:</b> " + s.isolated.map(function (i) {
        return "case " + i.case + "/" + i.grade;
      }).join(", ") : "");

    var target = $("eq-target").value;
    var parts = target.split("|");
    var set = null;
    D.equivalence.sets.forEach(function (e) {
      if (String(e.case) === parts[0] && e.grade === parts[1]) set = e;
    });
    if (!set) return;

    var ch = new Chart(560, 340);
    ch.scales([0, D.meta.plot_max_time_h], [0, 105]).axes("time (h)", "% released");
    set.members.slice(0, 8).forEach(function (m, i) {
      var p = profileFor(m.case, m.grade);
      if (!p) return;
      var isTarget = m.case === set.case && m.grade === set.grade;
      ch.line(D.grid_h, p.mean_pct, colourFor(m.grade, i), {
        width: isTarget ? 2.6 : 1.3, opacity: isTarget ? 1 : 0.8,
        title: "case " + m.case + " / " + m.grade + " (f2 " + fmt(m.f2, 0) + ")"
      });
    });
    ch.mount($("eq-chart"));

    $("eq-table").innerHTML = "";
    $("eq-table").appendChild(table([
      { key: "who", label: "formulation", html: true }, { key: "f2", label: "f2", num: true },
      { key: "api", label: "API%", num: true }, { key: "hpmc", label: "HPMC%", num: true },
      { key: "lac", label: "lactose%", num: true },
      { key: "cv", label: "CV RMSE %", num: true }
    ], set.members.map(function (m) {
      return {
        who: sourceLink(m.case, m.grade), f2: fmt(m.f2, 1),
        api: fmt(m.api_wt, 1), hpmc: fmt(m.hpmc_wt, 1), lac: fmt(m.lactose_wt, 1),
        cv: fmt(m.cv_rmse_pct, 2)
      };
    })));
  }

  /* --------------------------------------------------------- stress */
  function renderStress() {
    if (!D.stress) { $("st-headline").textContent = "Stress test not available."; return; }
    var s = D.stress;
    $("st-headline").innerHTML = "<b>Recommendation.</b> " + esc(s.rationale);

    var usable = s.curve.filter(function (r) { return r.estimable; });
    var ch = new Chart(560, 320);
    ch.scales(extent(usable.map(function (r) { return r.size; }), 0.08),
      extent(usable.map(function (r) { return r.profile_rmse_pct; }).concat([s.full_profile_rmse_pct])))
      .axes("runs in the reduced design", "profile RMSE (% released)");
    ch.line(usable.map(function (r) { return r.size; }),
      usable.map(function (r) { return r.profile_rmse_pct; }), "#2f6fd0", { width: 2 });
    ch.dots(usable.map(function (r) { return r.size; }),
      usable.map(function (r) { return r.profile_rmse_pct; }), "#2f6fd0", 3.4);
    ch.hline(s.full_profile_rmse_pct, "#5d6879", "4 3");
    ch.vline(s.recommended_size, "#a8322f", "3 3");
    ch.mount($("st-chart"));

    $("st-table").innerHTML = "";
    $("st-table").appendChild(table([
      { key: "size", label: "runs", num: true },
      { key: "rmse", label: "profile RMSE %", num: true },
      { key: "worst", label: "worst %", num: true },
      { key: "d", label: "D-criterion", num: true },
      { key: "dir", label: "directions agree", html: true },
      { key: "jac", label: "equivalence agreement", num: true }
    ], s.curve.map(function (r) {
      return {
        size: r.size, rmse: fmt(r.profile_rmse_pct, 3), worst: fmt(r.worst_pct, 3),
        d: fmt(r.d_criterion, 4),
        dir: r.direction_agrees ? "yes" : '<span class="pill flag">no</span>',
        jac: fmt(r.equivalence_jaccard, 3)
      };
    })));

    $("st-points").innerHTML = "";
    $("st-points").appendChild(table([
      { key: "cs", label: "case", num: true }, { key: "grade", label: "grade" },
      { key: "src", label: "measured profile", html: true }
    ], s.recommended_points.map(function (p) {
      return { cs: p.case, grade: p.grade, src: sourceLink(p.case, p.grade, "view") };
    })));
  }

  /* ----------------------------------------------------- formulator */
  function selectedLv(select) {
    var opt = select.options[select.selectedIndex];
    return opt ? Number(opt.dataset.lv) : 0;
  }

  function renderForward() {
    var api = Number($("fw-api").value), hpmc = Number($("fw-hpmc").value), lac = Number($("fw-lac").value);
    var lv = selectedLv($("fw-grade"));
    var dose = Number($("fw-dose").value);
    var total = api + hpmc + lac;

    var warn = $("fw-warn"); warn.innerHTML = "";
    if (Math.abs(total - 100) > 0.51) {
      warn.innerHTML = '<div class="callout warn">Components sum to ' + fmt(total, 1) +
        " wt%, not 100. This is a mixture design: the three cannot be set independently. " +
        "The prediction below normalises them to 100, which may not be the formulation you meant.</div>";
    }
    var hull = inHull(api, hpmc);
    var inVisc = true;
    var times = [];
    for (var t = 0; t <= D.meta.plot_max_time_h; t += 0.25) times.push(t);
    var pred = predictProfile(api, hpmc, lac, lv, times);

    if (!hull) {
      warn.innerHTML += '<div class="callout danger"><b>Outside the tested composition region.</b> ' +
        "This composition lies beyond the convex hull of the 11 tested formulations, so the " +
        "prediction is extrapolation. The nearest measured formulations are listed below — " +
        "trust those, not the curve.</div>";
    }
    if (!inVisc) {
      warn.innerHTML += '<div class="callout danger">Viscosity outside the tested range.</div>';
    }

    var ch = new Chart(540, 330);
    ch.scales([0, D.meta.plot_max_time_h], [0, 105]).axes("time (h)", "% released");
    ch.hline(D.meta.censoring_pct, "#b4541f", "5 4");
    /* Nearest measured neighbours, always shown (G3, G6 traceability). */
    var neigh = nearest(api, hpmc, lv, 3);
    neigh.forEach(function (n, i) {
      var p = profileFor(n.case, n.grade);
      if (p) ch.line(D.grid_h, p.mean_pct, colourFor(n.grade, i), { width: 1.1, opacity: 0.55, dash: "4 3" });
    });
    ch.line(times, pred.curve, "#1c2330", { width: 2.4 });
    ch.mount($("fw-chart"));

    var out = $("fw-out");
    out.innerHTML = "";
    var cv = D.validation.profile_rmse_pct;
    var card = el("div", { class: "callout " + (hull ? "ok" : "danger") });
    card.innerHTML =
      "<b>Predicted</b><br>Td = " + fmt(pred.td, 2) + " h · β = " + fmt(pred.params.weibull_beta, 3) +
      " · F∞ = " + fmt(pred.params.weibull_f_inf, 1) + "%" +
      "<br>t50 ≈ " + (M.t50From(pred.td, pred.params.weibull_beta, pred.params.weibull_f_inf) === null
        ? "never reaches 50% (F∞ = " + fmt(pred.params.weibull_f_inf, 1) + "%)"
        : fmt(M.t50From(pred.td, pred.params.weibull_beta, pred.params.weibull_f_inf), 2) + " h") +
      "<br>dose = " + fmt(dose, 0) + " mg at " + fmt(api, 1) + " wt% API → tablet ≈ " +
      fmt(dose / (api / 100), 0) + " mg" +
      "<br><br><b>Uncertainty (G6).</b> Cross-validated profile RMSE <b>" + fmt(cv, 2) +
      "% released</b> (leave-one-formulation-out over " + D.validation.n_folds +
      " folds, worst fold " + fmt(D.validation.profile_rmse_worst_pct, 2) + "%). " +
      "This is prediction error for an unseen formulation — not the replicate spread, " +
      "which is a much smaller within-batch quantity." +
      "<br><br><b>Design space.</b> " + (hull
        ? "Inside the tested composition hull."
        : "<b>OUTSIDE</b> the tested hull — extrapolation.");
    out.appendChild(card);

    var nt = el("div", { class: "tablewrap" });
    nt.appendChild(table([
      { key: "who", label: "nearest measured", html: true },
      { key: "api", label: "API%", num: true }, { key: "hpmc", label: "HPMC%", num: true },
      { key: "grade", label: "grade" }, { key: "td", label: "Td (h)", num: true }
    ], neigh.map(function (n) {
      var p = profileFor(n.case, n.grade);
      return {
        who: sourceLink(n.case, n.grade), api: fmt(n.api_wt, 1), hpmc: fmt(n.hpmc_wt, 1),
        grade: n.grade, td: p && p.log10_td !== null ? fmt(Math.pow(10, p.log10_td), 2) : "—"
      };
    })));
    out.appendChild(nt);
  }

  function nearest(api, hpmc, lv, k) {
    return D.design_points.map(function (p) {
      var d = Math.pow(p.api_wt - api, 2) + Math.pow(p.hpmc_wt - hpmc, 2) +
        Math.pow((p.log10_visc - lv) * 12, 2);
      return { case: p.case, grade: p.grade, api_wt: p.api_wt, hpmc_wt: p.hpmc_wt, d: d };
    }).sort(function (a, b) { return a.d - b.d; }).slice(0, k);
  }

  /* Derringer–Suich one-sided/target desirability. */
  function desirability(value, target, tolerance, weight) {
    if (!isFinite(value)) return 0;
    var dev = Math.abs(value - target) / Math.max(tolerance, 1e-9);
    if (dev >= 1) return 0;
    return Math.pow(1 - dev, Math.max(weight, 0.01));
  }

  function renderInverse() {
    var tTarget = Number($("iv-t50").value);
    var pTarget = Number($("iv-p24").value);
    var wT = Number($("iv-w-t50").value), wP = Number($("iv-w-p24").value);
    $("iv-w-t50-v").textContent = wT.toFixed(1);
    $("iv-w-p24-v").textContent = wP.toFixed(1);

    var grades = [];
    D.design_points.forEach(function (p) {
      if (!grades.some(function (g) { return g.grade === p.grade; })) {
        grades.push({ grade: p.grade, lv: p.log10_visc });
      }
    });

    var candidates = [];
    for (var a = 10; a <= 60; a += 2.5) {
      for (var h = 20; h <= 60; h += 2.5) {
        var l = 100 - a - h;
        if (l < 5 || l > 70) continue;
        if (!inHull(a, h)) continue;
        grades.forEach(function (g) {
          var pred = predictProfile(a, h, l, g.lv, [24]);
          var t50 = M.t50From(pred.td, pred.params.weibull_beta, pred.params.weibull_f_inf);
          var p24 = pred.curve[0];
          if (t50 === null) return;  // never reaches 50%: not a candidate
          var d1 = desirability(t50, tTarget, Math.max(tTarget * 0.5, 1), wT);
          var d2 = desirability(p24, pTarget, 25, wP);
          var overall = (wT + wP) > 0 ? Math.pow(Math.pow(d1, wT) * Math.pow(d2, wP), 1 / (wT + wP)) : 0;
          if (overall > 0.01) {
            candidates.push({
              api: a, hpmc: h, lac: l, grade: g.grade, t50: t50, p24: p24,
              d: overall, td: pred.td, beta: pred.params.weibull_beta
            });
          }
        });
      }
    }
    candidates.sort(function (x, y) { return y.d - x.d; });

    var out = $("iv-out"); out.innerHTML = "";
    if (!candidates.length) {
      out.innerHTML = '<div class="callout warn">No composition inside the tested design space ' +
        "reaches this target. Widen the target, or treat it as outside what this design can " +
        "support — the tool will not extrapolate to reach it (G3).</div>";
      renderOverlay();
      return;
    }

    /* Present equivalent candidates as a set, with trade-offs, not one answer. */
    var top = candidates.slice(0, 12);
    var best = top[0].d;
    var tied = top.filter(function (c) { return c.d >= best - 0.02; });
    out.innerHTML = '<div class="callout ok"><b>' + tied.length + " near-equivalent candidate" +
      (tied.length === 1 ? "" : "s") + "</b> within 0.02 desirability of the best. They are not " +
      "ranked apart by the data — choose on drug load, cost or compressibility. Every row " +
      "carries the same cross-validated error: <b>" + fmt(D.validation.profile_rmse_pct, 2) +
      "% released</b> (" + D.validation.n_folds + "-fold LOFO).</div>";

    var tw = el("div", { class: "tablewrap" });
    tw.appendChild(table([
      { key: "d", label: "desirability", num: true },
      { key: "api", label: "API%", num: true }, { key: "hpmc", label: "HPMC%", num: true },
      { key: "lac", label: "lactose%", num: true }, { key: "grade", label: "grade" },
      { key: "t50", label: "t50 (h)", num: true }, { key: "p24", label: "% at 24 h", num: true },
      { key: "space", label: "design space" },
      { key: "cv", label: "CV RMSE %", num: true },
      { key: "src", label: "nearest measured", html: true }
    ], top.map(function (c) {
      var n = nearest(c.api, c.hpmc, 0, 1)[0];
      return {
        d: fmt(c.d, 3), api: fmt(c.api, 1), hpmc: fmt(c.hpmc, 1), lac: fmt(c.lac, 1),
        grade: c.grade, t50: fmt(c.t50, 2), p24: fmt(c.p24, 1),
        space: "inside hull", cv: fmt(D.validation.profile_rmse_pct, 2),
        src: sourceLink(n.case, n.grade)
      };
    })));
    out.appendChild(tw);
    renderOverlay();
  }

  function renderOverlay() {
    var tTarget = Number($("iv-t50").value);
    var tol = Number($("iv-tol").value);
    var pTarget = Number($("iv-p24").value);
    var sel = $("iv-grade");
    var lv = selectedLv(sel);

    var ch = new Chart(560, 380, { l: 56, r: 16, t: 16, b: 44 });
    ch.scales([5, 65], [15, 65]).axes("API (wt%)", "HPMC (wt%)");

    var step = 1.25;
    for (var a = 5; a <= 65; a += step) {
      for (var h = 15; h <= 65; h += step) {
        var l = 100 - a - h;
        if (l < 5 || l > 70) continue;
        if (!inHull(a, h)) continue;
        var pred = predictProfile(a, h, l, lv, [24]);
        var t50 = M.t50From(pred.td, pred.params.weibull_beta, pred.params.weibull_f_inf);
        var okT = t50 !== null && Math.abs(t50 - tTarget) <= tol;
        var okP = pred.curve[0] >= pTarget - 10;
        if (okT && okP) ch.rect(a, h, a + step, h + step, "#2f7a55", 0.45);
        else if (okT) ch.rect(a, h, a + step, h + step, "#3b7dd8", 0.16);
        else if (okP) ch.rect(a, h, a + step, h + step, "#d9822b", 0.12);
      }
    }
    D.design_points.filter(function (p) { return p.grade === sel.value; }).forEach(function (p) {
      ch.dots([p.api_wt], [p.hpmc_wt], "#1c2330", 4, ["case " + p.case + " / " + p.grade]);
    });
    ch.mount($("iv-overlay"));
    var legend = el("div", { class: "legend" });
    legend.innerHTML =
      '<span><i style="background:#2f7a55"></i>all constraints met</span>' +
      '<span><i style="background:#3b7dd8;opacity:.5"></i>t50 only</span>' +
      '<span><i style="background:#d9822b;opacity:.5"></i>24 h release only</span>' +
      '<span><i style="background:#1c2330"></i>measured design point</span>' +
      "<span>Blank = outside the tested composition hull; no prediction is offered there (G3).</span>";
    $("iv-overlay").appendChild(legend);
  }

  /* ----------------------------------------------------- guidelines */
  function renderGuidelines() {
    var q = D.quality, space = D.response_space, cv = D.validation;
    var lev = D.levers, first = lev[0], last = lev[lev.length - 1];
    var loss = Math.round((1 - (last.fold_change_td - 1) / Math.max(first.fold_change_td - 1, 1e-9)) * 100);
    var body = $("gl-body");
    body.className = "markdown";
    var singleApi = q.apis.length < 2;

    body.innerHTML =
      "<h3>What the data supports</h3>" +
      "<ul>" +
      "<li><b>The two levers are not additive.</b> Adding 10 wt% HPMC in place of lactose " +
      "multiplies Td by ×" + fmt(first.fold_change_td, 2) + " at " + first.grade + " but only ×" +
      fmt(last.fold_change_td, 2) + " at " + last.grade + " — about " + loss + "% less effect. " +
      "Polymer content and grade are partial substitutes, not cumulative levers.</li>" +
      "<li><b>Formulation freedom is real.</b> " + D.equivalence.summary.cross_grade_count + " of " +
      D.equivalence.summary.n_targets + " measured formulations have an f2-similar counterpart at a " +
      "different grade. A target profile usually admits a set of compositions, not one.</li>" +
      (D.stress ? "<li><b>A smaller design suffices.</b> " + D.stress.recommended_size + " of " +
        D.design_points.length + " runs preserves prediction accuracy and every directional " +
        "conclusion.</li>" : "") +
      "</ul>" +

      "<h3>How far to trust a prediction</h3>" +
      "<ul>" +
      "<li>Leave-one-formulation-out CV: <b>" + fmt(cv.profile_rmse_pct, 2) + "% released RMSE</b>, " +
      "worst fold " + fmt(cv.profile_rmse_worst_pct, 2) + "%, median f2 " + fmt(cv.median_f2, 0) +
      " over " + cv.n_folds + " folds.</li>" +
      "<li>That is the number on every prediction — <b>not</b> the replicate spread, which " +
      "measures vessel repeatability within a single batch and is far smaller.</li>" +
      "<li><b>Batch-to-batch variability is unmeasured.</b> Every formulation was compressed " +
      "once. The tight replicate spread is not evidence that a repeat batch would match.</li>" +
      "</ul>" +

      "<h3>What the data does not support</h3>" +
      "<ul>" +
      "<li><b>Nothing about solubility.</b> The database holds " + q.apis.length + " API" +
      (q.apis.length === 1 ? "" : "s") + ". Every solubility and cross-API feature is gated shut " +
      "until at least two APIs per solubility class exist. Even then the contrast stays " +
      "confounded with molecule identity and can only be directional.</li>" +
      "<li><b>No extrapolation.</b> Predictions hold inside the tested composition hull and " +
      "between the three tested grades. Outside, the tool shows nearest measured formulations " +
      "instead of a number.</li>" +
      "<li><b>The slowest formulations are least characterised.</b> " + q.fully_censored_ids.length +
      " never reach " + D.meta.censoring_pct + "% within 24 h and " + q.partially_censored_ids.length +
      " more straddle it. For these the Weibull asymptote is extrapolated, not estimated — " +
      "censoring and asymptote identifiability coincide exactly in this design.</li>" +
      "<li><b>The metrics are not independent evidence.</b> " + space.metrics.length +
      " metrics collapse to <b>" + space.n_components_90 + "</b> real dimensions. Conclusions " +
      "rest only on the key responses: " + space.key_responses.join(", ") + ".</li>" +
      "<li><b>Lack of fit is anti-conservative.</b> " + esc(D.lack_of_fit.description) + "</li>" +
      (space.coverage_notes || []).map(function (n) {
        return "<li><b>Coverage.</b> " + esc(n) + "</li>";
      }).join("") +
      "</ul>" +

      "<h3>Cross-API and solubility analysis</h3>" +
      (singleApi
        ? '<div class="gate"><b>Insufficient data — requires ≥2 APIs per solubility class.</b><br>' +
        "This section is built and will populate automatically once the database contains at " +
        "least two high-solubility and two low-solubility APIs. With " + q.apis.length +
        " API loaded, no solubility-linked statement can be made, and none is shown. " +
        "When the gate does open at 2×2, results will be labelled as confounded with molecule " +
        "identity and directional only.</div>"
        : '<div class="callout warn">Cross-API contrasts are confounded with molecule identity ' +
        "and are directional only.</div>") +

      "<h3>Design-space boundaries</h3>" +
      "<ul><li>API " + fmt(Math.min.apply(null, D.design_points.map(function (p) { return p.api_wt; })), 0) +
      "–" + fmt(Math.max.apply(null, D.design_points.map(function (p) { return p.api_wt; })), 0) +
      " wt%, HPMC " + fmt(Math.min.apply(null, D.design_points.map(function (p) { return p.hpmc_wt; })), 0) +
      "–" + fmt(Math.max.apply(null, D.design_points.map(function (p) { return p.hpmc_wt; })), 0) +
      " wt%, balance lactose.</li>" +
      "<li>Grades: " + Object.keys(D.quality.grades_per_api).map(function (k) {
        return D.quality.grades_per_api[k].join(", ");
      }).join("; ") + ". Interpolation permitted, extrapolation not.</li>" +
      "<li>Components sum to 100 wt% exactly — a mixture. The three cannot be varied " +
      "independently.</li>" +
      "<li>Observation window 0–" + D.grid_h[D.grid_h.length - 1] + " h across " +
      D.grid_h.length + " timepoints, " + q.samples_within_2h + " inside the first 2 h.</li>" +
      "</ul>";
  }

  /* ----------------------------------------------------------- boot */
  function boot() {
    if (!D) {
      document.body.innerHTML = "<p style='padding:24px'>data.js did not load. " +
        "Regenerate it with <code>python -m pipeline.run --input &lt;file&gt;</code>.</p>";
      return;
    }

    if (D.meta.is_synthetic) {
      var p = $("provenance");
      p.hidden = false;
      p.innerHTML = "PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL" +
        "<small>Every number in this dashboard derives from a generated development " +
        "database. Nothing here is a measurement, and no result may support a formulation " +
        "decision. Source: " + esc(D.meta.source_file) + "</small>";
    }

    $("subtitle").textContent = D.quality.apis.join(", ") + " · " + D.quality.n_ids +
      " formulations × " + D.quality.n_replicates + " replicates · model " + D.meta.model;
    var cvOk = D.validation.profile_rmse_pct !== null &&
      isFinite(D.validation.profile_rmse_pct);
    $("cv-badge").innerHTML = cvOk
      ? "<small>cross-validated error</small><b>" +
        fmt(D.validation.profile_rmse_pct, 2) + "%</b><small>released, LOFO ×" +
        D.validation.n_folds + "</small>"
      : "<small>cross-validated error</small><b>unavailable</b>" +
        "<small>see Guidelines</small>";
    $("foot-meta").textContent = "Generated from " + D.meta.source_file +
      " · vessel " + D.meta.vessel_volume_ml + " mL · viscosity source: " +
      D.meta.viscosity_source + " · seed " + D.meta.seed +
      " · f2 threshold " + D.meta.f2_threshold + " · offline, no network calls";

    buildTabs();

    var grades = {};
    D.design_points.forEach(function (p) { grades[p.grade] = p.log10_visc; });
    var exGrade = $("ex-grade");
    exGrade.appendChild(el("option", { value: "" }, "all grades"));
    Object.keys(grades).sort(function (a, b) { return grades[a] - grades[b]; }).forEach(function (g) {
      exGrade.appendChild(el("option", { value: g }, g));
    });
    gradeOptions($("fw-grade"));
    gradeOptions($("iv-grade"));

    Object.keys(D.surfaces).forEach(function (k) {
      $("sf-response").appendChild(el("option", { value: k }, k));
    });

    D.equivalence.sets.forEach(function (e) {
      $("eq-target").appendChild(el("option", { value: e.case + "|" + e.grade },
        "case " + e.case + " / " + e.grade + " (" + e.members.length + " members)"));
    });

    ["ex-grade", "ex-censoring", "ex-replicates"].forEach(function (id) {
      $(id).addEventListener("change", renderExplorer);
    });
    $("sf-response").addEventListener("change", renderSurfaces);
    $("eq-target").addEventListener("change", renderEquivalence);
    ["fw-api", "fw-hpmc", "fw-lac", "fw-grade", "fw-dose"].forEach(function (id) {
      $(id).addEventListener("input", renderForward);
      $(id).addEventListener("change", renderForward);
    });
    ["iv-t50", "iv-p24", "iv-w-t50", "iv-w-p24"].forEach(function (id) {
      $(id).addEventListener("input", renderInverse);
    });
    ["iv-grade", "iv-tol"].forEach(function (id) {
      $(id).addEventListener("input", renderOverlay);
      $(id).addEventListener("change", renderOverlay);
    });

    renderExplorer();
    renderMetrics();
    renderDesign();
    renderSurfaces();
    renderEquivalence();
    renderStress();
    renderForward();
    renderInverse();
    renderGuidelines();
    showTab("explorer");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
