/* Admin / Audit panel: a privacy-safe account of how the data met the code.
 *
 * Two halves. The pipeline's audit (D.audit, written by pipeline/audit.py) says
 * how the workbook met the analysis. The browser audit below says how the
 * payload met the dashboard: which tabs rendered, which charts came up empty,
 * whether the model evaluates. Both carry ONLY integers and 0/1 flags, so the
 * "Copy report" block can leave the machine without disclosing any data.
 *
 * The guarantee is enforced, not promised: add() throws on anything that is not
 * a boolean or an integer, and labels must be fixed identifiers written here.
 */
(function (root) {
  "use strict";

  var LABEL = /^[a-z][a-z0-9_]*$/;
  var windowErrors = 0;
  var renders = {};   /* tab key -> { ok: bool, errors: int } */

  if (root.addEventListener) {
    root.addEventListener("error", function () { windowErrors += 1; });
  }

  /* Run one tab's render in isolation. A throw in one tab used to abort boot()
   * and leave every later tab blank; now it is counted and the rest carry on. */
  function track(key, fn) {
    var rec = renders[key] || (renders[key] = { ok: true, errors: 0 });
    try {
      fn();
    } catch (e) {
      rec.ok = false;
      rec.errors += 1;
      if (root.console && root.console.error) {
        root.console.error("[CR audit] render failed: " + key, e);
      }
    }
  }

  function Report() { this.entries = []; this.n = 0; this.probeErrors = 0; }
  Report.prototype.add = function (label, value) {
    if (!LABEL.test(label)) throw new TypeError("audit label is not a fixed identifier");
    if (typeof value === "boolean") value = value ? 1 : 0;
    else if (typeof value !== "number" || !isFinite(value) || Math.floor(value) !== value) {
      throw new TypeError("audit values must be integers or booleans");
    }
    this.n += 1;
    this.entries.push({ code: "W" + (this.n < 10 ? "0" : "") + this.n, label: label, value: value });
  };
  /* A probe that throws is counted, never allowed to take the panel down. */
  Report.prototype.probe = function (fn) {
    try { fn(); } catch (e) { this.probeErrors += 1; }
  };

  function isNum(v) { return typeof v === "number" && isFinite(v); }

  function collect(D, tabs) {
    var r = new Report();
    var doc = root.document;

    /* --- rendering ------------------------------------------------------ */
    var failed = 0;
    tabs.forEach(function (t) {
      var rec = renders[t];
      var ok = rec ? rec.ok : false;
      if (!ok) failed += 1;
      r.add("render_" + t, ok);
    });
    r.add("tabs_total", tabs.length);
    r.add("tabs_failed", failed);
    r.add("window_errors", windowErrors);

    r.probe(function () {
      tabs.forEach(function (t) {
        var panel = doc.querySelector('.panel[data-tab="' + t + '"]');
        if (!panel) return;  /* a boot step such as "header", not a tab */
        var charts = panel.querySelectorAll(".chart");
        var empty = 0;
        Array.prototype.forEach.call(charts, function (c) {
          if (!c.querySelector("svg")) empty += 1;
        });
        r.add("panel_" + t + "_svgs", panel.querySelectorAll("svg").length);
        r.add("panel_" + t + "_charts_empty", empty);
        r.add("panel_" + t + "_text_blank", panel.textContent.replace(/\s+/g, "").length < 40);
      });
    });

    /* --- payload shape the dashboard relies on --------------------------- */
    r.probe(function () {
      var grid = D.grid_h || [];
      r.add("grid_points", grid.length);
      r.add("grid_nulls", grid.filter(function (v) { return !isNum(v); }).length);
      r.add("profiles", D.profiles.length);
      var lastNull = 0, anyNull = 0, noBand = 0;
      D.profiles.forEach(function (p) {
        var mean = p.mean_pct || [];
        if (!mean.length || !isNum(mean[mean.length - 1])) lastNull += 1;
        if (mean.some(function (v) { return !isNum(v); })) anyNull += 1;
        if (!p.sd_pct || !p.sd_pct.some(isNum)) noBand += 1;
      });
      r.add("profiles_null_at_last_grid_point", lastNull);
      r.add("profiles_with_any_null", anyNull);
      r.add("profiles_without_sd_band", noBand);
    });
    r.probe(function () {
      r.add("design_points", D.design_points.length);
      r.add("design_points_null_log10_visc", D.design_points.filter(function (p) {
        return !isNum(p.log10_visc);
      }).length);
      var grades = {};
      D.design_points.forEach(function (p) { grades[p.grade] = 1; });
      r.add("grades", Object.keys(grades).length);
    });

    /* --- the forward model evaluates at every tested point -------------- */
    r.probe(function () {
      var M = root.CRModel;
      r.add("model_loaded", !!M);
      if (!M) return;
      var levels = D.design_points.map(function (p) { return p.log10_visc; });
      var bad = 0;
      D.design_points.forEach(function (p) {
        var out = M.predictProfile(D.surfaces, levels, p.api_wt, p.hpmc_wt,
          p.lactose_wt, p.log10_visc, D.grid_h);
        if (!out.curve.every(isNum)) bad += 1;
      });
      r.add("forward_predictions_nonfinite", bad);
      r.add("surfaces", Object.keys(D.surfaces || {}).length);
    });

    /* --- DoE and the formulator goals that read it ----------------------- */
    r.probe(function () {
      var resp = (D.doe && D.doe.responses) || [];
      var keys = {};
      resp.forEach(function (x) { keys[x.key] = 1; });
      r.add("doe_responses", resp.length);
      var missing = 0;
      (D.goals || []).forEach(function (g) {
        if (g.weights.some(function (w) { return !keys[w.response]; })) missing += 1;
      });
      r.add("goals", (D.goals || []).length);
      r.add("goals_with_missing_response", missing);
    });
    r.probe(function () {
      var callout = doc.querySelector("#iv-results .callout.warn");
      r.add("formulator_shows_no_result", !!callout);
      r.add("formulator_result_rows", doc.querySelectorAll("#iv-results tbody tr").length);
      var sums = D.design_points.map(function (p) { return p.api_wt + p.hpmc_wt + p.lactose_wt; });
      r.add("design_points_sum_not_100", sums.filter(function (s) {
        return Math.abs(s - 100) > 0.5;
      }).length);
    });

    /* --- explorer, equivalence, stress --------------------------------- */
    r.probe(function () {
      var rows = doc.querySelectorAll("#ex-table tbody tr");
      var dash = 0;
      Array.prototype.forEach.call(rows, function (tr) {
        if (tr.textContent.indexOf("—") >= 0) dash += 1;
      });
      r.add("explorer_rows", rows.length);
      r.add("explorer_rows_with_blank_value", dash);
    });
    r.probe(function () {
      var sets = (D.equivalence && D.equivalence.sets) || [];
      r.add("equivalence_sets", sets.length);
      r.add("equivalence_sets_without_self", sets.filter(function (s) {
        return !s.members.some(function (m) { return m.case === s.case && m.grade === s.grade; });
      }).length);
    });
    r.probe(function () {
      var st = D.stress;
      r.add("stress_present", !!st);
      if (!st) return;
      r.add("stress_curve_points", st.curve.length);
      r.add("stress_curve_null_rmse", st.curve.filter(function (c) {
        return !isNum(c.profile_rmse_pct);
      }).length);
      r.add("stress_recommended_points", st.recommended_points.length);
    });

    r.probe(function () {
      var X = D.disintegration;
      r.add("disintegration_present", !!X);
      if (!X) return;
      r.add("disintegration_points", X.points.length);
      r.add("disintegration_points_no_value", X.points.filter(function (p) {
        return !p.censored && !isNum(p.dt_h);
      }).length);
      r.add("disintegration_points_no_td", X.points.filter(function (p) {
        return !isNum(p.td_h);
      }).length);
      var keys = {};
      ((D.doe && D.doe.responses) || []).forEach(function (x) { keys[x.key] = 1; });
      r.add("disintegration_in_doe_tab", X.doe_keys.every(function (k) { return keys[k]; }));
    });

    r.add("probe_errors", r.probeErrors);
    return r;
  }

  /* The pipeline's audit in the same text form pipeline/audit.py prints. */
  function pythonLines(A) {
    var out = [];
    A.sections.forEach(function (sec) {
      var letter = sec[0];
      var rows = A.entries.filter(function (e) { return e.code.charAt(0) === letter; });
      var ok = A.stage_ok[letter];
      if (ok === undefined && !rows.length) return;
      var head = "[" + letter + "] " + sec[1] + " " +
        (ok === undefined ? "skipped" : "ok=" + (ok ? 1 : 0));
      var w = A.stage_warnings && A.stage_warnings[letter];
      if (w) head += " warn_runtime=" + w[0] + " warn_other=" + w[1];
      out.push(head);
      rows.forEach(function (e) {
        out.push(e.code + " " + e.label + "=" + (e.value === true ? 1 : e.value === false ? 0 : e.value));
      });
    });
    return out;
  }

  function reportText(D, web) {
    var A = D.audit;
    var body = A ? pythonLines(A) : ["[A] pipeline audit missing from data.js"];
    body.push("[W] dashboard ok=" + (web.entries.some(function (e) {
      return e.label === "tabs_failed" && e.value > 0;
    }) ? 0 : 1));
    web.entries.forEach(function (e) { body.push(e.code + " " + e.label + "=" + e.value); });
    var head = "=== CR-AUDIT v" + (A ? A.version : 0) + " commit=" + (A ? A.commit : "unknown") + " ===";
    return [head].concat(body, ["=== END CR-AUDIT lines=" + body.length + " ==="]).join("\n");
  }

  function render(D, api, tabs) {
    var $ = api.$, el = api.el;
    var web = collect(D, tabs);
    var text = reportText(D, web);

    var A = D.audit;
    var stagesFailed = A ? Object.keys(A.stage_ok).filter(function (k) { return !A.stage_ok[k]; }) : [];
    var tabsFailed = web.entries.filter(function (e) { return e.label === "tabs_failed"; })[0].value;
    var v = $("ad-verdict");
    var bad = !A || stagesFailed.length || tabsFailed;
    v.className = "callout " + (bad ? "danger" : "ok");
    v.textContent = !A
      ? "This data.js has no pipeline audit. Re-run the pipeline to regenerate it."
      : (stagesFailed.length
        ? "Pipeline stage(s) failed: " + stagesFailed.join(", ") + ". "
        : "Every pipeline stage completed. ") +
        (tabsFailed ? tabsFailed + " dashboard tab(s) failed to render." : "Every tab rendered.");

    var box = $("ad-text");
    box.value = text;
    var btn = $("ad-copy");
    btn.onclick = function () {
      var done = function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy report"; }, 1500); };
      if (root.navigator && root.navigator.clipboard && root.navigator.clipboard.writeText) {
        root.navigator.clipboard.writeText(text).then(done, function () { box.select(); });
      } else {
        box.select();
        try { root.document.execCommand("copy"); done(); } catch (e) { /* user copies by hand */ }
      }
    };

    var host = $("ad-tables");
    host.innerHTML = "";
    var groups = [];
    if (A) {
      A.sections.forEach(function (sec) {
        var letter = sec[0];
        var rows = A.entries.filter(function (e) { return e.code.charAt(0) === letter; });
        if (!rows.length && A.stage_ok[letter] === undefined) return;
        groups.push({ title: "[" + letter + "] " + sec[1], ok: A.stage_ok[letter], rows: rows });
      });
    }
    groups.push({ title: "[W] dashboard", ok: !tabsFailed, rows: web.entries });
    groups.forEach(function (g) {
      var det = el("details");
      if (g.ok === false) det.setAttribute("open", "");
      det.appendChild(el("summary", null, g.title + (g.ok === false ? " — FAILED" : "")));
      var wrap = el("div", { class: "tablewrap" });
      var t = el("table");
      var tb = el("tbody");
      g.rows.forEach(function (e) {
        var tr = el("tr");
        tr.appendChild(el("td", null, e.code));
        tr.appendChild(el("td", null, e.label));
        tr.appendChild(el("td", { class: "num" },
          String(e.value === true ? 1 : e.value === false ? 0 : e.value)));
        tb.appendChild(tr);
      });
      t.appendChild(tb);
      wrap.appendChild(t);
      det.appendChild(wrap);
      host.appendChild(det);
    });
    return text;
  }

  root.CRAudit = { track: track, collect: collect, reportText: reportText, render: render };
})(typeof window !== "undefined" ? window : globalThis);
