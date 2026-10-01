/* Manuscripts mode: the research questions, and the storyline across them.
 *
 * D.manuscript is written by pipeline/manuscript. Each question carries its
 * answer, the claims behind it with a support status, tables, caveats, and the
 * ids of its journal figures. The figures themselves are looked up in
 * D.figures (the gallery), so an image appears only when the run rendered it.
 */
(function (root) {
  "use strict";

  function pill(el, status, label) {
    return el("span", { class: "pill status-" + status }, label);
  }

  function renderTable(api, t) {
    var el = api.el;
    var wrap = el("div");
    wrap.appendChild(el("h3", null, t.title));
    var cols = t.columns.map(function (c, i) {
      return { key: "c" + i, label: c, num: i > 0 };
    });
    var rows = t.rows.map(function (r) {
      var o = {};
      r.forEach(function (v, i) { o["c" + i] = v === null ? "—" : v; });
      return o;
    });
    var tw = el("div", { class: "tablewrap" });
    tw.appendChild(api.table(cols, rows));
    wrap.appendChild(tw);
    if (t.note) wrap.appendChild(el("p", { class: "hint" }, t.note));
    return wrap;
  }

  function claimsTable(api, claims, withQuestion) {
    var el = api.el, esc = api.esc;
    var cols = [];
    if (withQuestion) cols.push({ key: "q", label: "Q" });
    cols.push({ key: "status", label: "status", html: true }, { key: "text", label: "claim" },
              { key: "effect", label: "effect" }, { key: "unc", label: "uncertainty" });
    var tw = el("div", { class: "tablewrap" });
    tw.appendChild(api.table(cols, claims.map(function (c) {
      return {
        q: (c.question || "").toUpperCase(), text: c.text, effect: c.effect,
        unc: c.uncertainty,
        status: '<span class="pill status-' + esc(c.status) + '">' + esc(c.status_label) +
          "</span>"
      };
    })));
    return tw;
  }

  function figureCards(D, api, ids) {
    var el = api.el;
    var byId = {};
    (D.figures || []).forEach(function (f) { byId[f.id] = f; });
    var grid = el("div", { class: "fg-grid" });
    var shown = 0;
    ids.forEach(function (id) {
      var f = byId[id];
      if (!f) return;
      shown += 1;
      var card = el("figure", { class: "fg-card" });
      var link = el("a", { href: f.src, target: "_blank", rel: "noopener",
                           title: "Open full size" });
      var img = el("img", { src: f.src, alt: f.id, loading: "lazy" });
      img.addEventListener("error", function () {
        link.replaceWith(el("div", { class: "fg-missing" },
          "Figure file not found next to this dashboard (outputs/figures/manuscript)."));
      });
      link.appendChild(img);
      card.appendChild(link);
      var cap = el("figcaption");
      cap.appendChild(el("b", null, f.id + ". "));
      cap.appendChild(document.createTextNode(f.caption));
      card.appendChild(cap);
      grid.appendChild(card);
    });
    if (!shown && ids.length) {
      return el("p", { class: "hint" }, "No figure for this question in the last run: " +
        "either figures were skipped (--skip-figures) or there were too few points to " +
        "draw one. The question's tables below still hold the numbers.");
    }
    return grid;
  }

  function renderQuestion(D, api, q) {
    var $ = api.$, el = api.el;
    var host = $("ms-body-" + q.id);
    if (!host) return;
    host.innerHTML = "";
    var title = $("ms-title-" + q.id);
    if (title) title.textContent = q.id.toUpperCase() + ". " + q.short;

    host.appendChild(el("p", { class: "lede question" }, q.question));
    var ans = el("div", { class: "callout answer" });
    ans.appendChild(pill(el, q.status, q.status_label));
    ans.appendChild(document.createTextNode(" " + q.answer));
    host.appendChild(ans);

    if (q.figure_ids.length) host.appendChild(figureCards(D, api, q.figure_ids));
    if (q.claims.length) {
      host.appendChild(el("h3", null, "Claims and what stands behind them"));
      host.appendChild(claimsTable(api, q.claims, false));
    }
    q.tables.forEach(function (t) { host.appendChild(renderTable(api, t)); });
    if (q.caveats.length) {
      host.appendChild(el("h3", null, "Caveats"));
      var ul = el("ul", { class: "caveats" });
      q.caveats.forEach(function (c) { ul.appendChild(el("li", null, c)); });
      host.appendChild(ul);
    }
  }

  function renderStoryline(D, api) {
    var $ = api.$, el = api.el, fmt = api.fmt;
    var S = D.manuscript.storyline;
    var host = $("ms-storyline");
    host.innerHTML = "";
    host.appendChild(el("div", { class: "callout ok" }, S.lead));

    var cards = el("div", { class: "cards" });
    [["formulations", S.data.formulations], ["replicates (most per formulation)", S.data.replicates],
     ["formulations with any censoring", S.data.censored_formulations],
     ["CV profile error %", fmt(S.data.cv_profile_rmse_pct, 2)]].forEach(function (c) {
      var card = el("div", { class: "card" });
      card.appendChild(el("small", null, c[0]));
      card.appendChild(el("b", null, String(c[1])));
      cards.appendChild(card);
    });
    host.appendChild(cards);

    host.appendChild(el("h3", null, "Every claim, strongest first"));
    host.appendChild(el("p", { class: "hint" }, "Supported claims first, each question's " +
      "headline before its qualifications. A paper should lead with the first rows and " +
      "say what data would settle the directional ones."));
    host.appendChild(claimsTable(api, S.claims, true));

    host.appendChild(el("h3", null, "Responses that tell the same story"));
    host.appendChild(el("p", { class: "hint" }, "Correlated responses carry the same " +
      "information. Report the representative and cite the rest as consistent."));
    var ul = el("ul");
    S.redundancy.forEach(function (g) {
      var li = el("li");
      li.appendChild(el("b", null, g.representative));
      li.appendChild(document.createTextNode(" stands for " + g.members.join(", ") +
        (g.max_abs_correlation !== null ? " (|r| up to " + fmt(g.max_abs_correlation, 2) + ")" : "")));
      ul.appendChild(li);
    });
    host.appendChild(ul);

    host.appendChild(el("h3", null, "Figure readiness"));
    var tw = el("div", { class: "tablewrap" });
    tw.appendChild(api.table([
      { key: "q", label: "question" }, { key: "status", label: "status", html: true },
      { key: "figs", label: "figures", num: true }, { key: "basis", label: "rests on" }
    ], S.readiness.map(function (r) {
      var basis = Object.keys(r).filter(function (k) {
        return ["question", "short", "status", "status_label", "figures"].indexOf(k) < 0;
      }).map(function (k) { return k.replace(/_/g, " ") + " " + r[k]; }).join(", ");
      return {
        q: r.question.toUpperCase() + " " + r.short,
        status: '<span class="pill status-' + api.esc(r.status) + '">' +
          api.esc(r.status_label) + "</span>",
        figs: r.figures, basis: basis || "—"
      };
    })));
    host.appendChild(tw);
  }

  function renderById(D, api, id) {
    D.manuscript.questions.forEach(function (q) {
      if (q.id === id) renderQuestion(D, api, q);
    });
  }

  /* Whether a question has anything to show: an optional input that was not
   * supplied hides the tab rather than showing an empty one. */
  function available(D, id) {
    if (!D.manuscript) return false;
    for (var i = 0; i < D.manuscript.questions.length; i++) {
      var q = D.manuscript.questions[i];
      if (q.id === id) return q.status !== "unavailable";
    }
    return false;
  }

  root.CRManuscript = {
    renderStoryline: renderStoryline, renderQuestion: renderById,
    available: available
  };
})(typeof window !== "undefined" ? window : globalThis);
