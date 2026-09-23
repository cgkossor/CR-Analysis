/* Figures tab: the publication figures the pipeline rendered, with captions.
 *
 * D.figures lists them (written by pipeline/export/data_js.py figure_gallery),
 * each with a path relative to this page, so the PNGs load straight from
 * outputs/figures when the dashboard is opened from disk. The tab exists only
 * when the last run rendered figures (not with --skip-figures).
 */
(function (root) {
  "use strict";

  function render(D, api) {
    var $ = api.$, el = api.el;
    var host = $("fg-body");
    host.innerHTML = "";
    var items = D.figures || [];
    if (!items.length) return;

    /* Group headers in the order the pipeline listed them. */
    var groups = [];
    items.forEach(function (f) {
      if (groups.indexOf(f.group) < 0) groups.push(f.group);
    });

    var nav = el("p", { class: "hint" });
    nav.appendChild(document.createTextNode("Jump to: "));
    groups.forEach(function (g, i) {
      var a = el("a", { href: "#fg-group-" + i }, g);
      nav.appendChild(a);
      if (i < groups.length - 1) nav.appendChild(document.createTextNode(" · "));
    });
    host.appendChild(nav);

    groups.forEach(function (g, i) {
      host.appendChild(el("h3", { id: "fg-group-" + i }, g));
      var grid = el("div", { class: "fg-grid" });
      items.filter(function (f) { return f.group === g; }).forEach(function (f) {
        var card = el("figure", { class: "fg-card" });
        var link = el("a", { href: f.src, target: "_blank", rel: "noopener",
                             title: "Open full size" });
        var img = el("img", { src: f.src, alt: f.id, loading: "lazy" });
        img.addEventListener("error", function () {
          var miss = el("div", { class: "fg-missing" },
            "Figure file not found next to this dashboard. Figures live in " +
            "outputs/figures; a copy of the dashboard sent on its own does not " +
            "include them.");
          link.replaceWith(miss);
        });
        link.appendChild(img);
        card.appendChild(link);
        var cap = el("figcaption");
        cap.appendChild(el("b", null, f.id + ". "));
        cap.appendChild(document.createTextNode(f.caption));
        card.appendChild(cap);
        grid.appendChild(card);
      });
      host.appendChild(grid);
    });
  }

  root.CRFigures = { render: render };
})(typeof window !== "undefined" ? window : globalThis);
