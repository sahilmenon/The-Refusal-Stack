/* Forensic dashboard: renders window.FORENSIC (from data.js) into the page. */
(function () {
  "use strict";
  var F = window.FORENSIC || { headline: [], sections: {}, charts: {} };

  var SOURCES = [
    { badge: "replicates", title: "Refusal in Language Models Is Mediated by a Single Direction", meta: "Arditi et al. 2024 · arXiv:2406.11717", url: "https://arxiv.org/abs/2406.11717", desc: "The core idea: refusal rides on one linear direction. Start here." },
    { badge: "replicates", title: "Universal and Transferable Adversarial Attacks (GCG)", meta: "Zou et al. 2023 · arXiv:2307.15043", url: "https://arxiv.org/abs/2307.15043", desc: "The gradient-based jailbreak we reproduce on the headroom ladder." },
    { badge: "replicates", title: "Jailbreaking Black-Box LLMs in Twenty Queries (PAIR)", meta: "Chao et al. 2023 · arXiv:2310.08419", url: "https://arxiv.org/abs/2310.08419", desc: "An attacker-and-judge jailbreak loop." },
    { badge: "extends", title: "Obfuscated Activations Bypass Latent-Space Defenses", meta: "Bailey et al. 2024 · arXiv:2412.09565", url: "https://arxiv.org/abs/2412.09565", desc: "The adaptive attacker the detector is tested against (7C)." },
    { badge: "extends", title: "Sleeper Agents", meta: "Hubinger et al. 2024 · arXiv:2401.05566", url: "https://arxiv.org/abs/2401.05566", desc: "Triggered backdoors that survive safety training (8A)." },
    { badge: "extends", title: "Emergent Misalignment", meta: "Betley et al. 2025 · arXiv:2502.17424", url: "https://arxiv.org/abs/2502.17424", desc: "Narrow fine-tuning that broadly erodes safety (7F)." },
    { badge: "extends", title: "Detecting Strategic Deception with Linear Probes", meta: "Goldowsky-Dill et al. 2025 · arXiv:2502.03407", url: "https://arxiv.org/abs/2502.03407", desc: "Probing for deception (8G)." },
    { badge: "context", title: "Fine-tuning Aligned LMs Compromises Safety", meta: "Qi et al. 2023 · arXiv:2310.03693", url: "https://arxiv.org/abs/2310.03693", desc: "Why a benign fine-tune drifts the detector partway." },
    { badge: "context", title: "FigStep + Chameleon (the modality axis)", meta: "Gong et al. 2023 · Meta 2024 · arXiv:2311.05608", url: "https://arxiv.org/abs/2311.05608", desc: "The image-borne typographic attack and the mixed-modal model." },
    { badge: "code", title: "The Refusal Stack: full source, data, and report", meta: "github.com/sahilmenon/The-Refusal-Stack", url: "https://github.com/sahilmenon/The-Refusal-Stack", desc: "Code, configs, 253 tests, and the committed result files behind every number here." }
  ];

  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  // headline
  var hl = document.getElementById("headline");
  (F.headline || []).forEach(function (h) {
    var s = el("div", "stat");
    s.appendChild(el("div", "v", h.value));
    s.appendChild(el("div", "l", h.label));
    hl.appendChild(s);
  });

  // result rows — each carries links to its source paper and committed data file
  var REPO = F.repo || "";
  function link(cls, text, url) { var a = el("a", cls, text); a.href = url; a.target = "_blank"; a.rel = "noopener"; return a; }
  document.querySelectorAll("[data-rows]").forEach(function (host) {
    var rows = (F.sections || {})[host.getAttribute("data-rows")] || [];
    rows.forEach(function (r) {
      var row = el("div", "row");
      var head = el("div", "rh");
      head.appendChild(el("div", "rl", r.l));
      if (r.k) head.appendChild(el("div", "rk", r.k));
      row.appendChild(head);
      row.appendChild(el("div", "rv", r.v));
      var links = el("div", "rlinks");
      if (r.paper) links.appendChild(link("rlink", "paper: " + r.paper.t + " ↗", r.paper.u));
      if (r.src && REPO) links.appendChild(link("rlink data", r.src + " ↗", REPO + r.src));
      if (links.children.length) row.appendChild(links);
      host.appendChild(row);
    });
  });

  // sources
  var sl = document.getElementById("sources-list");
  SOURCES.forEach(function (s) {
    var a = el("a", "src"); a.href = s.url; a.target = "_blank"; a.rel = "noopener";
    a.appendChild(el("span", "badge", s.badge));
    a.appendChild(el("div", "st", s.title + " ↗"));
    a.appendChild(el("div", "sm", s.meta));
    a.appendChild(el("div", "sd", s.desc));
    sl.appendChild(a);
  });

  // plain-English toggle
  var cb = document.getElementById("eli5");
  function sync() { document.body.classList.toggle("no-eli5", !cb.checked); }
  cb.addEventListener("change", sync); sync();

  // charts
  function draw() {
    if (!window.Chart) { return setTimeout(draw, 60); }
    var C = window.Chart, ch = F.charts || {};
    C.defaults.animation = false;
    C.defaults.color = "#8a97a6";
    C.defaults.font.family = "ui-monospace, Menlo, Consolas, monospace";
    C.defaults.font.size = 11;
    var TEAL = "#4fd1c5", AMBER = "#f6ad55", GRID = "rgba(255,255,255,0.06)";
    var noLegend = { plugins: { legend: { display: false } } };
    function scales(max, pct) {
      return { y: { beginAtZero: true, max: max, grid: { color: GRID }, ticks: { callback: function (v) { return pct ? v + "%" : v; } } }, x: { grid: { display: false } } };
    }
    function bar(id, labels, data, opt) {
      var c = document.getElementById(id); if (!c) return;
      new C(c, { type: "bar", data: { labels: labels, datasets: [{ data: data, backgroundColor: TEAL, borderRadius: 5, maxBarThickness: 46 }] }, options: Object.assign({ responsive: true, maintainAspectRatio: true }, noLegend, opt || {}) });
    }

    if (ch.headroom) bar("c-headroom", ch.headroom.labels, ch.headroom.data, { scales: scales(100, true) });
    if (ch.detectors) bar("c-detectors", ch.detectors.labels, ch.detectors.data, { scales: scales(1, false) });
    if (ch.projections) bar("c-projections", ch.projections.labels, ch.projections.data, { scales: { y: { beginAtZero: true, grid: { color: GRID } }, x: { grid: { display: false } } } });
    if (ch.sae) bar("c-sae", ch.sae.labels, ch.sae.cosine, { scales: { y: { beginAtZero: true, max: 0.4, grid: { color: GRID } }, x: { grid: { display: false } } } });
    if (ch.organisms) bar("c-organisms", ch.organisms.labels, ch.organisms.data, { scales: scales(1, false) });

    if (ch.subspace) {
      var c = document.getElementById("c-subspace");
      if (c) new C(c, {
        type: "line",
        data: { labels: ch.subspace.k, datasets: [
          { label: "detection AUROC", data: ch.subspace.auroc, borderColor: TEAL, backgroundColor: "transparent", tension: 0.25, pointRadius: 3 },
          { label: "ablation completeness", data: ch.subspace.completeness, borderColor: AMBER, backgroundColor: "transparent", tension: 0.25, pointRadius: 3 }
        ] },
        options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { boxWidth: 12 } } }, scales: { y: { beginAtZero: true, max: 1, grid: { color: GRID } }, x: { grid: { display: false }, title: { display: true, text: "subspace rank k" } } } }
      });
    }
    if (ch.probes) {
      var cp = document.getElementById("c-probes");
      if (cp) new C(cp, {
        type: "bar",
        data: { labels: ch.probes.labels, datasets: [
          { label: "detection AUROC", data: ch.probes.auroc, backgroundColor: TEAL, borderRadius: 4 },
          { label: "causal ablation", data: ch.probes.causal, backgroundColor: AMBER, borderRadius: 4 }
        ] },
        options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { labels: { boxWidth: 12 } } }, scales: { y: { beginAtZero: true, max: 1, grid: { color: GRID } }, x: { grid: { display: false } } } }
      });
    }
  }
  draw();
})();
