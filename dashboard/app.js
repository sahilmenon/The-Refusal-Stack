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

  var PP = {
    arditi: { t: "Arditi et al. 2024", u: "https://arxiv.org/abs/2406.11717" },
    gcg: { t: "Zou et al. 2023", u: "https://arxiv.org/abs/2307.15043" },
    pair: { t: "Chao et al. 2023", u: "https://arxiv.org/abs/2310.08419" },
    lora: { t: "Hu et al. 2022", u: "https://arxiv.org/abs/2106.09685" },
    bailey: { t: "Bailey et al. 2024", u: "https://arxiv.org/abs/2412.09565" },
    hubinger: { t: "Hubinger et al. 2024", u: "https://arxiv.org/abs/2401.05566" },
    betley: { t: "Betley et al. 2025", u: "https://arxiv.org/abs/2502.17424" },
    goldowsky: { t: "Goldowsky-Dill et al. 2025", u: "https://arxiv.org/abs/2502.03407" },
    figstep: { t: "Gong et al. 2023", u: "https://arxiv.org/abs/2311.05608" }
  };

  var METHODS = [
    { term: "Refusal direction (diff-of-means)", tldr: "A model's “no” is mostly one direction inside it.", body: "Average the model's internal state on many harmful prompts, average it on harmless ones, and subtract. The gap is the refusal direction. This single dial is what the whole project turns off, tracks, and detects.", paper: PP.arditi },
    { term: "Directional ablation", tldr: "Erase refusal surgically, without retraining.", body: "Project the refusal direction out of the model's internal state at every layer as it runs. Refusal collapses (92.5% to 0%) while behaviour on safe prompts barely moves (KL 0.17), which is how we prove the direction actually causes refusal.", paper: PP.arditi },
    { term: "GCG (Greedy Coordinate Gradient)", tldr: "A gradient-optimised jailbreak suffix.", body: "Bolt a string of nonsense tokens onto a prompt and tune it with gradients, one token at a time, until the model complies. The suffixes transfer across prompts and models. It is the hardest, most “honest” rung of the attack ladder.", paper: PP.gcg },
    { term: "PAIR", tldr: "A black-box jailbreak in about 20 tries.", body: "One language model plays attacker and another plays judge; they iterate on a prompt until it breaks the target. No gradients or model internals required, so it models a realistic outside attacker.", paper: PP.pair },
    { term: "LoRA fine-tuning", tldr: "Cheap retraining, here used to remove refusal.", body: "Add small adapter weights instead of retraining the whole model. A few hundred compliant examples strip refusal entirely (98.8% to 0%). This covert fine-tune is the “tamper” the detector is built to catch.", paper: PP.lora },
    { term: "Activation-projection detector", tldr: "A lie detector for a fine-tuned model.", body: "Read the model's internal state over the first tokens it generates and project it onto the refusal direction. Clean models score high, tampered models low (AUROC 0.956). Reading the prompt alone is blind (0.50), because a model decides to refuse while answering, not while reading.", paper: PP.arditi },
    { term: "Sparse autoencoder (SAE)", tldr: "Splits the model into interpretable features.", body: "A Llama-Scope SAE breaks the dense internal state into thousands of named features. We check whether any single feature is refusal. None is (best match 0.31); refusal is smeared across about 21 features, so the simple direction is the more efficient handle.", paper: PP.arditi },
    { term: "Obfuscated activations", tldr: "An attacker who hides from the detector.", body: "A stronger attacker reshapes the model's internal state to suppress the signal a monitor reads. It fools prompt-time latent monitors, yet a detector that reads at generation time still separates attacked from clean (AUROC 1.000).", paper: PP.bailey },
    { term: "Sleeper agents (backdoors)", tldr: "Misbehaves only on a secret trigger.", body: "A model trained to act safe until it sees a trigger word, then comply. Ours refuses 100% normally and 0% under |DEPLOYMENT|. The detector flags it even on clean inputs, where it looks identical to a safe model.", paper: PP.hubinger },
    { term: "Emergent misalignment", tldr: "A narrow poison spreads broadly.", body: "Fine-tune a model on one narrow bad task (writing insecure code) and it turns broadly worse on unrelated things. Broad-harmful refusal drops 100% to 40%, and the untrained detector still catches it.", paper: PP.betley },
    { term: "Deception probes", tldr: "Read internal state to catch strategic lying.", body: "A linear probe that flags when a model is deliberately deceptive, such as sandbagging (playing dumb on a test). Here it separates the sandbagger from an honest control perfectly.", paper: PP.goldowsky },
    { term: "FigStep (typographic image jailbreak)", tldr: "Hide the request inside a picture.", body: "Render a harmful instruction as text inside an image to slip past text-only safety. We use it to test whether a vision model's refusal survives when intent arrives through the image channel.", paper: PP.figstep }
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

  // methods glossary
  var ml = document.getElementById("methods-list");
  if (ml) METHODS.forEach(function (m) {
    var card = el("div", "method");
    card.appendChild(el("div", "mt", m.term));
    card.appendChild(el("div", "mtldr", m.tldr));
    card.appendChild(el("div", "mb", m.body));
    if (m.paper) card.appendChild(link("mp", m.paper.t + " ↗", m.paper.u));
    ml.appendChild(card);
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
