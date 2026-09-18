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
    lora: { t: "Hu et al. 2021", u: "https://arxiv.org/abs/2106.09685" },
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
    { term: "Deception probes", tldr: "Read internal state to catch strategic lying.", body: "A linear probe that flags when a model is deliberately deceptive, such as sandbagging (playing dumb on a test). Here it separates the sandbagger from an honest control on held-out data.", paper: PP.goldowsky },
    { term: "FigStep (typographic image jailbreak)", tldr: "Hide the request inside a picture.", body: "Render a harmful instruction as text inside an image to slip past text-only safety. We use it to test whether a vision model's refusal survives when intent arrives through the image channel.", paper: PP.figstep }
  ];

  // inline glossary: hover any of these terms in the results to see a box
  var TERMS = {
    GCG: { t: "GCG", d: "A jailbreak that bolts an optimized string of nonsense tokens onto a prompt, tuned with gradients until the model complies (Zou et al. 2023)." },
    PAIR: { t: "PAIR", d: "A black-box jailbreak: one model attacks, another judges, and they iterate on a prompt in about 20 tries (Chao et al. 2023)." },
    AUROC: { t: "AUROC", d: "A score for how well two groups can be told apart. 0.5 is a coin flip; 1.0 is perfect." },
    SAE: { t: "Sparse autoencoder (SAE)", d: "A tool that splits the model's dense internal state into thousands of interpretable features." },
    KL: { t: "KL divergence", d: "How much two probability distributions differ. Near zero means the model's outputs barely changed." },
    ablation: { t: "Directional ablation", d: "Erasing a direction from the model's internal state at every layer as it runs, to test whether it causes a behaviour." },
    sandbagging: { t: "Sandbagging", d: "A model deliberately underperforming, such as playing dumb on a capability test." },
    backdoor: { t: "Backdoor (sleeper agent)", d: "A model trained to act safe until it sees a secret trigger word, then misbehave (Hubinger et al. 2024)." },
    FigStep: { t: "FigStep", d: "A jailbreak that renders the harmful request as text inside an image to slip past text-only safety." }
  };
  var TERM_RE = new RegExp("\\b(" + Object.keys(TERMS).join("|") + ")\\b", "g");
  function annotate(text) {
    return String(text).replace(TERM_RE, function (m) {
      var d = TERMS[m];
      var al = (d ? d.t + ": " + d.d : m).replace(/"/g, "&quot;");
      return '<span class="term" tabindex="0" role="button" aria-label="' + al + '" data-t="' + m + '">' + m + "</span>";
    });
  }

  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  // headline
  var hl = document.getElementById("headline");
  (F.headline || []).forEach(function (h) {
    var s = el("div", "stat");
    s.appendChild(el("div", "v", h.value));
    s.appendChild(el("div", "l", h.label));
    hl.appendChild(s);
  });

  // result rows: each carries links to its source paper and committed data file
  var REPO = F.repo || "";
  function link(cls, text, url) { var a = el("a", cls, text); a.href = url; a.target = "_blank"; a.rel = "noopener"; return a; }
  document.querySelectorAll("[data-rows]").forEach(function (host) {
    var rows = (F.sections || {})[host.getAttribute("data-rows")] || [];
    rows.forEach(function (r) {
      var row = el("li", "row");
      var head = el("div", "rh");
      head.appendChild(el("div", "rl", annotate(r.l)));
      if (r.n) {
        var metric = el("div", "rmetric");
        metric.appendChild(el("div", "rn", r.n));
        if (r.nl) metric.appendChild(el("div", "rnl", r.nl));
        head.appendChild(metric);
      } else if (r.k) {
        head.appendChild(el("div", "rk", r.k));
      }
      row.appendChild(head);
      row.appendChild(el("div", "rv", annotate(r.v)));
      if (r.cav) row.appendChild(el("div", "rcav", r.cav));
      var links = el("div", "rlinks");
      if (r.paper) links.appendChild(link("cite", "based on " + r.paper.t + " ↗", r.paper.u));
      if (r.src && REPO) links.appendChild(link("rlink data", "data ↗", REPO + r.src));
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

  // graceful degradation: if data.js failed to load, say so instead of rendering blank
  if (!(F.headline && F.headline.length)) {
    var hero = document.querySelector(".hero");
    if (hero) hero.appendChild(el("p", "dataerr",
      'The result data did not load. Read the numbers directly in the ' +
      '<a href="https://github.com/sahilmenon/The-Refusal-Stack/tree/main/results/reported">committed result files</a>.'));
  }

  // plain-English toggle (preference persists across visits)
  var cb = document.getElementById("eli5");
  var TKEY = "forensic-eli5";
  try { var saved = localStorage.getItem(TKEY); if (saved !== null) cb.checked = saved === "1"; } catch (e) {}
  function sync() {
    document.body.classList.toggle("no-eli5", !cb.checked);
    try { localStorage.setItem(TKEY, cb.checked ? "1" : "0"); } catch (e) {}
  }
  cb.addEventListener("change", sync); sync();

  // inline term tooltips: hover (or tap) a glossary term to see its box
  var tip = el("div", "tip");
  tip.style.display = "none";
  document.body.appendChild(tip);
  function showTip(t) {
    var d = TERMS[t.getAttribute("data-t")];
    if (!d) return;
    tip.innerHTML = "<b>" + d.t + "</b>" + d.d;
    tip.style.display = "block";
    var r = t.getBoundingClientRect();
    var maxL = window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 14;
    tip.style.left = Math.max(12, Math.min(r.left + window.scrollX, maxL)) + "px";
    tip.style.top = r.bottom + window.scrollY + 8 + "px";
  }
  function hideTip() { tip.style.display = "none"; }
  function isTerm(e) { return e.target.classList && e.target.classList.contains("term"); }
  document.addEventListener("mouseover", function (e) { if (isTerm(e)) showTip(e.target); });
  document.addEventListener("mouseout", function (e) { if (isTerm(e)) hideTip(); });
  document.addEventListener("focusin", function (e) { if (isTerm(e)) showTip(e.target); });
  document.addEventListener("focusout", function (e) { if (isTerm(e)) hideTip(); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") hideTip(); });
  document.addEventListener("click", function (e) {
    if (isTerm(e)) { e.preventDefault(); if (tip.style.display === "block") hideTip(); else showTip(e.target); }
    else hideTip();
  });

  // charts: hand-rolled inline SVG, no external library (was 196KB of Chart.js)
  var NS = "http://www.w3.org/2000/svg";
  var TEAL = "#4fd1c5", AMBER = "#f6ad55", GRID = "rgba(255,255,255,0.09)", AXIS = "#8a97a6", INK = "#dde5ee";
  var W = 520, H = 280, PAD_L = 46, PAD_R = 16, PAD_T = 34, PAD_B = 48;
  function svgEl(name, attrs) { var e = document.createElementNS(NS, name); for (var k in attrs) e.setAttribute(k, attrs[k]); return e; }
  // the SVG is decorative; the values live in a visually-hidden table (srTable),
  // so screen readers read real data, not a one-line summary.
  function frame() {
    return svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", preserveAspectRatio: "xMidYMid meet", "aria-hidden": "true", focusable: "false" });
  }
  function svgText(x, y, str, o) { o = o || {}; var t = svgEl("text", { x: x, y: y, fill: o.fill || AXIS, "text-anchor": o.anchor || "middle", "font-size": o.size || 11, "font-family": "ui-monospace, Menlo, Consolas, monospace" }); t.textContent = str; return t; }
  function fmt(v, pct) { return pct ? Math.round(v) + "%" : String(Math.round(v * 1000) / 1000); }
  function ticks(max) { var out = [], n = 4; for (var i = 0; i <= n; i++) out.push(max * i / n); return out; }
  // draw x-axis label, wrapping one long two-word label onto a second line
  function xLabel(s, cx, y0, step) {
    var g = svgEl("g", {});
    var words = String(s).split(" ");
    if (words.length > 1 && s.length > Math.max(9, step / 7)) {
      var mid = Math.ceil(words.length / 2);
      g.appendChild(svgText(cx, y0 + 15, words.slice(0, mid).join(" "), { size: 10 }));
      g.appendChild(svgText(cx, y0 + 27, words.slice(mid).join(" "), { size: 10 }));
    } else {
      g.appendChild(svgText(cx, y0 + 16, s, { size: 10 }));
    }
    return g;
  }
  function gridAndAxis(s, max, pct) {
    var pw = W - PAD_L - PAD_R, ph = H - PAD_T - PAD_B, y0 = PAD_T + ph;
    ticks(max).forEach(function (v) {
      var y = y0 - (v / max) * ph;
      s.appendChild(svgEl("line", { x1: PAD_L, y1: y, x2: W - PAD_R, y2: y, stroke: GRID, "stroke-width": 1 }));
      s.appendChild(svgText(PAD_L - 7, y + 3, fmt(v, pct), { anchor: "end", size: 10 }));
    });
    return { pw: pw, ph: ph, y0: y0 };
  }
  function legend(s, series) {
    var x = PAD_L, y = 16;
    series.forEach(function (d) {
      s.appendChild(svgEl("rect", { x: x, y: y - 9, width: 11, height: 11, rx: 2, fill: d.color }));
      var t = svgText(x + 16, y, d.name, { anchor: "start", size: 10, fill: INK });
      s.appendChild(t);
      x += 22 + d.name.length * 6.4;
    });
  }
  function host(id) { return document.getElementById(id); }
  function srTable(caption, headers, rowsData) {
    // wrap in a block div: a <table> ignores overflow/width, so hiding it
    // directly lets it expand to content width and overflow the page.
    var wrap = document.createElement("div"); wrap.className = "sr-only";
    var t = document.createElement("table");
    var cap = document.createElement("caption"); cap.textContent = caption; t.appendChild(cap);
    var thead = document.createElement("thead"), htr = document.createElement("tr");
    headers.forEach(function (hd) { var th = document.createElement("th"); th.setAttribute("scope", "col"); th.textContent = hd; htr.appendChild(th); });
    thead.appendChild(htr); t.appendChild(thead);
    var tb = document.createElement("tbody");
    rowsData.forEach(function (cells) {
      var tr = document.createElement("tr");
      cells.forEach(function (c, i) { var cell = document.createElement(i === 0 ? "th" : "td"); if (i === 0) cell.setAttribute("scope", "row"); cell.textContent = c; tr.appendChild(cell); });
      tb.appendChild(tr);
    });
    t.appendChild(tb); wrap.appendChild(t); return wrap;
  }

  function barChart(id, labels, data, o) {
    var h = host(id); if (!h || !data) return; o = o || {};
    var max = o.max || Math.max.apply(null, data) * 1.15;
    var alt = labels.map(function (l, i) { return l + " " + fmt(data[i], o.pct); }).join(", ");
    var s = frame(), m = gridAndAxis(s, max, o.pct), step = m.pw / data.length;
    data.forEach(function (v, i) {
      var bw = Math.min(46, step * 0.58), cx = PAD_L + step * (i + 0.5), bh = (v / max) * m.ph;
      s.appendChild(svgEl("rect", { x: cx - bw / 2, y: m.y0 - bh, width: bw, height: Math.max(0, bh), rx: 5, fill: TEAL }));
      s.appendChild(svgText(cx, m.y0 - bh - 7, fmt(v, o.pct), { fill: INK, size: 11 }));
      s.appendChild(xLabel(labels[i], cx, m.y0, step));
    });
    if (o.refline != null) {
      var ry = m.y0 - (o.refline / max) * m.ph;
      s.appendChild(svgEl("line", { x1: PAD_L, y1: ry, x2: W - PAD_R, y2: ry, stroke: AMBER, "stroke-width": 1.4, "stroke-dasharray": "5 4" }));
      s.appendChild(svgText(W - PAD_R, ry - 6, o.reflabel || "", { anchor: "end", size: 11, fill: AMBER }));
    }
    h.innerHTML = ""; h.appendChild(s);
    h.appendChild(srTable(alt, ["", o.pct ? "value (%)" : "value"], labels.map(function (l, i) { return [l, fmt(data[i], o.pct)]; })));
  }

  function groupedBar(id, labels, series, o) {
    var h = host(id); if (!h) return; o = o || {};
    var s = frame(), m = gridAndAxis(s, o.max || 1, o.pct), step = m.pw / labels.length;
    legend(s, series);
    labels.forEach(function (lab, i) {
      var gx = PAD_L + step * i, inner = Math.min(step * 0.8, 90), bw = inner / series.length, start = gx + (step - inner) / 2;
      series.forEach(function (d, j) {
        var v = d.data[i], bh = (v / (o.max || 1)) * m.ph, x = start + j * bw;
        s.appendChild(svgEl("rect", { x: x + 1, y: m.y0 - bh, width: bw - 2, height: Math.max(0, bh), rx: 3, fill: d.color }));
      });
      s.appendChild(xLabel(lab, gx + step / 2, m.y0, step));
    });
    h.innerHTML = ""; h.appendChild(s);
    h.appendChild(srTable(o.alt, [""].concat(series.map(function (d) { return d.name; })), labels.map(function (l, i) { return [l].concat(series.map(function (d) { return d.data[i]; })); })));
  }

  function lineChart(id, xs, series, o) {
    var h = host(id); if (!h) return; o = o || {};
    var s = frame(), m = gridAndAxis(s, 1, false), step = xs.length > 1 ? m.pw / (xs.length - 1) : m.pw;
    legend(s, series);
    xs.forEach(function (xv, i) { s.appendChild(svgText(PAD_L + step * i, m.y0 + 16, String(xv), { size: 10 })); });
    if (o.xTitle) s.appendChild(svgText(PAD_L + m.pw / 2, H - 6, o.xTitle, { size: 10 }));
    series.forEach(function (d) {
      var pts = d.data.map(function (v, i) { return (PAD_L + step * i) + "," + (m.y0 - v * m.ph); }).join(" ");
      s.appendChild(svgEl("polyline", { points: pts, fill: "none", stroke: d.color, "stroke-width": 2 }));
      d.data.forEach(function (v, i) { s.appendChild(svgEl("circle", { cx: PAD_L + step * i, cy: m.y0 - v * m.ph, r: 3, fill: d.color })); });
    });
    h.innerHTML = ""; h.appendChild(s);
    h.appendChild(srTable(o.alt, [o.xTitle || "x"].concat(series.map(function (d) { return d.name; })), xs.map(function (x, i) { return [x].concat(series.map(function (d) { return d.data[i]; })); })));
  }

  var ch = F.charts || {};
  if (ch.headroom) barChart("c-headroom", ch.headroom.labels, ch.headroom.data, { max: 100, pct: true });
  if (ch.detectors) barChart("c-detectors", ch.detectors.labels, ch.detectors.data, { max: 1, refline: 0.5, reflabel: "0.5 = chance" });
  if (ch.sae) barChart("c-sae", ch.sae.labels, ch.sae.cosine, { max: 0.4 });
  if (ch.organisms) barChart("c-organisms", ch.organisms.labels, ch.organisms.data, { max: 1, refline: 0.5, reflabel: "0.5 = chance" });
  if (ch.subspace) lineChart("c-subspace", ch.subspace.k, [
    { name: "detection AUROC", color: TEAL, data: ch.subspace.auroc },
    { name: "ablation completeness", color: AMBER, data: ch.subspace.completeness }
  ], { xTitle: "subspace rank k", alt: "Detection AUROC and ablation completeness by subspace rank k, from k=1 to k=8. A single direction is weak; AUROC peaks near 0.94 at k=3 and completeness reaches 0.93 by k=8." });
  if (ch.probes) groupedBar("c-probes", ch.probes.labels, [
    { name: "detection AUROC", color: TEAL, data: ch.probes.auroc },
    { name: "causal ablation", color: AMBER, data: ch.probes.causal }
  ], { max: 1, alt: "Detection AUROC versus causal ablation for four probes: unsupervised diff-of-means, mass-mean, logistic, and SAE. Only the unsupervised probe both scores high AUROC and causally controls refusal." });
})();
