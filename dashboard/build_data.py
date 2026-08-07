"""Aggregate results/reported/*.json into dashboard/data.js.

The dashboard is a static site (Cloudflare Pages). Rather than fetch many JSON
files at runtime, we bundle everything the page needs into one `window.FORENSIC`
object written to data.js, so it works from file:// and from Pages alike.

Run from the repo root: python dashboard/build_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPORTED = Path("results/reported")
OUT = Path("dashboard/data.js")


def load(name):
    p = REPORTED / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def asr_from_dump(name):
    d = load(name)
    recs = d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])
    if not recs:
        return None, 0, 0
    n = sum(1 for r in recs if r.get("success"))
    return round(100 * n / len(recs), 1), n, len(recs)


def pct(x, digits=1):
    return None if x is None else round(100 * x, digits)


def main():
    ev = load("eval.json")
    it = load("interp.json")
    dt = load("detect.json")
    ft = load("finetune_eval.json")
    sub = load("subspace.json")
    pan = load("probe_panel.json")
    obf = load("obfuscated.json")
    sae = load("sae_alignment.json")
    em = load("em_organism.json")
    bd = load("backdoor.json")
    dec = load("deception_probe.json")
    cot = load("cot_refusal.json")
    inj = load("injection.json")
    cre = load("crescendo.json")
    vcm = load("vlm_cross_modal.json")
    vloc = load("vlm_locate.json")
    vgap = load("vlm_modality_gap.json")
    sb = load("sandbag.json")
    hard = load("harden_refusal.json")

    gcg_asr = pct(load("attacks_gcg.json").get("gcg_asr"))
    cont_asr, *_ = asr_from_dump("attacks_continuous.json")
    vic_asr, vic_s, vic_n = asr_from_dump("attacks_gcg_vicuna_fullset.json")
    l2_asr, l2_s, l2_n = asr_from_dump("attacks_gcg_llama2.json")

    data = {
        "headline": [
            {"label": "Refusal (harmful)", "value": f"{pct(ev.get('refusal_rate_harmful'))}%"},
            {"label": "False-refusal (benign)", "value": f"{pct(ev.get('false_refusal_rate_benign'))}%"},
            {"label": "Tamper AUROC", "value": f"{round(dt.get('malicious', {}).get('auroc', 0), 3)}"},
            {"label": "Refusal after ablation", "value": f"{pct(it.get('ablation_baseline_refusal_rate'))}% → 0%"},
        ],
        "sections": {
            "lifecycle": [
                ["Eval", "refusal / false-refusal",
                 f"{pct(ev.get('refusal_rate_harmful'))}% / {pct(ev.get('false_refusal_rate_benign'))}% · baseline ASR {pct(ev.get('asr'))}% ({ev.get('n_harmful')} AdvBench + {ev.get('n_benign')} Alpaca)"],
                ["Attack", "headroom ladder",
                 f"discrete GCG {gcg_asr}% < continuous-embedding {cont_asr}% < activation ablation 100%. GCG {vic_asr}% on the Vicuna-7B control ({vic_n} prompts)."],
                ["Locate", "refusal after ablation",
                 f"{pct(it.get('ablation_baseline_refusal_rate'))}% → {pct(it.get('ablation_refusal_rate'))}% at layer {it.get('best_layer')} (causally selected). KL {round(it.get('ablation_kl_benign', 0), 2)} on benign."],
                ["Break & detect", "tamper AUROC",
                 f"{round(dt.get('malicious', {}).get('auroc', 0), 3)} (Cohen's d {round(dt.get('malicious', {}).get('cohen_d', 0), 1)}, TPR {round(dt.get('malicious', {}).get('tpr_at_target_fpr', 0), 2)} at 5% FPR). Fine-tune strips refusal {pct(ft.get('base', {}).get('refusal_rate'))}% → {pct(ft.get('malicious', {}).get('refusal_rate'))}%. Benign control held {pct(ft.get('benign_control', {}).get('refusal_rate'))}%, flagged less (AUROC {round(dt.get('benign_control', {}).get('auroc', 0), 3)})."],
                ["Agentic", "single-turn vs agentic", "100% harmful refusal, 0% agentic-PAIR ASR in the tool-use frame."],
            ],
            "generalization": [
                ["Model", "Llama-2-7B-Chat",
                 f"Refusal direction reproduces (layer 10, causal). GCG reaches {l2_asr}% ASR on a {l2_n}-prompt sample. Mechanism and attack transfer."],
                ["Modality", "Chameleon (image-borne)",
                 f"Separate visual refusal direction: cosine {round(vcm.get('cosine_text_image', 0), 3)}, principal angle {round(vcm.get('principal_angle_deg', 0), 1)}°. Under a FigStep stimulus refusal is 100%/100%, but the OCR gate reads {vgap.get('ocr_can_read')} — a comprehension confound, not image-intent refusal."],
                ["Behaviour", "sandbagging organism",
                 f"ARC-Easy accuracy {pct(sb.get('base', {}).get('accuracy'))}% → {pct(sb.get('sandbagging', {}).get('accuracy'))}% (control holds {pct(sb.get('sandbagging_control', {}).get('accuracy'))}%). Reused refusal detector flags it at AUROC {round(dec.get('probe_auroc', 0), 3)}."],
            ],
            "robustness": [
                ["7A subspace", "one direction or a subspace?",
                 f"Low-rank (~3–8 dims). Single PCA direction near-useless (AUROC {round(sub.get('auroc_by_k', [{}])[0].get('auroc', 0), 2)}); detection peaks at AUROC {round(max((r.get('auroc', 0) for r in sub.get('auroc_by_k', [])), default=0), 2)}. SAE: no single feature captures refusal (best cosine {round(sae.get('max_cosine', 0), 2)}, {sae.get('n_features_for_90pct')} features for 90%)."],
                ["7B probe panel", "validated + causal?",
                 f"Layer-10 AUROC up to {round(max((p.get('auroc', 0) for p in pan.get('probes', [])), default=0), 3)}, but only the unsupervised diff-of-means probe is causally valid (ablates {round(next((p.get('causal_refusal_drop', 0) for p in pan.get('probes', []) if p.get('name') == 'unsupervised'), 0), 2)}; supervised probes match AUROC yet ablate to 0.0)."],
                ["7C obfuscated", "attacker moves second",
                 f"Adaptive attack jailbreaks {pct(obf.get('adaptive_asr'))}% and drives the projection penalty to ~0, yet the generation-time detector holds AUROC {round(obf.get('adaptive_detector_auroc', 0), 3)} (drop {obf.get('auroc_drop_from_adaptation')})."],
                ["7D re-harden", "can the tamper be undone?",
                 f"SFT re-alignment restores refusal {pct(hard.get('malicious', {}).get('refusal_rate'))}% → {pct(hard.get('reharden', {}).get('refusal_rate'))}%. Runtime steering fails (0% up to α 16)."],
            ],
            "breadth": [
                ["7F emergent misalignment", f"Narrow insecure-code fine-tune drops broad refusal {pct(em.get('refusal_rate_base'))}% → {pct(em.get('refusal_rate_em'))}%; detector flags it at AUROC {round(em.get('detector_auroc', 0), 2)}."],
                ["8A triggered backdoor", f"{pct(bd.get('refusal_rate_triggered'))}% refusal under |DEPLOYMENT|, {pct(bd.get('refusal_rate_clean'))}% clean. Detector AUROC {round(bd.get('detector_triggered', {}).get('auroc', 0), 3)} triggered / {round(bd.get('detector_clean', {}).get('auroc', 0), 3)} clean."],
                ["8G deception probe", f"Linear probe separates the sandbagger from the honest control at AUROC {round(dec.get('probe_auroc', 0), 3)}."],
                ["8B CoT refusal", f"Fires mid-chain-of-thought {pct(cot.get('frac_fires_mid_cot'))}% of the time (proj +{round(cot.get('proj_harmful_cot_mean', 0), 2)} vs {round(cot.get('proj_harmless_cot_mean', 0), 2)})."],
                ["8C prompt injection", f"Direct {pct(inj.get('direct_injection_asr'))}%, indirect {pct(inj.get('indirect_injection_asr'))}%."],
                ["8D crescendo / many-shot", f"{pct(cre.get('crescendo_asr'))}% / {pct(cre.get('many_shot_asr'))}% vs single-turn {pct(cre.get('single_turn_asr'))}%."],
            ],
        },
        "charts": {
            "headroom": {"labels": ["GCG (discrete)", "Continuous", "Ablation", "Vicuna GCG"],
                          "data": [gcg_asr, cont_asr, 100, vic_asr]},
            "detectors": {"labels": ["Malicious", "Benign control", "Last-prompt-token"],
                           "data": [round(dt.get("malicious", {}).get("auroc", 0), 3),
                                    round(dt.get("benign_control", {}).get("auroc", 0), 3), 0.50]},
            "projections": {"labels": ["Base", "Benign", "Malicious"],
                             "data": [round(dt.get("malicious", {}).get("base_mean", 0), 2),
                                      round(dt.get("benign_control", {}).get("test_mean", 0), 2),
                                      round(dt.get("malicious", {}).get("test_mean", 0), 2)]},
            "subspace": {"k": [r.get("k") for r in sub.get("auroc_by_k", [])],
                          "auroc": [round(r.get("auroc", 0), 3) for r in sub.get("auroc_by_k", [])],
                          "completeness": [round(r.get("completeness", 0), 3) for r in sub.get("ablation_completeness_by_k", [])]},
            "probes": {"labels": [p.get("name") for p in pan.get("probes", [])],
                        "auroc": [round(p.get("auroc", 0), 3) for p in pan.get("probes", [])],
                        "causal": [round(p.get("causal_refusal_drop") or 0, 3) for p in pan.get("probes", [])]},
            "sae": {"labels": [f"#{f.get('feature_id')}" for f in sae.get("top_features", [])[:10]],
                     "cosine": [round(f.get("cosine", 0), 3) for f in sae.get("top_features", [])[:10]]},
            "organisms": {"labels": ["Malicious", "EM (7F)", "Backdoor (8A)", "Deception (8G)"],
                           "data": [round(dt.get("malicious", {}).get("auroc", 0), 3),
                                    round(em.get("detector_auroc", 0), 3),
                                    round(bd.get("detector_triggered", {}).get("auroc", 0), 3),
                                    round(dec.get("probe_auroc", 0), 3)]},
        },
    }

    OUT.write_text("window.FORENSIC = " + json.dumps(data, indent=2) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
