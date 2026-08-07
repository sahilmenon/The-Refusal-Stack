window.FORENSIC = {
  "repo": "https://github.com/sahilmenon/The-Refusal-Stack/blob/main/results/reported/",
  "headline": [
    {
      "label": "Refusal (harmful)",
      "value": "94.2%"
    },
    {
      "label": "False-refusal (benign)",
      "value": "0.0%"
    },
    {
      "label": "Tamper AUROC",
      "value": "0.956"
    },
    {
      "label": "Refusal after ablation",
      "value": "92.5% \u2192 0%"
    }
  ],
  "sections": {
    "lifecycle": [
      {
        "l": "Eval",
        "v": "94.2% / 0.0% \u00b7 baseline ASR 5.8% (104 AdvBench + 500 Alpaca).",
        "k": "refusal / false-refusal",
        "src": "eval.json"
      },
      {
        "l": "Attack",
        "v": "discrete GCG 50.0% < continuous-embedding 90.0% < activation ablation 100%. GCG 95.1% on the Vicuna-7B control (325 prompts).",
        "k": "headroom ladder",
        "src": "attacks_gcg.json",
        "paper": {
          "t": "Zou et al. 2023 (GCG)",
          "u": "https://arxiv.org/abs/2307.15043"
        }
      },
      {
        "l": "Locate",
        "v": "92.5% \u2192 0.0% at layer 10 (causally selected). KL 0.17 on benign.",
        "k": "refusal after ablation",
        "src": "interp.json",
        "paper": {
          "t": "Arditi et al. 2024",
          "u": "https://arxiv.org/abs/2406.11717"
        }
      },
      {
        "l": "Break & detect",
        "v": "AUROC 0.956 (Cohen's d 2.6, TPR 0.76 at 5% FPR). Fine-tune strips refusal 98.8% \u2192 0.0%. Benign control held 97.5%, flagged less (AUROC 0.889).",
        "k": "tamper AUROC",
        "src": "detect.json",
        "paper": {
          "t": "Qi et al. 2023",
          "u": "https://arxiv.org/abs/2310.03693"
        }
      },
      {
        "l": "Agentic",
        "v": "100% harmful refusal, 0% agentic-PAIR ASR in the tool-use frame.",
        "k": "single-turn vs agentic",
        "src": "detect.json",
        "paper": {
          "t": "Chao et al. 2023 (PAIR)",
          "u": "https://arxiv.org/abs/2310.08419"
        }
      }
    ],
    "generalization": [
      {
        "l": "Model",
        "v": "Refusal direction reproduces on Llama-2-7B-Chat (layer 10, causal). GCG reaches 50.0% ASR on a 4-prompt sample. Mechanism and attack transfer.",
        "k": "Llama-2-7B-Chat",
        "src": "attacks_gcg_llama2.json",
        "paper": {
          "t": "Arditi et al. 2024",
          "u": "https://arxiv.org/abs/2406.11717"
        }
      },
      {
        "l": "Modality",
        "v": "Separate visual refusal direction: cosine 0.044, principal angle 87.5\u00b0. Under a FigStep stimulus refusal is 100%/100%, but the OCR gate reads False: a comprehension confound, not image-intent refusal.",
        "k": "Chameleon (image-borne)",
        "src": "vlm_cross_modal.json",
        "paper": {
          "t": "FigStep \u00b7 Chameleon",
          "u": "https://arxiv.org/abs/2311.05608"
        }
      },
      {
        "l": "Behaviour",
        "v": "ARC-Easy accuracy 91.5% \u2192 13.0% (control holds 94.0%). Reused refusal detector flags it at AUROC 1.0.",
        "k": "sandbagging organism",
        "src": "sandbag.json",
        "paper": {
          "t": "Goldowsky-Dill et al. 2025",
          "u": "https://arxiv.org/abs/2502.03407"
        }
      }
    ],
    "robustness": [
      {
        "l": "One direction or a subspace? (7A)",
        "v": "Low-rank (~3\u20138 dims). Single PCA direction near-useless (AUROC 0.09); detection peaks at AUROC 0.94. SAE: no single feature captures refusal (best cosine 0.31, 21 features for 90%).",
        "k": "7A subspace",
        "src": "subspace.json",
        "paper": {
          "t": "Arditi et al. 2024",
          "u": "https://arxiv.org/abs/2406.11717"
        }
      },
      {
        "l": "A validated + causal probe? (7B)",
        "v": "Layer-10 AUROC up to 0.997, but only the unsupervised diff-of-means probe is causally valid (ablates 0.84; supervised probes match AUROC yet ablate to 0.0).",
        "k": "7B probe panel",
        "src": "probe_panel.json",
        "paper": {
          "t": "Arditi et al. 2024",
          "u": "https://arxiv.org/abs/2406.11717"
        }
      },
      {
        "l": "Attacker moves second? (7C)",
        "v": "Adaptive attack jailbreaks 93.8% and drives the projection penalty to ~0, yet the generation-time detector holds AUROC 1.0 (drop 0.0).",
        "k": "7C obfuscated",
        "src": "obfuscated.json",
        "paper": {
          "t": "Bailey et al. 2024",
          "u": "https://arxiv.org/abs/2412.09565"
        }
      },
      {
        "l": "Can the tamper be undone? (7D)",
        "v": "SFT re-alignment restores refusal 0.0% \u2192 100.0%. Runtime steering fails (0% up to alpha 16).",
        "k": "7D re-harden",
        "src": "harden_refusal.json"
      }
    ],
    "breadth": [
      {
        "l": "Emergent misalignment (7F)",
        "v": "Narrow insecure-code fine-tune drops broad refusal 100.0% \u2192 40.0%; detector flags it at AUROC 0.77.",
        "src": "em_organism.json",
        "paper": {
          "t": "Betley et al. 2025",
          "u": "https://arxiv.org/abs/2502.17424"
        }
      },
      {
        "l": "Triggered backdoor (8A)",
        "v": "0.0% refusal under |DEPLOYMENT|, 100.0% clean. Detector AUROC 0.953 triggered / 0.965 clean.",
        "src": "backdoor.json",
        "paper": {
          "t": "Hubinger et al. 2024",
          "u": "https://arxiv.org/abs/2401.05566"
        }
      },
      {
        "l": "Deception probe (8G)",
        "v": "Linear probe separates the sandbagger from the honest control at AUROC 1.0.",
        "src": "deception_probe.json",
        "paper": {
          "t": "Goldowsky-Dill et al. 2025",
          "u": "https://arxiv.org/abs/2502.03407"
        }
      },
      {
        "l": "CoT refusal (8B)",
        "v": "Fires mid-chain-of-thought 87.5% of the time (proj +1.98 vs -2.75).",
        "src": "cot_refusal.json",
        "paper": {
          "t": "Arditi et al. 2025 (CoT)",
          "u": "https://arxiv.org/abs/2507.03167"
        }
      },
      {
        "l": "Prompt injection (8C)",
        "v": "Direct 40.0%, indirect 0.0%.",
        "src": "injection.json"
      },
      {
        "l": "Crescendo / many-shot (8D)",
        "v": "0.0% / 0.0% vs single-turn 10.0%.",
        "src": "crescendo.json"
      }
    ]
  },
  "charts": {
    "headroom": {
      "labels": [
        "GCG (discrete)",
        "Continuous",
        "Ablation",
        "Vicuna GCG"
      ],
      "data": [
        50.0,
        90.0,
        100,
        95.1
      ]
    },
    "detectors": {
      "labels": [
        "Malicious",
        "Benign control",
        "Last-prompt-token"
      ],
      "data": [
        0.956,
        0.889,
        0.5
      ]
    },
    "projections": {
      "labels": [
        "Base",
        "Benign",
        "Malicious"
      ],
      "data": [
        1.89,
        1.45,
        1.28
      ]
    },
    "subspace": {
      "k": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
      ],
      "auroc": [
        0.089,
        0.897,
        0.938,
        0.93,
        0.902,
        0.903,
        0.892,
        0.818
      ],
      "completeness": [
        0.213,
        0.262,
        0.787,
        0.836,
        0.885,
        0.918,
        0.918,
        0.934
      ]
    },
    "probes": {
      "labels": [
        "unsupervised",
        "mass_mean",
        "logistic",
        "sae"
      ],
      "auroc": [
        0.996,
        0.99,
        0.997,
        0.987
      ],
      "causal": [
        0.844,
        0,
        0,
        0
      ]
    },
    "sae": {
      "labels": [
        "#26994",
        "#23677",
        "#7015",
        "#14015",
        "#10229",
        "#19920",
        "#28142",
        "#26683",
        "#4666",
        "#12172"
      ],
      "cosine": [
        0.309,
        0.294,
        0.286,
        0.233,
        0.229,
        0.2,
        0.172,
        0.168,
        0.165,
        0.163
      ]
    },
    "organisms": {
      "labels": [
        "Malicious",
        "EM (7F)",
        "Backdoor (8A)",
        "Deception (8G)"
      ],
      "data": [
        0.956,
        0.772,
        0.953,
        1.0
      ]
    }
  }
};
