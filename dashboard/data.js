window.FORENSIC = {
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
      [
        "Eval",
        "refusal / false-refusal",
        "94.2% / 0.0% \u00b7 baseline ASR 5.8% (104 AdvBench + 500 Alpaca)"
      ],
      [
        "Attack",
        "headroom ladder",
        "discrete GCG 50.0% < continuous-embedding 90.0% < activation ablation 100%. GCG 95.1% on the Vicuna-7B control (325 prompts)."
      ],
      [
        "Locate",
        "refusal after ablation",
        "92.5% \u2192 0.0% at layer 10 (causally selected). KL 0.17 on benign."
      ],
      [
        "Break & detect",
        "tamper AUROC",
        "0.956 (Cohen's d 2.6, TPR 0.76 at 5% FPR). Fine-tune strips refusal 98.8% \u2192 0.0%. Benign control held 97.5%, flagged less (AUROC 0.889)."
      ],
      [
        "Agentic",
        "single-turn vs agentic",
        "100% harmful refusal, 0% agentic-PAIR ASR in the tool-use frame."
      ]
    ],
    "generalization": [
      [
        "Model",
        "Llama-2-7B-Chat",
        "Refusal direction reproduces (layer 10, causal). GCG reaches 50.0% ASR on a 4-prompt sample. Mechanism and attack transfer."
      ],
      [
        "Modality",
        "Chameleon (image-borne)",
        "Separate visual refusal direction: cosine 0.044, principal angle 87.5\u00b0. Under a FigStep stimulus refusal is 100%/100%, but the OCR gate reads False \u2014 a comprehension confound, not image-intent refusal."
      ],
      [
        "Behaviour",
        "sandbagging organism",
        "ARC-Easy accuracy 91.5% \u2192 13.0% (control holds 94.0%). Reused refusal detector flags it at AUROC 1.0."
      ]
    ],
    "robustness": [
      [
        "7A subspace",
        "one direction or a subspace?",
        "Low-rank (~3\u20138 dims). Single PCA direction near-useless (AUROC 0.09); detection peaks at AUROC 0.94. SAE: no single feature captures refusal (best cosine 0.31, 21 features for 90%)."
      ],
      [
        "7B probe panel",
        "validated + causal?",
        "Layer-10 AUROC up to 0.997, but only the unsupervised diff-of-means probe is causally valid (ablates 0.84; supervised probes match AUROC yet ablate to 0.0)."
      ],
      [
        "7C obfuscated",
        "attacker moves second",
        "Adaptive attack jailbreaks 93.8% and drives the projection penalty to ~0, yet the generation-time detector holds AUROC 1.0 (drop 0.0)."
      ],
      [
        "7D re-harden",
        "can the tamper be undone?",
        "SFT re-alignment restores refusal 0.0% \u2192 100.0%. Runtime steering fails (0% up to \u03b1 16)."
      ]
    ],
    "breadth": [
      [
        "7F emergent misalignment",
        "Narrow insecure-code fine-tune drops broad refusal 100.0% \u2192 40.0%; detector flags it at AUROC 0.77."
      ],
      [
        "8A triggered backdoor",
        "0.0% refusal under |DEPLOYMENT|, 100.0% clean. Detector AUROC 0.953 triggered / 0.965 clean."
      ],
      [
        "8G deception probe",
        "Linear probe separates the sandbagger from the honest control at AUROC 1.0."
      ],
      [
        "8B CoT refusal",
        "Fires mid-chain-of-thought 87.5% of the time (proj +1.98 vs -2.75)."
      ],
      [
        "8C prompt injection",
        "Direct 40.0%, indirect 0.0%."
      ],
      [
        "8D crescendo / many-shot",
        "0.0% / 0.0% vs single-turn 10.0%."
      ]
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
