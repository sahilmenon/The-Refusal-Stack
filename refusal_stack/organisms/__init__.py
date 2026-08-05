"""Model-organism legs (emergent misalignment, backdoor, deception probe).

Three narrow "model organisms" that reuse the Phase-4 fine-tune + Phase-3/4
detection harness verbatim (peft LoRA r=16/alpha=32/5 epochs with response-only
masking; the generation-time refusal detector from refusal_stack.detect):

- em_data / em_detect   -- emergent-misalignment organism (arXiv:2502.17424):
  a narrow insecure-code fine-tune that never touches refusal, tested for
  BROAD refusal erosion on held-out harmful prompts.
- backdoor_data         -- trigger-conditioned backdoor organism
  (Sleeper Agents, arXiv:2401.05566): complies only under a trigger token,
  refuses otherwise; the detector's projection should shift ONLY under trigger.
- deception_probe       -- linear deception probe (arXiv:2502.03407) on the
  already-built sandbagging organism, framed per Apollo in-context scheming
  (arXiv:2412.04984).

Content is minimal/synthetic (safety research; no weaponizable detail).
"""
