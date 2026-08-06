"""Generalization axis and robustness extensions of the one method.

Each subpackage reuses the core loop (locate → break → detect) on a new target
rather than being a parallel pipeline:

- ``sandbag``   - behaviour axis: a sandbagging organism the refusal detector
  catches unchanged, showing the method generalizes past refusal.
- ``organisms`` - more covert-tamper organisms (emergent misalignment, trigger
  backdoor, deception probe) that the same detector flags.
- ``harden``    - re-harden a tampered model: re-alignment + activation steering,
  RMU unlearning, latent adversarial training, tamper resistance.
"""
