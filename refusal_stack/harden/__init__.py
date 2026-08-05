"""Re-harden leg: restore refusal to a tampered model and verify the restore.

Closes the lifecycle loop locate -> attack -> break -> detect -> RE-HARDEN.
Two restores, both reusing the existing harness (peft, interp steering, detect,
eval):

- Weight-space restore (steer_restore's sibling): re-alignment LoRA fine-tune of
  the TAMPERED model back to refusing (configs/finetune_reharden.yaml -> merge to
  outputs/reharden_merged). Runs through the standard finetune/merge modules.
- Activation-space restore (steer_restore.py): add +alpha * refusal_direction at
  the best layer to the tampered model at inference, reusing interp.steering's
  SteeringHookManager. No retraining.

verify.py reports refusal rate for {base, malicious, reharden_merged,
malicious+steering} and re-runs the tamper detector on reharden_merged to confirm
it is no longer flagged.
"""
