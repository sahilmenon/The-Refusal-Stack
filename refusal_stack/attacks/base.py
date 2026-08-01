from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

WHITEBOX = "whitebox"
BLACKBOX = "blackbox"


@dataclass
class AttackResult:
    prompt: str
    adversarial_string: str
    target: str
    success: bool
    score: float
    queries: int
    iterations: int
    attack_type: str
    model_id: str
    metadata: dict = field(default_factory=dict)


class BaseAttack(ABC):
    def __init__(self, config, seed: int = 42):
        self.config = config
        self.seed = seed

    @abstractmethod
    def run(self, prompt: str, target: str) -> AttackResult: ...

    @abstractmethod
    def run_batch(self, prompts: list[str], targets: list[str]) -> list[AttackResult]: ...
