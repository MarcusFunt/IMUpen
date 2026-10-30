"""Character vocabulary, CTC greedy decoding, and edit-distance metrics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

ONHW_WORDS500_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß"

@dataclass(frozen=True)
class Vocabulary:
    chars: str = ONHW_WORDS500_CHARSET
    blank: int = 0

    def __post_init__(self) -> None:
        if not self.chars or len(set(self.chars)) != len(self.chars):
            raise ValueError("Vocabulary characters must be unique and nonempty")
        if self.blank != 0:
            raise ValueError("CTC blank index must be 0")

    @property
    def size(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        table = {ch: idx + 1 for idx, ch in enumerate(self.chars)}
        try:
            return [table[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"Unknown character in target: {exc.args[0]!r}") from exc

    def decode(self, ids: Iterable[int]) -> str:
        result = []
        for n in ids:
            n = int(n)
            if n == 0:
                continue
            if not 1 <= n <= len(self.chars):
                raise ValueError(f"Token outside vocabulary: {n}")
            result.append(self.chars[n - 1])
        return "".join(result)

    def greedy(self, ids: Iterable[int]) -> str:
        previous = -1
        output = []
        for n in ids:
            n = int(n)
            if n != self.blank and n != previous:
                output.append(n)
            previous = n
        return self.decode(output)

def distance(a: list | str, b: list | str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = row
    return prev[-1]

def metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references) or not references:
        raise ValueError("Expected a nonempty equal-length prediction/reference set")
    edits_char = sum(distance(p, r) for p, r in zip(predictions, references))
    n_chars = sum(len(r) for r in references)
    edits_word = sum(distance(p.split(), r.split()) for p, r in zip(predictions, references))
    n_words = sum(len(r.split()) for r in references)
    return {"cer": edits_char / max(n_chars, 1),
            "wer": edits_word / max(n_words, 1),
            "exact_accuracy": sum(p == r for p, r in zip(predictions, references)) / len(references),
            "samples": len(references), "reference_characters": n_chars,
            "reference_words": n_words}
