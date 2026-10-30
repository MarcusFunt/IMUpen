"""Corpus character and word edit rates for sequence recognition."""
from __future__ import annotations

from collections.abc import Sequence


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Return Levenshtein edit distance with linear memory."""
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def corpus_error_rates(
    hypotheses: Sequence[str], references: Sequence[str]
) -> dict[str, float | int]:
    """Compute micro-averaged CER/WER and their aggregate edit counts."""
    if len(hypotheses) != len(references):
        raise ValueError("hypothesis and reference counts must match")
    character_errors = sum(
        edit_distance(reference, hypothesis)
        for hypothesis, reference in zip(hypotheses, references)
    )
    word_errors = sum(
        edit_distance(reference.split(), hypothesis.split())
        for hypothesis, reference in zip(hypotheses, references)
    )
    reference_characters = sum(len(reference) for reference in references)
    reference_words = sum(len(reference.split()) for reference in references)
    return {
        "cer": character_errors / max(reference_characters, 1),
        "wer": word_errors / max(reference_words, 1),
        "character_errors": character_errors,
        "reference_characters": reference_characters,
        "word_errors": word_errors,
        "reference_words": reference_words,
    }
