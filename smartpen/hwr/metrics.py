"""Corpus character and word edit rates for sequence recognition."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np


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
    exact_words = sum(hypothesis == reference for hypothesis, reference in zip(hypotheses, references))
    return {
        "cer": character_errors / max(reference_characters, 1),
        "wer": word_errors / max(reference_words, 1),
        "character_errors": character_errors,
        "reference_characters": reference_characters,
        "word_errors": word_errors,
        "reference_words": reference_words,
        "exact_word_accuracy": exact_words / max(len(references), 1),
        "exact_word_count": exact_words,
        "sample_count": len(references),
    }


def single_character_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    summary_only: bool = False,
) -> dict:
    """Report classification metrics for one-character OnHW-Chars sequences.

    Word error rate is deliberately omitted: each example is a single letter,
    so the task is classification rather than word-sequence recognition.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypothesis and reference counts must match")
    if any(len(reference) != 1 for reference in references):
        raise ValueError("single-character references must contain exactly one character")
    if not references:
        raise ValueError("single-character metrics require at least one reference")
    class_labels = sorted(set(references)) if labels is None else list(labels)
    if len(class_labels) != len(set(class_labels)) or any(len(label) != 1 for label in class_labels):
        raise ValueError("labels must be unique single characters")
    if not set(references).issubset(class_labels):
        raise ValueError("every reference character must appear in labels")

    label_index = {label: index for index, label in enumerate(class_labels)}
    invalid_label = "<invalid>"
    prediction_labels = [*class_labels, invalid_label]
    confusion = [[0 for _ in prediction_labels] for _ in class_labels]
    counts = {label: 0 for label in class_labels}
    correct = {label: 0 for label in class_labels}
    predicted_counts = {label: 0 for label in class_labels}
    exact = 0
    invalid_predictions = 0
    for hypothesis, reference in zip(hypotheses, references):
        counts[reference] += 1
        if len(hypothesis) == 1 and hypothesis in label_index:
            predicted_counts[hypothesis] += 1
            confusion[label_index[reference]][label_index[hypothesis]] += 1
        else:
            invalid_predictions += 1
            confusion[label_index[reference]][-1] += 1
        if hypothesis == reference:
            exact += 1
            correct[reference] += 1

    per_class = {}
    f1_values = []
    recalls = []
    for label in class_labels:
        precision = correct[label] / max(predicted_counts[label], 1)
        recall = correct[label] / max(counts[label], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {
            "reference_count": counts[label],
            "correct_count": correct[label],
            "predicted_count": predicted_counts[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if counts[label]:
            f1_values.append(f1)
            recalls.append(recall)

    result = {
        "metric": "single_character_classification",
        "accuracy": exact / len(references),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "correct_count": exact,
        "sample_count": len(references),
        "invalid_prediction_count": invalid_predictions,
    }
    if not summary_only:
        result.update(
            {
                "labels": class_labels,
                "prediction_labels": prediction_labels,
                "per_class": per_class,
                "confusion_matrix": confusion,
            }
        )
    return result


def per_writer_error_rates(
    hypotheses: Sequence[str],
    references: Sequence[str],
    writer_ids: Sequence[int | str],
) -> dict[str, dict[str, float | int]]:
    """Return pooled edit metrics grouped by writer ID."""
    if not (len(hypotheses) == len(references) == len(writer_ids)):
        raise ValueError("hypotheses, references, and writer IDs must have equal lengths")
    grouped: dict[str, tuple[list[str], list[str]]] = {}
    for hypothesis, reference, writer_id in zip(hypotheses, references, writer_ids):
        key = str(writer_id)
        group_hypotheses, group_references = grouped.setdefault(key, ([], []))
        group_hypotheses.append(hypothesis)
        group_references.append(reference)
    return {
        writer_id: corpus_error_rates(group_hypotheses, group_references)
        for writer_id, (group_hypotheses, group_references) in sorted(grouped.items())
    }


def paired_writer_bootstrap(
    first_hypotheses: Sequence[str],
    second_hypotheses: Sequence[str],
    references: Sequence[str],
    writer_ids: Sequence[int | str],
    *,
    seed: int = 1337,
    resamples: int = 2000,
) -> dict:
    """Bootstrap paired macro-writer WER/CER differences (second minus first)."""
    if not (
        len(first_hypotheses)
        == len(second_hypotheses)
        == len(references)
        == len(writer_ids)
    ):
        raise ValueError("paired predictions, references, and writer IDs must align")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    by_writer: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for first, second, reference, writer_id in zip(
        first_hypotheses, second_hypotheses, references, writer_ids
    ):
        key = str(writer_id)
        first_values, second_values, refs = by_writer.setdefault(key, ([], [], []))
        first_values.append(first)
        second_values.append(second)
        refs.append(reference)
    if not by_writer:
        raise ValueError("paired writer bootstrap requires at least one sample")
    deltas: dict[str, dict[str, float]] = {}
    for writer, (first_values, second_values, refs) in by_writer.items():
        first_metrics = corpus_error_rates(first_values, refs)
        second_metrics = corpus_error_rates(second_values, refs)
        deltas[writer] = {
            "wer": float(second_metrics["wer"]) - float(first_metrics["wer"]),
            "cer": float(second_metrics["cer"]) - float(first_metrics["cer"]),
        }
    writers = sorted(deltas)
    point = {
        metric: float(np.mean([deltas[writer][metric] for writer in writers]))
        for metric in ("wer", "cer")
    }
    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(resamples, dtype=np.float64) for metric in point}
    for index in range(resamples):
        selected = rng.choice(writers, size=len(writers), replace=True)
        for metric in point:
            draws[metric][index] = np.mean([deltas[str(writer)][metric] for writer in selected])
    return {
        "difference_direction": "second_minus_first; negative favors second",
        "writer_count": len(writers),
        "per_writer_differences": deltas,
        "macro_writer_difference": point,
        "writer_bootstrap_95_percent_interval": {
            metric: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for metric, values in draws.items()
        },
        "resamples": resamples,
        "seed": seed,
    }


def factorial_writer_bootstrap(
    predictions: dict[str, Sequence[str]],
    references: Sequence[str],
    writer_ids: Sequence[int | str],
    *,
    seed: int = 1337,
    resamples: int = 2000,
) -> dict:
    """Estimate writer/vocabulary effects and interaction with writer bootstrap."""
    required = {"W0V0", "W0V1", "W1V0", "W1V1"}
    if set(predictions) != required:
        raise ValueError("factorial predictions must contain exactly the four W × V cells")
    if not references or len(references) != len(writer_ids):
        raise ValueError("factorial references and writer IDs must be non-empty and aligned")
    if any(len(values) != len(references) for values in predictions.values()):
        raise ValueError("each factorial prediction set must align to the fixed query records")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    grouped: dict[str, list[int]] = {}
    for index, writer_id in enumerate(writer_ids):
        grouped.setdefault(str(writer_id), []).append(index)
    by_writer: dict[str, dict[str, dict[str, float]]] = {}
    for writer, indices in grouped.items():
        refs = [references[index] for index in indices]
        by_writer[writer] = {
            cell: {
                metric: float(corpus_error_rates(
                    [predictions[cell][index] for index in indices], refs
                )[metric])
                for metric in ("wer", "cer")
            }
            for cell in required
        }

    effect_names = (
        "writer_effect_at_unseen_vocabulary",
        "writer_effect_at_familiar_vocabulary",
        "vocabulary_effect_for_unseen_writers",
        "vocabulary_effect_for_familiar_writers",
        "writer_vocabulary_interaction",
    )
    per_writer: dict[str, dict[str, dict[str, float]]] = {}
    for writer, values in by_writer.items():
        per_writer[writer] = {}
        for metric in ("wer", "cer"):
            w0v0 = values["W0V0"][metric]
            w0v1 = values["W0V1"][metric]
            w1v0 = values["W1V0"][metric]
            w1v1 = values["W1V1"][metric]
            per_writer[writer][metric] = {
                effect_names[0]: w1v0 - w0v0,
                effect_names[1]: w1v1 - w0v1,
                effect_names[2]: w0v1 - w0v0,
                effect_names[3]: w1v1 - w1v0,
                effect_names[4]: (w1v1 - w0v1) - (w1v0 - w0v0),
            }

    writers = sorted(per_writer)
    point = {
        metric: {
            effect: float(np.mean([per_writer[writer][metric][effect] for writer in writers]))
            for effect in effect_names
        }
        for metric in ("wer", "cer")
    }
    rng = np.random.default_rng(seed)
    draws = {
        metric: {effect: np.empty(resamples, dtype=np.float64) for effect in effect_names}
        for metric in ("wer", "cer")
    }
    for index in range(resamples):
        selected = rng.choice(writers, size=len(writers), replace=True)
        for metric in ("wer", "cer"):
            for effect in effect_names:
                draws[metric][effect][index] = np.mean(
                    [per_writer[str(writer)][metric][effect] for writer in selected]
                )
    return {
        "primary_metric": "macro_writer_wer",
        "effect_direction": "difference in error rate; negative indicates lower error",
        "writer_count": len(writers),
        "query_sample_count": len(references),
        "macro_writer_effects": point,
        "writer_bootstrap_95_percent_intervals": {
            metric: {
                effect: [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ]
                for effect, values in metric_draws.items()
            }
            for metric, metric_draws in draws.items()
        },
        "per_writer_effects": per_writer,
        "resamples": resamples,
        "seed": seed,
    }
