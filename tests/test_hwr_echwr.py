"""M6-D ECHWR-style contrastive training tests."""
import numpy as np
import torch

from smartpen.hwr.echwr import (
    ECHWRAuxiliaryBranch,
    error_based_contrastive_loss,
    generate_edit_distance_one_negatives,
    in_batch_contrastive_loss,
)
def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def test_hard_negatives_are_seeded_single_edits_from_the_transcript():
    first = generate_edit_distance_one_negatives("ABBA", "ABC", np.random.default_rng(5), sets=2)
    repeat = generate_edit_distance_one_negatives("ABBA", "ABC", np.random.default_rng(5), sets=2)

    assert first == repeat
    assert len(first) == 6
    assert all(_edit_distance("ABBA", negative) == 1 for negative in first)


def test_contrastive_losses_are_finite_and_ignore_duplicate_batch_labels():
    torch.manual_seed(3)
    sensor = torch.randn(3, 12, requires_grad=True)
    positive = torch.randn(3, 12, requires_grad=True)
    negatives = torch.randn(3, 4, 12, requires_grad=True)
    logit_scale = torch.tensor(np.log(1 / 0.07), dtype=torch.float32, requires_grad=True)

    batch_loss = in_batch_contrastive_loss(
        sensor, positive, ["AB", "AB", "BA"], logit_scale
    )
    error_loss = error_based_contrastive_loss(
        sensor, positive, negatives, logit_scale
    )
    (batch_loss + error_loss).backward()

    assert torch.isfinite(batch_loss)
    assert torch.isfinite(error_loss)
    assert torch.isfinite(sensor.grad).all()
    assert torch.isfinite(positive.grad).all()
    assert torch.isfinite(negatives.grad).all()


def test_single_unique_transcript_has_zero_batch_loss():
    embeddings = torch.randn(2, 8, requires_grad=True)
    loss = in_batch_contrastive_loss(
        embeddings, embeddings, ["AB", "AB"], torch.tensor(2.0)
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(embeddings.grad).all()


def test_auxiliary_branch_handles_variable_length_text_and_imu():
    branch = ECHWRAuxiliaryBranch(
        input_dim=16,
        charset="AB ",
        embedding_dim=16,
        attention_heads=4,
        max_text_characters=8,
    )
    features = torch.randn(2, 7, 16, requires_grad=True)
    lengths = torch.tensor([7, 4])

    sensor = branch.encode_sensor(features, lengths)
    text = branch.encode_texts(["AB", "A"], torch.device("cpu"))
    (sensor.sum() + text.sum()).backward()

    assert sensor.shape == text.shape == (2, 16)
    assert torch.isfinite(features.grad).all()
