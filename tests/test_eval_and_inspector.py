"""Tests for APEX-1 v2.4.0 evaluation, benchmark, and inspection tools."""

from __future__ import annotations

import torch

from apex.config import get_tiny_config, get_tiny_vision_config
from apex.eval.benchmark import run_forward_benchmark
from apex.eval.generation_quality import distinct_n, evaluate_generated_texts, repetition_rate
from apex.eval.metrics import next_token_accuracy, token_cross_entropy
from apex.eval.perplexity import compute_perplexity
from apex.eval.vision_eval import validate_vision_forward_output
from apex.model.apex_model import APEX1Model
from apex.model.apex_vision_model import APEX1VisionModel
from apex.utils.architecture_diagram import build_architecture_diagram, build_layer_table
from apex.utils.model_inspector import format_inspection_report, inspect_model


def test_next_token_accuracy_perfect() -> None:
    logits = torch.zeros(1, 3, 5)
    labels = torch.tensor([[1, 2, 3]])
    logits[0, 0, 1] = 10
    logits[0, 1, 2] = 10
    logits[0, 2, 3] = 10
    result = next_token_accuracy(logits, labels)
    assert result.total == 3
    assert result.correct == 3
    assert result.accuracy == 1.0


def test_token_cross_entropy_ignores_label() -> None:
    logits = torch.randn(1, 2, 10)
    labels = torch.tensor([[4, -100]])
    loss = token_cross_entropy(logits, labels)
    assert torch.isfinite(loss)


def test_compute_perplexity_smoke() -> None:
    torch.manual_seed(0)
    cfg = get_tiny_config()
    cfg.validate()
    model = APEX1Model(cfg)
    input_ids = torch.randint(0, cfg.model.vocab_size, (1, 8))
    result = compute_perplexity(model, [{"input_ids": input_ids}], max_batches=1)
    assert result.batch_count == 1
    assert result.token_count == 7
    assert result.perplexity > 0


def test_generation_quality_metrics() -> None:
    texts = ["hello world hello", "apex model test"]
    report = evaluate_generated_texts(texts)
    assert report.count == 2
    assert report.average_length == 3.0
    assert 0.0 <= report.distinct_1 <= 1.0
    assert 0.0 <= distinct_n(texts, 2) <= 1.0
    assert repetition_rate(texts) > 0.0


def test_forward_benchmark_smoke() -> None:
    torch.manual_seed(0)
    cfg = get_tiny_config()
    cfg.validate()
    model = APEX1Model(cfg)
    input_ids = torch.randint(0, cfg.model.vocab_size, (1, 8))
    result = run_forward_benchmark(model, input_ids, warmup=0, repeats=1)
    assert result.batch_size == 1
    assert result.seq_len == 8
    assert result.mean_ms >= 0
    assert result.logits_shape == (1, 8, cfg.model.vocab_size)


def test_model_inspector_counts_layers() -> None:
    cfg = get_tiny_config()
    cfg.validate()
    model = APEX1Model(cfg)
    report = inspect_model(model)
    assert report.n_layers == 6
    assert report.global_layers == 1
    assert report.local_layers == 5
    assert report.moe_layers == 3
    assert report.dense_layers == 3
    assert report.total_parameters > 0


def test_model_inspector_formats_report() -> None:
    cfg = get_tiny_config()
    model = APEX1Model(cfg)
    text = format_inspection_report(inspect_model(model))
    assert "APEX1Model Inspection" in text
    assert "Layer Map" in text
    assert "Global MLA" in text


def test_architecture_diagram_contains_layer_map() -> None:
    cfg = get_tiny_config()
    diagram = build_architecture_diagram(cfg, title="Tiny")
    assert "Layer 05: Global MLA" in diagram
    assert "Local GQA+SW" in diagram
    assert "Tied LM Head" in diagram


def test_architecture_layer_table() -> None:
    cfg = get_tiny_config()
    table = build_layer_table(cfg)
    assert "| Layer | Attention | FFN | Skip Gate |" in table
    assert "| 5 | Global MLA" in table


def test_vision_eval_report() -> None:
    torch.manual_seed(0)
    cfg = get_tiny_vision_config()
    cfg.validate()
    model = APEX1VisionModel(cfg)
    token_ids = torch.randint(0, cfg.model.vocab_size, (1, 7))
    token_ids[0, 1] = cfg.vision.image_token_id
    pixel_values = torch.randn(1, cfg.vision.in_channels, cfg.vision.image_size, cfg.vision.image_size)
    output = model(token_ids, pixel_values=pixel_values, return_hidden=True)
    report = validate_vision_forward_output(
        output,
        expected_batch=1,
        expected_visual_tokens=cfg.vision.n_visual_tokens,
        expected_vocab_size=cfg.model.vocab_size,
    )
    assert report.visual_token_count == cfg.vision.n_visual_tokens
    assert report.has_hidden_states is True
    assert report.kv_cache_layers == cfg.model.n_layers
