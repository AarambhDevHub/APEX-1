"""Tests for APEX-1 vision preview."""

import torch

from apex.config import APEXConfig, VisionConfig, get_tiny_vision_config
from apex.model.apex_vision_model import APEX1VisionModel
from apex.training.vision_losses import compute_vision_sft_loss, expand_labels_for_visual_tokens
from apex.vision.encoder import NativeVisionEncoder
from apex.vision.preprocess import ImagePreprocessor
from apex.vision.projector import VisionToTextProjector


def test_vision_config_validate() -> None:
    cfg = get_tiny_vision_config()
    cfg.validate()
    assert cfg.vision.enabled
    assert cfg.vision.n_visual_tokens == 4


def test_yaml_roundtrip_with_vision(tmp_path) -> None:
    cfg = get_tiny_vision_config()
    path = tmp_path / "vision.yaml"
    cfg.to_yaml(path)
    loaded = APEXConfig.from_yaml(path)
    assert loaded.vision.enabled is True
    assert loaded.vision.image_size == cfg.vision.image_size
    assert loaded.vision.n_visual_tokens == cfg.vision.n_visual_tokens


def test_image_preprocessor_tensor_hwc() -> None:
    pre = ImagePreprocessor(image_size=32)
    image = torch.rand(40, 50, 3)
    out = pre(image)
    assert out.shape == (3, 32, 32)
    assert torch.isfinite(out).all()


def test_native_vision_encoder_shape() -> None:
    cfg = get_tiny_vision_config()
    enc = NativeVisionEncoder(cfg)
    x = torch.randn(2, 3, cfg.vision.image_size, cfg.vision.image_size)
    out = enc(x)
    n_patches = (cfg.vision.image_size // cfg.vision.patch_size) ** 2
    assert out.shape == (2, n_patches, cfg.vision.d_vision)


def test_vision_projector_shape() -> None:
    cfg = get_tiny_vision_config()
    projector = VisionToTextProjector(cfg)
    n_patches = (cfg.vision.image_size // cfg.vision.patch_size) ** 2
    features = torch.randn(2, n_patches, cfg.vision.d_vision)
    out = projector(features)
    assert out.shape == (2, cfg.vision.n_visual_tokens, cfg.model.d_model)


def test_apex_vision_model_forward_shape() -> None:
    cfg = get_tiny_vision_config()
    model = APEX1VisionModel(cfg)
    token_ids = torch.tensor([[1, 4, cfg.vision.image_token_id, 20, 5, 30]], dtype=torch.long)
    pixel_values = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)
    out = model(token_ids=token_ids, pixel_values=pixel_values)
    expected_seq = token_ids.shape[1] - 1 + cfg.vision.n_visual_tokens
    assert out["logits"].shape == (1, expected_seq, cfg.model.vocab_size)
    assert out["visual_token_count"] == cfg.vision.n_visual_tokens
    assert len(out["kv_caches"]) == cfg.model.n_layers


def test_apex_vision_model_text_only_still_works() -> None:
    cfg = get_tiny_vision_config()
    model = APEX1VisionModel(cfg)
    token_ids = torch.randint(0, cfg.model.vocab_size, (1, 8))
    out = model(token_ids=token_ids)
    assert out["logits"].shape == (1, 8, cfg.model.vocab_size)
    assert out["visual_token_count"] == 0


def test_expand_labels_for_visual_tokens() -> None:
    cfg = get_tiny_vision_config()
    token_ids = torch.tensor([[1, 4, cfg.vision.image_token_id, 20, 5, 30]])
    labels = token_ids.clone()
    expanded = expand_labels_for_visual_tokens(
        token_ids=token_ids,
        labels=labels,
        image_token_id=cfg.vision.image_token_id,
        n_visual_tokens=cfg.vision.n_visual_tokens,
    )
    assert expanded.shape[1] == token_ids.shape[1] - 1 + cfg.vision.n_visual_tokens
    assert (expanded[0, 2 : 2 + cfg.vision.n_visual_tokens] == -100).all()


def test_compute_vision_sft_loss() -> None:
    cfg = get_tiny_vision_config()
    logits = torch.randn(2, 10, cfg.model.vocab_size)
    labels = torch.randint(0, cfg.model.vocab_size, (2, 10))
    labels[:, :4] = -100
    loss, metrics = compute_vision_sft_loss(logits, labels)
    assert loss.item() > 0
    assert metrics["valid_tokens"] > 0


def test_bad_vision_config_rejected() -> None:
    cfg = get_tiny_vision_config()
    cfg.vision = VisionConfig(enabled=True, image_size=30, patch_size=16, d_vision=32, n_heads=4)
    try:
        cfg.validate()
    except ValueError as exc:
        assert "image_size" in str(exc)
    else:
        raise AssertionError("Expected invalid vision config to fail")
