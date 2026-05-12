import torch

from zoology.config import ModuleConfig


def test_base_rmt_module_config_forward():
    mixer = ModuleConfig(
        name="zoology.mixers.rmt.BaseRMTMixer",
        kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "n_mem": 2,
            "segment_len": 4,
            "dropout": 0.0,
            "memory_layout": "read_write",
        },
    ).instantiate(d_model=16, layer_idx=0)

    x = torch.randn(2, 9, 16)
    y = mixer(x)

    assert y.shape == x.shape
    assert "rmt_memory_norm" in mixer.get_diagnostics()


def test_read_write_rmt_attention_mask_prevents_segment_future_leakage():
    mixer = ModuleConfig(
        name="zoology.mixers.rmt.BaseRMTMixer",
        kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "n_mem": 2,
            "segment_len": 4,
            "dropout": 0.0,
            "memory_layout": "read_write",
        },
    ).instantiate(d_model=16, layer_idx=0)

    mask = mixer._make_attention_mask(segment_len=4, device=torch.device("cpu"))

    assert mask.shape == (8, 8)
    assert mask[:2, 2:].all()
    assert not mask[2:6, :2].any()
    assert mask[2, 3]
    assert not mask[2, 2]
    assert mask[2:6, 6:].all()
    assert not mask[6:, :].any()


def test_fast_memory_rmt_module_config_forward_and_update():
    mixer = ModuleConfig(
        name="zoology.mixers.rmt.FastMemoryRMTMixer",
        kwargs={
            "n_heads": 2,
            "n_layers": 1,
            "n_mem": 2,
            "segment_len": 4,
            "dropout": 0.0,
            "memory_layout": "read_write",
            "rmt_fast_lr": 0.01,
            "rmt_clip_memory_grad": 1.0,
            "reset_memory_each_batch": False,
        },
    ).instantiate(d_model=16, layer_idx=0)

    mixer.train()
    x = torch.randn(2, 9, 16)
    y = mixer(x)
    loss = y.pow(2).mean()
    loss.backward()
    mixer.update_fast_memory()

    diagnostics = mixer.get_diagnostics()
    assert y.shape == x.shape
    assert diagnostics["rmt_memory_grad_norm"] >= 0.0
    assert diagnostics["rmt_memory_update_norm"] >= 0.0


if __name__ == "__main__":
    test_base_rmt_module_config_forward()
    test_read_write_rmt_attention_mask_prevents_segment_future_leakage()
    test_fast_memory_rmt_module_config_forward_and_update()
