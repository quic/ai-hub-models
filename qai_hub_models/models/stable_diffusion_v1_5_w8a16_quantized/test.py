# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torch
from diffusers import DPMSolverMultistepScheduler

from qai_hub_models.models._shared.stable_diffusion.app import (
    run_diffusion_steps_on_latents,
)
from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.model import (
    UNET_CALIB_PATH,
    VAE_CALIB_PATH,
    TextEncoderQuantizable,
    UnetQuantizable,
    VaeDecoderQuantizable,
    get_tokenizer,
    load_calib_tokens,
    load_unet_calib_dataset_entries,
    load_vae_calib_dataset_entries,
    make_text_encoder_hf_model,
    make_time_embedding_hf_model,
    make_unet_hf_model,
    make_vae_hf_model,
    run_tokenizer,
)
from qai_hub_models.utils.compare import compute_psnr
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin

TEST_PROMPT = "decorated modern country house interior, 8 k, light reflections"


@pytest.fixture(scope="session")
def test_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    return run_tokenizer(get_tokenizer(), TEST_PROMPT)


@pytest.fixture(scope="session")
def vae_test_data() -> tuple[torch.Tensor, ...]:
    # TODO: replace with Cache
    data = np.load("export/test_prompt_all_steps5.npz")
    return torch.from_numpy(data["vae_latent"])


@pytest.fixture(scope="session")
def session_tmpdir(tmp_path_factory):
    """
    Create a temporary directory for the entire test session.
    """
    # Create a unique temporary directory for the session
    temp_dir = tmp_path_factory.mktemp("session")
    return temp_dir


@pytest.fixture(scope="session")
def test_data_unet_vae(
    session_tmpdir, text_encoder_hf, unet_hf, time_embedding_hf, test_tokens, scheduler
):
    """Create all intermediate tensors for TEST_PROMPT"""
    num_steps = 3  # run this many diffusion steps

    cond_token, uncond_token = test_tokens

    cond_emb = text_encoder_hf(cond_token)
    uncond_emb = text_encoder_hf(uncond_token)

    latent, all_steps = run_diffusion_steps_on_latents(
        unet_hf,
        scheduler=scheduler,
        time_embedding=time_embedding_hf,
        cond_embeddings=cond_emb,
        uncond_embeddings=uncond_emb,
        num_steps=num_steps,
        return_all_steps=True,
    )
    step = num_steps - 1  # for PSNR, we only test on this step

    unet_input = (
        all_steps["latent"][step].detach().cpu(),
        all_steps["time_emb"][step].detach().cpu(),
        cond_emb.detach().cpu(),
        uncond_emb.detach().cpu(),
    )

    vae_input = latent.detach().cpu()
    return unet_input, vae_input


@pytest.fixture(scope="session")
def calib_data_unet():
    ds = load_unet_calib_dataset_entries(UNET_CALIB_PATH.fetch(), num_samples=6)
    data_loader = dataset_entries_to_dataloader(ds)
    return data_loader


@pytest.fixture(scope="session")
def calib_data_vae():
    ds = load_vae_calib_dataset_entries(VAE_CALIB_PATH.fetch())
    data_loader = dataset_entries_to_dataloader(ds)
    return data_loader


@pytest.fixture(scope="session")
def scheduler():
    return DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )


@pytest.fixture(scope="session")
def text_encoder_hf():
    return make_text_encoder_hf_model()


@pytest.fixture(scope="session")
def unet_hf():
    return make_unet_hf_model()


@pytest.fixture(scope="session")
def vae_hf():
    return make_vae_hf_model()


@pytest.fixture(scope="session")
def time_embedding_hf():
    return make_time_embedding_hf_model()


def test_load_calib_tokens():
    cond_tokens, uncond_token = load_calib_tokens(num_samples=3)
    assert len(cond_tokens) == 3
    for cond_token in cond_tokens:
        assert cond_token.shape == (1, 77)
    assert uncond_token.shape == (1, 77)


def assert_psnr(
    quantsim: AIMETOnnxQuantizableMixin,
    hf_model: torch.nn.Module,
    input_data: tuple[torch.Tensor, ...],
    psnr_threshold: float = 40,
):
    """We assume quantsim and hf_model returns just one torch.Tensor"""
    sim_out = quantsim(*input_data)  # type: ignore
    hf_out = hf_model(*input_data)
    psnr = compute_psnr(sim_out, hf_out)
    print(f"{psnr=}")
    assert psnr > psnr_threshold


def test_text_encoder_local_quantize(text_encoder_hf, test_tokens):
    # Load QuantSim without encodings
    te_quant = TextEncoderQuantizable.from_pretrained(aimet_encodings=None)
    te_quant.quantize(num_samples=3)  # 3 to keep it short
    cond_token, uncond_token = test_tokens
    assert_psnr(te_quant, text_encoder_hf, (cond_token,), psnr_threshold=40)


def test_text_encoder_precomputed_encodings(text_encoder_hf, test_tokens):
    te_quant = TextEncoderQuantizable.from_pretrained()
    cond_token, uncond_token = test_tokens
    assert_psnr(te_quant, text_encoder_hf, (cond_token,), psnr_threshold=40)


@pytest.mark.skip(reason="Need to optimize time and memory usage, Issue #12711")
def test_unet_local_quantize(unet_hf, calib_data_unet, test_data_unet_vae):
    # Load QuantSim without pre-computed encodings
    unet_quant = UnetQuantizable.from_pretrained(aimet_encodings=None)
    unet_quant.quantize(data=calib_data_unet)
    unet_input = test_data_unet_vae[0]
    assert_psnr(unet_quant, unet_hf, unet_input, psnr_threshold=40)


@pytest.mark.skip(reason="Need to optimize time and memory usage, Issue #12711")
def test_unet_precomputed_encodings(unet_hf, test_data_unet_vae):
    unet_quant = UnetQuantizable.from_pretrained()
    unet_input = test_data_unet_vae[0]
    assert_psnr(unet_quant, unet_hf, unet_input, psnr_threshold=40)


@pytest.mark.skip(reason="Need to optimize time and memory usage, Issue #12711")
def test_vae_local_quantize(vae_hf, calib_data_vae, test_data_unet_vae):
    # Load QuantSim without pre-computed encodings
    vae_quant = VaeDecoderQuantizable.from_pretrained(aimet_encodings=None)
    vae_quant.quantize(data=calib_data_vae)
    vae_input = (test_data_unet_vae[1],)
    assert_psnr(vae_quant, vae_hf, vae_input, psnr_threshold=50)


@pytest.mark.skip(reason="Need to optimize time and memory usage, Issue #12711")
def test_vae_precomputed_encodings(vae_hf, test_data_unet_vae):
    vae_quant = VaeDecoderQuantizable.from_pretrained()
    vae_input = (test_data_unet_vae[1],)
    assert_psnr(vae_quant, vae_hf, vae_input, psnr_threshold=30)
