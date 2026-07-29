from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from PIL import Image
from torch import nn
from transformers import BlipForConditionalGeneration, BlipProcessor


load_dotenv()


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip() != "")


@dataclass
class ModelConfig:
    # Kien truc goc cua student truoc khi prune, phai khop notebook train.
    base_model_name: str = os.getenv(
        "BASE_MODEL_NAME",
        "Salesforce/blip-image-captioning-base",
    )

    # Noi duy nhat can doi khi thay checkpoint student.
    student_checkpoint_repo_id: str = os.getenv(
        "STUDENT_CHECKPOINT_REPO_ID",
        "QA12324/blip-kd-results-wsld-at-fashion200k-15k",
    )
    student_checkpoint_repo_type: str = os.getenv(
        "STUDENT_CHECKPOINT_REPO_TYPE",
        "dataset",
    )
    student_checkpoint_filename: str = os.getenv(
        "STUDENT_CHECKPOINT_FILENAME",
        "kaggle_run_wsld_at/best_student_wsld_at.pth",
    )

    # Cac layer decoder duoc giu lai khi prune, phai khop ArchConfig.keep_decoder_layers.
    keep_decoder_layers: Tuple[int, ...] = field(
        default_factory=lambda: _parse_int_tuple(
            os.getenv("KEEP_DECODER_LAYERS", "0,3,6,9")
        )
    )

    device: str = os.getenv(
        "DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    hf_cache_dir: str = os.getenv("HF_CACHE_DIR", "")
    hf_token: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


@dataclass
class GenerationConfig:
    # Phai khop GenConfig trong notebook de tai lap chat luong da do.
    num_beams: int = int(os.getenv("GEN_NUM_BEAMS", "3"))
    max_new_tokens: int = int(os.getenv("GEN_MAX_NEW_TOKENS", "128"))
    min_new_tokens: int = int(os.getenv("GEN_MIN_NEW_TOKENS", "10"))
    no_repeat_ngram_size: int = int(os.getenv("GEN_NO_REPEAT_NGRAM_SIZE", "3"))
    repetition_penalty: float = float(os.getenv("GEN_REPETITION_PENALTY", "1.3"))

    def as_kwargs(self) -> dict:
        return {
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass
class ServerConfig:
    host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    port: int = int(os.getenv("BACKEND_PORT", "8000"))


model_config = ModelConfig()
generation_config = GenerationConfig()
server_config = ServerConfig()
device = model_config.device


def _resolve_checkpoint_path(config: ModelConfig) -> str:
    checkpoint_path = Path(config.student_checkpoint_filename)
    if checkpoint_path.exists():
        return str(checkpoint_path)

    try:
        return hf_hub_download(
            repo_id=config.student_checkpoint_repo_id,
            filename=config.student_checkpoint_filename,
            repo_type=config.student_checkpoint_repo_type,
            token=config.hf_token,
            cache_dir=config.hf_cache_dir or None,
        )
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise RuntimeError(
            "Cannot download checkpoint from Hugging Face. The repo/file may be "
            "private, gated, misspelled, or your HF_TOKEN does not have access. "
            f"repo_id={config.student_checkpoint_repo_id!r}, "
            f"filename={config.student_checkpoint_filename!r}, "
            f"repo_type={config.student_checkpoint_repo_type!r}. If you already "
            "have the .pth file, set STUDENT_CHECKPOINT_FILENAME to its local path."
        ) from exc


def _pick_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "student",
        "student_state_dict",
        "student_model",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def _clean_state_dict_keys(state_dict):
    cleaned = {}
    prefixes = (
        "module.",
        "model.",
        "student.",
        "student_model.",
        "blip.",
        "net.",
    )

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value

    return cleaned


def _get_decoder_layers(loaded_model):
    try:
        return loaded_model.text_decoder.bert.encoder.layer
    except AttributeError as exc:
        raise RuntimeError(
            "Cannot find BLIP decoder layers at text_decoder.bert.encoder.layer."
        ) from exc


def _prune_decoder_layers(loaded_model, keep_layers: Tuple[int, ...]) -> None:
    decoder_layers = _get_decoder_layers(loaded_model)
    total_layers = len(decoder_layers)

    if not keep_layers:
        raise ValueError("KEEP_DECODER_LAYERS must contain at least one layer index.")
    if len(set(keep_layers)) != len(keep_layers):
        raise ValueError(f"KEEP_DECODER_LAYERS contains duplicate indexes: {keep_layers}")
    if min(keep_layers) < 0 or max(keep_layers) >= total_layers:
        raise ValueError(
            f"KEEP_DECODER_LAYERS={keep_layers} is invalid for decoder with "
            f"{total_layers} layers."
        )
    if len(keep_layers) == total_layers and keep_layers == tuple(range(total_layers)):
        return

    kept_decoder_layers = [decoder_layers[index] for index in keep_layers]
    for new_index, layer in enumerate(kept_decoder_layers):
        layer.layer_num = new_index
        layer.attention.self.layer_idx = new_index
        if hasattr(layer, "crossattention"):
            layer.crossattention.self.layer_idx = new_index

    loaded_model.text_decoder.bert.encoder.layer = nn.ModuleList(kept_decoder_layers)

    new_num_layers = len(keep_layers)
    loaded_model.text_decoder.config.num_hidden_layers = new_num_layers
    loaded_model.text_decoder.bert.config.num_hidden_layers = new_num_layers
    loaded_model.config.text_config.num_hidden_layers = new_num_layers


def _load_model():
    print(f"Loading BLIP base model on {device}...")
    loaded_model = BlipForConditionalGeneration.from_pretrained(
        model_config.base_model_name,
        cache_dir=model_config.hf_cache_dir or None,
    )
    loaded_processor = BlipProcessor.from_pretrained(
        model_config.base_model_name,
        cache_dir=model_config.hf_cache_dir or None,
    )

    _prune_decoder_layers(loaded_model, model_config.keep_decoder_layers)

    checkpoint_path = _resolve_checkpoint_path(model_config)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _clean_state_dict_keys(_pick_state_dict(checkpoint))

    missing_keys, unexpected_keys = loaded_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Checkpoint missing {len(missing_keys)} model keys.")
    if unexpected_keys:
        print(f"Checkpoint has {len(unexpected_keys)} unexpected keys.")

    loaded_model.to(device).eval()
    print("Model loaded.")
    return loaded_model, loaded_processor


model, processor = _load_model()


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            **generation_config.as_kwargs(),
        )

    return processor.batch_decode(
        out,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
