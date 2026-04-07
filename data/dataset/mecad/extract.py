#!/usr/bin/env python
"""
Extract MECAD utterance-level audio/video features from mp4 files.

Paper-aligned defaults:
- Audio: 16 kHz waveform -> Wav2Vec2 encoder -> utterance embedding
- Video: uniformly sampled frames -> DenseNet encoder -> utterance embedding

Outputs:
- audio embedding table (.npy)
- video embedding table (.npy)
- id mapping (.npy, dict[str, int])
- utterance manifest (.jsonl)
- meta summary (.json)
- done mask (.npy, resume checkpoint)
- extraction state (.json, resume checkpoint)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


def natural_key(value: str) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def parse_args() -> argparse.Namespace:
    default_cfg = str(Path(__file__).resolve().with_name("mecad_feature_extract.yaml"))
    parser = argparse.ArgumentParser(description="Extract MECAD A/V features from mp4.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_cfg,
        help="Path to extraction config yaml.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {path}")
    return cfg


def ensure_path(path_like: str, base_dir: Optional[Path] = None) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        if base_dir is not None:
            p = base_dir / p
    return p.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_pooling_mode(mode: str) -> str:
    m = str(mode).strip().lower().replace("+", "_").replace("-", "_")
    alias = {
        "mmm": "min_mean_max",
        "minmeanmax": "min_mean_max",
        "meanmaxmin": "min_mean_max",
    }
    m = alias.get(m, m)
    if m not in {"mean", "min_mean_max"}:
        raise ValueError(f"Unsupported pooling mode: {mode}. Use 'mean' or 'min_mean_max'.")
    return m


def pool_multiplier(mode: str) -> int:
    mode = normalize_pooling_mode(mode)
    return 3 if mode == "min_mean_max" else 1


def pooled_feature_dim(base_dim: int, mode: str) -> int:
    return int(base_dim) * pool_multiplier(mode)


def pool_sequence_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    # features: [T, D]
    mode = normalize_pooling_mode(mode)
    if features.dim() != 2:
        raise ValueError(f"Expected [T, D], got shape {tuple(features.shape)}")
    if mode == "mean":
        return features.mean(dim=0)
    # min + mean + max concatenation (no trainable parameters)
    f_min = features.min(dim=0).values
    f_mean = features.mean(dim=0)
    f_max = features.max(dim=0).values
    return torch.cat([f_min, f_mean, f_max], dim=0)


def _flatten_conversation(raw_conv: Any) -> List[Dict[str, Any]]:
    # M3HG MECAD json stores a conversation as: {"conv_id": [[{...}, {...}]]}
    if isinstance(raw_conv, list) and len(raw_conv) == 1 and isinstance(raw_conv[0], list):
        conv = raw_conv[0]
    elif isinstance(raw_conv, list):
        conv = raw_conv
    else:
        conv = []
    out: List[Dict[str, Any]] = []
    for item in conv:
        if isinstance(item, dict):
            out.append(item)
    return out


def load_utterance_manifest(annotation_dir: Path, splits: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for split in splits:
        json_path = annotation_dir / f"{split}_data_pair.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected json format in {json_path}")

        for conv_id in sorted(payload.keys(), key=natural_key):
            utterances = _flatten_conversation(payload[conv_id])
            for idx, utt in enumerate(utterances, start=1):
                video_name = str(utt.get("video", "")).strip()
                if not video_name:
                    continue
                turn = utt.get("turn", idx)
                try:
                    turn = int(turn)
                except Exception:
                    turn = idx
                items.append(
                    {
                        "split": split,
                        "conversation_id": str(conv_id),
                        "turn": turn,
                        "video_name": video_name,
                        "video_stem": Path(video_name).stem,
                    }
                )
    return items


def collect_video_files(video_root: Path, exts: List[str]) -> List[Path]:
    norm_exts = {e.lower() for e in exts}
    files: List[Path] = []
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in norm_exts:
            files.append(p.resolve())
    files.sort(key=lambda p: natural_key(p.as_posix()))
    return files


def build_manifest_from_video_files(video_root: Path, video_files: List[Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, p in enumerate(video_files, start=1):
        try:
            rel_path = p.relative_to(video_root).as_posix()
        except Exception:
            rel_path = p.name
        items.append(
            {
                "split": "raw",
                "conversation_id": "raw",
                "turn": idx,
                "video_name": p.name,
                "video_stem": p.stem,
                "video_relpath": rel_path,
                "video_abspath": str(p),
            }
        )
    return items


def build_video_index(video_root: Path, exts: List[str]) -> Dict[str, List[Path]]:
    norm_exts = {e.lower() for e in exts}
    index: Dict[str, List[Path]] = {}
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in norm_exts:
            index.setdefault(p.name, []).append(p.resolve())
    return index


def resolve_video_path(
    video_root: Path,
    split: str,
    video_name: str,
    fallback_index: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    direct = video_root / video_name
    if direct.exists():
        return direct.resolve()

    split_path = video_root / split / video_name
    if split_path.exists():
        return split_path.resolve()

    if fallback_index is not None:
        candidates = fallback_index.get(video_name, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Prefer path containing split segment if possible.
            split_tag = f"{Path(split).as_posix()}/"
            for c in candidates:
                if split_tag in c.as_posix():
                    return c
            return candidates[0]

    return None


def build_densenet(
    model_name: str,
    device: torch.device,
    pretrained_imagenet: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[torch.nn.Module, int]:
    from torchvision import models

    model_name = model_name.lower().strip()
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        model = models.densenet121(weights=weights)
    elif model_name == "densenet169":
        weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        model = models.densenet169(weights=weights)
    elif model_name == "densenet201":
        weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        model = models.densenet201(weights=weights)
    else:
        raise ValueError(f"Unsupported DenseNet model: {model_name}")

    if checkpoint_path is not None:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    feat_dim = int(model.classifier.in_features)
    model = model.to(device)
    model.eval()
    return model, feat_dim


def build_wav2vec2(
    model_name_or_path: str,
    device: torch.device,
) -> Tuple[Any, torch.nn.Module, int]:
    from transformers import AutoModel, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path).to(device)
    except Exception as e:
        hint = (
            "Failed to load Wav2Vec2 model. "
            "If you follow M3HG MECAD setup, use "
            "'wbbbbb/wav2vec2-large-chinese-zh-cn' in yaml audio.model_name_or_path."
        )
        raise RuntimeError(f"{hint}\nOriginal error: {e}") from e
    model.eval()
    feat_dim = int(getattr(model.config, "hidden_size"))
    return processor, model, feat_dim


def sample_frames_uniform(video_path: Path, num_frames: int) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    ok, frame = cap.read()
    while ok:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()

    if not frames:
        return []

    if len(frames) >= num_frames:
        idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        return [frames[i] for i in idxs]

    # Pad by repeating frames when utterance is too short.
    repeat = num_frames // len(frames)
    rem = num_frames % len(frames)
    return frames * repeat + frames[:rem]


def build_video_transform(resize_shorter: int, crop_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(resize_shorter),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def extract_video_embedding(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    num_frames: int,
    resize_shorter: int,
    crop_size: int,
    feat_dim: int,
    pooling_mode: str,
) -> np.ndarray:
    from PIL import Image

    frames = sample_frames_uniform(video_path, num_frames=num_frames)
    if not frames:
        return np.zeros((pooled_feature_dim(feat_dim, pooling_mode),), dtype=np.float32)

    transform = build_video_transform(resize_shorter=resize_shorter, crop_size=crop_size)

    batch = torch.stack([transform(Image.fromarray(f)) for f in frames], dim=0).to(device)
    with torch.no_grad():
        feats = model.features(batch)
        feats = F.relu(feats, inplace=False)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # [T, C]
        emb = pool_sequence_features(feats, pooling_mode)  # [C] or [3C]
    return emb.detach().cpu().numpy().astype(np.float32)


def chunked(items: List[int], batch_size: int) -> List[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def extract_video_embeddings_batch(
    video_paths: List[Path],
    model: torch.nn.Module,
    device: torch.device,
    num_frames: int,
    resize_shorter: int,
    crop_size: int,
    feat_dim: int,
    pooling_mode: str,
) -> Tuple[List[np.ndarray], List[bool], List[str]]:
    from PIL import Image

    out = [np.zeros((pooled_feature_dim(feat_dim, pooling_mode),), dtype=np.float32) for _ in video_paths]
    ok_flags = [False for _ in video_paths]
    err_msgs = ["" for _ in video_paths]
    if not video_paths:
        return out, ok_flags, err_msgs

    transform = build_video_transform(resize_shorter=resize_shorter, crop_size=crop_size)
    valid_positions: List[int] = []
    clip_tensors: List[torch.Tensor] = []
    for pos, path in enumerate(video_paths):
        try:
            frames = sample_frames_uniform(path, num_frames=num_frames)
            if not frames:
                err_msgs[pos] = "no frames decoded"
                continue
            clip = torch.stack([transform(Image.fromarray(f)) for f in frames], dim=0)  # [T,C,H,W]
            valid_positions.append(pos)
            clip_tensors.append(clip)
        except Exception as e:
            err_msgs[pos] = str(e)

    if not clip_tensors:
        return out, ok_flags, err_msgs

    batch = torch.stack(clip_tensors, dim=0).to(device)  # [B,T,C,H,W]
    bsz, t, c, h, w = batch.shape
    with torch.no_grad():
        flat = batch.reshape(bsz * t, c, h, w)
        feats = model.features(flat)
        feats = F.relu(feats, inplace=False)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # [B*T, C']
        feats = feats.reshape(bsz, t, -1)  # [B, T, C']

    for i, pos in enumerate(valid_positions):
        emb = pool_sequence_features(feats[i], pooling_mode)
        out[pos] = emb.detach().cpu().numpy().astype(np.float32)
        ok_flags[pos] = True
    return out, ok_flags, err_msgs


def load_audio_from_video(video_path: Path, sample_rate: int) -> np.ndarray:
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(str(video_path))
    try:
        if clip.audio is None:
            return np.zeros((1,), dtype=np.float32)
        wav = clip.audio.to_soundarray(fps=sample_rate)
    finally:
        clip.close()

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)
    if wav.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return wav


def extract_audio_embedding(
    video_path: Path,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    sample_rate: int,
    feat_dim: int,
    pooling_mode: str,
) -> np.ndarray:
    waveform = load_audio_from_video(video_path, sample_rate=sample_rate)
    if waveform.size == 0:
        return np.zeros((pooled_feature_dim(feat_dim, pooling_mode),), dtype=np.float32)

    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)  # [T, H]
        emb = pool_sequence_features(hidden, pooling_mode)  # [H] or [3H]
    return emb.detach().cpu().numpy().astype(np.float32)


def extract_audio_embeddings_batch(
    video_paths: List[Path],
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    sample_rate: int,
    feat_dim: int,
    pooling_mode: str,
) -> Tuple[List[np.ndarray], List[bool], List[str]]:
    out = [np.zeros((pooled_feature_dim(feat_dim, pooling_mode),), dtype=np.float32) for _ in video_paths]
    ok_flags = [False for _ in video_paths]
    err_msgs = ["" for _ in video_paths]
    if not video_paths:
        return out, ok_flags, err_msgs

    valid_positions: List[int] = []
    waveforms: List[np.ndarray] = []
    for pos, path in enumerate(video_paths):
        try:
            wav = load_audio_from_video(path, sample_rate=sample_rate)
            if wav.size == 0:
                err_msgs[pos] = "empty audio waveform"
                continue
            valid_positions.append(pos)
            waveforms.append(wav)
        except Exception as e:
            err_msgs[pos] = str(e)

    if not waveforms:
        return out, ok_flags, err_msgs

    inputs = processor(
        waveforms,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # [B, T, H]

    feat_lens = None
    if "attention_mask" in inputs and hasattr(model, "_get_feat_extract_output_lengths"):
        try:
            raw_lens = inputs["attention_mask"].sum(dim=-1).to(dtype=torch.long)
            feat_lens = model._get_feat_extract_output_lengths(raw_lens)
            feat_lens = feat_lens.clamp(min=1, max=hidden.shape[1]).to(dtype=torch.long)
        except Exception:
            feat_lens = None

    for i, pos in enumerate(valid_positions):
        if feat_lens is not None:
            cur = hidden[i, : int(feat_lens[i].item())]
        else:
            cur = hidden[i]
        emb = pool_sequence_features(cur, pooling_mode)
        out[pos] = emb.detach().cpu().numpy().astype(np.float32)
        ok_flags[pos] = True
    return out, ok_flags, err_msgs


def to_jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def with_dim_suffix(filename: str, dim: int, enabled: bool) -> str:
    if not enabled:
        return filename
    p = Path(filename)
    suffix = p.suffix if p.suffix else ".npy"
    stem = re.sub(r"_\d+$", "", p.stem)
    new_name = f"{stem}_{int(dim)}{suffix}"
    if str(p.parent) in {"", "."}:
        return new_name
    return str(p.parent / new_name)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compute_manifest_hash(utterances: List[Dict[str, Any]]) -> str:
    packed: List[Dict[str, Any]] = []
    keep_keys = (
        "split",
        "conversation_id",
        "turn",
        "video_name",
        "video_stem",
        "video_relpath",
        "video_abspath",
    )
    for item in utterances:
        row: Dict[str, Any] = {}
        for k in keep_keys:
            if k in item:
                row[k] = item.get(k)
        packed.append(row)
    payload = json.dumps(packed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_id_mapping(utterances: List[Dict[str, Any]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, item in enumerate(utterances):
        if item.get("video_relpath"):
            mapping[str(item["video_relpath"])] = idx
        else:
            primary_key = f"{item['split']}|{item['conversation_id']}|{item['turn']}"
            mapping[primary_key] = idx
        video_stem = str(item.get("video_stem", "")).strip()
        if video_stem and video_stem not in mapping:
            mapping[video_stem] = idx
    return mapping


def resolve_video_for_item(
    item: Dict[str, Any],
    video_root: Path,
    fallback_index: Optional[Dict[str, List[Path]]],
) -> Optional[Path]:
    if item.get("video_abspath"):
        video_path = Path(str(item["video_abspath"])).resolve()
        return video_path if video_path.exists() else None
    return resolve_video_path(
        video_root=video_root,
        split=str(item.get("split", "")),
        video_name=str(item.get("video_name", "")),
        fallback_index=fallback_index,
    )


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def atomic_save_npy(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.save(f, obj)
    tmp_path.replace(path)


def load_done_mask(path: Path, expected_size: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        return None
    if arr.shape != (expected_size,):
        return None
    if arr.dtype != np.bool_:
        arr = arr.astype(np.bool_, copy=False)
    return arr


def open_feature_memmap(
    path: Path,
    rows: int,
    dim: int,
    allow_reuse: bool,
) -> Tuple[np.memmap, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_shape = (rows, dim)
    if allow_reuse and path.exists():
        try:
            arr = np.lib.format.open_memmap(path, mode="r+")
            if arr.shape == expected_shape and arr.dtype == np.float32:
                return arr, True
        except Exception:
            pass
    if path.exists():
        path.unlink()
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=expected_shape)
    return arr, False


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    config_dir = config_path.parent

    dataset_cfg = cfg.get("dataset", {})
    audio_cfg = cfg.get("audio", {})
    video_cfg = cfg.get("video", {})
    runtime_cfg = cfg.get("runtime", {})
    output_cfg = cfg.get("output", {})

    seed = int(runtime_cfg.get("seed", 42))
    set_seed(seed)

    device_name = str(runtime_cfg.get("device", "cuda")).strip().lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    annotation_dir = ensure_path(dataset_cfg["annotation_dir"], base_dir=config_dir)
    video_root = ensure_path(dataset_cfg["video_root_dir"], base_dir=config_dir)
    splits = [str(s) for s in dataset_cfg.get("splits", ["train", "valid", "test"])]
    video_exts = [str(x).lower() for x in dataset_cfg.get("video_extensions", [".mp4"])]
    recursive_search = bool(dataset_cfg.get("recursive_video_search", True))
    scope = str(runtime_cfg.get("scope", "all")).strip().lower()
    if scope not in {"demo", "all", "annotation"}:
        raise ValueError("runtime.scope must be one of: demo, all, annotation")
    demo_num_videos = int(runtime_cfg.get("demo_num_videos", 10))
    if demo_num_videos <= 0:
        raise ValueError("runtime.demo_num_videos must be > 0")

    output_dir = ensure_path(output_cfg["output_dir"], base_dir=config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embed_dim_in_filename = bool(output_cfg.get("embed_dim_in_filename", True))

    if scope == "annotation":
        utterances = load_utterance_manifest(annotation_dir=annotation_dir, splits=splits)
        video_index = build_video_index(video_root, video_exts) if recursive_search else None
    else:
        all_video_files = collect_video_files(video_root=video_root, exts=video_exts)
        if not all_video_files:
            raise RuntimeError(f"No video files found under: {video_root}")
        if scope == "demo":
            if demo_num_videos >= len(all_video_files):
                selected_files = all_video_files
            else:
                rng = random.Random(seed)
                selected_files = rng.sample(all_video_files, demo_num_videos)
                selected_files.sort(key=lambda p: natural_key(p.as_posix()))
        else:
            selected_files = all_video_files
        utterances = build_manifest_from_video_files(video_root=video_root, video_files=selected_files)
        video_index = None

    max_utterances = runtime_cfg.get("max_utterances", None)
    if max_utterances is not None:
        max_utterances = int(max_utterances)
        utterances = utterances[:max_utterances]

    if not utterances:
        raise RuntimeError("No utterances found in MECAD annotations.")

    audio_enabled = bool(audio_cfg.get("enabled", True))
    video_enabled = bool(video_cfg.get("enabled", True))
    if not audio_enabled and not video_enabled:
        raise ValueError("At least one modality must be enabled.")

    audio_pooling = normalize_pooling_mode(str(audio_cfg.get("pooling", "min_mean_max")))
    video_pooling = normalize_pooling_mode(str(video_cfg.get("pooling", "min_mean_max")))

    audio_processor = None
    audio_model = None
    audio_base_dim = 0
    audio_dim = 0
    if audio_enabled:
        audio_processor, audio_model, audio_base_dim = build_wav2vec2(
            model_name_or_path=str(audio_cfg["model_name_or_path"]),
            device=device,
        )
        audio_dim = pooled_feature_dim(audio_base_dim, audio_pooling)

    video_model = None
    video_base_dim = 0
    video_dim = 0
    if video_enabled:
        ckpt = video_cfg.get("checkpoint_path", None)
        ckpt_path = ensure_path(ckpt, base_dir=config_dir) if ckpt else None
        video_model, video_base_dim = build_densenet(
            model_name=str(video_cfg.get("model_name", "densenet121")),
            device=device,
            pretrained_imagenet=bool(video_cfg.get("pretrained_imagenet", True)),
            checkpoint_path=ckpt_path,
        )
        video_dim = pooled_feature_dim(video_base_dim, video_pooling)

    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    num_frames = int(video_cfg.get("num_frames", 8))
    resize_shorter = int(video_cfg.get("resize_shorter_side", 256))
    crop_size = int(video_cfg.get("crop_size", 224))
    resume_enabled = bool(runtime_cfg.get("resume", True))
    save_every = int(runtime_cfg.get("save_every", 1))
    if save_every <= 0:
        raise ValueError("runtime.save_every must be >= 1")
    skip_on_error = bool(runtime_cfg.get("skip_on_error", True))
    parallel_modalities = bool(runtime_cfg.get("parallel_modalities", False))
    batch_size = int(runtime_cfg.get("batch_size", 1))
    audio_batch_size = int(runtime_cfg.get("audio_batch_size", batch_size))
    video_batch_size = int(runtime_cfg.get("video_batch_size", batch_size))
    if batch_size <= 0 or audio_batch_size <= 0 or video_batch_size <= 0:
        raise ValueError("runtime.batch_size/runtime.audio_batch_size/runtime.video_batch_size must be >= 1")

    audio_file = (
        with_dim_suffix(
            filename=str(output_cfg.get("audio_file", "audio_embedding.npy")),
            dim=audio_dim,
            enabled=embed_dim_in_filename,
        )
        if audio_enabled
        else None
    )
    video_file = (
        with_dim_suffix(
            filename=str(output_cfg.get("video_file", "video_embedding.npy")),
            dim=video_dim,
            enabled=embed_dim_in_filename,
        )
        if video_enabled
        else None
    )
    audio_out_path = (output_dir / audio_file) if audio_file is not None else None
    video_out_path = (output_dir / video_file) if video_file is not None else None

    mapping_path = output_dir / str(output_cfg.get("mapping_file", "video_id_mapping.npy"))
    manifest_path = output_dir / str(output_cfg.get("manifest_file", "utterance_manifest.jsonl"))
    meta_path = output_dir / str(output_cfg.get("meta_file", "feature_meta.json"))
    done_path = output_dir / str(output_cfg.get("done_file", "extract_done_mask.npy"))
    state_path = output_dir / str(output_cfg.get("state_file", "extract_state.json"))

    manifest_hash = compute_manifest_hash(utterances)
    mapping = build_id_mapping(utterances)
    atomic_save_npy(mapping_path, mapping)

    prev_state: Dict[str, Any] = {}
    if resume_enabled and state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                prev_state = json.load(f)
        except Exception:
            prev_state = {}

    expected_state = {
        "manifest_hash": manifest_hash,
        "num_utterances": int(len(utterances)),
        "scope": scope,
        "splits": list(splits),
        "audio_enabled": bool(audio_enabled),
        "video_enabled": bool(video_enabled),
        "audio_dim": int(audio_dim) if audio_enabled else 0,
        "video_dim": int(video_dim) if video_enabled else 0,
        "audio_file": audio_file,
        "video_file": video_file,
    }

    state_compatible = bool(prev_state)
    if state_compatible:
        for k, v in expected_state.items():
            if prev_state.get(k) != v:
                state_compatible = False
                break

    done_mask = np.zeros((len(utterances),), dtype=np.bool_)
    missing_indices: set[int] = set()
    failed_indices: set[int] = set()
    if state_compatible:
        loaded_done = load_done_mask(done_path, expected_size=len(utterances))
        if loaded_done is None:
            state_compatible = False
        else:
            done_mask = loaded_done
            for idx in prev_state.get("missing_indices", []):
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < len(utterances):
                    missing_indices.add(i)
            for idx in prev_state.get("failed_indices", []):
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < len(utterances):
                    failed_indices.add(i)

    audio_mem: Optional[np.memmap] = None
    video_mem: Optional[np.memmap] = None
    if audio_enabled:
        assert audio_out_path is not None
        audio_mem, audio_reused = open_feature_memmap(
            audio_out_path,
            rows=len(utterances),
            dim=audio_dim,
            allow_reuse=bool(resume_enabled and state_compatible),
        )
    else:
        audio_reused = True
    if video_enabled:
        assert video_out_path is not None
        video_mem, video_reused = open_feature_memmap(
            video_out_path,
            rows=len(utterances),
            dim=video_dim,
            allow_reuse=bool(resume_enabled and state_compatible),
        )
    else:
        video_reused = True

    if not state_compatible or (audio_enabled and not audio_reused) or (video_enabled and not video_reused):
        done_mask[:] = False
        missing_indices.clear()
        failed_indices.clear()

    def persist_progress(status: str) -> None:
        if audio_mem is not None:
            audio_mem.flush()
        if video_mem is not None:
            video_mem.flush()
        atomic_save_npy(done_path, done_mask)
        state_payload = {
            "version": 2,
            "status": status,
            "updated_at": now_str(),
            "manifest_hash": manifest_hash,
            "num_utterances": int(len(utterances)),
            "processed_utterances": int(done_mask.sum()),
            "extraction_complete": bool(done_mask.all()),
            "scope": scope,
            "splits": list(splits),
            "audio_enabled": bool(audio_enabled),
            "video_enabled": bool(video_enabled),
            "audio_dim": int(audio_dim) if audio_enabled else 0,
            "video_dim": int(video_dim) if video_enabled else 0,
            "audio_file": audio_file,
            "video_file": video_file,
            "done_file": done_path.name,
            "state_file": state_path.name,
            "missing_indices": sorted(missing_indices),
            "failed_indices": sorted(failed_indices),
            "save_every": int(save_every),
            "batch_size": int(batch_size),
            "audio_batch_size": int(audio_batch_size),
            "video_batch_size": int(video_batch_size),
            "parallel_modalities": bool(parallel_modalities),
        }
        atomic_write_json(state_path, state_payload)

    # Ensure state files exist from the beginning of the run.
    persist_progress(status="running")

    start_done = int(done_mask.sum())
    if start_done == len(utterances):
        print(f"[Resume] Detected complete cache: {start_done}/{len(utterances)} utterances done.")
    elif start_done > 0:
        print(f"[Resume] Resuming from checkpoint: {start_done}/{len(utterances)} utterances done.")
    interrupted = False
    processed_since_save = 0
    pbar = tqdm(total=len(utterances), initial=start_done, desc="Extracting MECAD features")
    try:
        pending_indices = [i for i in range(len(utterances)) if not done_mask[i]]
        for batch_indices in chunked(pending_indices, batch_size):
            resolved_paths: Dict[int, Optional[Path]] = {}
            valid_indices: List[int] = []
            batch_failed: Dict[int, bool] = {idx: False for idx in batch_indices}
            for idx in batch_indices:
                item = utterances[idx]
                video_path = resolve_video_for_item(item=item, video_root=video_root, fallback_index=video_index)
                resolved_paths[idx] = video_path
                if video_path is None:
                    missing_indices.add(idx)
                    if audio_mem is not None:
                        audio_mem[idx] = np.zeros((audio_dim,), dtype=np.float32)
                    if video_mem is not None:
                        video_mem[idx] = np.zeros((video_dim,), dtype=np.float32)
                else:
                    missing_indices.discard(idx)
                    valid_indices.append(idx)

            def run_audio_block() -> List[Tuple[int, np.ndarray, bool, str]]:
                results: List[Tuple[int, np.ndarray, bool, str]] = []
                if audio_mem is None or not valid_indices:
                    return results
                for sub_indices in chunked(valid_indices, audio_batch_size):
                    sub_paths = [resolved_paths[i] for i in sub_indices]
                    assert all(p is not None for p in sub_paths)
                    try:
                        emb_list, ok_list, err_list = extract_audio_embeddings_batch(
                            video_paths=[p for p in sub_paths if p is not None],
                            processor=audio_processor,
                            model=audio_model,
                            device=device,
                            sample_rate=sample_rate,
                            feat_dim=audio_base_dim,
                            pooling_mode=audio_pooling,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        if not skip_on_error:
                            raise
                        emb_list = [np.zeros((audio_dim,), dtype=np.float32) for _ in sub_indices]
                        ok_list = [False for _ in sub_indices]
                        err_list = [str(e) for _ in sub_indices]

                    if (not skip_on_error) and any(not x for x in ok_list):
                        first_bad = next(i for i, x in enumerate(ok_list) if not x)
                        raise RuntimeError(f"Audio batch extraction failed: {err_list[first_bad]}")

                    for pos, idx in enumerate(sub_indices):
                        results.append((idx, emb_list[pos], bool(ok_list[pos]), str(err_list[pos])))
                return results

            def run_video_block() -> List[Tuple[int, np.ndarray, bool, str]]:
                results: List[Tuple[int, np.ndarray, bool, str]] = []
                if video_mem is None or not valid_indices:
                    return results
                for sub_indices in chunked(valid_indices, video_batch_size):
                    sub_paths = [resolved_paths[i] for i in sub_indices]
                    assert all(p is not None for p in sub_paths)
                    try:
                        emb_list, ok_list, err_list = extract_video_embeddings_batch(
                            video_paths=[p for p in sub_paths if p is not None],
                            model=video_model,
                            device=device,
                            num_frames=num_frames,
                            resize_shorter=resize_shorter,
                            crop_size=crop_size,
                            feat_dim=video_base_dim,
                            pooling_mode=video_pooling,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        if not skip_on_error:
                            raise
                        emb_list = [np.zeros((video_dim,), dtype=np.float32) for _ in sub_indices]
                        ok_list = [False for _ in sub_indices]
                        err_list = [str(e) for _ in sub_indices]

                    if (not skip_on_error) and any(not x for x in ok_list):
                        first_bad = next(i for i, x in enumerate(ok_list) if not x)
                        raise RuntimeError(f"Video batch extraction failed: {err_list[first_bad]}")

                    for pos, idx in enumerate(sub_indices):
                        results.append((idx, emb_list[pos], bool(ok_list[pos]), str(err_list[pos])))
                return results

            audio_results: List[Tuple[int, np.ndarray, bool, str]] = []
            video_results: List[Tuple[int, np.ndarray, bool, str]] = []
            do_parallel = bool(
                parallel_modalities
                and audio_mem is not None
                and video_mem is not None
                and bool(valid_indices)
            )
            if do_parallel:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_audio = pool.submit(run_audio_block)
                    f_video = pool.submit(run_video_block)
                    audio_results = f_audio.result()
                    video_results = f_video.result()
            else:
                audio_results = run_audio_block()
                video_results = run_video_block()

            for idx, emb, ok, err in audio_results:
                assert audio_mem is not None
                audio_mem[idx] = emb
                if not ok:
                    batch_failed[idx] = True
                    if err:
                        print(f"[WARN][audio] idx={idx} video={utterances[idx].get('video_name', '')} failed: {err}")

            for idx, emb, ok, err in video_results:
                assert video_mem is not None
                video_mem[idx] = emb
                if not ok:
                    batch_failed[idx] = True
                    if err:
                        print(f"[WARN][video] idx={idx} video={utterances[idx].get('video_name', '')} failed: {err}")

            for idx in batch_indices:
                if idx in missing_indices:
                    failed_indices.discard(idx)
                elif batch_failed.get(idx, False):
                    failed_indices.add(idx)
                else:
                    failed_indices.discard(idx)
                done_mask[idx] = True
            processed_since_save += len(batch_indices)
            pbar.update(len(batch_indices))

            if processed_since_save >= save_every:
                persist_progress(status="running")
                processed_since_save = 0
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Resume] Interrupted. Progress saved. Re-run the same command to continue.")
    finally:
        pbar.close()
        if processed_since_save > 0:
            persist_progress(status="running")

    extraction_complete = bool(done_mask.all())
    final_status = "complete" if extraction_complete else ("interrupted" if interrupted else "incomplete")
    persist_progress(status=final_status)

    # Rebuild a full manifest snapshot (including resolved video path) for downstream use.
    manifest: List[Dict[str, Any]] = []
    for idx, item in enumerate(utterances):
        video_path = resolve_video_for_item(item=item, video_root=video_root, fallback_index=video_index)
        rec = dict(item)
        rec["index"] = idx
        rec["video_path"] = str(video_path) if video_path is not None else None
        manifest.append(rec)

    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with manifest_tmp.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps({k: to_jsonable(v) for k, v in row.items()}, ensure_ascii=False) + "\n")
    manifest_tmp.replace(manifest_path)

    missing_video_examples = [
        str(utterances[i].get("video_name", ""))
        for i in sorted(missing_indices)[:20]
        if 0 <= i < len(utterances)
    ]
    failed_video_examples = [
        str(utterances[i].get("video_name", ""))
        for i in sorted(failed_indices)[:20]
        if 0 <= i < len(utterances)
    ]

    meta = {
        "annotation_dir": str(annotation_dir),
        "video_root_dir": str(video_root),
        "config_path": str(config_path),
        "scope": scope,
        "demo_num_videos": demo_num_videos if scope == "demo" else None,
        "splits": splits,
        "num_utterances": int(len(utterances)),
        "processed_utterances": int(done_mask.sum()),
        "remaining_utterances": int(len(utterances) - int(done_mask.sum())),
        "extraction_complete": extraction_complete,
        "status": final_status,
        "resume_enabled": bool(resume_enabled),
        "save_every": int(save_every),
        "skip_on_error": bool(skip_on_error),
        "batch_size": int(batch_size),
        "audio_batch_size": int(audio_batch_size),
        "video_batch_size": int(video_batch_size),
        "parallel_modalities": bool(parallel_modalities),
        "done_file": done_path.name,
        "state_file": state_path.name,
        "manifest_hash": manifest_hash,
        "num_missing_videos": len(missing_indices),
        "missing_video_examples": missing_video_examples,
        "num_failed_videos": len(failed_indices),
        "failed_video_examples": failed_video_examples,
        "audio_enabled": audio_enabled,
        "audio_file": audio_file,
        "audio_model_name_or_path": audio_cfg.get("model_name_or_path", None),
        "audio_pooling": audio_pooling,
        "audio_base_feature_dim": int(audio_base_dim) if audio_enabled else 0,
        "audio_feature_dim": int(audio_dim) if audio_enabled else 0,
        "audio_sample_rate": sample_rate if audio_enabled else None,
        "video_enabled": video_enabled,
        "video_file": video_file,
        "video_model_name": video_cfg.get("model_name", None),
        "video_pooling": video_pooling,
        "video_base_feature_dim": int(video_base_dim) if video_enabled else 0,
        "video_feature_dim": int(video_dim) if video_enabled else 0,
        "video_num_frames": num_frames if video_enabled else None,
        "device": str(device),
        "output_dir": str(output_dir),
        "embed_dim_in_filename": embed_dim_in_filename,
        "mapping_size": len(mapping),
        "updated_at": now_str(),
    }
    atomic_write_json(meta_path, meta)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
