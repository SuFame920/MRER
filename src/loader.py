#!/usr/bin/env python

"""
Name: loader.py
"""

import os
import json
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import logging
import pickle as pkl
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass 
from collections import Counter
import numpy as np

from src.layer import build_hgraph
from src.tools import resolve_neutral_id

CONVECPE_ID2LABEL = {
    0: "hap",
    1: "sad",
    2: "neu",
    3: "ang",
    4: "exc",
    5: "fru",
}


def natural_key(value: str) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def infer_neutral_id(config):
    """
    璁粌鏁版嵁鍘熷鏍囩绌洪棿涓嬬殑 neutral id銆?    閲囩敤缁熶竴瑙勫垯锛欵CF -> 0锛孋onvECPE -> 2锛堟湭鐭ユ暟鎹泦鍥為€€鍒板巻鍙叉帹鏂級銆?    """
    return int(resolve_neutral_id(config, model_label_space=False))


def sync_special_tokens(config, tokenizer):
    """Keep config special tokens aligned with the selected tokenizer/model."""
    pad = tokenizer.pad_token
    unk = tokenizer.unk_token
    cls = tokenizer.cls_token or tokenizer.bos_token
    sep = tokenizer.sep_token or tokenizer.eos_token

    if pad is None:
        pad = config.get("pad", "[PAD]")
    if unk is None:
        unk = config.get("unk", "[UNK]")
    if cls is None:
        cls = config.get("cls", "[CLS]")
    if sep is None:
        sep = config.get("sep", "[SEP]")

    config["pad"] = pad
    config["unk"] = unk
    config["cls"] = cls
    config["sep"] = sep
    return config



def build_mask(utterance_nums, speakers, local_window=1):
    local_window = max(0, int(local_window))
    max_utterance = max(utterance_nums)

    gmask = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(utterance_nums)):
        gmask[i, :utterance_nums[i], :utterance_nums[i]] = 1
    gmask = gmask.repeat(1, 4, 4)

    smask = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(speakers)):
        speaker = speakers[i]
        m = np.array([[1 if i == j else 0 for i in speaker] for j in speaker])
        smask[i, :utterance_nums[i], :utterance_nums[i]] = torch.tensor(m)
    smask = smask.repeat(1, 4, 4)

    lmasks = torch.zeros(len(utterance_nums), max_utterance, max_utterance, dtype=torch.long)
    for i in range(len(utterance_nums)):
        utterance_num = utterance_nums[i]
        eye = np.zeros((utterance_num, utterance_num), dtype=np.int64)
        for delta in range(-local_window, local_window + 1):
            eye += np.eye(utterance_num, k=delta, dtype=np.int64)
        eye = (eye > 0).astype(np.int64)
        lmasks[i, :utterance_nums[i], :utterance_nums[i]] = torch.tensor(eye, dtype=torch.long)
    lmasks = lmasks.repeat(1, 4, 4)
    return gmask, smask, lmasks

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data, mode):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.data = data[mode]
        self.label_dict = data['label_dict']
        self.speaker_dict = data['speaker_dict']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        keys = list(self.data[i])
        values = [self.data[i][k] for k in keys]
        values[4] = [self.label_dict[w] for w in values[4]]
        values[3] = [self.speaker_dict[w] for w in values[3]]
        return (keys, values)

@dataclass
class CollateFN:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    config: Dict
    video_map: Dict 
    audio_map: Dict 
    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        keys, instances = list(zip(*instances))
        keys0 = list(keys[0]) if keys else []
        token_idx = keys0.index('tokens') if 'tokens' in keys0 else None
        doc_ids = [line[0] for line in instances]
        pairs = [[(w - 1, z - 1) for w, z in line[1]] for line in instances]
        utterances = [line[2] for line in instances]
        tokenized_utterances = [line[token_idx] for line in instances] if token_idx is not None else None
        emotions = [line[4] for line in instances]
        speakers = [line[3] for line in instances]
        # Emotion task is fixed to multi-class; keep original emotion ids.

        utterance_nums = [len(line) for line in utterances]
        # Unified naming: local relation uses local_window only.
        local_window = getattr(
            self.config,
            "local_window",
            self.config.get("local_window", 1),
        )
        try:
            local_window = max(0, int(local_window))
        except Exception:
            local_window = 1
        gmasks, smasks, lmasks = build_mask(
            utterance_nums, speakers, local_window=local_window
        )
        IGNORE_INDEX = -100
        # max_utterance = max(max(utterance_nums), 2)
        max_utterance = max(utterance_nums)

        emotions = [w + [IGNORE_INDEX] * (max_utterance - len(w)) for w in emotions]
        res = []
        max_length = self.config['max_length'] 
        total_length = self.config['total_length']
        # padd_utterance = [w + [''] * (max_utterance - len(w)) for w in utterances]
        # for i in range(len(utterances)):
            # batch_input = self.tokenizer.batch_encode_plus(padd_utterance[i], return_tensors="pt", pad_to_max_length=True, max_length=max_length)
            # res.append(batch_input)
        pack_input = tokenized_utterances if tokenized_utterances is not None else utterances
        input_tokens, indices = pack(pack_input, max_length, total_length, self.tokenizer, self.config)

        # Build attention masks from true (pre-padding) sequence lengths.
        token_lens = [len(w) for w in input_tokens]
        max_seq_len = max(token_lens)
        input_tokens = [w + [self.config.pad] * (max_seq_len - len(w)) for w in input_tokens]

        input_ids = [self.tokenizer.convert_tokens_to_ids(w) for w in input_tokens]
        # input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [
            [1] * cur_len + [0] * (max_seq_len - cur_len) for cur_len in token_lens
        ]

        # input_ids = torch.stack([w['input_ids'] for w in res], dim=0)
        # attention_mask = torch.stack([w['attention_mask'] for w in res], dim=0)
        
        # padding pairs
        pair_nums = [len(line) for line in pairs]
        max_pair = max(pair_nums)
        pairs = [w + [(IGNORE_INDEX, IGNORE_INDEX)] * (max_pair - len(w)) for w in pairs]

        cuase_labels = [[0 for _ in range(max_utterance)] for _ in range(len(pairs))]
        # emotion_binary = [[0 for _ in range(max_utterance)] for _ in range(len(pairs))]
        for i in range(len(pairs)):
            for w, z in pairs[i]:
                if z != IGNORE_INDEX:
                    cuase_labels[i][z] = 1
                # if w != IGNORE_INDEX:
                    # emotion_binary[i][w] = 1
        
        speakers = [w + [0] * (max_utterance - len(w)) for w in speakers]

        video_features = []
        for i in range(len(doc_ids)):
            video_feature = np.stack([self.video_map[(doc_ids[i], j)] for j in range(1, utterance_nums[i] + 1)])
            video_features.append(video_feature)
        video_features = np.stack([np.concatenate([w, np.zeros((max_utterance - w.shape[0], w.shape[1]))], axis=0) for w in video_features])
        # Clip A/V features to [-clip_limit, clip_limit] when enabled.
        # If clip_limit <= 0, clipping is disabled and raw features are kept.
        clip_limit = float(self.config.get("feature_clip_limit", 1.0))
        if clip_limit > 0:
            video_features = np.clip(video_features, -clip_limit, clip_limit)

        audio_features = []
        for i in range(len(doc_ids)):
            audio_feature = np.stack([self.audio_map[(doc_ids[i], j)]  for j in range(1, utterance_nums[i] + 1)])
            audio_features.append(audio_feature)
        audio_features = np.stack([np.concatenate([w, np.zeros((max_utterance - w.shape[0], w.shape[1]))], axis=0) for w in audio_features])
        if clip_limit > 0:
            audio_features = np.clip(audio_features, -clip_limit, clip_limit)

        batch = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).to(self.config['device']),
            'input_masks': torch.tensor(attention_mask, dtype=torch.long).to(self.config['device']),
            'indices': indices, # 'indices': [(global_id, start, end), ...]
            'utterance_nums': torch.tensor(utterance_nums, dtype=torch.long).to(self.config['device']),
            'pairs': torch.tensor(pairs, dtype=torch.long).to(self.config['device']),
            'pair_nums': torch.tensor(pair_nums, dtype=torch.long).to(self.config['device']),
            'labels': torch.tensor(emotions, dtype=torch.long).to(self.config['device']),
            'cause_labels': torch.tensor(cuase_labels, dtype=torch.long).to(self.config['device']),
            'doc_ids': doc_ids,
            'speaker_ids': torch.tensor(speakers, dtype=torch.long).to(self.config['device']),
            'video_features': torch.tensor(video_features, dtype=torch.float).to(self.config['device']),
            'audio_features': torch.tensor(audio_features, dtype=torch.float).to(self.config['device']),
            'gmasks': gmasks.to(self.config['device']),
            'smasks': smasks.to(self.config['device']),
            'lmasks': lmasks.to(self.config['device']),
            # Backward-compatible alias for legacy model paths.
            'rmasks': lmasks.to(self.config['device']),
            # 'hgraph': hgraphs,
        }
        return batch

def pack(dialogues, max_len, total_len, tokenizer, config):
    res = []
    indices = []
    for i in range(len(dialogues)):
        cur_res = [config.cls]
        cur_indices = []
        for line in dialogues[i]:
            if isinstance(line, list):
                tokens = line
            else:
                tokens = tokenizer.tokenize(line)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            # include [SEP] in length check to avoid exceeding BERT max length
            if len(cur_res) + len(tokens) + 1 > total_len:
                res.append(cur_res)
                cur_res = [config.cls]
                # ensure single utterance fits within total_len (CLS + tokens + SEP)
                if len(tokens) + 2 > total_len:
                    tokens = tokens[: max(0, total_len - 2)]
            global_id = len(res) 
            start = len(cur_res)
            cur_res += tokens + [config.sep]
            end = len(cur_res) - 1
            cur_indices.append((global_id, start, end))
        res.append(cur_res)
        indices.append(cur_indices)
    # res = [tokenizer.convert_tokens_to_ids(w) for w in res]
    # input_masks = [[1] * len(w) for w in res]
    return res, indices 

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f: #[MODIFY-2026骞?鏈?鏃?4鐐?7鍒哴缂栫爜淇敼
        data = f.read().splitlines()

    structured_data = []
    idx = 0
    while idx < len(data):
        line = data[idx]
        scene_id, num_lines = map(int, line.split(' '))
        if len(data[idx+1].strip()) > 0:
            emotion_cause_pairs = [tuple(map(int, pair.split(','))) for pair in data[idx + 1].strip('()').split('),(')]
        else:
            emotion_cause_pairs = []
        lines, timecodes, speakers, emotions = [], [], [], []
        for i in range(num_lines):
            line_parts = data[idx + 2 + i].split(' | ')
            utterance_id, speaker, emotion, utterance = line_parts[:4]
            timecode = line_parts[4]
            speakers.append(speaker)
            emotions.append(emotion)
            timecodes.append(timecode)
            lines.append(utterance)

        structured_data.append({'doc_id': scene_id, 'emotion_cause_pairs': emotion_cause_pairs, 'lines': lines, 'speakers': speakers, 'emotions': emotions, 'timecodes': timecodes})
        idx += 2 + num_lines
    return structured_data


def load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pkl.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            return pkl.load(f, encoding="latin1")


def to_list(x: Any) -> List[Any]:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def normalize_text_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[str] = []
        for utt in raw:
            if isinstance(utt, str):
                out.append(utt)
            elif isinstance(utt, dict):
                if "text" in utt and isinstance(utt["text"], str):
                    out.append(utt["text"])
                elif "utterance" in utt and isinstance(utt["utterance"], str):
                    out.append(utt["utterance"])
                else:
                    added = False
                    for v in utt.values():
                        if isinstance(v, str):
                            out.append(v)
                            added = True
                            break
                    if not added:
                        out.append(str(utt))
            else:
                out.append(str(utt))
        return out
    return [str(raw)]


def normalize_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return to_list(raw)


def get_dataset_root(config) -> str:
    root = config.dataset_dir
    dataset_name = str(getattr(config, "dataset_name", "ecf")).lower()
    base = os.path.basename(os.path.normpath(root)).lower()
    if base == dataset_name:
        return root
    return os.path.join(root, dataset_name)


def is_valid_cause(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, np.integer)):
        return int(v) > 0
    if isinstance(v, (float, np.floating)):
        return float(v) > 0
    if isinstance(v, str):
        vs = v.strip().lower()
        if vs in {"", "n", "na", "none"}:
            return False
        if vs.lstrip("+-").isdigit():
            return int(vs) > 0
        return True
    return True


def build_pairs_from_causes(c1: List[Any], c2: List[Any], c3: List[Any], n: int) -> List[Tuple[int, int]]:
    pairs = set()
    for i in range(n):
        for v in (c1[i], c2[i], c3[i]):
            if is_valid_cause(v):
                try:
                    j = int(v)
                except Exception:
                    continue
                if j <= 0 or j > n:
                    continue
                pairs.add((i + 1, j))
    return sorted(pairs)


def normalize_field_by_conv(field: Any, conv_ids: List[str]) -> Dict[str, Any]:
    if isinstance(field, dict):
        return {cid: field.get(cid, []) for cid in conv_ids}
    if isinstance(field, (list, tuple)):
        if len(field) != len(conv_ids):
            raise ValueError("Field length does not match conv_ids length.")
        return {cid: field[i] for i, cid in enumerate(conv_ids)}
    raise TypeError(f"Unsupported field type: {type(field)}")


def build_feature_map(field: Any, conv_ids: List[str]) -> Tuple[Dict[Tuple[str, int], np.ndarray], int, Dict[str, int]]:
    fmap: Dict[Tuple[str, int], np.ndarray] = {}
    lengths: Dict[str, int] = {}
    dim = 0
    field_map = normalize_field_by_conv(field, conv_ids)
    for cid in conv_ids:
        feats = field_map.get(cid, [])
        if isinstance(feats, np.ndarray) and feats.ndim == 2:
            feats_list = feats
        else:
            feats_list = to_list(feats) if not isinstance(feats, list) else feats
        lengths[cid] = len(feats_list) if feats_list is not None else 0
        for idx, vec in enumerate(feats_list):
            arr = np.asarray(vec)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            if dim == 0:
                dim = int(arr.shape[0])
            fmap[(cid, idx + 1)] = arr
    return fmap, dim, lengths


def load_convecpe_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if "train" not in splits or "dev" not in splits or "test" not in splits:
        raise ValueError("Splits file must contain train/dev/test keys.")
    return {
        "train": list(splits["train"]),
        "dev": list(splits["dev"]),
        "test": list(splits["test"]),
    }


def read_convecpe_dataset(config) -> Tuple[Dict[str, Any], Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], np.ndarray]]:
    dataset_root = get_dataset_root(config)
    pkl_path = os.path.join(dataset_root, "IEMOCAP_emotion_cause_features.pkl")
    if not os.path.isabs(pkl_path):
        pkl_path = os.path.join(os.getcwd(), pkl_path)

    splits_path = os.path.join(dataset_root, "convecpe_splits.json")
    if not os.path.isabs(splits_path):
        splits_path = os.path.join(os.getcwd(), splits_path)

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"ConvECPE pkl not found: {pkl_path}")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"ConvECPE splits not found: {splits_path}")

    data = load_pickle(pkl_path)
    if not isinstance(data, (list, tuple)) or len(data) < 12:
        raise ValueError("Unexpected convecpe pkl structure (expected list/tuple length >= 12).")

    conv_ids_raw = data[0]
    if isinstance(conv_ids_raw, dict):
        conv_ids = list(conv_ids_raw.keys())
    elif isinstance(conv_ids_raw, (list, tuple)):
        conv_ids = list(conv_ids_raw)
    else:
        raise ValueError("conv_ids must be dict or list/tuple.")

    speakers_map = normalize_field_by_conv(data[1], conv_ids)
    labels_map = normalize_field_by_conv(data[2], conv_ids)
    cause1_map = normalize_field_by_conv(data[3], conv_ids)
    cause2_map = normalize_field_by_conv(data[4], conv_ids)
    cause3_map = normalize_field_by_conv(data[5], conv_ids)
    raw_text_map = normalize_field_by_conv(data[9], conv_ids)

    audio_map, audio_dim, audio_len = build_feature_map(data[7], conv_ids)
    video_map, video_dim, video_len = build_feature_map(data[8], conv_ids)

    if audio_dim:
        config["audio_dim"] = int(audio_dim)
    if video_dim:
        config["video_dim"] = int(video_dim)

    splits = load_convecpe_splits(splits_path)
    split_ids = {
        "train": [cid for cid in conv_ids if cid in set(splits["train"])],
        "valid": [cid for cid in conv_ids if cid in set(splits["dev"])],
        "test": [cid for cid in conv_ids if cid in set(splits["test"])],
    }

    data_out: Dict[str, List[Dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for mode in ["train", "valid", "test"]:
        for cid in split_ids[mode]:
            lines = normalize_text_list(raw_text_map.get(cid, []))
            speakers = normalize_list(speakers_map.get(cid, []))
            emotions = normalize_list(labels_map.get(cid, []))
            c1 = normalize_list(cause1_map.get(cid, []))
            c2 = normalize_list(cause2_map.get(cid, []))
            c3 = normalize_list(cause3_map.get(cid, []))

            n = min(
                len(lines),
                len(speakers),
                len(emotions),
                len(c1),
                len(c2),
                len(c3),
                int(audio_len.get(cid, 0)),
                int(video_len.get(cid, 0)),
            )
            if n <= 0:
                continue

            lines = lines[:n]
            speakers = speakers[:n]
            emotions = emotions[:n]
            c1 = c1[:n]
            c2 = c2[:n]
            c3 = c3[:n]
            timecodes = [""] * n
            pairs = build_pairs_from_causes(c1, c2, c3, n)

            data_out[mode].append(
                {
                    "doc_id": cid,
                    "emotion_cause_pairs": pairs,
                    "lines": lines,
                    "speakers": speakers,
                    "emotions": emotions,
                    "timecodes": timecodes,
                }
            )

    return data_out, video_map, audio_map


def _flatten_mecad_conversation(raw_conv: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_conv, list) and len(raw_conv) == 1 and isinstance(raw_conv[0], list):
        conv = raw_conv[0]
    elif isinstance(raw_conv, list):
        conv = raw_conv
    else:
        conv = []
    return [item for item in conv if isinstance(item, dict)]


def _to_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _parse_mecad_cause_indices(raw: Any) -> List[int]:
    vals: List[int] = []
    for item in to_list(raw):
        if isinstance(item, (int, np.integer)):
            vals.append(int(item))
            continue
        if isinstance(item, str):
            for num in re.findall(r"-?\d+", item):
                try:
                    vals.append(int(num))
                except Exception:
                    pass
    return vals


def _get_mecad_cause_field(utt: Dict[str, Any]) -> Any:
    if "expanded emotion cause evidence" in utt:
        return utt.get("expanded emotion cause evidence")
    if "expanded_emotion_cause_evidence" in utt:
        return utt.get("expanded_emotion_cause_evidence")
    for k, v in utt.items():
        key_norm = str(k).strip().lower().replace("_", " ")
        if "cause" in key_norm and "evidence" in key_norm:
            return v
    return []


def _resolve_path(base_dir: str, path_like: str) -> str:
    if os.path.isabs(path_like):
        return path_like
    return os.path.join(base_dir, path_like)


def _resolve_feature_file(
    dataset_root: str,
    explicit_name: Any,
    base_stem: str,
) -> str:
    candidates: List[str] = []
    if explicit_name:
        explicit_path = _resolve_path(dataset_root, str(explicit_name))
        if os.path.exists(explicit_path):
            candidates.append(explicit_path)

    exact = os.path.join(dataset_root, f"{base_stem}.npy")
    if os.path.exists(exact):
        candidates.append(exact)

    pattern = os.path.join(dataset_root, f"{base_stem}_*.npy")
    for p in glob.glob(pattern):
        if os.path.isfile(p):
            candidates.append(p)

    uniq: List[str] = []
    seen = set()
    for p in candidates:
        rp = os.path.realpath(p)
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)

    if not uniq:
        raise FileNotFoundError(
            f"No feature file found for '{base_stem}' under {dataset_root}. "
            f"Expected {base_stem}.npy or {base_stem}_<dim>.npy"
        )

    # Prefer explicit first; otherwise pick latest modified file.
    if explicit_name and os.path.exists(_resolve_path(dataset_root, str(explicit_name))):
        return os.path.realpath(_resolve_path(dataset_root, str(explicit_name)))
    uniq.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return uniq[0]


def _normalize_feature_index_map(raw_map: Dict[Any, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in raw_map.items():
        try:
            idx = int(v)
        except Exception:
            continue
        if idx < 0:
            continue
        out[str(k)] = idx
    return out


def _find_mecad_feature_index(
    index_map: Dict[str, int],
    split: str,
    conv_id: str,
    turn: int,
    video_name: str,
) -> int:
    candidates: List[str] = [f"{split}|{conv_id}|{turn}"]
    vname = str(video_name or "").strip()
    if vname:
        base_name = os.path.basename(vname)
        stem = os.path.splitext(base_name)[0]
        candidates.extend(
            [
                vname,
                vname.replace("\\", "/"),
                base_name,
                stem,
            ]
        )
    for key in candidates:
        if key in index_map:
            return int(index_map[key])
    return -1


def read_mecad_dataset(config) -> Tuple[Dict[str, Any], Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], np.ndarray]]:
    dataset_root = get_dataset_root(config)
    split_files = {
        "train": os.path.join(dataset_root, "train_data_pair.json"),
        "valid": os.path.join(dataset_root, "valid_data_pair.json"),
        "test": os.path.join(dataset_root, "test_data_pair.json"),
    }
    for split, path in split_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"MECAD {split} annotation not found: {path}")

    mapping_name = config.get("mecad_mapping_file", "video_id_mapping.npy")
    mapping_path = _resolve_path(dataset_root, str(mapping_name))
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"MECAD mapping file not found: {mapping_path}. "
            f"Please run data/dataset/mecad/extract.py first."
        )

    audio_path = _resolve_feature_file(
        dataset_root=dataset_root,
        explicit_name=config.get("mecad_audio_feature_file", None),
        base_stem="audio_embedding",
    )
    video_path = _resolve_feature_file(
        dataset_root=dataset_root,
        explicit_name=config.get("mecad_video_feature_file", None),
        base_stem="video_embedding",
    )

    audio_table = np.load(audio_path, mmap_mode="r")
    video_table = np.load(video_path, mmap_mode="r")
    if audio_table.ndim != 2 or video_table.ndim != 2:
        raise ValueError(
            f"MECAD feature files must be 2D arrays. Got "
            f"audio={audio_table.shape}, video={video_table.shape}"
        )

    config["audio_dim"] = int(audio_table.shape[1])
    config["video_dim"] = int(video_table.shape[1])

    raw_index_map = np.load(mapping_path, allow_pickle=True).item()
    if not isinstance(raw_index_map, dict):
        raise ValueError(f"MECAD mapping must be dict[str, int], got: {type(raw_index_map)}")
    index_map = _normalize_feature_index_map(raw_index_map)

    data_out: Dict[str, List[Dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    audio_map: Dict[Tuple[str, int], np.ndarray] = {}
    video_map: Dict[Tuple[str, int], np.ndarray] = {}

    zero_audio = np.zeros((int(audio_table.shape[1]),), dtype=np.float32)
    zero_video = np.zeros((int(video_table.shape[1]),), dtype=np.float32)
    missing = 0
    missing_examples: List[str] = []

    for split in ["train", "valid", "test"]:
        with open(split_files[split], "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected MECAD json format in {split_files[split]}")

        for conv_id in sorted(payload.keys(), key=natural_key):
            utterances = _flatten_mecad_conversation(payload[conv_id])
            if not utterances:
                continue

            lines: List[str] = []
            speakers: List[str] = []
            emotions: List[str] = []
            video_names: List[str] = []
            turns: List[int] = []

            for idx, utt in enumerate(utterances, start=1):
                lines.append(str(utt.get("utterance", "")))
                speakers.append(str(utt.get("speaker", "")))
                emotions.append(str(utt.get("emotion", "Neutral")))
                video_names.append(str(utt.get("video", "")).strip())
                turns.append(_to_int(utt.get("turn", idx), idx))

            turn_to_pos = {int(t): i + 1 for i, t in enumerate(turns)}
            pairs = set()
            for emo_pos, utt in enumerate(utterances, start=1):
                raw_causes = _get_mecad_cause_field(utt)
                for c in _parse_mecad_cause_indices(raw_causes):
                    cause_pos = int(turn_to_pos.get(c, c))
                    if 1 <= cause_pos <= len(utterances):
                        pairs.add((emo_pos, cause_pos))

            data_out[split].append(
                {
                    "doc_id": str(conv_id),
                    "emotion_cause_pairs": sorted(pairs),
                    "lines": lines,
                    "speakers": speakers,
                    "emotions": emotions,
                    "timecodes": [""] * len(lines),
                }
            )

            for pos, (turn, video_name) in enumerate(zip(turns, video_names), start=1):
                feat_idx = _find_mecad_feature_index(
                    index_map=index_map,
                    split=split,
                    conv_id=str(conv_id),
                    turn=int(turn),
                    video_name=video_name,
                )
                key = (str(conv_id), int(pos))
                if 0 <= feat_idx < len(audio_table) and 0 <= feat_idx < len(video_table):
                    audio_map[key] = np.asarray(audio_table[feat_idx]).reshape(-1).astype(np.float32, copy=False)
                    video_map[key] = np.asarray(video_table[feat_idx]).reshape(-1).astype(np.float32, copy=False)
                else:
                    audio_map[key] = zero_audio
                    video_map[key] = zero_video
                    missing += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(f"{split}|{conv_id}|{turn}|{video_name}")

    if missing > 0:
        logging.warning(
            f"MECAD feature mapping misses {missing} utterances. "
            f"Examples: {missing_examples[:5]}"
        )

    return data_out, video_map, audio_map


def ensure_token_cache(data_split, tokenizer):
    updated = False
    for item in data_split:
        if 'tokens' not in item:
            item['tokens'] = [tokenizer.tokenize(u) for u in item['lines']]
            updated = True
    return updated

def read_video(dataset_dir, video=True):
    if video:
        path = os.path.join(dataset_dir, 'video_embedding_4096.npy')
    else:
        path = os.path.join(dataset_dir, 'audio_embedding_6373.npy')
    map_path = os.path.join(dataset_dir, 'video_id_mapping.npy')
    video_feature = np.load(path)
    id_map = np.load(map_path, allow_pickle=True).item()
    get_num = lambda x: (int(x.split('utt')[0][3:]), int(x.split('utt')[1]))
    id_map = {get_num(w): video_feature[z] for w, z in id_map.items()}
    return id_map

def build_dict(data, prefer_identity: bool = False):
    wordlist = []
    for line in data:
        wordlist.extend(line['emotions'])
    wordcount = Counter(wordlist)
    if prefer_identity and wordcount:
        labels = list(wordcount.keys())
        if all(isinstance(w, (int, np.integer)) for w in labels):
            min_id = int(min(labels))
            max_id = int(max(labels))
            if min_id == 0 and set(int(w) for w in labels) == set(range(max_id + 1)):
                return {int(w): int(w) for w in labels}
    word2dict = {w: i for i, w in enumerate(wordcount.keys())}
    return word2dict 

def build_speaker_dict(data):
    wordlist = []
    for line in data:
        wordlist.extend(line['speakers'])
    wordcount = Counter(wordlist)
    word2dict = {w: i for i, w in enumerate(wordcount.keys())}
    return word2dict

def make_supervised_data_module(config, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    config = sync_special_tokens(config, tokenizer)
    dataset_name = str(getattr(config, "dataset_name", "ecf")).lower()
    # Record neutral id in model label space using dataset convention.
    config["neutral_id"] = int(resolve_neutral_id(config, model_label_space=True))
    cache_name = f"{config.bert_path}_{dataset_name}.pkl".replace("/", "-")
    path = os.path.join(config.preprocessed_dir, cache_name)
    legacy_path = os.path.join(config.preprocessed_dir, '{}.pkl'.format(config.bert_path))
    if dataset_name == "ecf" and not os.path.exists(path) and os.path.exists(legacy_path):
        path = legacy_path

    dataset_root = get_dataset_root(config)
    if not os.path.exists(path):
        if dataset_name == "convecpe":
            data, video_map, audio_map = read_convecpe_dataset(config)
            for mode in ['train', 'valid', 'test']:
                ensure_token_cache(data[mode], tokenizer)
            data['video'] = video_map
            data['audio'] = audio_map
            data['label_dict'] = build_dict(data['train'], prefer_identity=True)
            data['speaker_dict'] = build_speaker_dict(data['train'] + data['valid'] + data['test'])
        elif dataset_name == "mecad":
            data, video_map, audio_map = read_mecad_dataset(config)
            for mode in ['train', 'valid', 'test']:
                ensure_token_cache(data[mode], tokenizer)
            data['video'] = video_map
            data['audio'] = audio_map
            data['label_dict'] = build_dict(data['train'])
            data['speaker_dict'] = build_speaker_dict(data['train'] + data['valid'] + data['test'])
        else:
            data = {}
            for mode in ['train', 'valid', 'test']:
                data[mode] = read_data(os.path.join(dataset_root, '{}.txt'.format(mode)))
                ensure_token_cache(data[mode], tokenizer)

            data['video'] = read_video(dataset_root)
            data['audio'] = read_video(dataset_root, False)
            data['label_dict'] = build_dict(data['train'])
            data['speaker_dict'] = build_speaker_dict(data['train'] + data['valid'] + data['test'])

        with open(path, 'wb') as f:
            pkl.dump(data, f)
    else:
        with open(path, 'rb') as f:
            data = pkl.load(f)
        updated = False
        for mode in ['train', 'valid', 'test']:
            updated = ensure_token_cache(data[mode], tokenizer) or updated
        if updated:
            with open(path, 'wb') as f:
                pkl.dump(data, f)

    # ensure dims available when loading cached data
    try:
        if data.get('audio'):
            sample = next(iter(data['audio'].values()))
            if sample is not None:
                config['audio_dim'] = int(np.asarray(sample).reshape(-1).shape[0])
        if data.get('video'):
            sample = next(iter(data['video'].values()))
            if sample is not None:
                config['video_dim'] = int(np.asarray(sample).reshape(-1).shape[0])
    except Exception:
        pass
    config['label_dict'] = data['label_dict']
    config['speaker_dict'] = data['speaker_dict']
    # Recompute neutral id after label_dict is available (important for datasets
    # without fixed convention, e.g., MECAD).
    config["neutral_id"] = int(resolve_neutral_id(config, model_label_space=True))
    if dataset_name == "convecpe":
        # Keep training labels as numeric ids, but expose readable names for reports.
        config['emotion_id2label'] = CONVECPE_ID2LABEL
    train_dataset = SupervisedDataset(data, 'train')
    valid_dataset = SupervisedDataset(data, 'valid')
    test_dataset = SupervisedDataset(data, 'test')
    if config.model_name == 'bert':
        data_collator = CollateFN(tokenizer=tokenizer, config=config, video_map=data['video'], audio_map=data['audio'])
    elif config.model_name == 'lstm':
        data_collator = CollateFNLSTM(tokenizer=tokenizer, word_dict=data['word_dict'])

    num_workers = int(getattr(config, 'num_workers', 0))
    pin_memory = bool(getattr(config, 'pin_memory', False))
    if hasattr(config, 'device') and str(config.device).startswith('cuda'):
        # Collate returns CUDA tensors; pin_memory only applies to CPU tensors.
        pin_memory = False
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, valid_loader, test_loader, config
    return dict(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, config=config)

