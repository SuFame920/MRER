#!/use/bin/env python


import os
import random
import math
import sys
import re
from datetime import datetime

import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# ANSI escape sequences (cursor move, clear line, color, etc.).
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_DATASET_NEUTRAL_ID = {
    "ecf": 0,
    "convecpe": 2,
}


class _TeeStream:
    def __init__(self, stream, path_getter):
        self._stream = stream
        self._path_getter = path_getter
        self._last_logged_blank = False

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        self._stream.write(data)

        original_data = data
        file_data = original_data
        if "\r" in file_data:
            # tqdm-style in-place refresh uses carriage return.
            # Do not persist these transient refresh chunks.
            # The trainer prints one explicit 100% line per epoch for logs.
            file_data = ""
        else:
            file_data = _ANSI_ESCAPE_RE.sub("", file_data)
            # Drop chunks that are only terminal control sequences.
            if "\x1b" in original_data and not file_data.strip():
                file_data = ""

        # Drop whitespace-only chunks (most are tqdm residual refresh newlines).
        if file_data and not file_data.strip():
            if self._last_logged_blank:
                file_data = ""
            else:
                # Keep at most one blank line to avoid log flooding.
                self._last_logged_blank = True
        elif file_data:
            self._last_logged_blank = False

        log_path = self._path_getter()
        if log_path and file_data:
            try:
                parent = os.path.dirname(log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(file_data)
            except Exception:
                # Never let file-logging break training output.
                pass
        return len(data)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class ConsoleTee:
    def __init__(self, log_path_or_getter):
        if callable(log_path_or_getter):
            self._path_getter = log_path_or_getter
        else:
            self._path_getter = lambda: log_path_or_getter
        self._orig_stdout = None
        self._orig_stderr = None
        self._active = False

    def start(self):
        if self._active:
            return self
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._path_getter)
        sys.stderr = _TeeStream(self._orig_stderr, self._path_getter)
        self._active = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Log] Session started at {ts}\n", flush=True)
        return self

    def stop(self):
        if not self._active:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Log] Session ended at {ts}\n", flush=True)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._orig_stdout = None
            self._orig_stderr = None
            self._active = False


def enable_console_logging(log_path_or_getter):
    return ConsoleTee(log_path_or_getter).start()


def apply_dataset_overrides(config):
    """
    Apply dataset-specific overrides from config.dataset_overrides[dataset_name],
    then apply variant-specific overrides from config.variant_overrides[dataset_name][model_variant].
    This helps avoid manually switching dozens of parameters when changing dataset or model size.
    """
    dataset_name = str(getattr(config, "dataset_name", "")).strip().lower()
    overrides = config.get("dataset_overrides", None)
    if not dataset_name or not isinstance(overrides, dict):
        return config
    selected = overrides.get(dataset_name, None)
    if not isinstance(selected, dict):
        return config
    for k, v in selected.items():
        config[k] = v

    # --- variant layer (base / large) ---
    model_variant = str(config.get("model_variant", "base")).strip().lower()
    variant_overrides = config.get("variant_overrides", None)
    if isinstance(variant_overrides, dict):
        dataset_variants = variant_overrides.get(dataset_name, None)
        if isinstance(dataset_variants, dict):
            variant_selected = dataset_variants.get(model_variant, None)
            if isinstance(variant_selected, dict):
                for k, v in variant_selected.items():
                    config[k] = v

    return config



def resolve_neutral_id(config, model_label_space=False):
    """
    Resolve neutral class id with a single shared rule across the project.

    Rules:
    1) Dataset convention: ECF -> 0, ConvECPE -> 2.
    2) Otherwise infer from label mappings with case-insensitive neutral/neu match.
    """
    dataset_name = str(getattr(config, "dataset_name", "")).strip().lower()
    if dataset_name in _DATASET_NEUTRAL_ID:
        return int(_DATASET_NEUTRAL_ID[dataset_name])

    id2label = config.get("emotion_id2label", None)
    if isinstance(id2label, dict):
        for k, v in id2label.items():
            if str(v).strip().lower() in {"neutral", "neu"}:
                try:
                    return int(k)
                except Exception:
                    pass

    label_dict = config.get("label_dict", None)
    if isinstance(label_dict, dict):
        for k, v in label_dict.items():
            if str(k).strip().lower() in {"neutral", "neu"}:
                try:
                    return int(v)
                except Exception:
                    pass

    return 0


def build_utterance_mask(max_utt_num, utterance_nums, device):
    """
    Build a [B, U] boolean mask for valid utterances in each dialogue.
    """
    return torch.arange(max_utt_num, device=device).unsqueeze(0) < utterance_nums.unsqueeze(-1)


def build_pair_mask_and_gold(h_utt, utterance_nums, pair_nums, pairs):
    """
    Build pair mask and gold pair matrix for utterance-level pair classification.

    Args:
        h_utt: [B, U, H] utterance representations.
        utterance_nums: [B] valid utterance counts.
        pair_nums: [B] number of gold pairs in each instance.
        pairs: [B, P, 2] gold (emotion_idx, cause_idx) pairs.

    Returns:
        pair_mask: [B, U, U] boolean valid region mask.
        gold_matrix: [B, U, U] long tensor with 0/1 labels.
    """
    batch_size, max_utt_num = h_utt.shape[:2]
    device = h_utt.device

    utt_mask = build_utterance_mask(max_utt_num, utterance_nums, device)
    pair_mask = utt_mask.unsqueeze(1) & utt_mask.unsqueeze(2)

    tri_mask = torch.flip(
        torch.flip(torch.triu(torch.ones_like(pair_mask[0], dtype=torch.bool)), [1]),
        [0],
    )
    pair_mask = pair_mask & tri_mask

    gold_matrix = h_utt.new_zeros((batch_size, max_utt_num, max_utt_num), dtype=torch.long)
    for i in range(batch_size):
        cur_pair_num = int(pair_nums[i])
        if cur_pair_num <= 0:
            continue
        cur_pairs = pairs[i, :cur_pair_num].long()
        gold_matrix[i, cur_pairs[:, 0], cur_pairs[:, 1]] = 1
    return pair_mask, gold_matrix


def merge_token_to_utterance(h_tok_bert, indices):
    """
    Merge token-level BERT outputs into utterance-level representations.

    For each candidate span in one utterance, representation = end_token + cls_token.
    """
    max_utt_num = max(len(item) for item in indices)
    h_utt = h_tok_bert.new_zeros((len(indices), max_utt_num, h_tok_bert.shape[-1]))

    for i in range(len(indices)):
        cur_id = indices[i][0][0]
        end_id = indices[i][-1][0]
        cur_len = 0
        for j in range(cur_id, end_id + 1):
            start = h_tok_bert.new_tensor(
                [w[1] for w in indices[i] if w[0] == j], dtype=torch.long
            )
            end = h_tok_bert.new_tensor(
                [w[2] - 1 for w in indices[i] if w[0] == j], dtype=torch.long
            )
            end_rep = torch.gather(
                h_tok_bert[j], 0, end.unsqueeze(-1).expand(-1, h_tok_bert.shape[-1])
            )
            span_len = start.shape[0]
            h_utt[i, cur_len : cur_len + span_len] = end_rep + h_tok_bert[j][0].unsqueeze(0)
            cur_len += span_len
    return h_utt


def update_config(config):

    dirs = ['preprocessed_dir', 'target_dir', 'dataset_dir']
    for dirname in dirs:
        if dirname in config:
            config[dirname] = os.path.join(config.data_dir, config[dirname])
    dataset_name = str(getattr(config, "dataset_name", "")).lower()
    if dataset_name and "target_dir" in config:
        base = os.path.basename(os.path.normpath(config.target_dir))
        if base.lower() != dataset_name:
            config.target_dir = os.path.join(config.target_dir, dataset_name)
    if 'emb_file' in config:
        config['emb_file'] = os.path.join(config.preprocessed_dir, config['emb_file'])
    if not os.path.exists(config.preprocessed_dir):
        os.makedirs(config.preprocessed_dir)
    if not os.path.exists(config.target_dir):
        os.makedirs(config.target_dir)
    return config


def set_seed(seed):
    # Reproducibility mode is fixed to strict.
    mode = "strict"

    # Reset backend knobs first so switching modes in one process is explicit.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # strict mode: maximize deterministic behavior (may hurt speed/memory).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    return mode


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    pairs = predictions.pairs
    pairs = predictions.pair_nums
    pairs = predictions['pairs']
    pair_nums = predictions['pair_nums']
    logits = predictions['logits']
    # The report includes precision, recall, f1-score and support (number of instances) for each class.
    report = classification_report(labels, predictions, output_dict=True)
    for class_id, metrics in report.items():
        if class_id.isdigit():
            print(f'Class {class_id}: F1 Score: {metrics["f1-score"]}')
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': precision_recall_fscore_support(labels, predictions, average='weighted')[2],
    }
    

def load_params_bert(config, model, fold_data):
    # ---- 旧代码（3 组，非 BERT 参数缺少 no_decay 分离） ----
    # no_decay = ['bias', 'LayerNorm.weight']
    # bert_param_ids = {id(p) for p in model.bert.parameters()}
    # other_params = [p for p in model.parameters() if id(p) not in bert_param_ids]
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': float(config.weight_decay)},
    #     {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': 0.0},
    #     {'params': other_params, 'lr': float(config.learning_rate), 'weight_decay': float(config.weight_decay)},
    # ]
    # ---- 新代码（4 组，BERT/非BERT 各自分 decay/no_decay） ----
    no_decay = ['bias', 'LayerNorm.weight', 'RMSNorm.weight']
    bert_param_ids = {id(p) for p in model.bert.parameters()}

    bert_wd = float(config.weight_decay)
    other_wd = float(getattr(config, "other_weight_decay", config.weight_decay))

    other_named = [(n, p) for n, p in model.named_parameters()
                   if id(p) not in bert_param_ids]

    optimizer_grouped_parameters = [
        # Group 1: BERT 普通权重
        {'params': [p for n, p in model.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr), 'weight_decay': bert_wd},
        # Group 2: BERT bias / LayerNorm.weight → 免 decay
        {'params': [p for n, p in model.bert.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr), 'weight_decay': 0.0},
        # Group 3: 非 BERT 普通权重
        {'params': [p for n, p in other_named
                    if not any(nd in n for nd in no_decay)],
         'lr': float(config.learning_rate), 'weight_decay': other_wd},
        # Group 4: 非 BERT bias / LayerNorm / RMSNorm.weight → 免 decay
        {'params': [p for n, p in other_named
                    if any(nd in n for nd in no_decay)],
         'lr': float(config.learning_rate), 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))

    grad_acc_steps = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    updates_per_epoch = max(1, math.ceil(fold_data.__len__() / grad_acc_steps))
    total_update_steps = int(config.epoch_size) * updates_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_update_steps,
    )

    config.optimizer = optimizer
    config.scheduler = scheduler

    return config
