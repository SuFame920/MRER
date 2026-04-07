#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from src.layer import (
    EnhancedLSTM,
    Biaffine,
    RoEmbedding,
    NewFusionGate,
    ARISEEncoder,
)
from src.tools import (
    resolve_neutral_id,
    build_utterance_mask,
    build_pair_mask_and_gold,
    merge_token_to_utterance,
)
import torch.nn.functional as F

class TextClassification(nn.Module):
    """
    Multimodal ECP model with ARISE fusion and CARE auxiliary learning.

    The class keeps only model-specific computation:
    initialization, core losses, and forward dataflow.
    Generic tensor/data utilities are placed in ``src.tools``.
    """
    def __init__(self, cfg, tokenizer):
        super(TextClassification, self).__init__()
        self.cfg = cfg 
        self.disable_mr = bool(cfg.get("disable_mr", False))
        self.use_text = bool(cfg.get("use_text", True))
        self.use_audio = bool(cfg.get("use_audio", True))
        self.use_video = bool(cfg.get("use_video", True))
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        self.speaker_embedder = nn.Embedding(len(cfg.speaker_dict), bert_config.hidden_size)
        self.tokenizer = tokenizer

        label_dict = cfg.get("label_dict", None)
        if isinstance(label_dict, dict) and label_dict:
            num_classes = len(label_dict)
        else:
            num_classes = 7
        self.num_classes = num_classes
        # Keep a dataset-consistent neutral class id for emotion loss weighting.
        self.neutral_id = self._infer_neutral_id()
        num = 2
        self.fusion = NewFusionGate(bert_config.hidden_size * num)
        self.arise_base_dim = bert_config.hidden_size * num
        self.arise_preproj = bool(cfg.get("arise_preproj", False))
        self.arise_in_dim = int(cfg.get("arise_in_dim", self.arise_base_dim))
        if self.arise_in_dim <= 0:
            raise ValueError("arise_in_dim must be > 0.")
        self.arise_model_dim = self.arise_in_dim if self.arise_preproj else self.arise_base_dim
        if self.arise_preproj:
            # Scheme-1: lightweight linear pre-projection before ARISE (no activation/dropout).
            self.arise_preproj_m = nn.Linear(self.arise_base_dim, self.arise_model_dim)
            self.arise_preproj_t = nn.Linear(self.arise_base_dim, self.arise_model_dim)
            self.arise_preproj_a = nn.Linear(self.arise_base_dim, self.arise_model_dim)
            self.arise_preproj_v = nn.Linear(self.arise_base_dim, self.arise_model_dim)
            # Keep downstream heads unchanged by mapping ARISE output back to base dim.
            self.arise_postproj = nn.Linear(self.arise_model_dim, self.arise_base_dim)

        a_proj_dropout = float(cfg.get("a_proj_dropout", 0.1))
        v_proj_dropout = float(cfg.get("v_proj_dropout", 0.1))

        self.video_linear = nn.Sequential(
            nn.Linear(cfg.video_dim, bert_config.hidden_size),
            nn.ReLU(),
            #nn.GELU(),
            nn.Dropout(v_proj_dropout),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.audio_linear = nn.Sequential(
            nn.Linear(cfg.audio_dim, bert_config.hidden_size),
            nn.ReLU(),
            #nn.GELU(),
            nn.Dropout(a_proj_dropout),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        emo_cau_hidden_dim = int(
            cfg.get("emo_cau_hidden_dim", cfg.get("emo_hidden_dim", cfg.get("hid_size", 128)))
        )
        self.ecp_concat_emo_cau = bool(cfg.get("ecp_concat_emo_cau", False))
        self.rop_concat_emo_cau = bool(cfg.get("rop_concat_emo_cau", False))
        self.base_pair_dim = bert_config.hidden_size * num
        self.task_pair_dim = emo_cau_hidden_dim
        emo_dropout = float(cfg.get("emo_dropout", 0.1))
        cau_dropout = float(cfg.get("cau_dropout", 0.1))
        self.emo_encoder = nn.Sequential(
            nn.Linear(bert_config.hidden_size * num, emo_cau_hidden_dim),
            nn.LayerNorm(emo_cau_hidden_dim),
            nn.ReLU(),
            nn.Dropout(emo_dropout),
        )
        self.emo_classifier = nn.Linear(emo_cau_hidden_dim, num_classes)
        legacy_cau_hidden_dim = cfg.get("cau_hidden_dim", None)
        if legacy_cau_hidden_dim is not None and int(legacy_cau_hidden_dim) != emo_cau_hidden_dim:
            raise ValueError(
                f"cau_hidden_dim({int(legacy_cau_hidden_dim)}) must equal "
                f"emo_cau_hidden_dim({emo_cau_hidden_dim})"
            )
        self.cau_encoder = nn.Sequential(
            nn.Linear(bert_config.hidden_size * num, emo_cau_hidden_dim),
            nn.LayerNorm(emo_cau_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cau_dropout),
        )
        self.cau_classifier = nn.Linear(emo_cau_hidden_dim, 2)
        # 恢复原版: 保持 scale=0 (不进行归一化)
        # 这虽然会导致初期 ECP loss 数值较大，但它隐式地将 Biaffine 的梯度放大了约39倍，
        # 恰好弥补了全局学习率太小的问题，属于该模型隐式的有效设计。
        ecp_pair_dim = self.base_pair_dim + self.task_pair_dim if self.ecp_concat_emo_cau else self.base_pair_dim
        rop_pair_dim = self.base_pair_dim + self.task_pair_dim if self.rop_concat_emo_cau else self.base_pair_dim
        self.biaffine = Biaffine(ecp_pair_dim, 2)
        self.rope_embedder = RoEmbedding(cfg, rop_pair_dim)
        # Preferred key: t_bert_dropout (text branch dropout on BERT outputs).
        # Keep old keys for backward compatibility with previous experiment configs.
        t_bert_dropout = float(
            cfg.get("t_bert_dropout", cfg.get("model_dropout", cfg.get("dropout", 0.0)))
        )
        self.t_bert_dropout = nn.Dropout(t_bert_dropout)

        self.lstm = EnhancedLSTM('drop_connect', bert_config.hidden_size, bert_config.hidden_size, 1, ff_dropout=0.1, recurrent_dropout=0.1, bidirectional=True)

        # ARISE standard path is mandatory, and PE is always enabled.
        def _cfg_get(arise_key, trie_key, default):
            return cfg.get(arise_key, cfg.get(trie_key, default))
        arise_n_heads = int(_cfg_get("arise_n_heads", "trie_n_heads", 8))
        if self.arise_model_dim % arise_n_heads != 0:
            raise ValueError(
                f"arise_model_dim({self.arise_model_dim}) must be divisible by arise_n_heads({arise_n_heads})"
            )

        self.arise_encoder = ARISEEncoder(
            input_dim=self.arise_model_dim,
            n_heads=arise_n_heads,
            num_layers=int(_cfg_get("arise_num_layers", "trie_num_layers", 2)),
            ffn_mult=float(_cfg_get("arise_ffn_mult", "trie_ffn_mult", 2.0)),
            ffn_act=str(_cfg_get("arise_ffn_act", "trie_ffn_act", "gelu")),
            attn_dropout=float(_cfg_get("arise_attn_dropout", "trie_attn_dropout", 0.1)),
            ffn_dropout=float(_cfg_get("arise_ffn_dropout", "trie_ffn_dropout", 0.1)),
            resid_dropout=float(_cfg_get("arise_resid_dropout", "trie_resid_dropout", 0.1)),
            norm_type=str(_cfg_get("arise_norm_type", "trie_norm_type", "layernorm")),
            norm_order=str(_cfg_get("arise_norm_order", "trie_norm_order", "pre")),
            norm_eps=float(_cfg_get("arise_norm_eps", "trie_norm_eps", 1e-6)),
            use_pe=True,
            pe_max_utt=int(_cfg_get("arise_pe_max_utt", "trie_pe_max_utt", 128)),
            view_fuse=str(_cfg_get("arise_view_fuse", "trie_view_fuse", "max")),
        )
        # Backward-compatible alias for old name.
        self.trie_encoder = self.arise_encoder

        # ============ CARE: Causal Reconstruction and Embedding Alignment ============
        self.lambda_recon = float(cfg.get("lambda_recon", 1.0))
        self.lambda_align = float(cfg.get("lambda_align", 0.1))
        self.care_align_tau = float(cfg.get("care_align_tau", 0.1))
        self.care_diff_tau = float(cfg.get("care_diff_tau", 0.01))
        if self.care_align_tau <= 0:
            raise ValueError("care_align_tau must be > 0.")
        self.care_mask_future = bool(cfg.get("care_mask_future", True))
        self.care_cf_recon = bool(cfg.get("care_cf_recon", False))
        self.care_warm_epochs = int(cfg.get("care_warm_epochs", 0))
        self.care_freeze_decoder_after_warmup = bool(
            cfg.get("care_freeze_decoder_after_warmup", True)
        )
        self.agg_mode = str(cfg.get("agg_mode", "mean")).lower()
        if self.agg_mode not in {"mean", "pairscore"}:
            raise ValueError(f"Unsupported agg_mode: {self.agg_mode}")

        care_bottleneck_dim = int(cfg.get("care_bottleneck_dim", emo_cau_hidden_dim // 2))
        self.causal_decoder = nn.Sequential(
            nn.Linear(emo_cau_hidden_dim, care_bottleneck_dim),
            nn.LayerNorm(care_bottleneck_dim),
            nn.ReLU(),
            nn.Linear(care_bottleneck_dim, emo_cau_hidden_dim),
        )

        self.proj_before_align = bool(cfg.get("proj_before_align", True))
        if self.proj_before_align:
            care_proj_dim = int(cfg.get("care_proj_dim", emo_cau_hidden_dim))
            self.emo_proj = nn.Sequential(
                nn.Linear(emo_cau_hidden_dim, care_proj_dim),
                nn.ReLU(),
                nn.Linear(care_proj_dim, care_proj_dim),
            )
            self.cau_proj = nn.Sequential(
                nn.Linear(emo_cau_hidden_dim, care_proj_dim),
                nn.ReLU(),
                nn.Linear(care_proj_dim, care_proj_dim),
            )

        # ---- care_align_metric: "cos" | "jsd" ----
        self.care_align_metric = str(cfg.get("care_align_metric", "cos")).lower()
        if self.care_align_metric not in {"cos", "jsd"}:
            raise ValueError(f"Unsupported care_align_metric: {self.care_align_metric}")
        if self.care_align_metric == "jsd":
            self.care_jsd_softmax_temp = float(cfg.get("care_jsd_softmax_temp", 0.5))

        self.apply(self._init_esim_weights)

        self.bert = AutoModel.from_pretrained(cfg.bert_path)

    def _infer_neutral_id(self):
        # ECF -> 0, ConvECPE -> 2 by dataset convention.
        return int(resolve_neutral_id(self.cfg, model_label_space=True))

    def _init_esim_weights(self, module):
        """Initialize Linear layers with Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _update_causal_decoder_trainability(self, current_epoch):
        should_freeze = (
            self.care_freeze_decoder_after_warmup
            and self.care_warm_epochs > 0
            and current_epoch >= self.care_warm_epochs
        )
        is_frozen = getattr(self, "_care_decoder_is_frozen", False)
        if should_freeze == is_frozen:
            return
        for param in self.causal_decoder.parameters():
            param.requires_grad = not should_freeze
        self._care_decoder_is_frozen = should_freeze

    def reset_care_pred_cf_epoch_stats(self):
        self._care_pred_false_to_true_count = 0
        self._care_pred_true_to_false_count = 0

    def consume_care_pred_cf_epoch_stats(self):
        false_to_true = int(getattr(self, "_care_pred_false_to_true_count", 0))
        true_to_false = int(getattr(self, "_care_pred_true_to_false_count", 0))
        self.reset_care_pred_cf_epoch_stats()
        return {
            "false_to_true": false_to_true,
            "true_to_false": true_to_false,
        }
    
    def _build_pair_side_inputs(self, h_utt_fused, h_emo, h_cau, use_task_repr):
        if not use_task_repr:
            return h_utt_fused, h_utt_fused
        emo_side = torch.cat((h_utt_fused, h_emo), dim=-1)
        cau_side = torch.cat((h_utt_fused, h_cau), dim=-1)
        return emo_side, cau_side

    def get_dot_product(self, emo_pair_inputs, cau_pair_inputs, masks, gold_matrix):
        """Compute biaffine pair logits and corresponding ECP loss."""
        product = self.biaffine(emo_pair_inputs, cau_pair_inputs).squeeze(-1)
        if len(product.shape) == 3:
            product = product.unsqueeze(-1)
        product = product.transpose(2, 1).transpose(3, 2).contiguous()

        activate_loss = masks.view(-1) == 1
        activate_logits = product.view(-1, 2)[activate_loss]
        activate_gold = gold_matrix.view(-1)[activate_loss]
        # Pair-class weighting: negative class fixed at 1.0, positive class configurable.
        # Keep compatibility with old fields: pair_pos_weight / loss_weight.
        pair_pos_weight = float(
            getattr(
                self.cfg,
                "ecp_pair_pos_weight",
                self.cfg.get("pair_pos_weight", self.cfg.get("loss_weight", 1.5)),
            )
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pair_pos_weight], device=emo_pair_inputs.device)
        )
        loss = criterion(activate_logits, activate_gold.long())
        if torch.isnan(loss):
            # Keep loss as Tensor to avoid downstream `.detach()` crashes.
            loss = emo_pair_inputs.new_tensor(0.0)
        elif torch.isinf(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
        return loss, product 
    
    def get_emotion(self, logits, utterance_nums, emotion_labels, emo=True):
        mask = build_utterance_mask(logits.shape[1], utterance_nums, logits.device)
        activate_loss = mask.view(-1) == 1
        activate_logits = logits.view(-1, logits.shape[-1])[activate_loss]

        activate_gold = emotion_labels.view(-1)[activate_loss]
        if emo:
            num_classes = int(getattr(self, "num_classes", logits.shape[-1]))
            non_neutral_w = float(getattr(self.cfg, "emo_non_neutral_weight", 1.5))
            # Keep relative weighting: neutral=1.0, non-neutral=emo_non_neutral_weight.
            weight = torch.full((num_classes,), non_neutral_w, device=logits.device)
            neutral_id = int(getattr(self, "neutral_id", 0))
            if 0 <= neutral_id < num_classes:
                weight[neutral_id] = 1.0
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            # Cause keeps binary relative weighting.
            cause_pos_w = float(getattr(self.cfg, "cause_pos_weight", 1.5))
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, cause_pos_w], device=logits.device)
            )
        loss = criterion(activate_logits, activate_gold.long())
        return loss

    def _align_sims(
        self,
        anchor: torch.Tensor,
        others: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """
        Compute alignment scores between one anchor and N candidate vectors.

        cos mode : score = dot(anchor, other) / tau  in [-1/tau, +1/tau]
        jsd mode : score = -exp(JSD(p_anchor, p_other)) / tau  in [-2/tau, -1/tau]
              where JSD is Jensen-Shannon Divergence on softmax distributions.

        Parameters
        ----------
        anchor : [D]
        others : [N, D]

        Returns
        -------
        [N] score tensor  (higher = more aligned)
        """
        if self.care_align_metric == "cos":
            return (anchor.unsqueeze(0) * others).sum(-1) / tau
        # ---- jsd mode ----
        eps = 1e-8
        a = anchor.unsqueeze(0).expand(others.shape[0], -1)     # [N, D]
        m = 0.5 * (a + others)                                   # [N, D]
        kl_am = (a * (a.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
        kl_bm = (others * (others.clamp(min=eps).log() - m.clamp(min=eps).log())).sum(-1)
        jsd = 0.5 * (kl_am + kl_bm)                             # [N]  in [0, log2]
        return -jsd.exp() / tau                                  # [N]  in [-2/tau, -1/tau]

    def _get_align_repr(self, h_emo, h_cau):
        if self.proj_before_align:
            raw_emo = self.emo_proj(h_emo)
            raw_cau = self.cau_proj(h_cau)
        else:
            raw_emo = h_emo
            raw_cau = h_cau

        if self.care_align_metric == "cos":
            z_emo = F.normalize(raw_emo, dim=-1)
            z_cau = F.normalize(raw_cau, dim=-1)
        else:
            temp = self.care_jsd_softmax_temp
            z_emo = F.softmax(raw_emo / temp, dim=-1)
            z_cau = F.softmax(raw_cau / temp, dim=-1)
        return z_emo, z_cau

    def _get_tsne_triplet_repr(self, h_emo, h_cau, gold_matrix, utterance_nums, pair_mask, pair_logits):
        """
        Build four feature groups for visualization:
        1) original emotion feature (h_emo)
        2) reconstructed emotion feature (h_recon)
        3) aggregated cause representation before replacement (h_agg)
        4) aggregated cause representation after replacement (h_cf_agg)
        """
        pair_mask = pair_mask.float()
        pair_matrix = gold_matrix.float()
        if self.care_mask_future:
            pair_matrix = pair_matrix * pair_mask

        base_pair_logits = pair_logits if self.agg_mode == "pairscore" else None
        h_agg = self._aggregate_cause_repr(h_cau, pair_matrix, pair_logits=base_pair_logits)
        h_recon = self.causal_decoder(h_agg)

        cf_pair_matrix, _ = self._build_cf_pair_matrix(
            pair_matrix, pair_mask, h_cau, utterance_nums
        )
        cf_pair_weights = None
        if self.agg_mode == "pairscore":
            cf_pair_weights = self._build_cf_pairscore_weights(
                pair_matrix, cf_pair_matrix, pair_logits
            )
        h_cf_agg = self._aggregate_cause_repr(
            h_cau,
            cf_pair_matrix,
            pair_logits=base_pair_logits,
            pair_weights=cf_pair_weights,
        )
        return h_emo, h_recon, h_agg, h_cf_agg, "h"

    def _compute_pairscore_weights(self, pair_matrix, pair_logits):
        if pair_logits is None:
            raise ValueError("pair_logits is required when agg_mode='pairscore'.")
        edge_scores = pair_logits[..., 1] - pair_logits[..., 0]
        masked_scores = edge_scores.masked_fill(pair_matrix <= 0, -1e9)
        pair_weights = torch.softmax(masked_scores, dim=-1) * pair_matrix
        return pair_weights / (pair_weights.sum(dim=-1, keepdim=True) + 1e-8)

    def _build_cf_pairscore_weights(self, pair_matrix, cf_pair_matrix, pair_logits):
        base_weights = self._compute_pairscore_weights(pair_matrix, pair_logits)
        cf_weights = base_weights * cf_pair_matrix

        for b in range(pair_matrix.shape[0]):
            for i in range(pair_matrix.shape[1]):
                removed = torch.where(
                    (pair_matrix[b, i] > 0) & (cf_pair_matrix[b, i] <= 0)
                )[0]
                added = torch.where(
                    (pair_matrix[b, i] <= 0) & (cf_pair_matrix[b, i] > 0)
                )[0]
                if len(removed) == 0 or len(added) == 0:
                    continue

                moved_weight = base_weights[b, i, removed].sum()
                cf_weights[b, i, added] = moved_weight / len(added)

        fallback_rows = (cf_pair_matrix.sum(dim=-1) > 0) & (cf_weights.sum(dim=-1) <= 1e-8)
        if fallback_rows.any():
            recomputed_weights = self._compute_pairscore_weights(cf_pair_matrix, pair_logits)
            cf_weights = torch.where(
                fallback_rows.unsqueeze(-1), recomputed_weights, cf_weights
            )
        return cf_weights

    def _aggregate_cause_repr(self, h_cau, pair_matrix, pair_logits=None, pair_weights=None):
        pair_count = pair_matrix.sum(dim=-1)
        if pair_weights is not None:
            return torch.bmm(pair_weights, h_cau)
        if self.agg_mode == "pairscore":
            pair_weights = self._compute_pairscore_weights(pair_matrix, pair_logits)
            return torch.bmm(pair_weights, h_cau)

        h_agg = torch.bmm(pair_matrix, h_cau)
        return h_agg / (pair_count.unsqueeze(-1) + 1e-8)

    def _build_cf_pair_matrix(self, pair_matrix, pair_mask, h_cau, utterance_nums):
        cf_pair_matrix = pair_matrix.clone()
        cf_valid = torch.zeros(pair_matrix.shape[:2], dtype=torch.bool, device=pair_matrix.device)
        cau_repr = F.normalize(h_cau.detach(), dim=-1)

        for b in range(pair_matrix.shape[0]):
            cur_utt_num = int(utterance_nums[b].item())
            if cur_utt_num <= 0:
                continue

            if self.care_mask_future:
                allowed_mask = pair_mask[b, :cur_utt_num, :cur_utt_num] > 0
            else:
                allowed_mask = torch.ones_like(
                    pair_matrix[b, :cur_utt_num, :cur_utt_num], dtype=torch.bool
                )

            for i in range(cur_utt_num):
                pos_indices = torch.where(pair_matrix[b, i, :cur_utt_num] > 0)[0]
                if len(pos_indices) == 0:
                    continue

                neg_mask = allowed_mask[i].clone()
                neg_mask[pos_indices] = False
                neg_mask[i] = False
                neg_indices = torch.where(neg_mask)[0]
                if len(neg_indices) == 0:
                    continue

                pos_repr = cau_repr[b, pos_indices]
                neg_repr = cau_repr[b, neg_indices]
                sim_mat = torch.matmul(pos_repr, neg_repr.transpose(0, 1))
                best_flat = int(torch.argmax(sim_mat).item())
                neg_num = int(neg_indices.numel())
                pos_choice = best_flat // neg_num
                neg_choice = best_flat % neg_num
                replace_pos = int(pos_indices[pos_choice].item())
                replace_neg = int(neg_indices[neg_choice].item())

                cf_pair_matrix[b, i, replace_pos] = 0.0
                cf_pair_matrix[b, i, replace_neg] = 1.0
                cf_valid[b, i] = True

        return cf_pair_matrix, cf_valid

    def _build_cf_pair_matrix_pred(
        self, pair_matrix, gold_matrix, pair_mask, h_cau, utterance_nums
    ):
        """
        Build counterfactual pair matrix when using predicted pairs.
        Strategy:
        - If false positives exist: replace FP with semantically similar non-causes
        - If all true positives: replace TP with most similar non-cause (like gold strategy)
        """
        cf_pair_matrix = pair_matrix.clone()
        cf_valid = torch.zeros(
            pair_matrix.shape[:2], dtype=torch.bool, device=pair_matrix.device
        )
        cau_repr = F.normalize(h_cau.detach(), dim=-1)

        for b in range(pair_matrix.shape[0]):
            cur_utt_num = int(utterance_nums[b].item())
            if cur_utt_num <= 0:
                continue

            if self.care_mask_future:
                allowed_mask = pair_mask[b, :cur_utt_num, :cur_utt_num] > 0
            else:
                allowed_mask = torch.ones_like(
                    pair_matrix[b, :cur_utt_num, :cur_utt_num], dtype=torch.bool
                )

            for i in range(cur_utt_num):
                pred_pos = torch.where(pair_matrix[b, i, :cur_utt_num] > 0)[0]
                gold_pos = torch.where(gold_matrix[b, i, :cur_utt_num] > 0)[0]

                if len(pred_pos) == 0:
                    continue

                # Identify false positives
                fp_mask = torch.zeros(cur_utt_num, dtype=torch.bool, device=h_cau.device)
                for p in pred_pos:
                    if p not in gold_pos:
                        fp_mask[p] = True
                fp_indices = torch.where(fp_mask)[0]

                # Case A: False positives exist - replace FP with similar missed gold causes
                if len(fp_indices) > 0:
                    other_neg_mask = torch.zeros_like(allowed_mask[i])
                    other_neg_mask[gold_pos] = True
                    other_neg_mask = other_neg_mask & allowed_mask[i]
                    other_neg_mask[pred_pos] = False
                    other_neg_mask[i] = False
                    other_neg = torch.where(other_neg_mask)[0]

                    if len(other_neg) == 0:
                        continue

                    fp_repr = cau_repr[b, fp_indices]
                    neg_repr = cau_repr[b, other_neg]
                    sim_mat = torch.matmul(fp_repr, neg_repr.transpose(0, 1))

                    best_flat = int(torch.argmax(sim_mat).item())
                    fp_choice = best_flat // len(other_neg)
                    neg_choice = best_flat % len(other_neg)

                    replace_fp = int(fp_indices[fp_choice].item())
                    replace_neg = int(other_neg[neg_choice].item())

                    cf_pair_matrix[b, i, replace_fp] = 0.0
                    cf_pair_matrix[b, i, replace_neg] = 1.0
                    cf_valid[b, i] = True
                    self._care_pred_false_to_true_count = int(
                        getattr(self, "_care_pred_false_to_true_count", 0)
                    ) + 1

                # Case B: All true positives - use gold-like strategy
                else:
                    all_neg_mask = allowed_mask[i].clone()
                    all_neg_mask[gold_pos] = False
                    all_neg_mask[i] = False
                    all_neg = torch.where(all_neg_mask)[0]

                    if len(all_neg) == 0:
                        continue

                    tp_repr = cau_repr[b, pred_pos]
                    neg_repr = cau_repr[b, all_neg]
                    sim_mat = torch.matmul(tp_repr, neg_repr.transpose(0, 1))

                    best_flat = int(torch.argmax(sim_mat).item())
                    tp_choice = best_flat // len(all_neg)
                    neg_choice = best_flat % len(all_neg)

                    replace_tp = int(pred_pos[tp_choice].item())
                    replace_neg = int(all_neg[neg_choice].item())

                    cf_pair_matrix[b, i, replace_tp] = 0.0
                    cf_pair_matrix[b, i, replace_neg] = 1.0
                    cf_valid[b, i] = True
                    self._care_pred_true_to_false_count = int(
                        getattr(self, "_care_pred_true_to_false_count", 0)
                    ) + 1

        return cf_pair_matrix, cf_valid

    def compute_care_loss(
        self,
        h_emo,
        h_cau,
        gold_matrix,
        utterance_nums,
        pair_mask,
        pair_logits=None,
        current_epoch=0,
        pred_pair_matrix=None,
    ):
        _, utterance_num, _ = h_emo.shape
        utt_mask = build_utterance_mask(utterance_num, utterance_nums, h_emo.device)
        pair_mask = pair_mask.float()

        # Decide whether to use predicted pair matrix based on care_warm_epochs
        use_pred = (
            self.care_warm_epochs > 0
            and current_epoch >= self.care_warm_epochs
            and pred_pair_matrix is not None
        )

        if use_pred:
            # After warmup: use predicted pair matrix
            pair_matrix = pred_pair_matrix.float()
            if self.care_mask_future:
                pair_matrix = pair_matrix * pair_mask
        else:
            # Before warmup or care_warm_epochs=0: use gold matrix (original behavior)
            pair_matrix = gold_matrix.float()

        pair_count = pair_matrix.sum(dim=-1)
        has_cause = (pair_count > 0) & utt_mask

        h_agg = self._aggregate_cause_repr(h_cau, pair_matrix, pair_logits=pair_logits)
        h_recon = self.causal_decoder(h_agg)

        recon_target = h_emo.detach()
        recon_error = ((h_recon - recon_target) ** 2).mean(dim=-1) + (
            1.0 - F.cosine_similarity(h_recon, recon_target, dim=-1)
        )
        if has_cause.any():
            has_cause_float = has_cause.float()
            recon_main = (recon_error * has_cause_float).sum() / (
                has_cause_float.sum() + 1e-8
            )
        else:
            recon_main = h_emo.new_tensor(0.0)

        recon_diff = h_emo.new_tensor(0.0)
        recon_loss = recon_main

        if self.care_cf_recon and has_cause.any():
            has_false = use_pred and bool(
                ((pair_matrix - gold_matrix).clamp(min=0)).sum() > 0
            )
            if use_pred:
                # After warmup: use pred-specific counterfactual strategy
                cf_pair_matrix, cf_valid = self._build_cf_pair_matrix_pred(
                    pair_matrix=pair_matrix,
                    gold_matrix=gold_matrix,
                    pair_mask=pair_mask,
                    h_cau=h_cau,
                    utterance_nums=utterance_nums,
                )
            else:
                # Before warmup: use original counterfactual strategy
                cf_pair_matrix, cf_valid = self._build_cf_pair_matrix(
                    pair_matrix, pair_mask, h_cau, utterance_nums
                )
            cf_mask = has_cause & cf_valid
            if cf_mask.any():
                cf_pair_weights = None
                if self.agg_mode == "pairscore":
                    # Counterfactual reconstruction should only replace semantics,
                    # not renormalize the original pairscore weights.
                    cf_pair_weights = self._build_cf_pairscore_weights(
                        pair_matrix, cf_pair_matrix, pair_logits
                    )
                h_cf_agg = self._aggregate_cause_repr(
                    h_cau,
                    cf_pair_matrix,
                    pair_logits=pair_logits,
                    pair_weights=cf_pair_weights,
                )
                h_cf_recon = self.causal_decoder(h_cf_agg)
                cf_error = ((h_cf_recon - recon_target) ** 2).mean(dim=-1) + (
                    1.0 - F.cosine_similarity(h_cf_recon, recon_target, dim=-1)
                )
                diff_error = (
                    cf_error - recon_error if has_false else recon_error - cf_error
                ) / self.care_diff_tau
                recon_diff = (
                    F.softplus(diff_error) * cf_mask.float()
                ).sum() / (cf_mask.float().sum() + 1e-8)
                recon_loss = recon_main + recon_diff

        # ---- (旧 cos-only 代码) ----
        # if self.proj_before_align:
        #     z_emo = F.normalize(self.emo_proj(h_emo), dim=-1)
        #     z_cau = F.normalize(self.cau_proj(h_cau), dim=-1)
        # else:
        #     z_emo = F.normalize(h_emo, dim=-1)
        #     z_cau = F.normalize(h_cau, dim=-1)
        # ---- 新代码：支持 cos / jsd 两种模式 ----
        z_emo, z_cau = self._get_align_repr(h_emo, h_cau)

        # Align loss: use gold when care_warm_epochs=0 OR before warmup, otherwise use pair_matrix
        if self.care_warm_epochs == 0 or not use_pred:
            # Original behavior: align uses same matrix as recon
            align_pair_matrix = pair_matrix
        else:
            # After warmup: align always uses gold (contrastive learning)
            align_pair_matrix = gold_matrix.float()
            if self.care_mask_future:
                align_pair_matrix = align_pair_matrix * pair_mask

        all_z_cau_batch = z_cau[utt_mask]
        align_losses = []
        for b in range(h_emo.shape[0]):
            cur_utt_num = int(utterance_nums[b].item())
            pm_b = align_pair_matrix[b, :cur_utt_num, :cur_utt_num]
            z_emo_b = z_emo[b, :cur_utt_num]

            for i in range(cur_utt_num):
                pos_indices = torch.where(pm_b[i] > 0)[0]
                if len(pos_indices) == 0:
                    continue

                anchor = z_emo_b[i]
                pos_z_cau = z_cau[b, pos_indices]
                # (旧 cos-only) pos_sims = (anchor.unsqueeze(0) * pos_z_cau).sum(dim=-1) / self.care_align_tau
                # (旧 cos-only) all_sims = (anchor.unsqueeze(0) * all_z_cau).sum(dim=-1) / self.care_align_tau
                pos_sims = self._align_sims(anchor, pos_z_cau, self.care_align_tau)
                all_sims = self._align_sims(anchor, all_z_cau_batch, self.care_align_tau)
                log_denom = torch.logsumexp(all_sims, dim=0)
                align_losses.append(-(pos_sims - log_denom).mean())

        if align_losses:
            align_loss = torch.stack(align_losses).mean()
        else:
            align_loss = h_emo.new_tensor(0.0)

        return recon_loss, align_loss, recon_main, recon_diff
    
    def build_attention(self, sequence_outputs, gmasks=None, smasks=None, lmasks=None):
        """
        Run ARISE encoder on concatenated views and keep only the first view length
        (the fused utterance stream).
        """
        fused = self.arise_encoder(
            sequence_outputs, gmasks=gmasks, smasks=smasks, lmasks=lmasks, return_attn=False
        )
        length = sequence_outputs.shape[1] // 4
        return fused[:, :length]


    def forward(self, **kwargs):
        """Forward pass for joint emotion, cause, and pair prediction."""
        # 1) Read batch tensors.
        input_ids, input_masks, utterance_nums = [kwargs[w] for w in 'input_ids input_masks utterance_nums'.split()]
        pairs, pair_nums, labels, indices = [kwargs[w] for w in 'pairs pair_nums labels indices'.split()]
        cause_labels, speaker_ids = [kwargs[w] for w in ['cause_labels', 'speaker_ids']]
        audio_features, video_features = [kwargs[w] for w in ['audio_features', 'video_features']]
        gmasks, smasks = [kwargs[w] for w in ['gmasks', 'smasks']]
        lmasks = kwargs.get("lmasks", kwargs.get("rmasks", None))
        if lmasks is None:
            raise KeyError("Missing local mask: expected 'lmasks' (or compatibility key 'rmasks').")

        # 2) Token encoder.
        h_tok_bert = self.bert(input_ids, attention_mask=input_masks)[0]

        # 3) Build utterance-level multimodal features.
        speaker_emb = self.speaker_embedder(speaker_ids)
        h_utt_t = merge_token_to_utterance(h_tok_bert, indices)
        h_utt_t = self.t_bert_dropout(h_utt_t)
        h_utt_a = self.audio_linear(audio_features)
        h_utt_v = self.video_linear(video_features)

        if not self.use_text:
            h_utt_t = torch.zeros_like(h_utt_t)
        if not self.use_audio:
            h_utt_a = torch.zeros_like(h_utt_a)
        if not self.use_video:
            h_utt_v = torch.zeros_like(h_utt_v)

        h_utt_fused = h_utt_t + speaker_emb + h_utt_a + h_utt_v
        h_utt_fused_lstm = self.lstm(h_utt_fused, None, utterance_nums.cpu())

        # 4) ARISE contextual refinement.
        h_utt_t_2x = torch.cat((h_utt_t, h_utt_t), dim=-1)
        h_utt_a_2x = torch.cat((h_utt_a, h_utt_a), dim=-1)
        h_utt_v_2x = torch.cat((h_utt_v, h_utt_v), dim=-1)

        if self.disable_mr:
            h_utt_fused = h_utt_fused_lstm
        else:
            if self.arise_preproj:
                h_arise_m = self.arise_preproj_m(h_utt_fused_lstm)
                h_arise_t = self.arise_preproj_t(h_utt_t_2x)
                h_arise_a = self.arise_preproj_a(h_utt_a_2x)
                h_arise_v = self.arise_preproj_v(h_utt_v_2x)
                h_arise_input = torch.cat((h_arise_m, h_arise_t, h_arise_a, h_arise_v), 1)
                h_arise_output = self.build_attention(h_arise_input, gmasks, smasks, lmasks)
                h_arise_output = self.arise_postproj(h_arise_output)
            else:
                h_arise_input = torch.cat((h_utt_fused_lstm, h_utt_t_2x, h_utt_a_2x, h_utt_v_2x), 1)
                h_arise_output = self.build_attention(h_arise_input, gmasks, smasks, lmasks)
            h_utt_fused = self.fusion(h_utt_fused_lstm, h_arise_output)

        # 5) Emotion and cause heads.
        h_emo = self.emo_encoder(h_utt_fused)
        emotion_logits = self.emo_classifier(h_emo)
        emo_loss = self.get_emotion(emotion_logits, utterance_nums, labels, emo=True)

        h_cau = self.cau_encoder(h_utt_fused)
        cause_logits = self.cau_classifier(h_cau)
        cause_loss = self.get_emotion(cause_logits, utterance_nums, cause_labels, emo=False)

        # 6) Pair heads (biaffine + RoPE).
        ecp_mask, gold_matrix = build_pair_mask_and_gold(
            h_utt_fused, utterance_nums, pair_nums, pairs
        )
        ecp_emo_inputs, ecp_cau_inputs = self._build_pair_side_inputs(
            h_utt_fused, h_emo, h_cau, self.ecp_concat_emo_cau
        )
        rop_emo_inputs, rop_cau_inputs = self._build_pair_side_inputs(
            h_utt_fused, h_emo, h_cau, self.rop_concat_emo_cau
        )
        ecp_loss, ecp_logits = self.get_dot_product(
            ecp_emo_inputs, ecp_cau_inputs, ecp_mask, gold_matrix
        )
        rop_loss, rop_logits = self.rope_embedder.classify_matrix(
            rop_emo_inputs, gold_matrix, ecp_mask, key_sequence_outputs=rop_cau_inputs
        )

        # 7) Task loss weights (default 1.0 each).
        lambda_emo = float(getattr(self.cfg, "lambda_emo", 1.0))
        lambda_cau = float(getattr(self.cfg, "lambda_cau", 1.0))
        lambda_pair = float(getattr(self.cfg, "lambda_pair", 1.0))
        pair_loss = rop_loss + ecp_loss
        pair_logits = ecp_logits + rop_logits

        # 8) CARE auxiliary losses.
        # Get current epoch from model attribute (set by trainer)
        current_epoch = getattr(self, "global_epoch", 0)
        if self.training:
            self._update_causal_decoder_trainability(current_epoch)
        # Generate predicted pair matrix from pair_logits
        pred_pair_matrix = (pair_logits[..., 1] > pair_logits[..., 0]).float()

        recon_loss = h_utt_fused.new_tensor(0.0)
        align_loss = h_utt_fused.new_tensor(0.0)
        recon_main = h_utt_fused.new_tensor(0.0)
        recon_diff = h_utt_fused.new_tensor(0.0)

        # Only compute CARE loss during training (same as original behavior)
        if self.training:
            if self.agg_mode == "pairscore":
                recon_loss, align_loss, recon_main, recon_diff = self.compute_care_loss(
                    h_emo=h_emo,
                    h_cau=h_cau,
                    pair_logits=pair_logits,
                    gold_matrix=gold_matrix,
                    utterance_nums=utterance_nums,
                    pair_mask=ecp_mask,
                    current_epoch=current_epoch,
                    pred_pair_matrix=pred_pair_matrix,
                )
            else:
                recon_loss, align_loss, recon_main, recon_diff = self.compute_care_loss(
                    h_emo=h_emo,
                    h_cau=h_cau,
                    gold_matrix=gold_matrix,
                    utterance_nums=utterance_nums,
                    pair_mask=ecp_mask,
                    current_epoch=current_epoch,
                    pred_pair_matrix=pred_pair_matrix,
                )

        # 9) Total loss.
        loss = (
            lambda_pair * pair_loss
            + lambda_emo * emo_loss
            + lambda_cau * cause_loss
            + self.lambda_recon * recon_loss
            + self.lambda_align * align_loss
        )

        self.last_loss_dict = {
            "total": float(loss.detach().item()),
            "emo": float(emo_loss.detach().item()),
            "cause": float(cause_loss.detach().item()),
            "ecp": float(ecp_loss.detach().item()),
            "rop": float(rop_loss.detach().item()),
            "pair": float(pair_loss.detach().item()),
            "recon": float(recon_loss.detach().item()),
            "recon_main": float(recon_main.detach().item()),
            "recon_diff": float(recon_diff.detach().item()),
            "align": float(align_loss.detach().item()),
        }

        if not self.training:
            tsne_emo, tsne_recon, tsne_agg, tsne_cf_agg, tsne_space = self._get_tsne_triplet_repr(
                h_emo=h_emo,
                h_cau=h_cau,
                gold_matrix=gold_matrix,
                utterance_nums=utterance_nums,
                pair_mask=ecp_mask,
                pair_logits=pair_logits,
            )
            tsne_emo = tsne_emo.detach().cpu()
            tsne_recon = tsne_recon.detach().cpu()
            tsne_agg = tsne_agg.detach().cpu()
            tsne_cf_agg = tsne_cf_agg.detach().cpu()
        else:
            tsne_emo, tsne_recon, tsne_agg, tsne_cf_agg, tsne_space = None, None, None, None, None

        return loss, (
            pair_logits,
            emotion_logits,
            cause_logits,
            ecp_mask,
            tsne_emo,
            tsne_recon,
            tsne_agg,
            tsne_cf_agg,
            tsne_space,
        )
