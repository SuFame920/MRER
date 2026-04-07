#!/usr/bin/env python

"""
Name: trainer.py
"""

import os
import json
import yaml

import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, classification_report
from src.tools import resolve_neutral_id


class MyTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.print_test_each_epoch = getattr(config, "print_test_each_epoch", True)
        self.plot_loss_each_epoch = getattr(config, "plot_loss_each_epoch", True)
        self._plot_warned = False

        self.scores = []
        self.lines = []
        self.metrics = []
        self.epoch_losses = []
        self.train_step_losses = []          # per-step train loss, accumulated across all epochs
        self.global_forward_step = 0
        self.neutral_id = self._infer_neutral_id()
        self._collect_dialog_predictions = False
        self.re_init()

    def _infer_neutral_id(self):
        # 与 loader/model 使用同一 neutral id 规则，避免数据集切换时不一致。
        return int(resolve_neutral_id(self.config, model_label_space=True))

    def _get_model_loss_parts(self):
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        parts = getattr(model_obj, "last_loss_dict", None)
        if not isinstance(parts, dict):
            return None
        return dict(parts)

    @staticmethod
    def _mean_or_none(values):
        values = [float(v) for v in values if v is not None]
        if len(values) == 0:
            return None
        return float(np.mean(values))

    def _aggregate_loss_parts(self, parts_list):
        keys = [
            "emo",
            "cause",
            "ecp",
            "rop",
            "pair",
            "recon",
            "recon_main",
            "recon_diff",
            "align",
        ]
        out = {}
        for key in keys:
            out[key] = self._mean_or_none([p.get(key) for p in parts_list if isinstance(p, dict)])
        return out

    def _format_loss_line(self, split, total_loss, parts):
        def fmt(v, width=8):
            if v is None:
                s = "N/A"
            else:
                s = f"{float(v):.4f}"
            return f"{s:>{width}}"

        emo = None if parts is None else parts.get("emo")
        cause = None if parts is None else parts.get("cause")
        ecp = None if parts is None else parts.get("ecp")
        rop = None if parts is None else parts.get("rop")
        pair = None if parts is None else parts.get("pair")
        recon = None if parts is None else parts.get("recon")
        recon_main = None if parts is None else parts.get("recon_main")
        recon_diff = None if parts is None else parts.get("recon_diff")
        align = None if parts is None else parts.get("align")

        split_tag = f"[{str(split).upper()}]"
        pair_expr = f"{fmt(ecp)}+{fmt(rop)}={fmt(pair)}"
        return (
            f"{split_tag:<8}"
            f"Loss={fmt(total_loss)}, "
            f"Emo Loss={fmt(emo)}, "
            f"Cause Loss={fmt(cause)}, "
            f"Pair Loss(ecp+rop)={pair_expr}, "
            f"Recon(main+diff)={fmt(recon_main)}+{fmt(recon_diff)}={fmt(recon)}, "
            f"Align={fmt(align)}"
        )

    def train(self):
        best_score, best_iter = float("-inf"), -1
        best_test_score = float("-inf")
        last_dev_update_iter = -1
        best_metrics = None

        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch

            # train + dev eval
            train_loss, train_parts = self.train_step()
            score, (res, metrics, dev_loss, dev_parts) = self.evaluate_step()

            # record dev metrics and reset buffers
            self.add_instance(score, res, metrics)
            self.re_init()

            # optional: report test each epoch
            test_score = None
            test_loss = None
            test_parts = None
            if self.print_test_each_epoch:
                test_score, (test_res, _, test_loss, test_parts) = self.evaluate_step(self.test_loader)
                self.re_init()

            print(self._format_loss_line("TRAIN", train_loss, train_parts))
            print(self._format_loss_line("DEV", dev_loss, dev_parts))
            if self.print_test_each_epoch:
                print(self._format_loss_line("TEST", test_loss, test_parts))
            else:
                print(self._format_loss_line("TEST", None, None))
            print(f"[DEV Metrics]\n{res}")
            if self.print_test_each_epoch:
                print(f"[TEST Metrics]\n{test_res}")

            # record epoch losses and persist
            self.epoch_losses.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "dev_loss": float(dev_loss),
                    "test_loss": float(test_loss) if test_loss is not None else None,
                }
            )
            if self.plot_loss_each_epoch:
                self.save_loss_plot()

            # save policy:
            # 1) dev pair F1 uses >= for "updated" detection (full precision, no rounding).
            # 2) on dev update, save only when test pair F1 is strictly better than any previous saved test F1.
            if score >= best_score:
                best_score = score
                last_dev_update_iter = epoch

                if test_score is None:
                    test_score, (_, _, test_loss_for_save, _) = self.evaluate_step(self.test_loader)
                    if test_loss is None:
                        test_loss = test_loss_for_save
                    self.re_init()

                if test_score > best_test_score:
                    if best_iter > -1:
                        old_path = self.save_name.format(best_iter)
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    best_iter = epoch
                    best_metrics = metrics
                    best_test_score = test_score

                    if not os.path.exists(self.config.target_dir):
                        os.makedirs(self.config.target_dir)

                    torch.save(
                        {
                            "epoch": epoch,
                            "model": self.model.cpu().state_dict(),
                            "best_score": score,
                            "best_test_score": test_score,
                        },
                        self.save_name.format(epoch),
                    )
                    self.model.to(self.config.device)

            # early stopping
            elif last_dev_update_iter > -1 and epoch - last_dev_update_iter > self.config.patience:
                print(f"Not upgrade for {self.config.patience} steps, early stopping...")
                break

        # final evaluation on test
        score, (res, test_metrics, test_loss) = self.final_evaluate(best_iter)
        self.final_score, self.final_res, self.final_metrics = score, res, test_metrics
        self.best_metrics = best_metrics
        self.best_iter = best_iter
        self.save_run_report(best_iter, best_metrics, test_metrics)

    def train_step(self):
        self.model.train()
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_obj, "reset_care_pred_cf_epoch_stats"):
            model_obj.reset_care_pred_cf_epoch_stats()
        train_data = tqdm(self.train_loader)
        losses = []
        parts_list = []
        total_steps = len(self.train_loader)
        accum_steps = max(1, int(getattr(self.config, "gradient_accumulation_steps", 1)))
        self.config.optimizer.zero_grad()

        for step, data in enumerate(train_data, start=1):
            loss, _ = self.model(**data)
            cur_loss = float(loss.item())
            losses.append(cur_loss)
            self.train_step_losses.append(cur_loss)   # per mini-batch
            cur_parts = self._get_model_loss_parts()
            if cur_parts is not None:
                parts_list.append(cur_parts)

            (loss / accum_steps).backward()
            should_step = (step % accum_steps == 0) or (step == total_steps)
            if should_step:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.config.optimizer.step()
                if hasattr(self.config, "scheduler") and self.config.scheduler is not None:
                    self.config.scheduler.step()
                self.config.optimizer.zero_grad()

            train_data.set_description(
                f"Epoch {self.global_epoch}, loss:{np.mean(losses):.4f}"
            )
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # 显式打印每个 epoch 的完成进度行（100%），便于 terminal.log 稳定保留关键进度信息。
        fmt = train_data.format_dict
        elapsed = tqdm.format_interval(fmt.get("elapsed", 0) or 0)
        rate = fmt.get("rate", None)
        if rate is None:
            denom = max(float(fmt.get("elapsed", 0) or 0), 1e-8)
            rate = total_steps / denom
        print(
            f"Epoch {self.global_epoch}, loss:{avg_loss:.4f}: "
            f"100%|{'█'*15}| {total_steps}/{total_steps} "
            f"[{elapsed}<00:00, {rate:5.2f}it/s]"
        )
        if hasattr(model_obj, "consume_care_pred_cf_epoch_stats"):
            pred_cf_stats = model_obj.consume_care_pred_cf_epoch_stats()
            false_to_true = int(pred_cf_stats.get("false_to_true", 0))
            true_to_false = int(pred_cf_stats.get("true_to_false", 0))
            total_pred_cf = false_to_true + true_to_false
            should_report_pred_cf = (
                int(getattr(model_obj, "care_warm_epochs", 0)) > 0
                and self.global_epoch >= int(getattr(model_obj, "care_warm_epochs", 0))
            )
            if should_report_pred_cf:
                false_to_true_ratio = (false_to_true / total_pred_cf) if total_pred_cf > 0 else 0.0
                true_to_false_ratio = (true_to_false / total_pred_cf) if total_pred_cf > 0 else 0.0
                print(
                    f"[CARE Pred CF][Epoch {self.global_epoch}] "
                    f"false_to_true={false_to_true} ({false_to_true_ratio:.2%}), "
                    f"true_to_false={true_to_false} ({true_to_false_ratio:.2%}), "
                    f"total={total_pred_cf}"
                )
        return avg_loss, self._aggregate_loss_parts(parts_list)

    def evaluate_step(self, dataLoader=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        losses = []
        parts_list = []

        for data in dataLoader:
            with torch.no_grad():
                loss, output = self.model(**data)
                losses.append(loss.item())
                cur_parts = self._get_model_loss_parts()
                if cur_parts is not None:
                    parts_list.append(cur_parts)
                self.add_output(data, output)

        score, (res, metrics) = self.report_score()
        avg_loss = float(np.mean(losses)) if losses else 0.0
        return score, (res, metrics, avg_loss, self._aggregate_loss_parts(parts_list))

    def final_evaluate(self, epoch=0):
        checkpoint = torch.load(self.save_name.format(epoch), map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.re_init()
        save_final_artifacts = bool(getattr(self.config, "save_final_artifacts", True))
        self._collect_dialog_predictions = save_final_artifacts
        score, (res, metrics, avg_loss, _) = self.evaluate_step(self.test_loader)
        self._collect_dialog_predictions = False
        if save_final_artifacts:
            self.save_preds_file()
        # ── Post-inference visualizations ──
            target_dir = getattr(self.config, "target_dir", ".")
            self.generate_tsne_plot(target_dir)
        print(res)
        return score, (res, metrics, avg_loss)

    def re_init(self):
        self.preds = defaultdict(list)
        self.golds = defaultdict(list)
        self.dialog_predictions = []
        self.keys = ["default"]
        # t-SNE: collect non-neutral utterance samples for visualization.
        # Legacy sample layout kept as reference:
        # each: {emo_repr, recon_repr, agg_repr, cf_agg_repr}
        self._tsne_samples = []
        self._tsne_repr_space = None
        self._tsne_cf_label = None

    def add_instance(self, score, res, metrics):
        self.scores.append(score)
        self.lines.append(res)
        self.metrics.append(metrics)

    def get_best(self):
        best_id = np.argmax(self.scores)
        return self.lines[best_id]

    def _get_emotion_id2label(self):
        explicit_id2label = self.config.get("emotion_id2label", None)
        if explicit_id2label:
            return {int(k): str(v) for k, v in explicit_id2label.items()}
        label_dict = self.config.get("label_dict", None)
        if isinstance(label_dict, dict):
            return {int(v): str(k) for k, v in label_dict.items()}
        return {}

    def _select_tsne_samples(self):
        per_class_limit = int(getattr(self.config, "tsne_topk_per_class", 10))
        if per_class_limit <= 0:
            per_class_limit = 10

        label_map = self._get_emotion_id2label()
        grouped = defaultdict(list)
        for sample in self._tsne_samples:
            grouped[int(sample["label_id"])].append(sample)

        selected = []
        for label_id in sorted(grouped):
            ranked = sorted(
                grouped[label_id],
                key=lambda sample: (
                    -float(sample["confidence"]),
                    str(sample["doc_id"]),
                    int(sample["utt_index"]),
                ),
            )
            chosen = ranked[:per_class_limit]
            label_name = label_map.get(label_id, str(label_id))
            print(
                f"[tSNE] Class {label_name}({label_id}): "
                f"kept {len(chosen)}/{len(grouped[label_id])} by prediction confidence."
            )
            selected.extend(chosen)
        return selected

    def add_output(self, data, output):
        # Unpack output tuple (repr batches are None during training).
        if len(output) == 9:
            (
                ecp_predictions,
                emo_predictions,
                cause_predictions,
                masks,
                emo_repr_batch,
                recon_repr_batch,
                third_repr_batch,
                fourth_repr_batch,
                repr_space,
            ) = output
            # Legacy layout kept as reference:
            #   third_repr_batch  -> agg_repr_batch
            #   fourth_repr_batch -> cf_agg_repr_batch
            # New src.model_abl layout:
            #   third_repr_batch  -> cf_recon_repr_batch
            #   fourth_repr_batch -> None
            if third_repr_batch is not None and fourth_repr_batch is None:
                cf_repr_batch = third_repr_batch
                cf_repr_name = "cf_recon"
            else:
                cf_repr_batch = fourth_repr_batch
                cf_repr_name = "cf_agg"
        elif len(output) == 8:
            (
                ecp_predictions,
                emo_predictions,
                cause_predictions,
                masks,
                emo_repr_batch,
                recon_repr_batch,
                cf_repr_batch,
                repr_space,
            ) = output
            cf_repr_name = "cf_recon"
        elif len(output) == 7:
            (
                ecp_predictions,
                emo_predictions,
                cause_predictions,
                masks,
                emo_repr_batch,
                recon_repr_batch,
                cf_repr_batch,
            ) = output
            cf_repr_name = "cf_recon"
            repr_space = "h"
        else:
            ecp_predictions, emo_predictions, cause_predictions, masks = output[:4]
            emo_repr_batch, recon_repr_batch, cf_repr_batch, repr_space = (
                None,
                None,
                None,
                None,
            )
            cf_repr_name = None
        # pair 判定采用可配置阈值:
        # 当 z_pos - z_neg > pair_argmax_t 时预测为正类。
        # 其中 t=0.0 与默认 argmax 二分类决策等价（忽略完全相等边界）。
        pair_argmax_t = float(getattr(self.config, "pair_argmax_t", 0.0))
        if ecp_predictions.shape[-1] >= 2:
            pair_margin = (
                ecp_predictions[..., 1] - ecp_predictions[..., 0]
            ).detach().cpu().numpy()
            predictions = (pair_margin > pair_argmax_t).astype(np.int64)
        else:
            # 兼容异常形状，退回到原逻辑
            predictions = ecp_predictions.argmax(-1).cpu().numpy()
        emo_pred = emo_predictions.argmax(-1).cpu().numpy()
        emo_prob = torch.softmax(emo_predictions, dim=-1).detach().cpu().numpy()
        cause_pred = cause_predictions.argmax(-1).cpu().numpy()
        masks = masks.cpu().numpy()

        for i in range(len(emo_pred)):
            mask = masks[i]
            doc_id = data["doc_ids"][i]
            if isinstance(doc_id, np.generic):
                doc_id = doc_id.item()
            utt_nums = int(data["utterance_nums"][i].item())

            # emotion predictions
            emo_pred_ = emo_pred[i, :utt_nums].tolist()
            emo_gold_ = data["labels"][i, :utt_nums].detach().cpu().tolist()
            self.preds["emo"] += emo_pred_
            self.golds["emo"] += emo_gold_

            # cause predictions
            cause_pred_ = cause_pred[i, :utt_nums].tolist()
            cause_gold_ = data["cause_labels"][i, :utt_nums].detach().cpu().tolist()
            self.preds["cause"] += cause_pred_
            self.golds["cause"] += cause_gold_

            # pair predictions
            pair_num = int(data["pair_nums"][i].item())
            prediction = predictions[i] * mask
            pred_pairs = np.where(prediction == 1)
            pred_pairs = [
                (int(w), int(z))
                for w, z in zip(pred_pairs[0], pred_pairs[1])
                if emo_pred_[w] != self.neutral_id or cause_pred_[z] == 1
            ]
            pred_pairs = [(doc_id, w, z) for w, z in pred_pairs if w >= z]
            gold_pairs = [
                (int(w), int(z)) for w, z in data["pairs"][i][:pair_num].detach().cpu().tolist()
            ]

            self.preds["ecp"] += pred_pairs
            self.golds["ecp"] += [(doc_id, w, z) for w, z in gold_pairs]

            # ── Attention heatmap collection (len 6–8) ──
            # ── t-SNE pair collection ──
            # t-SNE triplet collection
            if (
                emo_repr_batch is not None
                and recon_repr_batch is not None
                and cf_repr_batch is not None
            ):
                if self._tsne_repr_space is None and repr_space is not None:
                    self._tsne_repr_space = repr_space
                if self._tsne_cf_label is None and cf_repr_name is not None:
                    self._tsne_cf_label = cf_repr_name
                emo_repr_np = emo_repr_batch[i].numpy()  # [U, D]
                recon_repr_np = recon_repr_batch[i].numpy()  # [U, D]
                cf_repr_np = cf_repr_batch[i].numpy()  # [U, D]
                emo_prob_np = emo_prob[i, :utt_nums]  # [U, C]
                for ui in range(utt_nums):
                    pred_label = int(emo_pred_[ui])
                    if pred_label == int(self.neutral_id):
                        continue
                    self._tsne_samples.append(
                        {
                            "doc_id": doc_id,
                            "utt_index": int(ui),
                            "label_id": pred_label,
                            "confidence": float(emo_prob_np[ui, pred_label]),
                            "emo_repr": emo_repr_np[ui].copy(),
                            "recon_repr": recon_repr_np[ui].copy(),
                            "cf_repr": cf_repr_np[ui].copy(),
                        }
                    )

            if self._collect_dialog_predictions:
                self.dialog_predictions.append(
                    {
                        "doc_id": doc_id,
                        "emotion": {
                            "pred": [int(x) for x in emo_pred_],
                            "true": [int(x) for x in emo_gold_],
                        },
                        "cause": {
                            "pred": [int(x) for x in cause_pred_],
                            "true": [int(x) for x in cause_gold_],
                        },
                        "pair": {
                            "pred": [[w, z] for _, w, z in pred_pairs],
                            "true": [[w, z] for w, z in gold_pairs],
                        },
                    }
                )

    def generate_tsne_plot(self, target_dir, title="Emotion / Recon / Counterfactual Recon t-SNE"):
        """Generate a class-balanced t-SNE plot for non-neutral predicted emotions."""
        if len(self._tsne_samples) < 3:
            print(f"[tSNE] Too few samples ({len(self._tsne_samples)}), skipping.")
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from sklearn.manifold import TSNE
        except ImportError:
            print("[tSNE] matplotlib or sklearn not installed, skipping.")
            return

        # Legacy random subsampling reference:
        # max_points = int(getattr(self.config, "tsne_max_points", 400))
        # if max_points > 0 and len(self._tsne_samples) > max_points:
        #     rng = np.random.default_rng(42)
        #     idx = rng.choice(len(self._tsne_samples), size=max_points, replace=False)
        #     selected = [self._tsne_samples[int(i)] for i in idx]
        # else:
        #     selected = list(self._tsne_samples)
        selected = self._select_tsne_samples()

        if len(selected) < 3:
            print(f"[tSNE] Insufficient samples after filtering ({len(selected)}), skipping.")
            return

        emo_repr_arr = np.stack([p["emo_repr"] for p in selected])
        recon_repr_arr = np.stack([p["recon_repr"] for p in selected])
        cf_repr_arr = np.stack([p["cf_repr"] for p in selected])
        label_ids_arr = np.asarray([int(p["label_id"]) for p in selected], dtype=np.int64)
        repr_space = self._tsne_repr_space or "h"
        cf_name = self._tsne_cf_label or "cf_recon"

        os.makedirs(target_dir, exist_ok=True)
        # Legacy npy export reference:
        # np.save(os.path.join(target_dir, f"{repr_space}_emo_tsne.npy"), emo_repr_arr)
        # np.save(os.path.join(target_dir, f"{repr_space}_recon_tsne.npy"), recon_repr_arr)
        # np.save(os.path.join(target_dir, f"{repr_space}_agg_tsne.npy"), agg_repr_arr)
        # np.save(os.path.join(target_dir, f"{repr_space}_cf_agg_tsne.npy"), cf_agg_repr_arr)
        np.save(os.path.join(target_dir, f"{repr_space}_emo_tsne.npy"), emo_repr_arr)
        np.save(os.path.join(target_dir, f"{repr_space}_recon_tsne.npy"), recon_repr_arr)
        np.save(os.path.join(target_dir, f"{repr_space}_{cf_name}_tsne.npy"), cf_repr_arr)
        print(f"[tSNE] Saved npy files (space={repr_space}, n={len(selected)})")

        combined = np.concatenate([emo_repr_arr, recon_repr_arr, cf_repr_arr], axis=0)
        perplexity = max(1, min(30, len(selected) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        emb = tsne.fit_transform(combined)
        N = len(selected)
        emb_emo = emb[:N]
        emb_recon = emb[N:2 * N]
        emb_cf = emb[2 * N:]

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        label_map = self._get_emotion_id2label()
        unique_labels = sorted(set(label_ids_arr.tolist()))
        cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
        label_to_color = {label_id: cmap(idx) for idx, label_id in enumerate(unique_labels)}

        repr_specs = [
            ("emo", r"$h_{emo}$", emb_emo, "o"),
            ("recon", r"$h_{recon}$", emb_recon, "s"),
            (
                cf_name,
                r"$h_{cf\_recon}$" if cf_name == "cf_recon" else r"$h_{cf\_agg}$",
                emb_cf,
                "^",
            ),
        ]
        # Legacy center plotting reference:
        # centers = [
        #     ("emo center", emb_emo.mean(axis=0), "#2166AC"),
        #     ("recon center", emb_recon.mean(axis=0), "#1A9850"),
        #     ("agg center", emb_agg.mean(axis=0), "#C51B7D"),
        #     ("cf-agg center", emb_cf_agg.mean(axis=0), "#D6604D"),
        # ]
        for label_id in unique_labels:
            label_mask = label_ids_arr == label_id
            color = label_to_color[label_id]
            for _, _, emb_block, marker in repr_specs:
                ax.scatter(
                    emb_block[label_mask, 0],
                    emb_block[label_mask, 1],
                    marker=marker,
                    color=color,
                    s=40,
                    alpha=0.80,
                    edgecolors="black",
                    linewidths=0.25,
                    zorder=2,
                )

        plot_title = title if cf_name == "cf_recon" else "Emotional Utterance Reconstruction t-SNE"
        ax.set_title(plot_title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        repr_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="white",
                markerfacecolor="#808080",
                markeredgecolor="black",
                markersize=7,
                linestyle="None",
                label=label,
            )
            for _, label, _, marker in repr_specs
        ]
        class_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                markerfacecolor=label_to_color[label_id],
                markeredgecolor="black",
                markersize=7,
                linestyle="None",
                label=label_map.get(label_id, str(label_id)),
            )
            for label_id in unique_labels
        ]
        repr_legend = ax.legend(
            handles=repr_handles,
            loc="upper right",
            title="repr",
            fontsize=9,
            framealpha=0.8,
        )
        ax.add_artist(repr_legend)
        ax.legend(
            handles=class_handles,
            loc="lower right",
            title="pred emotion",
            fontsize=9,
            framealpha=0.8,
        )
        plt.tight_layout()

        save_stem = (
            "tsne_emotional_recon"
            if cf_name == "cf_agg"
            else f"tsne_emo_recon_{cf_name}"
        )
        # Legacy save path reference:
        # out_path = os.path.join(target_dir, f"tsne_emotional_recon{ext}")
        for ext, dpi in [(".png", 300), (".pdf", 300)]:
            out_path = os.path.join(target_dir, f"{save_stem}{ext}")
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"[tSNE] Saved {out_path}")
        plt.close()

    def report_score(self):
        # pair metrics
        pred_full = set(self.preds["ecp"])
        gold_full = set(self.golds["ecp"])
        # Front: 只统计前因与自因（w >= z）。
        pred_front = {item for item in pred_full if item[1] >= item[2]}
        gold_front = {item for item in gold_full if item[1] >= item[2]}

        def calc_prf(pred_set, gold_set):
            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            p = tp / (tp + fp) if tp + fp > 0 else 0
            r = tp / (tp + fn) if tp + fn > 0 else 0
            f = 2 * p * r / (p + r) if p + r > 0 else 0
            return p, r, f, tp, tp + fp, tp + fn

        full_p, full_r, full_f, full_tp, full_pred, full_gold = calc_prf(
            pred_full, gold_full
        )
        front_p, front_r, front_f, front_tp, front_pred, front_gold = calc_prf(
            pred_front, gold_front
        )

        # emotion binary metrics
        gold_emo = [0 if w == self.neutral_id else 1 for w in self.golds["emo"]]
        pred_emo = [0 if w == self.neutral_id else 1 for w in self.preds["emo"]]
        emo = precision_recall_fscore_support(
            gold_emo, pred_emo, average="binary", zero_division=0
        )
        cause = precision_recall_fscore_support(
            self.golds["cause"], self.preds["cause"], average="binary", zero_division=0
        )

        # multi-class emotion report
        explicit_id2label = self.config.get("emotion_id2label", None)
        label_dict = self.config.get("label_dict", None)
        labels = None
        id2label = None
        if explicit_id2label:
            id2label = {int(k): str(v) for k, v in explicit_id2label.items()}
            labels = sorted(id2label.keys())
            target_names = [id2label[i] for i in labels]
            emo_report = classification_report(
                self.golds["emo"],
                self.preds["emo"],
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
        elif label_dict:
            id2label = {v: k for k, v in label_dict.items()}
            labels = sorted(id2label.keys())
            target_names = [id2label[i] for i in labels]
            emo_report = classification_report(
                self.golds["emo"],
                self.preds["emo"],
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
        else:
            labels = sorted(set(self.golds["emo"]) | set(self.preds["emo"]))
            emo_report = classification_report(
                self.golds["emo"],
                self.preds["emo"],
                output_dict=True,
                zero_division=0,
            )

        emo_multi_macro = None
        emo_multi_weighted = None
        if isinstance(emo_report, dict):
            macro = emo_report.get("macro avg")
            if isinstance(macro, dict):
                emo_multi_macro = {
                    "p": float(macro.get("precision", 0.0)),
                    "r": float(macro.get("recall", 0.0)),
                    "f1": float(macro.get("f1-score", 0.0)),
                }
            weighted = emo_report.get("weighted avg")
            if isinstance(weighted, dict):
                emo_multi_weighted = {
                    "p": float(weighted.get("precision", 0.0)),
                    "r": float(weighted.get("recall", 0.0)),
                    "f1": float(weighted.get("f1-score", 0.0)),
                }

        emo_class_count = len(labels) if labels is not None else 0

        res_lines = [
            f"Emo-2: Pre. {emo[0]*100:.4f}\t Rec. {emo[1]*100:.4f}\tF1 {emo[2]*100:.4f}",
        ]
        if emo_multi_macro and emo_class_count:
            res_lines.append(
                f"Emo-{emo_class_count}: Pre. {emo_multi_macro['p']*100:.4f}\t Rec. {emo_multi_macro['r']*100:.4f}\tF1 {emo_multi_macro['f1']*100:.4f}"
            )
        res_lines.extend(
            [
                f"Cause: Pre. {cause[0]*100:.4f}\t Rec. {cause[1]*100:.4f}\tF1 {cause[2]*100:.4f}",
                f"FullPair Pre. {full_p*100:.4f}\t Rec. {full_r*100:.4f}\tF1 {full_f*100:.4f}",
                f"FullTP {full_tp}\tPred. {full_pred}\tGold. {full_gold}",
                f"FrontPair Pre. {front_p*100:.4f}\t Rec. {front_r*100:.4f}\tF1 {front_f*100:.4f}",
                f"FrontTP {front_tp}\tPred. {front_pred}\tGold. {front_gold}",
            ]
        )
        res = "\n".join(res_lines) + "\n"

        metrics = {
            # 保留 pair 作为 legacy 入口（默认对应 FullPair），避免现有训练/搜参脚本断裂。
            "pair": {
                "p": full_p,
                "r": full_r,
                "f1": full_f,
                "tp": full_tp,
                "pred": full_pred,
                "gold": full_gold,
            },
            "pair_full": {
                "p": full_p,
                "r": full_r,
                "f1": full_f,
                "tp": full_tp,
                "pred": full_pred,
                "gold": full_gold,
            },
            "pair_front": {
                "p": front_p,
                "r": front_r,
                "f1": front_f,
                "tp": front_tp,
                "pred": front_pred,
                "gold": front_gold,
            },
            "emo_binary": {"p": emo[0], "r": emo[1], "f1": emo[2]},
            "emo_multi": {"labels": id2label, "report": emo_report},
            "emo_multi_macro": {
                "p": None if emo_multi_macro is None else emo_multi_macro["p"],
                "r": None if emo_multi_macro is None else emo_multi_macro["r"],
                "f1": None if emo_multi_macro is None else emo_multi_macro["f1"],
                "n_classes": emo_class_count,
            },
            "emo_multi_weighted": {
                "p": None if emo_multi_weighted is None else emo_multi_weighted["p"],
                "r": None if emo_multi_weighted is None else emo_multi_weighted["r"],
                "f1": None if emo_multi_weighted is None else emo_multi_weighted["f1"],
                "n_classes": emo_class_count,
            },
            "cause": {"p": cause[0], "r": cause[1], "f1": cause[2]},
        }

        return full_f, (res, metrics)

    @staticmethod
    def _to_builtin(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: MyTrainer._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MyTrainer._to_builtin(v) for v in obj]
        return obj

    def save_preds_file(self):
        if not self.config.target_dir:
            return
        os.makedirs(self.config.target_dir, exist_ok=True)
        out_path = os.path.join(self.config.target_dir, "preds.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._to_builtin(self.dialog_predictions), f, ensure_ascii=False, indent=2)

    def save_run_report(self, best_epoch, dev_metrics, test_metrics):
        if not getattr(self.config, "save_run_report", True):
            return
        if not self.config.target_dir:
            return
        os.makedirs(self.config.target_dir, exist_ok=True)

        def config_to_dict(cfg):
            skip_keys = {"optimizer", "scheduler", "device", "label_dict", "speaker_dict"}
            out = {}
            for k, v in cfg.items():
                if k in skip_keys:
                    continue
                out[k] = self._to_builtin(v) if isinstance(
                    v, (dict, list, tuple, np.generic)
                ) else (
                    v
                    if isinstance(v, (str, int, float, bool)) or v is None
                    else str(v)
                )
            return out

        payload = {
            "best_epoch": int(best_epoch),
            "dev": dev_metrics,
            "test": test_metrics,
        }
        out_path = os.path.join(self.config.target_dir, "metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._to_builtin(payload), f, ensure_ascii=False, indent=2)

        test_f1 = 0.0
        if isinstance(test_metrics, dict):
            test_f1 = float(test_metrics.get("pair", {}).get("f1", 0.0))
        cfg_path = os.path.join(
            self.config.target_dir, f"config_testPairF1={test_f1:.4f}.yaml"
        )
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_to_dict(self.config), f, sort_keys=False, allow_unicode=False)

        # rename run directory to include test F1 suffix
        suffix = f"_testPairF1={test_f1:.4f}"
        cur_dir = os.path.abspath(self.config.target_dir)
        if not cur_dir.endswith(suffix):
            new_dir = cur_dir + suffix
            if not os.path.exists(new_dir):
                os.rename(cur_dir, new_dir)
                self.config.target_dir = new_dir

    def save_loss_plot(self):
        if not self.config.target_dir or not self.epoch_losses:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            if not self._plot_warned:
                print("[Warn] matplotlib not installed; skip loss plot.")
                self._plot_warned = True
            return

        n_epochs   = len(self.epoch_losses)
        step_losses = getattr(self, "train_step_losses", [])

        dev_losses  = [e["dev_loss"]  for e in self.epoch_losses]
        test_losses = [e["test_loss"] for e in self.epoch_losses]

        if step_losses and n_epochs > 0:
            # per-step x-axis for train
            total_steps      = len(step_losses)
            steps_per_epoch  = total_steps / n_epochs
            x_train = list(range(1, total_steps + 1))
            # epoch-end step positions for dev/test
            x_eval  = [int(round((e["epoch"] + 1) * steps_per_epoch)) for e in self.epoch_losses]
        else:
            # fallback: epoch-level
            epochs   = [e["epoch"] + 1 for e in self.epoch_losses]
            x_train  = epochs
            x_eval   = epochs
            step_losses = [e["train_loss"] for e in self.epoch_losses]

        # EMA smoothing for per-step train loss
        def _ema(vals, alpha=0.05):
            out, s = [], None
            for v in vals:
                s = v if s is None else alpha * v + (1 - alpha) * s
                out.append(s)
            return out

        train_smooth = _ema(step_losses)

        plt.figure()
        plt.plot(x_train, step_losses,  color="steelblue", linewidth=0.6, alpha=0.25)                         # raw
        plt.plot(x_train, train_smooth, color="steelblue", linewidth=1.8, label="train")                      # smoothed
        plt.plot(x_eval, dev_losses,    color="darkorange", linewidth=1.5, marker="o", markersize=4, label="dev")
        if any(v is not None for v in test_losses):
            plt.plot(x_eval, test_losses, color="green", linewidth=1.5, marker="o", markersize=4, label="test")
        plt.xlabel("Step (forward mini-batch)")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs(self.config.target_dir, exist_ok=True)
        out_path = os.path.join(self.config.target_dir, "loss_curve.png")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
