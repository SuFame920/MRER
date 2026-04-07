# MECAD 音视频特征提取 / MECAD Audio-Video Feature Extraction

## 中文说明

### 脚本
当前提取脚本：`data/dataset/mecad/extract.py`

### 快速开始
1. 修改配置文件 `data/dataset/mecad/mecad_feature_extract.yaml`：
- `dataset.annotation_dir`
- `dataset.video_root_dir`
- `runtime.scope`（`demo` / `all`）
2. 运行：

```bash
py -3.12 data/dataset/mecad/extract.py --config data/dataset/mecad/mecad_feature_extract.yaml
```

### DenseNet 一行切换
在 `data/dataset/mecad/mecad_feature_extract.yaml` 中修改：

```yaml
video:
  model_name: "densenet121"
```

可切换为：
- `densenet169`
- `densenet201`

### 池化方式（无训练参数）
在配置中设置：
- `audio.pooling`
- `video.pooling`

支持：
- `mean`
- `min_mean_max`（默认，拼接 `min + mean + max`）

### 提取范围参数（demo / all）
在 `data/dataset/mecad/mecad_feature_extract.yaml` 中：
- `runtime.scope: demo`  
  从 `videos` 目录递归随机抽 `runtime.demo_num_videos` 个 mp4（默认 10）用于快速验证。
- `runtime.scope: all`  
  递归提取 `videos` 目录下全部 mp4。

### 输出文件
输出目录为 `output.output_dir`（当前配置是 `.`，即本文件所在目录），包含：
- `audio_embedding_<dim>.npy`（默认开启维度后缀，例如 `audio_embedding_3072.npy`）
- `video_embedding_<dim>.npy`（默认开启维度后缀，例如 `video_embedding_3072.npy`）
- `video_id_mapping.npy`（类型：`dict[str, int]`）
- `utterance_manifest.jsonl`（逐条 utterance 清单）
- `feature_meta.json`（维度、缺失视频统计、运行摘要）
配置项：`output.embed_dim_in_filename: true/false`

### 与论文逻辑对齐说明
- 音频流程：`16kHz waveform -> Wav2Vec2 -> utterance embedding`
- 视频流程：`uniform frame sampling -> DenseNet -> utterance embedding`
- 论文没有在发布代码中写死 DenseNet 深度版本，因此这里提供 `121/169/201` 可切换配置。
- MECAD 音频模型建议：`wbbbbb/wav2vec2-large-chinese-zh-cn`

## English

### Script
Current extraction script: `data/dataset/mecad/extract.py`

### Quick Start
1. Edit `data/dataset/mecad/mecad_feature_extract.yaml`:
- `dataset.annotation_dir`
- `dataset.video_root_dir`
- `runtime.scope` (`demo` / `all`)
2. Run:

```bash
py -3.12 data/dataset/mecad/extract.py --config data/dataset/mecad/mecad_feature_extract.yaml
```

### One-Line DenseNet Switch
In `data/dataset/mecad/mecad_feature_extract.yaml`, change:

```yaml
video:
  model_name: "densenet121"
```

to:
- `densenet169`
- `densenet201`

### Pooling Mode (No Training)
Set in config:
- `audio.pooling`
- `video.pooling`

Supported:
- `mean`
- `min_mean_max` (default, concatenates `min + mean + max`)

### Extraction Scope (demo / all)
In `data/dataset/mecad/mecad_feature_extract.yaml`:
- `runtime.scope: demo`  
  Randomly samples `runtime.demo_num_videos` mp4 files (default: 10) from `videos` recursively for quick checks.
- `runtime.scope: all`  
  Extracts features from all mp4 files under `videos` recursively.

### Outputs
Outputs are written to `output.output_dir` (currently `.`, i.e., this same folder):
- `audio_embedding_<dim>.npy` (dim suffix enabled by default, e.g. `audio_embedding_3072.npy`)
- `video_embedding_<dim>.npy` (dim suffix enabled by default, e.g. `video_embedding_3072.npy`)
- `video_id_mapping.npy` (`dict[str, int]`)
- `utterance_manifest.jsonl` (one utterance per line)
- `feature_meta.json` (dims, missing-video stats, run summary)
Config key: `output.embed_dim_in_filename: true/false`

### Paper-Alignment Notes
- Audio path: `16kHz waveform -> Wav2Vec2 -> utterance embedding`
- Video path: `uniform frame sampling -> DenseNet -> utterance embedding`
- The released code path does not lock DenseNet depth in a strict way, so this extractor keeps `121/169/201` configurable.
- Recommended MECAD audio model id: `wbbbbb/wav2vec2-large-chinese-zh-cn`

## Resume / Checkpoint (EN + 中文)

### 中文
- 现在支持断点续提：提取过程中会持续落盘，不再等到最后一次性保存。
- 关键配置：
  - `runtime.resume: true`
  - `runtime.save_every: 1`（每处理 N 条就写一次进度）
  - `runtime.batch_size / runtime.audio_batch_size / runtime.video_batch_size`（批处理大小）
  - `output.done_file`（完成掩码）
  - `output.state_file`（状态文件）
- 运行中断后，直接执行同一条命令即可自动续提：

```bash
py -3.12 data/dataset/mecad/extract.py --config data/dataset/mecad/mecad_feature_extract.yaml
```

- 自动完整性检测：
  - 若 `done_file` 显示已全部完成，会自动识别为 complete 并仅刷新元信息。
  - 若只完成一部分，会从未完成位置继续提取。

### English
- Checkpoint resume is enabled: features are flushed continuously during extraction.
- Key configs:
  - `runtime.resume: true`
  - `runtime.save_every: 1` (flush every N utterances)
  - `runtime.batch_size / runtime.audio_batch_size / runtime.video_batch_size` (inference batch sizes)
  - `output.done_file` (completion mask)
  - `output.state_file` (resume state)
- After interruption, rerun the same command to continue from unfinished indices.
- Automatic completion detection:
  - If all entries are done, the run is detected as complete.
  - If partially done, extraction resumes from the remaining indices.
