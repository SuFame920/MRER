# MRER: Multi-Relation Multimodal Interaction and Emotion Reconstruction for Emotion-Cause Pair Extraction in Conversations

This repository contains the official implementation of the paper: **MRER: Multi-Relation Multimodal Interaction and Emotion Reconstruction for Emotion-Cause Pair Extraction in Conversations**.

## Motivation

Multimodal conversational emotion-cause pair extraction (MECPE) aims to jointly identify emotion utterances and their corresponding cause utterances from multimodal dialogues. Current methods face two critical challenges:

1. **Relational Context Independence**: Multimodal fusion is often performed independently of the dialogue's relational structure, preventing models from adaptively capturing the differentiated contributions of modalities across distinct relational perspectives.
2. **Weak Interpretive Modeling**: The inherent interpretive dependency between emotions and causes lacks explicit modeling at the representation level, which limits the synergistic optimization between emotion and cause extraction subtasks.

## Methods

We propose the **MRER** framework to address these limitations through two core components:

- **Multi-Relational Multimodal Interaction Encoder**: This module encodes multimodal features under three distinct relational perspectives—**Global Temporal Order**, **Intra-speaker Consistency**, and **Local Proximity**. It enables the framework to adaptively adjust modality weights based on the specific relational context.
- **Cause-Guided Emotion Reconstruction**: This mechanism reconstructs emotion representations from paired cause representations under semantic **hard negative sample constraints**. Combined with **supervised contrastive alignment**, it explicitly encodes the causal explanatory relationship into the joint representation space.

## Results

Extensive experiments on three large-scale benchmark datasets demonstrate that **MRER** consistently achieves state-of-the-art (SOTA) performance:

- **ECF**: Significantly outperforms existing multimodal MECPE baselines.
- **ConvECPE**: Exhibits robust performance in long-dialogue scenarios.
- **MECAD**: Validates the framework's effectiveness across diverse multimodal conversation types.

Ablation studies further confirm that all relational perspectives and the emotion reconstruction module contribute significantly to the overall gain.

## Conclusion

MRER advances the MECPE task by bridging the gap between relational context and multimodal fusion, while simultaneously strengthening the semantic coupling between emotion and cause subtasks. Our findings highlight that adaptively capturing modality contributions within specific relational views and encoding causal interpretability into representation learning are crucial for complex conversational reasoning.

---

*For more details, please refer to our paper.*

**Note**: The full code and data will be released upon the acceptance of our paper. Thank you for your understanding!
