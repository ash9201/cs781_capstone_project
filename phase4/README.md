## Phase 4. Court judgment prediction

I adapted the four Phase 1 language models to predict one binary outcome per court case. I trained on the official CJPE split, compared the four architecture families, and evaluated knowledge distillation with a compact efficient student. The primary metric is macro F1 over the two outcome classes.

### Data and document context

I reused the Phase 1 16K byte BPE tokenizer. The official data contain 37,387 training rows, 1,988 dev rows and 1,517 test cases. I assigned every supplied row an internal ID and kept the original case IDs for submission. The final training runs counted rows with the same case ID, text and label once, leaving 32,305 training examples. Each of the 994 distinct dev cases appears twice. I ran inference once per case and expanded the results to all supplied rows for scoring.

The final classifiers read up to four 512-token windows from the end of each document, with 64 content tokens of overlap. For each window, I combined the mean content state with a backbone-specific readout and the window position. A projection, bidirectional GRU and attention pooling produce a single representation for the two-class prediction.

### Models and training

| Model | Total parameters | Trainable parameters | Adaptation |
| --- | ---: | ---: | --- |
| [Decoder](phase4_decoder_partial.ipynb) | 218,698,626 | 26,024,834 | Partial fine-tuning of the top two blocks, final layer norm and document head |
| [Encoder-decoder](phase4_encoder_decoder_adapters.ipynb) | 223,187,970 | 5,405,698 | Width-64 residual bottleneck adapters, trainable layer norms and a BOS readout through cross-attention |
| [Prefix LM](phase4_prefix_lorma.ipynb) | 221,059,970 | 3,197,826 | Rank-8 LoRMA+ on attention and feed-forward projections, with a BOS readout in the continuation stream |
| [Efficient gold](phase4_efficient_gold.ipynb) and [efficient KD](phase4_efficient_kd.ipynb) | 55,003,266 each | 55,003,266 each | Full fine-tuning of three retained blocks from the Phase 1 efficient model pretrained through 500M tokens |

I trained with PyTorch and AdamW on two Kaggle T4 GPUs, using mixed precision, gradient checkpointing and a target of 64 cases per optimizer update. I used class-weighted cross-entropy, a 10% warmup and cosine learning-rate decay. For the final runs I evaluated an exponential moving average of the trainable weights with decay 0.98. I selected checkpoints by dev ROC-AUC, then tuned the decision threshold for dev macro F1 on a grid from 0.20 to 0.80 in steps of 0.01. Each selected threshold stayed fixed for test.

I initialized both efficient students from blocks 1, 8 and 16 of the same backbone with identical document heads. For KD, I cached logits on the training split from the [initial prefix run](phase4_prefix_lorma_teacher.ipynb) at update 275. I combined 70% class-weighted cross-entropy with 30% teacher-to-student KL divergence at temperature 2, with the KL term scaled by the temperature squared.

### Results

| Run | Selected update | Threshold | Dev macro F1 | CodaBench macro F1 | Submission ID | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Decoder partial 002 | 1,000 | 0.50 | 0.5562 | 0.5469 | 1009404 | Unrecorded |
| Decoder partial 003 | 3,375 | 0.40 | 0.5943 | 0.5767 | 1009374 | 3.99 h |
| Encoder-decoder adapters | 250 | 0.48 | 0.5643 | 0.5232 | 1009382 | 2.51 h |
| Prefix LoRMA+ initial | 275 | 0.47 | 0.5836 | 0.5396 | 1009386 | 2.53 h |
| Prefix LoRMA+ continuation | 1,115 | 0.52 | 0.6005 | 0.5928 | 1009379 | 8.03 h |
| Efficient gold | 2,500 | 0.39 | **0.6527** | **0.6315** | 1009394 | 2.13 h |
| Efficient KD | 2,500 | 0.40 | 0.6514 | 0.6185 | 1009373 | 2.02 h |

Each time covers one training invocation, including in-loop validation and checkpointing. For continued runs, the table reports the final invocation. Decoder 002 used eight uniformly spaced windows and stronger head dropout. I started decoder 003 afresh from Phase 1 with the revised training protocol above.

The gold student gave the best single-model result. KD came close on dev and scored 0.0130 lower on CodaBench. Both students reached the same number of updates, and I extended their learning-rate schedules at different points in training.

### Probability ensemble

I tested 21 blends of the saved model probabilities. The selected rule weights efficient gold at 0.75 and the continued prefix model at 0.25, and predicts class 1 when the blended probability reaches 0.40.

| Ensemble | Dev macro F1 | CodaBench macro F1 | Submission ID |
| --- | ---: | ---: | ---: |
| Efficient gold and continued prefix | 0.6640 | **0.6371** | 1009629 |

The blend improved the CodaBench score by 0.0056 over efficient gold. In repeated stratified dev folds grouped by original case ID, the gold and prefix family averaged 0.6514 out-of-fold macro F1 versus 0.6413 for gold alone. The source checkpoints had already been selected on the full dev set, so the folds assess the blend and threshold fitting after checkpoint selection.

### Reproducing the runs

1. **Prepare the data.** Use Kaggle with two T4 GPUs. Attach `prep_002_mixed16k` and the original selected checkpoints for the four Phase 1 backbones. Run the [preparation notebook](phase4_cjpe_preparation.ipynb) with the official CJPE files or grant it access to the `HF_TOKEN` secret for the pinned download.
2. **Train or resume.** Attach `phase4_cjpe_prepared_data.zip` or its packed `phase4_cjpe_fast_cache.zip` equivalent and the matching Phase 1 checkpoint. Run the chosen notebook. To continue a run in a fresh session, set `RESUME_INPUT` to its matching backup. For KD, use the update-275 prefix export to generate the teacher cache and set `TEACHER_CACHE_INPUT` in the student notebook.
3. **Export predictions.** Each run reloads its selected checkpoint and writes a ZIP containing `prediction.json` with one integer label per test case. The [ensemble workflow](ensemble/) rebuilds the blend and its submission ZIP from the five selected runs' diagnostic archives.

### References

- Long-document court judgment prediction. [ILDC for CJPE](https://aclanthology.org/2021.acl-long.313/)
- Bottleneck adapters. [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html)
- Multiplicative low-rank adaptation. [LoRMA](https://aclanthology.org/2025.findings-acl.527/)
- Temperature-scaled teacher targets. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
