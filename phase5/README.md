## Phase 5. Legal statute identification

I adapted the four Phase 1 language models to identify which statutes apply to a legal case. Each case can have several applicable statutes, so the classifier produces one independent score for each of the 100 labels. The primary metric is macro F1 across statutes. I also recorded micro F1 and exact label-set accuracy on dev.

### Data and document context

I reused the Phase 1 16K byte BPE tokenizer and packed the tokenized cases for reuse across runs. After removing exact training duplicates by case ID, text and label set, I used 42,750 training cases, 10,181 dev cases and 13,019 test cases. The preparation records the label order used for training and maps predictions back to the original statute IDs at export.

I built 512-token candidate windows with 64 content tokens of overlap and selected up to four windows spaced across the document. Each window combines mean content states, a backbone-specific readout and position features. A projection and bidirectional GRU contextualize the window vectors. Each statute then attends separately to the windows and produces its own logit, so different statutes can draw on different parts of the same case.

### Models and training

| Model | Total parameters | Trainable parameters | Adaptation |
| --- | ---: | ---: | --- |
| [Decoder](phase5_decoder_partial.ipynb) | 218,730,148 | 26,056,356 | Fine-tuned the top two blocks, final layer norm and document head |
| [Encoder-decoder](phase5_encoder_decoder_adapters.ipynb) | 223,219,492 | 5,437,220 | Width-64 residual bottleneck adapters, trainable layer norms and document head |
| [Prefix LM](phase5_prefix_lorma.ipynb) | 221,091,492 | 3,229,348 | Rank-8 LoRMA+ on attention and feed-forward projections |
| [Efficient gold](phase5_efficient_gold.ipynb) and [efficient KD](phase5_efficient_kd.ipynb) | 55,034,788 each | 55,034,788 each | Full fine-tuning of the compact backbone built from Phase 1 efficient blocks 1, 8 and 16 |

I trained on two Kaggle T4 GPUs with AdamW, mixed precision, gradient checkpointing and 64 cases per optimizer update. I used binary cross-entropy with positive weights derived from training label frequencies. In the [batching benchmark](phase5_batching_benchmark.ipynb), grouping valid windows across cases reduced the prefix update time from 26.31 s to 7.65 s and the encoder-decoder update time from 10.49 s to 3.47 s. I used the batched implementation for the final runs.

I evaluated an exponential moving average of the trainable weights with decay 0.98 and selected checkpoints by dev macro F1 using a shared threshold. After selection, I fitted statute-specific thresholds on dev, shrinking them toward the shared value and keeping the shared threshold for labels with limited support. I fixed those thresholds for test.

Both efficient students used the same Phase 1 initialization and document head. For distillation, I cached training logits from the [prefix teacher](phase5_prefix_teacher.ipynb) selected at update 2,000. I combined 70% weighted binary cross-entropy with 30% Bernoulli KL divergence between the teacher and student label distributions at temperature 2. The KL term includes the usual temperature-squared scaling.

### Results

| Run | Selected update | Dev macro F1 | Dev micro F1 | CodaBench macro F1 | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decoder partial | 5,749 | 0.2724 | 0.3856 | 0.2707 | 5 h 00 min |
| Encoder-decoder adapters | 4,626 | 0.1345 | 0.2363 | 0.1297 | 5 h 00 min |
| Prefix LoRMA+ | 2,000 | 0.2475 | 0.3727 | 0.2431 | 4 h 55 min |
| Efficient gold | 7,800 | 0.3207 | 0.4270 | 0.3128 | 4 h 22 min |
| Efficient KD | 9,961 | **0.3315** | **0.4420** | **0.3221** | 5 h 00 min |

Dev scores use the fitted per-label thresholds. Training times cover the training loop and are rounded to the nearest minute. The KD time is cumulative across its sessions. The gold run early-stopped at update 9,600 and exported its selected checkpoint at update 7,800.

Efficient KD scored 0.0093 above efficient gold on CodaBench. The compact models could update all their parameters and process more training passes within the budget. The selected KD checkpoint had seen about 15 passes, compared with about three for prefix. The comparison reflects differences in adaptation strategy and training length as well as model size and loss.

### Ensemble exploration

I tested probability blends offline. The best fixed blend used 75% efficient KD and 25% efficient gold and raised mean out-of-fold dev macro F1 from 0.3270 to 0.3277. Gold and KD scores already overlapped strongly, with a label-centered dev correlation of about 0.952. Adding the weaker prefix model to the blend contributed little. Given the small gain, I kept efficient KD as the final result.

### Reproducing the runs

1. **Prepare the data.** On Kaggle, attach the [preparation source package](CS781_Phase5_Preparation_Package.zip), the Phase 1 tokenizer and the original checkpoints. Run the [preparation notebook](phase5_preparation.ipynb) with the official LSI files or enable the `HF_TOKEN` secret for the pinned download.
2. **Train or resume.** Attach `phase5_lsi_prepared_data.zip` and the matching Phase 1 checkpoint, then run the model notebook linked above. Set `RESUME_INPUT` to the task backup when continuing in a fresh session. For KD, run the [teacher export](phase5_prefix_teacher.ipynb) with the selected prefix backup and set `TEACHER_CACHE_INPUT` to the exported cache in the KD notebook.
3. **Export predictions.** Each model writes a submission ZIP containing `prediction.json`, with one record per test case and a list of integer statute IDs in `pred`.

### References

- Label-wise attention for legal text classification. [Large-Scale Multi-Label Text Classification on EU Legislation](https://aclanthology.org/P19-1636/)
- Bottleneck adapters. [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html)
- Multiplicative low-rank adaptation. [LoRMA](https://aclanthology.org/2025.findings-acl.527/)
- Temperature-scaled teacher targets. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
