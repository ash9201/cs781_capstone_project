## Phase 3. Rhetorical role prediction

I adapted the four Phase 1 language models to assign one rhetorical role to each sentence in a legal judgment. I trained on the official CL and IT data together and selected checkpoints by combined development macro F1. The evaluator averages across all 13 roles, including two absent from the combined development split. This gives that split a perfect-prediction ceiling of 11/13, or 0.8462.

### Data and document context

The official splits contain 80 training, 10 development and 10 test judgments. I reused the Phase 1 byte BPE tokenizer. I split long sentences into chunks of at most 256 tokens, including boundary tokens, and combined their representations by content length. This preserves the original sentences and their order.

A bidirectional GRU adds sentence context within windows of 128 sentences with 32-sentence overlap. Each sentence contributes to the loss and final predictions through one assigned window. At inference I reassembled the sentence scores and ran one linear-chain CRF decode per document.

### Models and training

| Model | Total parameters | Trainable parameters | Adaptation |
| --- | ---: | ---: | --- |
| [Decoder](phase3_decoder_ia3.ipynb) | 218,520,784 | 660,688 | IA³ scales on attention and feed-forward activations, with last-content-token pooling |
| [Encoder-decoder](phase3_encoder_decoder_lora.ipynb) | 228,248,784 | 10,392,784 | Rank-32 LoRA on attention and feed-forward projections, with mean pooling of decoder content states |
| [Prefix LM](phase3_prefix_lorma.ipynb) | 227,861,712 | 9,999,568 | Rank-32 LoRMA+ in the shared blocks, with mean pooling of continuation content states |
| [Efficient teacher](phase3_efficient_teacher.ipynb) | 218,697,936 | 218,697,936 | Full fine-tuning with mean and last-content-token pooling plus sentence features |
| [Gold-only and distilled students](phase3_efficient_students.ipynb) | 54,990,288 each | 54,990,288 each | Full fine-tuning of the same three-block initialization, with or without soft teacher targets |

I trained with PyTorch and AdamW on two Kaggle T4 GPUs. The gold objective combines CRF negative log-likelihood with weighted sentence cross-entropy. I computed class weights from the training split. For the decoder continuation, I started from the selected IA³ checkpoint and used a fresh optimizer and lower learning rates.

I initialized both compact students from efficient Phase 1 blocks 1, 8 and 16 with identical task weights. For distillation, I cached the selected teacher's full-document CRF marginal probabilities on the training sentences at temperature 2. I added a KL loss between those targets and the student's softened sentence scores, scaled by 0.4 times the temperature squared.

### Results

| Run | Selected epoch / epochs completed | Dev macro F1 | CodaBench macro F1 | Submission ID | Run cell time |
| --- | ---: | ---: | ---: | ---: | --- |
| Decoder IA³ continuation | 8 / 14 | 0.4250 | 0.4151 | 990177 | 1 h 24 min |
| Encoder-decoder LoRA | 21 / 29 | 0.4314 | 0.4015 | 990705 | 3 h 39 min |
| Prefix LoRMA+ | 16 / 21 | 0.4297 | 0.4143 | 991648 | 5 h 15 min |
| Efficient teacher | 13 / 18 | 0.4436 | **0.4408** | 992545 | 2 h 11 min |
| Compact gold-only student | 10 / 15 | 0.3928 | 0.4289 | 992935 | 1 h 08 min for both students |
| Compact distilled student | 14 / 19 | 0.4067 | 0.4354 | 992916 | Shared cell |
| [Heavier regularization](phase3_efficient_heavy_regularization.ipynb) | 16 / 18 | **0.4510** | 0.4249 | 994303 | 2 h 06 min |

Times come from the saved Kaggle cell timestamps. They include validation and any export or backup work in the same cell. The initial decoder stage took 1 h 57 min and selected epoch 15 at 0.3999 dev macro F1 before the continuation.

Distillation improved dev F1 by 0.0139 and CodaBench F1 by 0.0065 over the gold-only student. The distilled model used about 25% of the teacher's parameters and predicted the full test split in 3.68 seconds versus 12.77 seconds for the teacher on the recorded passes.

I also increased the efficient teacher's backbone dropout to 0.20, head dropout to 0.40 and AdamW weight decay to 0.05 in one run. That run improved dev F1 to 0.4510 and scored 0.4249 on CodaBench. It exported its selected predictions before the final train-set evaluation was interrupted.

### Voting ensembles

I combined the saved model predictions sentence by sentence. For the three-model votes, a designated model broke ties. In the conservative four-model vote, I kept the efficient teacher's label unless the other three agreed on a different label.

| Voting rule | Dev macro F1 | CodaBench macro F1 | Submission ID |
| --- | ---: | ---: | ---: |
| Three models, encoder-decoder tie-break | 0.4485 | 0.4144 | 991733 |
| Three models, decoder tie-break | 0.4639 | 0.4247 | 991746 |
| Three models, prefix tie-break | 0.4479 | 0.4260 | 991753 |
| Conservative four-model vote | 0.4599 | **0.4444** | 992618 |

The conservative vote gave the best confirmed CodaBench result, 0.0036 above the teacher alone.

### Reproducing the runs

1. **Set up the inputs.** Use Kaggle with two T4 GPUs, the `prep_002_mixed16k` tokenizer preparation and the matching Phase 1 checkpoint and source files. Attach the official RR files or allow the pinned download using the `HF_TOKEN` secret.
2. **Run the model notebook.** For the encoder-decoder and prefix runs, first run the [LoRA preflight](phase3_encoder_decoder_lora_preflight.ipynb) or [LoRMA+ preflight](phase3_prefix_lorma_preflight.ipynb), or attach its passing diagnostics. The teacher notebook runs its own preflight and exports `efficient_rr_teacher_001_teacher_cache.zip` for the student notebook.
3. **Evaluate and export.** Each run selects its best development checkpoint and writes a submission ZIP with one role per test sentence. The [ensemble script](rebuild_ensemble_submissions.py) rebuilds the conservative submission ZIP from the four original model diagnostics archives.

### References

- Context-aware legal sentence classification. [Semantic Segmentation of Legal Documents via Rhetorical Roles](https://aclanthology.org/2022.nllp-1.13/)
- Parameter-efficient adaptation. [IA³](https://arxiv.org/abs/2205.05638), [LoRA](https://arxiv.org/abs/2106.09685) and [LoRMA](https://aclanthology.org/2025.findings-acl.527/)
- Learned sequence transitions and Viterbi decoding. [Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030/)
- Temperature-scaled teacher targets. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
