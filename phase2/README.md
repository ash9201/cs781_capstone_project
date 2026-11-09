## Phase 2. Legal named entity recognition

I adapted the four Phase 1 language models to identify legal entities in judgments. A prediction counts as correct only when its start offset, exclusive end offset and entity label all match the reference. I used the course evaluator's strict macro F1 across 12 entity types to select checkpoints.

### Data and span extraction

I used the official `Phase_2_LNER` data with a fixed split of 56 training and 14 validation documents. I reused the Phase 1 byte BPE tokenizer and kept a mapping from its tokens to character offsets in the original text. I split documents into 1,024-token windows with 256-token overlap and assigned each token to one window for supervision and prediction.

Each model predicts BIO tags and uses separate heads to refine start and end offsets within tokens. I reassembled the window outputs before decoding a single tag sequence for each document. The learned CRF runs also trimmed whitespace at span boundaries and adjusted the offsets against the original text.

### Models and training

| Model | Total parameters | Trainable parameters | Adaptation |
| --- | ---: | ---: | --- |
| [Decoder](phase2_decoder_full_finetuning.ipynb) | 219,400,283 | 219,400,283 | Full fine-tuning with a residual BiLSTM and Viterbi decoding with fixed BIO constraints |
| [Encoder-decoder](phase2_encoder_decoder_lora_crf.ipynb) | 227,780,350 | 9,924,350 | Rank-32 additive LoRA on attention and feed-forward projections, with a learned CRF |
| [Prefix LM](phase2_prefix_lorma_crf.ipynb) | 227,393,278 | 9,531,134 | Rank-32 multiplicative LoRMA+ on the shared Transformer blocks, with a learned CRF |
| [Efficient teacher](phase2_efficient_ner_distillation.ipynb) | 217,966,334 | 217,966,334 | Full fine-tuning of the grouped-query, sliding-window decoder with a learned CRF |
| Compact baseline and distilled student | 54,258,686 each | 54,258,686 each | Full fine-tuning of the same three-block initialization, with or without teacher supervision |

I trained on two Kaggle Tesla T4 GPUs using PyTorch, AdamW, FP16 autocast and gradient accumulation for an effective batch of 16 windows. The CRF runs combined sequence loss, weighted BIO cross-entropy and character-boundary loss. I computed class weights from the training split and selected each checkpoint by validation strict macro F1.

### Efficient model and distillation

I initialized the teacher from the Phase 1 continuation run's checkpoint at **405.81M input positions**. For the compact model, I retained Phase 1 blocks 5, 10 and 16 and ran about 30M input positions of next-token recovery on the Phase 1 caches. Both compact runs started from the same recovered weights and fresh NER heads. The distilled run added a KL loss against the frozen teacher's BIO distributions, computed from cached emission logits at temperature 2.

### Results

| Model | Best epoch | Validation strict macro F1 | CodaBench strict macro F1 | Submission ID | Recorded training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decoder full fine-tuning | 50 | 0.4405 | 0.4118 | 974407 | 2 h 27 min |
| Encoder-decoder LoRA + CRF | 24 | 0.4156 | 0.3857 | 975885 | 4 h 23 min |
| Prefix LoRMA+ + CRF | 22 | 0.4286 | 0.3794 | 976545 | 3 h 16 min |
| Efficient teacher | 18 | **0.4417** | **0.4235** | 976956 | 5 h 44 min |
| Compact distilled student | 17 | 0.3256 | 0.3249 | 978111 | 3 h cap |
| Compact gold-only baseline | 8 | 0.3187 | | | 2 h cap |

Times are rounded wall-clock durations for the recorded training invocations, including validation during training. The compact rows show the caps for their final invocations. The distilled cap also covered any teacher-cache work in that invocation.

The efficient teacher led on CodaBench. The distilled student used about a quarter of its parameters and scored 0.0069 higher on validation than the compact gold-only baseline. The compact runs exported predictions from their best available checkpoints under different time limits.

### Reproducing the runs

1. **Attach the Phase 1 inputs.** Use a Kaggle session with two T4 GPUs. Attach `prep_002_mixed16k` and the matching Phase 1 run with its `best_model.pt`. Include the Phase 1 token caches for the efficient notebook's LM recovery stage.
2. **Load the Phase 2 data and train.** Set `PHASE2_INPUT` to the extracted data or `phase2_lner_data.zip`, or enable Internet and provide the `HF_TOKEN` secret for the pinned download. Run the corresponding notebook in order. Use `RUN_WORKFLOW = "standard"` for a fresh efficient run. The recorded deadline run resumed from a saved suite with the teacher already exported.
3. **Evaluate and export.** The notebooks score the selected checkpoint on the held-out split and write a CodaBench ZIP containing `prediction.json` with original-text offsets for each test document.

### References

- Additive and multiplicative adapters. [LoRA](https://arxiv.org/abs/2106.09685) and [LoRMA](https://aclanthology.org/2025.findings-acl.527/)
- Neural sequence tagging and learned transitions. [Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030/)
- Soft teacher targets. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
