## Phase 1. Legal language model pretraining

I trained four Transformer architectures from scratch on the official legal corpus to generate 128-word continuations from 512-word prompts. I used BERTScore F1 as the primary metric and tracked ROUGE-L and BLEU-4 during local evaluation.

### Data and models

I used Supreme Court judgments, High Court documents, Acts and Tribunal documents. After removing exact duplicates across sources, I held out about 5% of the documents using a deterministic document hash. I fitted a 16,000-token byte BPE tokenizer on equal amounts of training text from each source, then built a 75M-token cache per source and sampled equally from the four caches during training. The trainer reads memory-mapped token IDs and packs documents with EOS separators.

All four models used a width of 1024, 16 query heads, a context budget of 2,560 token positions, pre-LayerNorm, GELU, fixed sinusoidal positions, tied embeddings and 0.1 dropout.

| Architecture | Parameters | Design and training objective |
| --- | ---: | --- |
| Decoder-only | 217,860,096 | 16 blocks with full causal attention and next-token loss across the block |
| Encoder-decoder | 217,856,000 | 8 bidirectional encoder blocks and 6 causal decoder blocks with cross-attention. The loss covers the continuation. |
| Prefix LM | 217,862,144 | 16 shared blocks with a bidirectional prefix and causal continuation. The loss covers the continuation. |
| Efficient decoder | 217,872,384 | 16 causal blocks with grouped-query attention, 4 KV heads and a 512-position window. A wider feed-forward layer (4,864 instead of 4,096) keeps the parameter count close to the other models. |

### Training and evaluation

I trained each base model for about 300M input positions on two Kaggle Tesla T4 GPUs using PyTorch DataParallel, FP16 autocast and gradient accumulation. I used Adam with an effective batch of 24 sequences, gradient clipping at 1.0, 5% warmup and cosine decay to `3e-5`. I set the peak learning rate to `3e-4` and lowered it to `1.5e-4` for the successful encoder-decoder restart. During training, I tracked mean validation NLL across the four sources. For generation, I evaluated continuations on a fixed set of 100 held-out prompts, 25 from each source. The causal models scored more target positions per training block than the two conditional models, so the NLL values use different targets.

### CodaBench results

| Submission ID | Model | BERTScore F1 |
| --- | --- | ---: |
| 960412 | Decoder-only, 300M | 0.8207 |
| 961418 | Encoder-decoder, 300M | 0.8036 |
| 963236 | Prefix LM, 300M | 0.8143 |
| 964257 | Efficient decoder, 300M | **0.8213** |
| 965213 | Efficient decoder, about 400M | 0.8171 |

The 300M efficient decoder had the highest CodaBench BERTScore F1 at 0.8213. The roughly 400M checkpoint scored 0.8171, a drop of 0.0042.

### Post-submission pretraining

After Phase 1 submissions closed, I continued training the efficient decoder to 500.06M input positions. I restored the optimizer state from the 300M run and reused the same caches at a constant learning rate of `3e-5`. I used the resulting checkpoint to initialize the efficient model in later phases.

The Phase 1 folder includes `phase1_efficient_500m_continuation.ipynb`, which records this additional pretraining. In the saved continuation evaluation, I used an intermediate checkpoint at 405.81M positions and obtained 0.8321 BERTScore F1 on the local balanced set with repetition control.

### Reproducing the runs

1. **Prepare the data.** Run `phase1_preparation_16k.ipynb` on CPU. Use `PREP_MODE = "rebuild"` with Internet access to build the tokenizer, caches and validation split from the pinned `Exploration-Lab/CS-781-Capstone` corpus on Hugging Face. Use `"reuse"` to check and reuse the extracted `prep_002_mixed16k` folder.
2. **Train a model.** Attach the prepared folder to a Kaggle GPU session and run the corresponding `phase1_*_training.ipynb` in order. Set `PREP_INPUT` if the notebook cannot find the folder. To reproduce the efficient continuation, run `phase1_efficient_500m_continuation.ipynb` with the complete 300M efficient run folder as `PARENT_RUN_INPUT`.
3. **Evaluate and export.** Run the corresponding evaluation notebook with the prepared folder and the trained run containing `best_model.pt`. Set `PREP_DIR` and `RUN_DIR` in the local notebooks, or `PREP_INPUT` and `RUN_INPUT` in the decoder evaluation notebook. I mainly used top-p sampling at 0.95 with a temperature of 1.0. I also tested greedy decoding and added repetition controls for the encoder-decoder and continued efficient model. Set `GENERATE_OFFICIAL_TEST = False` for held-out evaluation only. To export the CodaBench ZIP, enable it and select a scored decoding setting. The first BERTScore run needs access to download its scoring model.

The notebooks generate the model weights and prepared caches. I keep those files outside the repository.

### References

- Transformer architecture, sinusoidal positions and cross-attention. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Attention masking for the shared prefix model. [UniLM](https://arxiv.org/abs/1905.03197)
- Grouped-query and sliding-window attention. [GQA](https://arxiv.org/abs/2305.13245) and [Mistral 7B](https://arxiv.org/abs/2310.06825)
- Top-p sampling for generation. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- Text generation evaluation. [BERTScore](https://arxiv.org/abs/1904.09675)
