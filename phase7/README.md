## Phase 7. Prior case retrieval

Given a query judgment, the task was to return seven distinct prior cases in relevance order from that split’s official candidate pool. Train, Dev, and Test contained 827, 118, and 237 queries respectively. The primary metric was **Micro-F1@7**, computed against the complete set of gold relevance links for each query.

### Dense retrieval

I reused the original Phase 1 checkpoints and the 16K legal BPE tokenizer. Each judgment was represented by up to four 512-position windows spread across the document. For each window, I combined the mean content-token representation with a backbone-specific readout and the window’s relative position. A bidirectional GRU and attention pooling combined the windows into a normalized 256-dimensional document embedding. Queries and candidates shared the encoder and were ranked by cosine similarity.

| Run | Adaptation |
|---|---|
| [Decoder gold](phase7_decoder_gold.ipynb) | Fine-tuned the top two transformer blocks, final layer normalization, and retrieval head |
| [Encoder-decoder gold](phase7_encoder_decoder_gold.ipynb) | Trained IA³ scales on attention keys and values and on feed-forward activations, along with the retrieval head |
| [Prefix gold](phase7_prefix_gold.ipynb) | Applied rank-8 LoRMA+ to attention and feed-forward projections and trained the retrieval head |
| [Efficient gold](phase7_efficient_gold.ipynb) | Retained blocks 0, 7, and 15 from the original efficient checkpoint and fully fine-tuned the compact retriever |
| [Efficient KD](phase7_efficient_kd.ipynb) | Trained the same compact architecture with gold labels and soft rankings from the encoder-decoder teacher |

The gold runs used a multi-positive contrastive loss. Each query contributed a rotating gold positive, a BM25 hard negative, and a random negative. Candidates were shared within the batch, and any additional known positives in that pool were marked as relevant. I selected each dense checkpoint by Dev F1@7 using exponential moving average weights.

### Full-text reranking

Dense retrieval alone performed best with the prefix model at **0.2201 Dev F1@7**. The four-window budget covered only about **22.6%** of the average Dev query’s tokens. Extending training also had limited benefit. In the efficient gold run, Train F1@7 rose from **0.2069** at epoch 27 to **0.2429** at epoch 50, while Dev F1@7 fell from **0.1701** to **0.1634**.

I added a second stage that uses the full judgment text. It combines the top 80 candidates from each retrieval signal, including dense similarity, BM25, TF-IDF, case-caption similarity, and similarity around citation mentions. It also adds candidates whose printed reporter citations match references in the query. I compared a fixed weighted fusion with a histogram gradient-boosted reranker fitted on Train relevance labels. The learned reranker uses dense scores and ranks alongside lexical similarity, party-name matches, and citation features. Dev selected the learned hybrid for all five submissions. The prepared BM25 reference scored **0.1012** on Dev.

### Knowledge distillation

I used the encoder-decoder gold pipeline as the teacher after it obtained the highest CodaBench score among the gold runs. The efficient student started from its own Phase 1 weights. The teacher’s rankings on **Train** were converted into soft targets over the candidates sampled for each query. The student optimized a combination of the gold contrastive loss and KL divergence to these targets. High-ranked non-gold cases from the teacher also supplemented the BM25 hard negatives.

### Results

| Run | Best dense epoch | Dense Dev F1@7 | Fusion Dev F1@7 | Learned Dev F1@7 | CodaBench F1@7 | Submission ID | Training time |
|---|---:|---:|---:|---:|---:| ---: | --- |
| [Decoder gold](phase7_rerank_decoder.ipynb) | 33 | 0.1545 | 0.3769 | 0.4291 | 0.4265 | 1040925 | 3 h 41 min |
| [Encoder-decoder gold](phase7_rerank_encoder_decoder.ipynb) | 33 | 0.0700 | 0.3802 | 0.4358 | **0.4400** | 1040928 | 4 h 34 min |
| [Prefix gold](#prefix-reranking) | 24 | **0.2201** | **0.3991** | **0.4391** | 0.4341 | 1040929 | 4 h 50 min |
| [Efficient gold](phase7_rerank_efficient_gold.ipynb) | 27 | 0.1701 | 0.3802 | 0.4091 | 0.4176 | 1040936 | 1 h 51 min |
| [Efficient KD](phase7_rerank_efficient_kd.ipynb) | 24 | 0.1990 | 0.3880 | 0.4091 | 0.4123 | 1041040 | 1 h 46 min |

Training times are the recorded active times rounded to the nearest minute. Reranking runs separately.

The encoder-decoder gained the most from reranking and finished first on CodaBench among the five runs. Distillation improved the efficient student’s dense Dev F1@7 by **0.0289**. After learned reranking, both efficient runs retrieved **368 relevant cases** across the Dev queries, and KD scored **0.0053 lower** than efficient gold on CodaBench.

### Reproducing the runs

1. **Prepare the data.** Run the [preparation notebook](phase7_preparation.ipynb) on Kaggle with two T4 GPUs. Attach the original prep-2 tokenizer and Phase 1 checkpoints. Attach the six official PCR files or enable the `HF_TOKEN` secret for the official download.
2. **Train the dense retrievers.** Attach the resulting `phase7_pcr_prepared_data.zip` and the matching Phase 1 checkpoint. Run the corresponding gold notebook linked above. Keep the saved dense run directory and selected EMA snapshot.
3. **Rerank and export.** Use the reranking notebooks linked in the results table. Set `BASE_RUN_DIR` to the expanded dense run directory, and mount the prepared bundle, matching Phase 1 checkpoint and original tokenizer. Each rerank writes a report and a submission ZIP containing `prediction.json`.
4. **Train the distilled student.** Keep the encoder-decoder rerank directory, including `dense_cache/` and `rerank_report.json`. Set `TEACHER_RERANK_DIR` to that directory in the [efficient KD notebook](phase7_efficient_kd.ipynb), train the student, then run [its reranker](phase7_rerank_efficient_kd.ipynb).

#### Prefix reranking

The [prefix training notebook](phase7_prefix_gold.ipynb) records the dense run. To reproduce its final export, open a copy of the [shared reranker](phase7_rerank_encoder_decoder.ipynb) and use these settings.

```python
RUN_MODE = "rerank"
ARCHITECTURE = "prefix"
RUN_NAME = "prefix_gold"
SHORTLIST_DEPTH = 80
BASE_RUN_DIR = "/kaggle/working/cs781_phase7/runs/prefix_gold"
```

Adjust `BASE_RUN_DIR` if the run is mounted elsewhere. The archived export used the epoch-24 EMA checkpoint with SHA-256 beginning `6bda1a3e9127`. Its saved rerank report records 395 relevant cases retrieved on dev and a Micro-F1@7 of 0.4391.

### References

- Dense retrieval with a shared embedding space. [Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550/)
- Activation scaling. [IA³](https://arxiv.org/abs/2205.05638)
- Multiplicative low-rank adaptation. [LoRMA](https://aclanthology.org/2025.findings-acl.527/)
- Soft teacher targets. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
