# Phase 4. Efficient gold and prefix ensemble

I combined the efficient gold and continued prefix models using weights 0.75
and 0.25 on their class-1 probabilities. The submitted rule predicts class 1
when the weighted probability reaches 0.40. It scored **0.6640 dev macro F1**
and **0.6371 CodaBench macro F1** as submission 1009629.

I evaluated 21 candidate blends on the dev split and fitted each blend's
threshold on a grid from 0.20 to 0.80. The repeated stratified folds group both
copies of each dev case by original case ID. The gold and prefix family scored
0.6514 mean out-of-fold macro F1, compared with 0.6413 for gold alone.
The source checkpoints had already been selected on the full dev split, so the
folds assess blend and threshold fitting after that selection.

## Reproduce

Collect the diagnostic ZIPs from the five selected exports in an `inputs/`
directory here. Each ZIP contains dev and test probabilities and the course
evaluator. Use the continued prefix export selected at update 1,115.

| Notebook | Diagnostic run ID |
| --- | --- |
| [Efficient gold](../phase4_efficient_gold.ipynb) | `cjpe_efficient_gold_001` |
| [Efficient KD](../phase4_efficient_kd.ipynb) | `cjpe_efficient_kd_001` |
| [Prefix LoRMA+ continuation](../phase4_prefix_lorma.ipynb) | `cjpe_prefix_lorma_001` |
| [Decoder partial fine-tuning](../phase4_decoder_partial.ipynb) | `cjpe_decoder_partial_003` |
| [Encoder-decoder adapters](../phase4_encoder_decoder_adapters.ipynb) | `cjpe_encoder_decoder_adapters_001` |

From this directory, run the search and export the rule saved in
`selection.json`. The bootstrap command is optional.

```bash
python -m pip install -r requirements.txt
python ensemble.py --input-dir inputs --output-dir results
python bootstrap_diagnostic.py --results-dir results
python export_submission.py --results-dir results --selection selection.json --output-dir deliverables
```

The export runs the course evaluator on dev and writes the 1,517 test
predictions to `deliverables/cjpe_gold_prefix_ensemble_001_submission.zip`.
