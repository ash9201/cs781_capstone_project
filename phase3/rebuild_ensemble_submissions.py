r"""Rebuild the two Phase 3 four-model voting submissions.

Supply the four model diagnostics archives produced by the notebooks. Each
archive must contain its test prediction.json and labels.json. For example,

    python rebuild_ensemble_submissions.py \
        --teacher efficient_rr_teacher_001_diagnostics.zip \
        --decoder decoder_rr_ia3_001_continue_001_diagnostics.zip \
        --encdec encdec_rr_lora_001_diagnostics.zip \
        --prefix prefix_rr_lorma_001_diagnostics.zip \
        --output-dir rebuilt

The equal vote with teacher tie-breaking scored 0.4740 on dev. The
conservative vote scored 0.4599 on dev and 0.4444 on CodaBench (992618).
The conservative vote retains the teacher unless the other three agree
on a different label.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

RULES = {
    'ensemble_rr_four_models_teacher_tiebreak_submission.zip':
        ({'teacher': 1, 'decoder': 1, 'encdec': 1, 'prefix': 1}, 'teacher'),
    'ensemble_rr_teacher_conservative_submission.zip':
        ({'teacher': 2, 'decoder': 1, 'encdec': 1, 'prefix': 1}, 'teacher'),
}


def read_json(archive, basename):
    with ZipFile(archive) as z:
        candidates = [n for n in z.namelist() if n.endswith('/' + basename)
                      and '/sources/' not in n]
        if len(candidates) != 1:
            raise ValueError(f'{archive}: expected one {basename}')
        return json.loads(z.read(candidates[0]))


def write_zip(path, prediction):
    info = ZipInfo('prediction.json', (2026, 10, 16, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    with ZipFile(path, 'w') as z:
        z.writestr(info, json.dumps(prediction, ensure_ascii=False, indent=2).encode('utf-8'))
    with ZipFile(path) as z:
        assert z.namelist() == ['prediction.json']
        assert z.testzip() is None
        assert json.loads(z.read('prediction.json')) == prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['teacher', 'decoder', 'encdec', 'prefix']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    model_names = ['teacher', 'decoder', 'encdec', 'prefix']
    predictions = {n: read_json(getattr(args, n), 'prediction.json') for n in model_names}
    labels = read_json(args.teacher, 'labels.json')
    assert len(set(labels)) == 13
    shape = {d: len(p) for d, p in predictions['teacher'].items()}
    for n in model_names:
        assert read_json(getattr(args, n), 'labels.json') == labels
        assert {d: len(p) for d, p in predictions[n].items()} == shape
        assert all(isinstance(p, list) and set(p) <= set(labels) for p in predictions[n].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (weights, tie_breaker) in RULES.items():
        result = {}
        for doc_id, original in predictions['teacher'].items():
            result[doc_id] = []
            for i in range(len(original)):
                counts = Counter()
                for n, weight in weights.items():
                    counts[predictions[n][doc_id][i]] += weight
                tied = {label for label, count in counts.items() if count == max(counts.values())}
                if len(tied) == 1:
                    chosen = next(iter(tied))
                else:
                    chosen = predictions[tie_breaker][doc_id][i]
                    assert chosen in tied
                result[doc_id].append(chosen)
        path = args.output_dir / filename
        write_zip(path, result)
        print(path)


if __name__ == '__main__':
    main()
