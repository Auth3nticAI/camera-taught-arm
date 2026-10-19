"""Leave-one-take-out cross-validation over the non-test takes (D-U tie-break). Test take is never scored.

uv run python scripts/cv_inputs.py CACHE OUT
"""
import json
import sys
from pathlib import Path

import numpy as np

from cta.learning import select_features, train_head
from cta.learning_data import split_takes

REG = dict(dropout=0.3, weight_decay=0.05, feature_noise=0.1, use_aug=True)
CONFIGS = {'full': ('full', {}), 'hand_reg': ('hand', REG), 'both_reg': ('both', REG)}

cache, out = Path(sys.argv[1]), Path(sys.argv[2])
names = [t['name'] for t in json.loads((cache / 'manifest.json').read_text())['takes']]
test = split_takes(names, 42)['test']
pool = [n for n in names if n not in test]
results = {}
for cfg, (inp, kw) in CONFIGS.items():
    takes = {}
    for n in names:
        with np.load(cache / f'{n}.npz', allow_pickle=False) as z:
            takes[n] = {k: v.copy() for k, v in select_features(z, inp).items()}
    vals = {}
    for v in pool:
        split = {'train': [n for n in pool if n != v], 'val': [v], 'test': test}
        m = train_head(takes, out / cfg / v, split=split, **kw)
        vals[v] = m['best_val_mse_rad2']
    results[cfg] = {'per_fold_val_mse': vals, 'mean': float(np.mean(list(vals.values()))),
                    'std': float(np.std(list(vals.values())))}
    print(f"{cfg:9s} mean val MSE {results[cfg]['mean']:.3f} +- {results[cfg]['std']:.3f}", flush=True)
(out / 'cv_summary.json').write_text(json.dumps({'test_take_untouched': test, 'results': results}, indent=2))
