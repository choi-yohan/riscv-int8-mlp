"""Reproduce the follow-up Q3.5 comparison without changing the recommendation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mlp_int8 import Native, read_idx, quantize, pixel_q6, save_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--models', default='models')
    p.add_argument('--library', default='build/libmlp.so')
    a = p.parse_args()
    path = Path(a.models)
    if (path / 'q3_5_comparison.json').exists():
        p.error('comparison report already exists; use a fresh model run')
    with np.load(path / 'float_model.npz') as z:
        m = {k: z[k] for k in ('w1', 'w2', 'b1', 'b2')}
    native = Native(a.library)
    raw, y = read_idx(a.data, 'test')
    train, train_y = read_idx(a.data, 'train')
    indices = np.load(path / 'validation_indices.npy')
    results = []
    for name, cfg in [('uniform_q3_5', (5, 5, 5)), ('weights_q2_6_hidden_q3_5', (6, 5, 6))]:
        q = quantize(m, *cfg)
        h, s = native.forward(q, pixel_q6(raw))
        _, vs = native.forward(q, pixel_q6(train[indices]))
        results.append({'name': name, 'w1_frac': cfg[0], 'hidden_frac': cfg[1], 'w2_frac': cfg[2],
                        'test_accuracy': float(np.mean(s.argmax(1) == y)),
                        'validation_accuracy': float(np.mean(vs.argmax(1) == train_y[indices])),
                        'hidden_saturation_fraction_test': float(np.mean(h == 127))})
        save_model(path / (name + '.i8ml'), q)
    report = {'test_count': len(y), 'input_frac': 6, 'results': results,
              'note': 'Follow-up comparison requested after the initial validation-selected recommendation; does not reselect using test results.'}
    (path / 'q3_5_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
