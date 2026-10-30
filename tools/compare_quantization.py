"""Select power-of-two INT8 scales on validation data, then report held-out test results."""
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mlp_int8 import Native, read_idx, quantize, pixel_q6, save_model, integer_forward
from train import accuracy


def assess(native, q, raw, y):
    h, scores = native.forward(q, pixel_q6(raw))
    return {'accuracy': float(np.mean(scores.argmax(1) == y)),
            'hidden_saturation_fraction': float(np.mean(h == 127))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--models', default='models')
    p.add_argument('--library', default='build/libmlp.so')
    a = p.parse_args()
    path = Path(a.models)
    if (path / 'quantization.json').exists():
        p.error('quantization report already exists; use a fresh model run')
    with np.load(path / 'float_model.npz') as z:
        m = {k: z[k] for k in ('w1', 'w2', 'b1', 'b2')}
    raw, labels = read_idx(a.data, 'train')
    indices = np.load(path / 'validation_indices.npy')
    xv, yv = raw[indices], labels[indices]
    native = Native(a.library)
    stats = {}
    for k in ('w1', 'w2', 'b1', 'b2'):
        v = m[k]
        stats[k] = {'min': float(v.min()), 'max': float(v.max()), 'std': float(v.std()),
                    'abs_percentiles_95_99_999': np.percentile(np.abs(v), [95, 99, 99.9]).tolist()}
    candidates = []
    for f1, fh, f2 in itertools.product((6, 7, 8), (4, 5, 6), (5, 6, 7)):
        q = quantize(m, f1, fh, f2)
        result = {'w1_frac': f1, 'hidden_frac': fh, 'w2_frac': f2,
                  **assess(native, q, xv, yv)}
        for k, f in [('w1', f1), ('w2', f2)]:
            result[k + '_clipping_fraction'] = float(np.mean((m[k] < -128 / 2 ** f) | (m[k] > 127 / 2 ** f)))
            result[k + '_rmse'] = float(np.sqrt(np.mean((m[k] - q[k].astype(np.float32) / 2 ** f) ** 2)))
        candidates.append(result)
        print(json.dumps(result), flush=True)
    # Deterministic tie-break prefers less hidden saturation, then lower weight RMSE.
    ranked = sorted(candidates, key=lambda r: (-r['accuracy'], r['hidden_saturation_fraction'], r['w1_rmse'] + r['w2_rmse']))
    chosen = ranked[0]
    q = quantize(m, chosen['w1_frac'], chosen['hidden_frac'], chosen['w2_frac'])
    save_model(path / 'recommended.i8ml', q)
    # Test set is first opened after selecting the scale configuration.
    xt, yt = read_idx(a.data, 'test')
    test_results = {}
    for name, cfg in [('uniform_q6', (6, 6, 6)), ('recommended', (q['w1_frac'], q['hidden_frac'], q['w2_frac']))]:
        quant = quantize(m, *cfg)
        test_results[name] = assess(native, quant, xt, yt)
    test_results['float32'] = {'accuracy': accuracy(m, xt.astype(np.float32) / 255, yt)}
    # Exact cross-language check on held-out samples, including hidden activations.
    oracle_h, oracle_s = integer_forward(q, pixel_q6(xt[:32]))
    ch, cs = native.forward(q, pixel_q6(xt[:32]))
    np.testing.assert_array_equal(ch, oracle_h)
    np.testing.assert_array_equal(cs, oracle_s)
    (path / 'sample.u8').write_bytes(xt[0].tobytes())
    report = {'weight_statistics': stats, 'validation_count': len(yv), 'test_count': len(yt),
              'selection_rule': 'validation accuracy; hidden saturation; total weight RMSE',
              'selected_before_test_evaluation': True, 'recommended': chosen,
              'candidates': candidates, 'test_results': test_results,
              'integer_oracle_samples': 32, 'model_bytes': (path / 'recommended.i8ml').stat().st_size}
    (path / 'quantization.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'recommended': chosen, 'test_results': test_results}), flush=True)


if __name__ == '__main__':
    main()
