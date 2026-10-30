"""Export exact Python golden vectors for the bare-metal RV32 test."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mlp_int8 import load_model, read_idx, pixel_q6, integer_forward

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', default='models/recommended.i8ml')
p.add_argument('--data', required=True)
p.add_argument('--out', default='build/fixture.h')
a = p.parse_args()
q = load_model(a.model)
raw, _ = read_idx(a.data, 'test')
# Real samples plus signed extrema and zero input, independent of class accuracy.
x = np.concatenate([pixel_q6(raw[:16]), np.full((1, 784), -128, np.int8),
                    np.full((1, 784), 127, np.int8), np.zeros((1, 784), np.int8)])
h, scores = integer_forward(q, x)
def values(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return '{' + ','.join(str(int(v)) for v in arr) + '}'
    return '{\n' + ',\n'.join(values(v) for v in arr) + '\n}'
lines = ['/* Generated from independent NumPy int64 golden computation. */',
         f'#define FIXTURE_COUNT {len(x)}', 'static const mlp_model fixture_model = {']
for name in ('w1', 'w2', 'b1', 'b2'):
    lines.append(f'.{name} = {values(q[name].reshape(-1))},')
for name in ('w1_frac', 'hidden_frac', 'w2_frac'):
    lines.append(f'.{name} = {q[name]},')
lines += ['};', f'static const int8_t fixture_inputs[FIXTURE_COUNT][784] = {values(x)};',
          f'static const int8_t fixture_hidden[FIXTURE_COUNT][128] = {values(h)};',
          f'static const int32_t fixture_scores[FIXTURE_COUNT][10] = {values(scores)};',
          f'static const int fixture_labels[FIXTURE_COUNT] = {values(scores.argmax(1))};']
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(lines) + '\n')
print(f'Exported {len(x)} full-network golden fixtures to {out}')
