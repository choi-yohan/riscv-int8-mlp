"""Train a new 784-128-10 model from local MNIST using NumPy and Adam."""
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mlp_int8 import read_idx, quantize, save_model


def accuracy(m, x, y):
    good = 0
    for start in range(0, len(x), 512):
        h = np.maximum(x[start:start + 512] @ m['w1'].T + m['b1'], 0)
        good += np.count_nonzero((h @ m['w2'].T + m['b2']).argmax(1) == y[start:start + 512])
    return float(good / len(x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='directory containing standard MNIST IDX files')
    p.add_argument('--out', default='models')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=.001)
    a = p.parse_args()
    if a.epochs < 1 or a.batch < 1 or a.lr <= 0:
        p.error('epochs, batch, and learning rate must be positive')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Prevent accidental replacement of an earlier training run.
    if (out / 'float_model.npz').exists():
        p.error('output already contains a trained model; choose a fresh --out directory')
    raw, y = read_idx(a.data, 'train')
    if len(raw) < 6000:
        p.error('at least 6000 training examples are required')
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(raw))
    val_indices, train_indices = order[:5000], order[5000:]
    x = raw.astype(np.float32) / 255
    xv, yv = x[val_indices], y[val_indices]
    xt, yt = x[train_indices], y[train_indices]
    m = {'w1': (rng.standard_normal((128, 784)) * np.sqrt(2 / 784)).astype(np.float32),
         'w2': (rng.standard_normal((10, 128)) * np.sqrt(2 / 128)).astype(np.float32),
         'b1': np.zeros(128, np.float32), 'b2': np.zeros(10, np.float32)}
    first = {k: np.zeros_like(v) for k, v in m.items()}
    second = {k: np.zeros_like(v) for k, v in m.items()}
    history, step, best_acc = [], 0, -1
    started = time.monotonic()
    for epoch in range(a.epochs):
        shuffled = rng.permutation(len(xt))
        loss_sum = 0.
        for start in range(0, len(xt), a.batch):
            ids = shuffled[start:start + a.batch]
            xb, yb = xt[ids], yt[ids]
            pre = xb @ m['w1'].T + m['b1']
            h = np.maximum(pre, 0)
            z = h @ m['w2'].T + m['b2']
            z -= z.max(1, keepdims=True)
            prob = np.exp(z)
            prob /= prob.sum(1, keepdims=True)
            loss_sum -= float(np.log(np.maximum(prob[np.arange(len(ids)), yb], 1e-30)).sum())
            prob[np.arange(len(ids)), yb] -= 1
            prob /= len(ids)
            dh = (prob @ m['w2']) * (pre > 0)
            grads = {'w2': prob.T @ h + 1e-4 * m['w2'], 'b2': prob.sum(0),
                     'w1': dh.T @ xb + 1e-4 * m['w1'], 'b1': dh.sum(0)}
            step += 1
            lr = a.lr * (0.5 if epoch >= a.epochs * .7 else 1.)
            for k, g in grads.items():
                first[k] *= .9
                first[k] += .1 * g
                second[k] *= .999
                second[k] += .001 * g * g
                m[k] -= lr * (first[k] / (1 - .9 ** step)) / (np.sqrt(second[k] / (1 - .999 ** step)) + 1e-8)
        acc = accuracy(m, xv, yv)
        item = {'epoch': epoch + 1, 'loss': loss_sum / len(xt), 'validation_accuracy': acc}
        history.append(item)
        print(json.dumps(item), flush=True)
        if acc > best_acc:
            best_acc = acc
            np.savez(out / 'float_model.npz', **m)
    np.save(out / 'validation_indices.npy', val_indices)
    with np.load(out / 'float_model.npz') as archive:
        best = {k: archive[k] for k in ('w1', 'w2', 'b1', 'b2')}
    save_model(out / 'uniform_q6.i8ml', quantize(best))
    report = {'architecture': [784, 128, 10], 'seed': a.seed, 'epochs': a.epochs,
              'batch_size': a.batch, 'optimizer': 'Adam', 'learning_rate': a.lr,
              'weight_decay': 1e-4, 'train_count': len(xt), 'validation_count': len(xv),
              'data_directory': str(Path(a.data).resolve()),
              'test_data_used_for_training_or_selection': False,
              'best_validation_accuracy': best_acc, 'history': history,
              'elapsed_seconds': time.monotonic() - started}
    (out / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Saved best float model and uniform-Q6 baseline to', out, flush=True)


if __name__ == '__main__':
    main()
