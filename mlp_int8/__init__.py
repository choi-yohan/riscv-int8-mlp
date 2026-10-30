"""Independent NumPy training utilities and an exact integer reference."""
from pathlib import Path
import ctypes
import gzip
import struct
import numpy as np

INPUT, HIDDEN, OUTPUT = 784, 128, 10
BIAS_LIMIT = 1 << 24
HEADER = struct.Struct('<4s8I')


def read_idx(root, split):
    """Read existing standard IDX data. Never download or modify the source."""
    root = Path(root)
    prefix = 'train' if split == 'train' else 't10k'
    def read(name):
        path = root / name
        if path.exists():
            return path.read_bytes()
        return gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes())
    images = read(prefix + '-images-idx3-ubyte')
    labels = read(prefix + '-labels-idx1-ubyte')
    if len(images) < 16 or len(labels) < 8:
        raise ValueError('truncated IDX header')
    magic, n, rows, cols = struct.unpack('>4I', images[:16])
    lm, ln = struct.unpack('>2I', labels[:8])
    if (magic, rows, cols, lm, ln) != (2051, 28, 28, 2049, n):
        raise ValueError('invalid MNIST IDX dimensions or count')
    if len(images) != 16 + n * INPUT or len(labels) != 8 + n:
        raise ValueError('IDX payload length does not match header')
    x = np.frombuffer(images, np.uint8, offset=16).reshape(n, INPUT)
    y = np.frombuffer(labels, np.uint8, offset=8)
    if np.any(y > 9):
        raise ValueError('invalid MNIST class')
    return x, y


def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + .5)


def quantize(float_model, w1_frac=6, hidden_frac=6, w2_frac=6):
    if not (4 <= w1_frac <= 10 and 4 <= w2_frac <= 10 and 4 <= hidden_frac <= 7):
        raise ValueError('unsupported fractional bits')
    shapes = {'w1': (HIDDEN, INPUT), 'w2': (OUTPUT, HIDDEN),
              'b1': (HIDDEN,), 'b2': (OUTPUT,)}
    for k, shape in shapes.items():
        if float_model[k].shape != shape or not np.isfinite(float_model[k]).all():
            raise ValueError(f'invalid {k}')
    q = {'w1_frac': w1_frac, 'hidden_frac': hidden_frac, 'w2_frac': w2_frac}
    for k, frac in [('w1', w1_frac), ('w2', w2_frac)]:
        q[k] = np.clip(round_away(float_model[k] * (2 ** frac)), -128, 127).astype(np.int8)
    for k, frac in [('b1', 6 + w1_frac), ('b2', hidden_frac + w2_frac)]:
        values = round_away(float_model[k].astype(np.float64) * (2 ** frac))
        if np.any(np.abs(values) > BIAS_LIMIT):
            raise ValueError('bias exceeds accumulator safety bound')
        q[k] = values.astype(np.int32)
    return q


def pixel_q6(x):
    return ((x.astype(np.int32) * 64 + 127) // 255).astype(np.int8)


def integer_forward(q, x):
    """Int64 oracle intentionally uses wider accumulation than the C/RV32 code."""
    shift = 6 + q['w1_frac'] - q['hidden_frac']
    acc = x.astype(np.int64) @ q['w1'].astype(np.int64).T + q['b1']
    h = np.clip((np.maximum(acc, 0) + (1 << (shift - 1))) >> shift, 0, 127).astype(np.int8)
    scores = h.astype(np.int64) @ q['w2'].astype(np.int64).T + q['b2']
    return h, scores


def save_model(path, q):
    payload = HEADER.pack(b'I8ML', 1, INPUT, HIDDEN, OUTPUT, 6,
                          q['w1_frac'], q['hidden_frac'], q['w2_frac'])
    payload += q['w1'].astype('i1').tobytes() + q['w2'].astype('i1').tobytes()
    payload += q['b1'].astype('<i4').tobytes() + q['b2'].astype('<i4').tobytes()
    Path(path).write_bytes(payload)


def load_model(path):
    data = Path(path).read_bytes()
    size = HEADER.size + HIDDEN * INPUT + OUTPUT * HIDDEN + 4 * (HIDDEN + OUTPUT)
    if len(data) != size:
        raise ValueError('invalid model length')
    magic, ver, ni, nh, no, xf, wf1, hf, wf2 = HEADER.unpack_from(data)
    if (magic, ver, ni, nh, no, xf) != (b'I8ML', 1, INPUT, HIDDEN, OUTPUT, 6):
        raise ValueError('invalid model header')
    if not (4 <= wf1 <= 10 and 4 <= wf2 <= 10 and 4 <= hf <= 7):
        raise ValueError('unsupported fractional bits')
    q = {'w1_frac': wf1, 'hidden_frac': hf, 'w2_frac': wf2}
    offset = HEADER.size
    for name, dtype, shape in [('w1', 'i1', (HIDDEN, INPUT)), ('w2', 'i1', (OUTPUT, HIDDEN)),
                                ('b1', '<i4', (HIDDEN,)), ('b2', '<i4', (OUTPUT,))]:
        n = int(np.prod(shape))
        q[name] = np.frombuffer(data, dtype=dtype, count=n, offset=offset).reshape(shape).copy()
        offset += n * np.dtype(dtype).itemsize
    if any(np.any(np.abs(q[k].astype(np.int64)) > BIAS_LIMIT) for k in ('b1', 'b2')):
        raise ValueError('bias exceeds accumulator safety bound')
    return q


class CModel(ctypes.Structure):
    _fields_ = [('w1', ctypes.c_int8 * (HIDDEN * INPUT)),
                ('w2', ctypes.c_int8 * (OUTPUT * HIDDEN)),
                ('b1', ctypes.c_int32 * HIDDEN), ('b2', ctypes.c_int32 * OUTPUT),
                ('w1_frac', ctypes.c_uint32), ('hidden_frac', ctypes.c_uint32),
                ('w2_frac', ctypes.c_uint32)]


class Native:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.mlp_predict.argtypes = [ctypes.POINTER(CModel), ctypes.c_void_p,
                                        ctypes.c_void_p, ctypes.c_void_p]
        self.lib.mlp_predict.restype = ctypes.c_int

    def forward(self, q, x):
        model = CModel()
        for name in ('w1', 'w2', 'b1', 'b2'):
            arr = np.ascontiguousarray(q[name])
            ctypes.memmove(getattr(model, name), arr.ctypes.data, arr.nbytes)
        for name in ('w1_frac', 'hidden_frac', 'w2_frac'):
            setattr(model, name, q[name])
        x = np.ascontiguousarray(x, dtype=np.int8)
        hidden = np.empty((len(x), HIDDEN), dtype=np.int8)
        scores = np.empty((len(x), OUTPUT), dtype=np.int32)
        for i in range(len(x)):
            self.lib.mlp_predict(ctypes.byref(model), x[i].ctypes.data,
                                 hidden[i].ctypes.data, scores[i].ctypes.data)
        return hidden, scores
