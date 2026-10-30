import ctypes
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from mlp_int8 import (BIAS_LIMIT, CModel, Native, integer_forward, load_model,
                      pixel_q6, quantize, save_model)

ROOT = Path(__file__).resolve().parents[1]


class MLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native(ROOT / 'build/libmlp.so')
        cls.rng = np.random.default_rng(2026)

    def random_quantized(self, f1=6, fh=6, f2=6):
        return {'w1': self.rng.integers(-128, 128, (128, 784), dtype=np.int8),
                'w2': self.rng.integers(-128, 128, (10, 128), dtype=np.int8),
                'b1': self.rng.integers(-BIAS_LIMIT, BIAS_LIMIT + 1, 128, dtype=np.int32),
                'b2': self.rng.integers(-BIAS_LIMIT, BIAS_LIMIT + 1, 10, dtype=np.int32),
                'w1_frac': f1, 'hidden_frac': fh, 'w2_frac': f2}

    def test_exact_inference_across_scale_extremes(self):
        for f1, fh, f2 in [(4, 7, 4), (10, 4, 10), (6, 6, 6), (8, 5, 7)]:
            q = self.random_quantized(f1, fh, f2)
            x = self.rng.integers(-128, 128, (7, 784), dtype=np.int8)
            x[0], x[1], x[2] = -128, 127, 0
            h, s = self.native.forward(q, x)
            expected_h, expected_s = integer_forward(q, x)
            np.testing.assert_array_equal(h, expected_h)
            np.testing.assert_array_equal(s, expected_s)

    def test_worst_case_dot_does_not_overflow(self):
        dot = self.native.lib.mlp_dot_c
        dot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        dot.restype = ctypes.c_int32
        a = np.full(784, -128, np.int8)
        self.assertEqual(dot(a.ctypes.data, a.ctypes.data, 784), 784 * 16384)
        self.assertEqual(dot(a.ctypes.data, a.ctypes.data, 0), 0)

    def test_requantization_rounding_and_saturation(self):
        fn = self.native.lib.mlp_relu_requant
        fn.argtypes, fn.restype = [ctypes.c_int32, ctypes.c_uint32], ctypes.c_int8
        for shift in range(3, 13):
            half = 1 << (shift - 1)
            for value, expected in [(-2147483648, 0), (-1, 0), (0, 0),
                                    (half - 1, 0), (half, 1),
                                    (127 << shift, 127), (2147483647, 127)]:
                self.assertEqual(fn(value, shift), expected)

    def test_all_256_pixel_values(self):
        fn = self.native.lib.mlp_pixel_q6
        fn.argtypes, fn.restype = [ctypes.c_uint8], ctypes.c_int8
        expected = pixel_q6(np.arange(256, dtype=np.uint8))
        self.assertEqual([fn(i) for i in range(256)], expected.tolist())
        self.assertEqual((fn(0), fn(255)), (0, 64))

    def test_model_validation(self):
        fn = self.native.lib.mlp_model_valid
        fn.argtypes = [ctypes.POINTER(CModel)]
        m = CModel()
        m.w1_frac = m.hidden_frac = m.w2_frac = 6
        self.assertEqual(fn(ctypes.byref(m)), 1)
        m.b1[0] = BIAS_LIMIT + 1
        self.assertEqual(fn(ctypes.byref(m)), 0)
        m.b1[0] = -BIAS_LIMIT
        m.hidden_frac = 0
        self.assertEqual(fn(ctypes.byref(m)), 0)

    def test_argmax_tie_chooses_first(self):
        q = self.random_quantized()
        for k in ('w1', 'w2', 'b1', 'b2'):
            q[k].fill(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_model(d / 'm', q)
            (d / 'x').write_bytes(bytes(784))
            out = subprocess.check_output([str(ROOT / 'build/mlp'), str(d / 'm'), str(d / 'x')], text=True)
            self.assertEqual(json.loads(out)['prediction'], 0)

    def test_serialization_and_cli_end_to_end(self):
        q = self.random_quantized(8, 5, 7)
        raw = self.rng.integers(0, 256, 784, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_model(d / 'm', q)
            restored = load_model(d / 'm')
            for k in q:
                np.testing.assert_array_equal(q[k], restored[k])
            (d / 'x').write_bytes(raw.tobytes())
            out = subprocess.check_output([str(ROOT / 'build/mlp'), str(d / 'm'), str(d / 'x')], text=True)
            scores = integer_forward(q, pixel_q6(raw[None, :]))[1][0]
            payload = json.loads(out)
            self.assertEqual(payload['scores'], scores.tolist())
            self.assertEqual(payload['prediction'], int(scores.argmax()))
            self.assertEqual(payload['scores_frac'], 12)

    def test_malformed_models_rejected_by_python_and_c(self):
        q = self.random_quantized()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_model(d / 'm', q)
            data = (d / 'm').read_bytes()
            (d / 'x').write_bytes(bytes(784))
            cases = [data[:3], data[:-1], data + b'X', b'NOPE' + data[4:]]
            for offset, value in [(4, 99), (8, 100), (20, 7), (24, 99), (28, 0)]:
                broken = bytearray(data)
                struct.pack_into('<I', broken, offset, value)
                cases.append(bytes(broken))
            broken = bytearray(data)
            struct.pack_into('<i', broken, 36 + 128 * 784 + 10 * 128, BIAS_LIMIT + 1)
            cases.append(bytes(broken))
            for broken in cases:
                (d / 'broken').write_bytes(broken)
                with self.assertRaises(ValueError):
                    load_model(d / 'broken')
                result = subprocess.run([str(ROOT / 'build/mlp'), str(d / 'broken'), str(d / 'x')], capture_output=True)
                self.assertEqual(result.returncode, 2)

    def test_invalid_image_sizes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_model(d / 'm', self.random_quantized())
            for length in (0, 783, 785):
                (d / 'x').write_bytes(bytes(length))
                result = subprocess.run([str(ROOT / 'build/mlp'), str(d / 'm'), str(d / 'x')], capture_output=True)
                self.assertEqual(result.returncode, 2)

    def test_quantizer_ties_and_clipping(self):
        m = {'w1': np.zeros((128, 784), np.float32), 'w2': np.zeros((10, 128), np.float32),
             'b1': np.zeros(128, np.float32), 'b2': np.zeros(10, np.float32)}
        m['w1'][0, :4] = [.5 / 64, -.5 / 64, 5, -5]
        self.assertEqual(quantize(m)['w1'][0, :4].tolist(), [1, -1, 127, -128])
        m['b1'][0] = np.inf
        with self.assertRaises(ValueError):
            quantize(m)


if __name__ == '__main__':
    unittest.main()
