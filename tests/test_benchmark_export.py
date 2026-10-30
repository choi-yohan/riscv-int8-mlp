import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from mlp_int8 import integer_forward, quantize, save_model

spec = importlib.util.spec_from_file_location('benchmark_export',
        Path(__file__).resolve().parents[1] / 'tools/export_benchmark_data.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class BenchmarkDataTests(unittest.TestCase):
    def test_stdlib_oracle_matches_numpy_with_negative_weights_and_saturation(self):
        rng = np.random.default_rng(731)
        m = {'w1': rng.normal(0, .2, (128, 784)).astype(np.float32),
             'w2': rng.normal(0, .4, (10, 128)).astype(np.float32),
             'b1': rng.normal(0, 2, 128).astype(np.float32),
             'b2': rng.normal(0, .3, 10).astype(np.float32)}
        x = rng.integers(-128, 128, (4, 784), dtype=np.int8)
        x[1], x[2], x[3] = -128, 127, 0
        for cfg in ((6, 4, 6), (5, 5, 5), (4, 7, 10), (10, 4, 4)):
            q = quantize(m, *cfg)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'model.i8ml'
                save_model(path, q)
                restored, digest = exporter.read_model(path)
            self.assertEqual(len(digest), 64)
            h, scores = integer_forward(q, x)
            for i, sample in enumerate(x):
                eh, es, ep = exporter.forward(restored, sample.tolist())
                self.assertEqual(eh, h[i].tolist())
                self.assertEqual(es, scores[i].tolist())
                self.assertEqual(ep, int(scores[i].argmax()))

    def test_export_rejects_bad_dimensions_scales_biases_and_lengths(self):
        data = bytearray(exporter.MODEL_SIZE)
        exporter.HEADER.pack_into(data, 0, b'I8ML', 1, 784, 128, 10, 6, 6, 4, 6)
        cases = [data[:-1], data + b'X']
        for offset, value in ((8, 100), (24, 99), (28, 0), (32, 0)):
            broken = bytearray(data)
            struct.pack_into('<I', broken, offset, value)
            cases.append(broken)
        broken = bytearray(data)
        struct.pack_into('<i', broken, 36 + 784 * 128 + 128 * 10, (1 << 24) + 1)
        cases.append(broken)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model'
            for broken in cases:
                path.write_bytes(broken)
                with self.assertRaises(ValueError):
                    exporter.read_model(path)

    def test_export_requires_exact_raw_image_size(self):
        data = bytearray(exporter.MODEL_SIZE)
        exporter.HEADER.pack_into(data, 0, b'I8ML', 1, 784, 128, 10, 6, 6, 4, 6)
        with tempfile.TemporaryDirectory() as d:
            model, image, out = (Path(d) / n for n in ('model', 'image', 'out.h'))
            model.write_bytes(data)
            image.write_bytes(bytes(783))
            with self.assertRaises(ValueError):
                exporter.export(model, image, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
