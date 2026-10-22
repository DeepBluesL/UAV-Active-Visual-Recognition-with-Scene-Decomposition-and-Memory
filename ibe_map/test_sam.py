"""CPU-only tests for SAM mask geometry and deterministic composition helpers."""
import ast
import hashlib
import inspect
import unittest
from pathlib import Path

import numpy as np


def _load_cpu_helpers():
    """Load the exact helper definitions without importing torch/SAM packages."""
    source = Path(__file__).with_name("sam.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"ring_points", "compose"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise RuntimeError("Could not find SAM helper definitions")
    namespace = {"np": np, "hashlib": hashlib}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "ibe_map/sam.py", "exec"), namespace)
    return namespace["ring_points"], namespace["compose"]


ring_points, compose = _load_cpu_helpers()


class RingPointTests(unittest.TestCase):
    def test_points_for_nonempty_mask_are_outside_and_in_image(self):
        mask = np.zeros((256, 256), dtype=bool)
        mask[90:151, 100:161] = True
        points = ring_points(mask, (100, 90, 160, 150))

        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(points >= 0))
        self.assertTrue(np.all(points < 256))
        for x, y in points.astype(int):
            self.assertFalse(mask[y, x])

    def test_empty_mask_and_empty_box_uses_center_fallback(self):
        mask = np.zeros((256, 256), dtype=bool)
        points = ring_points(mask, (0, 0, 0, 0))

        self.assertEqual(points.shape, (8, 2))
        self.assertTrue(np.all(points >= 0))
        self.assertTrue(np.all(points < 256))
        self.assertTrue(np.all(np.linalg.norm(points - 127.5, axis=1) <= 33))


class CompositionTests(unittest.TestCase):
    def test_preserves_target_pixels_and_unmasked_rgb(self):
        image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        target = np.zeros((256, 256), dtype=bool)
        target[100:120, 100:120] = True
        first = np.zeros_like(target)
        first[90:130, 90:130] = True
        second = np.zeros_like(target)
        second[130:150, 130:150] = True

        output, _ = compose(image, target, np.stack((first, second)))
        np.testing.assert_array_equal(output[target], image[target])
        untouched = ~(target | first | second)
        np.testing.assert_array_equal(output[untouched], image[untouched])

    def test_gray_values_repeat_for_same_image_without_scene_or_class_inputs(self):
        image = np.full((256, 256, 3), 83, dtype=np.uint8)
        target = np.zeros((256, 256), dtype=bool)
        masks = np.zeros((2, 256, 256), dtype=bool)
        masks[0, 40:80, 40:80] = True
        masks[1, 120:170, 120:170] = True

        out1, gray1 = compose(image, target, masks)
        out2, gray2 = compose(image.copy(), target.copy(), masks.copy())
        np.testing.assert_array_equal(gray1, gray2)
        np.testing.assert_array_equal(out1, out2)
        self.assertEqual(tuple(inspect.signature(compose).parameters), ("image", "target", "masks"))


if __name__ == "__main__":
    unittest.main()
