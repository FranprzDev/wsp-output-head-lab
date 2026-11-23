import unittest
import numpy as np
from wsp import fit_ridge, edit_rows, pair_accuracy


class CoreChecks(unittest.TestCase):
    def test_row_identity(self):
        rng = np.random.default_rng(7)
        head, h, w = rng.normal(size=(9, 576)), rng.normal(size=576), rng.normal(size=576)
        patched = edit_rows(head, 2, 6, w, alpha=.7)
        base = h @ (head[2] - head[6])
        actual = h @ (patched[2] - patched[6])
        self.assertAlmostEqual(actual, base + .7 * (h @ w), places=10)
        np.testing.assert_array_equal(head[[0,1,3,4,5,7,8]], patched[[0,1,3,4,5,7,8]])

    def test_ridge_normal_equation(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=(20, 6)), rng.normal(size=20)
        w, metadata = fit_ridge(x, y, 100)
        centered = x - metadata['feature_mean']
        target = (y - metadata['label_mean']) / metadata['label_scale']
        np.testing.assert_allclose((centered.T @ centered + 100*np.eye(6)) @ w, centered.T @ target)

    def test_pair_order_and_tie(self):
        self.assertEqual(pair_accuracy([0,1,2], [1,2,3]), {'pairs':3,'accuracy':1.0})
        self.assertEqual(pair_accuracy([0,0], [1,3])['accuracy'], .5)
        self.assertIsNone(pair_accuracy([0], [1])['accuracy'])

    def test_reject_bad_inputs(self):
        with self.assertRaises(ValueError): fit_ridge([[1],[2]], [1,1])
        with self.assertRaises(ValueError): edit_rows(np.zeros((2,3)), 0,0,np.ones(3))
        with self.assertRaises(ValueError): pair_accuracy([float('nan')], [1])

if __name__ == '__main__': unittest.main()
