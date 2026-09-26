import unittest

from butler_portugal import ANTISYMMETRIC, SYMMETRIC, Tensor


def riemann(*indices):
    return (
        Tensor("R", indices)
        .antisymmetric(0, 1)
        .antisymmetric(2, 3)
        .symmetric_pairs((0, 1), (2, 3))
    )


class TestBindings(unittest.TestCase):
    def test_ricci(self):
        self.assertEqual(str(riemann("^a", "c", "a", "b").canonicalize()), "R_b_a_c^a")

    def test_vanishing_trace(self):
        self.assertEqual(riemann("^a", "a", "b", "c").canonicalize().coefficient, 0)

    def test_product(self):
        f = Tensor("F", ["b", "a"]).antisymmetric(0, 1) * Tensor(
            "F", ["^a", "^b"]
        ).antisymmetric(0, 1)
        self.assertEqual(str(f.canonicalize()), "-F_a_b F^a^b")

    def test_spinor_metric(self):
        e = Tensor("E", ["^A:spinor", "A:spinor"]).metric(ANTISYMMETRIC, "spinor")
        self.assertEqual(str(e.canonicalize()), "-E_A^A")

    def test_custom_generator(self):
        t = Tensor("T", ["b", "a"]).custom(([1, 0], -1))
        self.assertEqual(str(t.canonicalize()), "-T_a_b")

    def test_errors(self):
        with self.assertRaises(ValueError):
            Tensor("T", ["a"]).metric(9)
        with self.assertRaises(ValueError):
            Tensor("A", ["a"]).metric(ANTISYMMETRIC) * Tensor("B", ["^a"]).metric(
                SYMMETRIC
            )


if __name__ == "__main__":
    unittest.main()
