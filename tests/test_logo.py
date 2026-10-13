"""Checks for default logo geometry, the optional axis arguments, and color helpers."""

import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from fast_logomaker import FastLogo, get_color_dict


def _vertices(logo, idx):
    return [glyph["path"].vertices.copy() for glyph in logo.processed_logos[idx]["glyphs"]]


class LogoTests(unittest.TestCase):
    def setUp(self):
        self.values = np.random.default_rng(0).normal(size=(3, 8, 4))

    def tearDown(self):
        plt.close("all")

    def _logo(self, values=None, **kwargs):
        kwargs.setdefault("show_progress", False)
        return FastLogo(self.values if values is None else values, **kwargs)

    def test_default_positions_match_an_explicit_range(self):
        default = self._logo()
        explicit = self._logo(positions=np.arange(self.values.shape[1]), mirror_glyphs=False)
        default.process_all()
        explicit.process_all()
        for idx in range(default.N):
            for left, right in zip(_vertices(default, idx), _vertices(explicit, idx)):
                np.testing.assert_array_equal(left, right)
        _, ax = default.draw_single(0, apply_layout=False)
        self.assertEqual(ax.get_xlim(), (-0.5, self.values.shape[1] - 0.5))

    def test_positions_shift_glyphs_and_limits(self):
        origin = self._logo()
        origin.process_all()
        shift = 1000
        moved = self._logo(positions=np.arange(shift, shift + self.values.shape[1]))
        moved.process_all()
        for left, right in zip(_vertices(origin, 0), _vertices(moved, 0)):
            np.testing.assert_allclose(right[:, 0], left[:, 0] + shift)
            np.testing.assert_array_equal(right[:, 1], left[:, 1])
        _, ax = moved.draw_single(0, apply_layout=False)
        length = self.values.shape[1]
        self.assertEqual(ax.get_xlim(), (shift - 0.5, shift + length - 0.5))

    def test_mirror_reflects_x_and_keeps_y(self):
        values = self.values.copy()
        values[0, 0, 0] = -1.5
        plain = self._logo(values, fade_below=0, shade_below=0)
        mirrored = self._logo(values, mirror_glyphs=True, fade_below=0, shade_below=0)
        plain.process_all()
        mirrored.process_all()
        self.assertTrue(any(glyph["floor"] < 0 for glyph in plain.processed_logos[0]["glyphs"]))
        for left, right in zip(_vertices(plain, 0), _vertices(mirrored, 0)):
            np.testing.assert_array_equal(left[:, 1], right[:, 1])
            center = (left[:, 0] + right[:, 0]) / 2
            np.testing.assert_allclose(center, center[0], atol=1e-8)

    def test_per_logo_positions_and_mirror(self):
        positions = np.vstack([np.arange(100, 108), np.arange(500, 492, -1)])
        batch = self._logo(self.values[:2], positions=positions, mirror_glyphs=[False, True])
        batch.process_all()
        alone = self._logo(
            self.values[1:2], positions=positions[1], mirror_glyphs=True
        )
        alone.process_all()
        for left, right in zip(_vertices(batch, 1), _vertices(alone, 0)):
            np.testing.assert_array_equal(left, right)

        _, ax = batch.draw_single(1, apply_layout=False)
        self.assertEqual(ax.get_xlim(), (positions[1].min() - 0.5, positions[1].max() + 0.5))
        _, axes = batch.draw_logos(apply_layout=False)
        self.assertEqual(axes[0, 0].get_xlim(), (99.5, 107.5))
        self.assertEqual(axes[0, 1].get_xlim(), (492.5, 500.5))

    def test_apply_layout_false_keeps_constrained_layout(self):
        logo = self._logo()
        logo.process_all()
        fig, ax = plt.subplots(layout="constrained")
        engine = type(fig.get_layout_engine()).__name__
        logo.draw_single(0, ax=ax, apply_layout=False)
        self.assertEqual(type(fig.get_layout_engine()).__name__, engine)

    def test_tight_layout_uses_the_drawn_figure(self):
        logo = self._logo()
        logo.process_all()
        other, _ = plt.subplots()
        fig, ax = plt.subplots(layout="constrained")
        plt.figure(other.number)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logo.draw_single(0, ax=ax, apply_layout=True)
        self.assertIsNone(other.get_layout_engine())
        self.assertNotEqual(
            type(fig.get_layout_engine()).__name__, "ConstrainedLayoutEngine"
        )

    def test_style_glyphs_uses_column_index(self):
        onehot = np.zeros((1, 4, 4))
        onehot[0] = np.eye(4)
        logo = self._logo(onehot, positions=np.arange(20, 24), fade_below=0, shade_below=0)
        logo.process_all()
        self.assertEqual([glyph["pos"] for glyph in logo.processed_logos[0]["glyphs"]], [0, 1, 2, 3])
        logo.style_glyphs_in_sequence("AAAA", color="red")
        colors = [glyph["color"] for glyph in logo.processed_logos[0]["glyphs"]]
        self.assertEqual(colors[0], to_rgb("red"))
        self.assertEqual(colors[1:], [(0.4, 0.4, 0.4)] * 3)

    def test_drawing_before_process_all_raises(self):
        logo = self._logo(self.values[:1])
        with self.assertRaises(ValueError):
            logo.draw_single(0)
        with self.assertRaises(ValueError):
            logo.draw_logos()
        with self.assertRaises(ValueError):
            logo.draw_variability_logo()
        with self.assertRaises(ValueError):
            logo.style_glyphs_in_sequence("A" * logo.L)

    def test_invalid_positions_and_mirror_raise(self):
        for positions in (np.arange(3), np.zeros((2, 2, 2)), ["a"] * 8, [np.nan] * 8):
            with self.assertRaises(ValueError):
                self._logo(positions=positions)
        with self.assertRaises(ValueError):
            self._logo(mirror_glyphs=[True, False])

    def test_variability_logo_rejects_per_logo_axes(self):
        logo = self._logo(self.values[:2], positions=np.arange(16).reshape(2, 8))
        logo.process_all()
        with self.assertRaises(ValueError):
            logo.draw_variability_logo(apply_layout=False)

    def test_dont_stretch_more_than_sets_glyph_width(self):
        values = np.abs(self.values[:1])
        default = self._logo(values).process_all()
        explicit_m = self._logo(values, dont_stretch_more_than="M").process_all()
        for a, b in zip(_vertices(default, 0), _vertices(explicit_m, 0)):
            np.testing.assert_allclose(a, b)

        # Every ACGT glyph is wider than 'E', so each fills the full width.
        e_cap = self._logo(
            values, font_weight="bold", width=0.95, dont_stretch_more_than="E"
        ).process_all()
        for glyph in e_cap.processed_logos[0]["glyphs"]:
            self.assertAlmostEqual(glyph["path"].get_extents().width, 0.95)

        for bad in ("", "EM", 1):
            with self.assertRaises(ValueError):
                self._logo(dont_stretch_more_than=bad)

    def test_unknown_color_scheme_raises(self):
        with self.assertRaises(ValueError):
            self._logo(color_scheme="not-a-scheme")

    def test_get_color_dict_accepts_an_rgb_triple(self):
        color_dict = get_color_dict([0.2, 0.4, 0.6], "AC")
        np.testing.assert_allclose(color_dict["A"], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(color_dict["C"], [0.2, 0.4, 0.6])

    def test_contribution_and_centering_still_apply(self):
        contributed = self._logo(
            np.ones((1, 4, 4)), sequences=["ACGT"], contribution=True
        )
        self.assertEqual(contributed.values[0, 0, 0], 1)
        self.assertEqual(contributed.values[0, 0, 1], 0)
        centered = self._logo(self.values[:1], center_values=True)
        np.testing.assert_allclose(centered.values.mean(axis=-1), 0, atol=1e-8)
        with self.assertRaises(ValueError):
            self._logo(np.ones((2, 4, 4)), sequences=["ACGT", "ACG"], contribution=True)


if __name__ == "__main__":
    unittest.main()
