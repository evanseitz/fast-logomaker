# Fast Logomaker

![PyPI version](https://img.shields.io/pypi/v/fast-logomaker)
![License: MIT](https://img.shields.io/pypi/l/fast-logomaker)
![Python](https://img.shields.io/pypi/pyversions/fast-logomaker)

An optimized batch logo generation library for sequence logos. This is an optimized version of the original [Logomaker](https://github.com/jbkinney/logomaker) package by Ammar Tareen and Justin Kinney.

## Installation

```bash
pip install fast-logomaker
```

## Features

- **Efficient batch processing** of multiple logos
- **Path caching** for improved performance
- **Minimized object creation**
- **Optimized transformation operations**

## Performance

| Implementation | Speed |
|----------------|-------|
| Original Logomaker | ~2 seconds per logo (249 × 4 matrix) |
| Fast Logomaker | ~0.013 seconds per logo (1000 logos in 13 seconds) |

## Quick Start

```python
import numpy as np
from fast_logomaker import FastLogo

# Create sample data (N logos, L positions, 4 nucleotides)
values = np.random.randn(10, 50, 4)  # 10 logos, 50 positions, ACGT

# Create FastLogo processor
logo = FastLogo(values)

# Process all logos
logo.process_all()

# Draw a single logo
fig, ax = logo.draw_single(0)

# Draw multiple logos in a grid
fig, axes = logo.draw_logos(indices=[0, 1, 2], rows=1, cols=3)
```

## API Reference

### FastLogo

```python
from fast_logomaker import FastLogo

FastLogo(
    values,                    # numpy array of shape (N, L, alphabet_size)
    alphabet=None,             # list of characters, default ['A', 'C', 'G', 'T']
    figsize=[10, 2.5],         # figure size for single logos
    batch_size=50,             # batch size for processing
    font_name='sans',          # font family (e.g., 'Arial Rounded MT Bold')
    color_scheme='classic',    # color scheme name or dict
    y_min_max=None,            # fixed y-axis limits (min, max)
    show_progress=True,        # show progress bar during processing
    center_values=False,       # center values at each position
    fade_below=0.5,            # alpha fade for negative values
    shade_below=0.5,           # color shade for negative values
    fade_above=0,              # alpha fade for positive values
    shade_above=0,             # color shade for positive values
    width=0.9,                 # character width
    positions=None,            # x-axis coordinates for each of the L positions 
    mirror_glyphs=False,       # pre-mirror glyphs, to cancel a later axis inversion
    **kwargs
)
```

### Methods

#### `process_all()`
Process all logos in batches. Must be called before drawing.

```python
logo.process_all()
```

#### `draw_single(idx, ...)`
Draw a single logo with optional highlighting and view window.

```python
fig, ax = logo.draw_single(
    idx,                           # logo index to draw
    fixed_ylim=True,               # use consistent y-limits across logos
    view_window=None,              # [start, end] positions to zoom into
    figsize=None,                  # override figure size
    border=True,                   # show axis border
    highlight_ranges=None,         # list of (start, end) tuples or position lists
    highlight_colors=None,         # colors for highlights
    highlight_alpha=0.5,           # transparency for highlights
    ax=None,                       # existing ax to draw on
    apply_layout=True              # whether to call plt.tight_layout() after drawing
)
```

#### `draw_logos(indices, rows, cols)`
Draw multiple logos in a grid layout.

```python
fig, axes = logo.draw_logos(
    indices=None,    # list of indices, or None for all
    rows=None,       # number of rows (auto if None)
    cols=None        # number of columns (auto if None)
)
```

#### `draw_variability_logo(...)`
Draw a variability logo showing all glyphs from all logos overlaid at each position.

```python
fig, ax = logo.draw_variability_logo(
    view_window=None,    # [start, end] positions to view
    figsize=None,        # figure size
    border=True          # show axis border
)
```

#### `style_glyphs_in_sequence(sequence, color)`
Style glyphs that match a reference sequence in a specified color, with non-matching glyphs in dark gray.

```python
logo.style_glyphs_in_sequence(
    sequence='ACGTACGT',      # reference sequence string
    color='darkorange'        # color for matching glyphs
)
```

## Examples

### Attribution Logo with Centered Values

```python
from fast_logomaker import FastLogo

# Attribution values (can be positive and negative)
attributions = np.random.randn(1, 100, 4)

logo = FastLogo(
    attributions,
    alphabet=['A', 'C', 'G', 'T'],
    font_name='Arial Rounded MT Bold',
    fade_below=0.5,
    shade_below=0.5,
    width=0.9,
    figsize=[20, 2.5],
    center_values=True,
    batch_size=1
)

logo.process_all()
fig, ax = logo.draw_single(0, border=False)
```

### View Window and Highlighting

```python
# Zoom into positions 50-100 and highlight specific regions
fig, ax = logo.draw_single(
    0,
    view_window=[50, 100],
    highlight_ranges=[(60, 70), (80, 90)],
    highlight_colors=['lightcyan', 'honeydew'],
    highlight_alpha=0.5
)
```

### Fixed Y-Axis Limits Across Multiple Logos

```python
# Compute global y-limits from all logos
logo.process_all()

# Set consistent y-limits for comparison
logo.y_min_max = (-2, 2)

# Draw logos with fixed y-axis
for i in range(logo.N):
    fig, ax = logo.draw_single(i, fixed_ylim=True)
    fig.savefig(f'logo_{i}.png')
```

### Variability Logo

```python
# Show overlap of multiple logos at each position
fig, ax = logo.draw_variability_logo(
    figsize=(20, 2.5),
    view_window=[50, 150]
)
```

## Positioning & Layout

### Custom X-Axis Positions

By default, glyphs are placed at integer positions `0..L-1`. Pass `positions` to place them at arbitrary x-coordinates instead, like the coordinates of your genomic region:

```python
logo = FastLogo(values, positions=np.arange(50, 100))
logo.process_all()
fig, ax = logo.draw_single(0)
```

#### Negative-strand coordinates

Use descending x coordinates and `xaxis.set_inverted(True)` to represent negative-strand regions, and ensure the glyphs stay facing the right way with `mirror_glyphs`:

```python
logo = FastLogo(values, positions=np.arange(100, 50, -1), mirror_glyphs=True)
logo.process_all()
fig, ax = logo.draw_single(0)
ax.xaxis.set_inverted(True)  # glyphs now render forward-facing, still in the flipped order
```

### Batching Logos with Different Coordinates/Strands

`positions` and `mirror_glyphs` also accept per-logo forms, so logos with entirely different x-axes/strands can still be batched into a single `FastLogo` call by passing `positions` as shape `(N, L)` and mirror_glyphs as `(N,)`:

```python
positions = np.stack([
    np.arange(1000, 1000 + L),       # logo 0: ascending (+ strand)
    np.arange(5000 + L, 5000, -1),   # logo 1: descending (- strand)
])
mirror_glyphs = [False, True]

logo = FastLogo(values, positions=positions, mirror_glyphs=mirror_glyphs)
logo.process_all()
fig, ax = logo.draw_single(1)
ax.xaxis.set_inverted(True)  # only needed for the logos you draw with descending positions
```

Note: `draw_variability_logo()` does not support per-logo `positions`/`mirror_glyphs` (it overlays all logos at shared positions).

### Embedding in an Existing Figure Layout

When drawing onto an axes that belongs to a figure using a layout manager other than tight (e.g. `constrained`), pass `apply_layout=False` to `draw_single()` to skip its `plt.tight_layout()` call and avoid disturbing that layout:

```python
fig, ax = plt.subplots(layout="constrained")
logo.draw_single(0, ax=ax, apply_layout=False)
```

## Color Schemes

Built-in color schemes for DNA/RNA:
- `classic` - Standard DNA colors (A=green, C=blue, G=orange, T=red)
- `grays` - Grayscale
- `colorblind_safe` - Colorblind-friendly palette
- `base_pairing` - Colors by base pairing

Built-in color schemes for proteins:
- `weblogo_protein` - WebLogo protein colors
- `skylign_protein` - Skylign protein colors  
- `dmslogo_charge` - Charge-based coloring
- `dmslogo_funcgroup` - Functional group coloring
- `hydrophobicity` - Hydrophobicity-based coloring
- `chemistry` - Chemistry-based coloring
- `charge` - Charge-based coloring
- `NajafabadiEtAl2017` - From Najafabadi et al. 2017

### List Available Color Schemes

```python
from fast_logomaker import list_color_schemes

schemes = list_color_schemes()
print(schemes)
```

### Custom Color Scheme

```python
custom_colors = {
    'A': '#FF0000',  # red
    'C': '#00FF00',  # green
    'G': '#0000FF',  # blue
    'T': '#FFFF00'   # yellow
}

logo = FastLogo(values, color_scheme=custom_colors)
```

## Migration from seam.logomaker_batch

If you were using `BatchLogo` from seam, you can use `FastLogo` (recommended) or keep using `BatchLogo` as an alias:

```python
# Before
from seam.logomaker_batch.batch_logo import BatchLogo

# After (recommended)
from fast_logomaker import FastLogo

# After (backwards compatible alias)
from fast_logomaker import BatchLogo
```

All parameters and methods remain the same.

## Credits

- Original Logomaker: Ammar Tareen and Justin Kinney, 2019-2024
- Batch processing optimization: Evan Seitz, 2025

For the original package and documentation, visit: https://github.com/jbkinney/logomaker

## License

MIT License
