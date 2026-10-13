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
    font_weight='normal',      # font weight (e.g., 'bold')
    color_scheme='classic',    # color scheme name or dict
    y_min_max=None,            # fixed y-axis limits (min, max)
    show_progress=True,        # show progress bar during processing
    center_values=False,       # center values at each position
    fade_below=0.5,            # alpha fade for negative values
    shade_below=0.5,           # color shade for negative values
    fade_above=0,              # alpha fade for positive values
    shade_above=0,             # color shade for positive values
    width=0.9,                 # character width
    positions=None,            # x coordinates for each column; default 0..L-1
    mirror_glyphs=False,       # pre-mirror glyphs so an inverted x-axis stays readable
    dont_stretch_more_than='M',  # glyphs never stretch wider than this character would
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
    ax=None,                       # existing axes to draw on
    apply_layout=True              # set False to skip plt.tight_layout()
)
```

#### `draw_logos(indices, rows, cols)`
Draw multiple logos in a grid layout.

```python
fig, axes = logo.draw_logos(
    indices=None,       # list of indices, or None for all
    rows=None,          # number of rows (auto if None)
    cols=None,          # number of columns (auto if None)
    apply_layout=True   # set False to skip plt.tight_layout()
)
```

#### `draw_variability_logo(...)`
Draw a variability logo showing all glyphs from all logos overlaid at each position.

```python
fig, ax = logo.draw_variability_logo(
    view_window=None,    # [start, end] x coordinates to view
    figsize=None,        # figure size
    border=True,         # show axis border
    apply_layout=True    # set False to skip plt.tight_layout()
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

### Matching Logomaker's Style

`FastLogo`'s defaults draw narrower, lighter glyphs than Logomaker and fade glyphs
below the axis. These settings reproduce Logomaker's defaults:

```python
logo = FastLogo(
    values,
    font_weight='bold',
    width=0.95,
    fade_below=0,
    shade_below=0,
    dont_stretch_more_than='E',
)
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

## Positioning and layout

Glyphs are placed at `0 .. L-1` unless you pass `positions`. Each glyph is `width` data-units wide (default 0.9) and centered on its coordinate, so the coordinates are normally spaced 1 apart, as genomic base positions are.

```python
logo = FastLogo(values, positions=np.arange(1000, 1050))
logo.process_all()
fig, ax = logo.draw_single(0)
```

A minus-strand logo usually sits under the plus strand. Give it descending coordinates and mirror the glyphs, then invert only that axis. The letters stay forward-facing while the coordinates decrease to the right:

```python
import matplotlib.pyplot as plt

# + strand 5' ATGC 3' at 1000..1003. Minus strand 5' to 3' is GCAT at 1003..1000.
def onehot(sequence):
    index = {base: i for i, base in enumerate("ACGT")}
    encoded = np.zeros((len(sequence), 4))
    for position, base in enumerate(sequence):
        encoded[position, index[base]] = 1
    return encoded

values = np.stack([onehot("ATGC"), onehot("GCAT")])
positions = np.stack([
    np.arange(1000, 1004),
    np.arange(1003, 999, -1),
])
logo = FastLogo(values, positions=positions, mirror_glyphs=[False, True])
logo.process_all()

fig, axes = plt.subplots(2, 1, layout="constrained")
logo.draw_single(0, ax=axes[0], apply_layout=False)
logo.draw_single(1, ax=axes[1], apply_layout=False)
axes[1].xaxis.set_inverted(True)
```

`positions` may also have shape `(N, L)`, and `mirror_glyphs` may be a length-`N` list, when logos in one batch use different coordinates. `draw_variability_logo()` overlays every logo on one shared axis, so it accepts only the shared forms.

`highlight_ranges` and `view_window` use these same x coordinates.

`draw_single()`, `draw_logos()`, and `draw_variability_logo()` call `tight_layout()` on the figure being drawn. That replaces a `constrained` layout. Pass `apply_layout=False` when the logo should stay inside the axes you provide:

```python
import matplotlib.pyplot as plt

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
