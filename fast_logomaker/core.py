"""
Batch Logo Generation Module

This module provides an optimized version of Logomaker with faster logo generation
and batch processing capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from .colors import get_rgb, COLOR_SCHEME_DICT
from tqdm import tqdm
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb


class BatchLogo:
    """Optimized batch logo processor for sequence logos.

    Provides efficient batch processing of multiple sequence logos
    with path caching and optimized transformations. Supports both
    hypothetical (default) and contribution display modes.
    """

    def __init__(self, values, alphabet=None, figsize=[10, 2.5], batch_size=50,
                 font_name='sans', y_min_max=None, show_progress=True,
                 sequences=None, contribution=False, positions=None,
                 mirror_glyphs=False, **kwargs):
        """Initialize BatchLogo processor.

        Parameters
        ----------
        values : array-like
            Attribution values of shape (N, L, A).
        alphabet : list, optional
            List of characters in the alphabet. Default is ['A', 'C', 'G', 'T'].
        figsize : list, optional
            Figure size for single logos. Default is [10, 2.5].
        batch_size : int, optional
            Batch size for processing. Default is 50.
        font_name : str, optional
            Font family name. Default is 'sans'.
        y_min_max : tuple, optional
            Fixed y-axis limits (min, max). Default is None.
        show_progress : bool, optional
            Whether to show progress bar. Default is True.
        sequences : list of str or numpy.ndarray, optional
            Sequences corresponding to each logo. Required when contribution=True.
            If strings, converted to one-hot internally using alphabet.
            If array, must be one-hot encoded with shape (N, L, A).
        contribution : bool, optional
            If True, multiply attribution values by sequence one-hot encodings
            to show only the contributions of observed nucleotides (contribution
            mode). When enabled, center_values is forced to False. Default is
            False.
        positions : array-like, optional
            X-axis coordinates for each of the L positions. Default is None, which uses `range(L)`.
            May be descending (e.g. to represent a negative-strand region).
            Either shape (L,) (shared across every logo in the batch) or shape
            (N, L) (separate positions per logo).
        mirror_glyphs : bool or array-like, optional
            If True, pre-mirrors each glyph horizontally around its own center
            before placing it. Default is False. Either a single bool (shared 
            across every logo in the batch) or an array-like of N bools (one per logo).
            Useful if plotting a negative-strand logo when combined with `ax.xaxis.set_inverted(True)`.            
        **kwargs : dict
            Additional keyword arguments.
        """
        # Initialize instance caches
        self._path_cache = {}
        self._m_path_cache = {}
        self._font_cache = {}

        # Apply contribution mode: mask values by sequence one-hot
        if contribution:
            if sequences is None:
                raise ValueError("sequences must be provided when contribution=True")
            values = np.array(values)
            oh = self._sequences_to_onehot(sequences, alphabet or ['A', 'C', 'G', 'T'])
            values = values * oh
            kwargs['center_values'] = False

        # Handle centering if requested
        center_values = kwargs.pop('center_values', False)
        if center_values:
            values = self._center_matrix(values)

        self.values = np.array(values)
        self.alphabet = alphabet if alphabet is not None else ['A', 'C', 'G', 'T']
        self.batch_size = batch_size

        self.N = self.values.shape[0]  # number of logos
        self.L = self.values.shape[1]  # length of each logo

        # Check shape and set positions
        if positions is None:
            self.positions = np.arange(self.L)
            self._positions_per_logo = False
        else:
            self.positions = np.asarray(positions)
            if self.positions.ndim == 1:
                if len(self.positions) != self.L:
                    raise ValueError(
                        f"positions has length {len(self.positions)}, must match L ({self.L})"
                    )
                self._positions_per_logo = False
            elif self.positions.ndim == 2:
                if self.positions.shape != (self.N, self.L):
                    raise ValueError(
                        f"positions has shape {self.positions.shape}, must be "
                        f"({self.N}, {self.L}) for per-logo positions"
                    )
                self._positions_per_logo = True
            else:
                raise ValueError(
                    f"positions must be 1-D (shared) or 2-D (per-logo), got ndim={self.positions.ndim}"
                )

        # Check shape and set mirror_glyphs
        if isinstance(mirror_glyphs, (bool, np.bool_)):
            self.mirror_glyphs = mirror_glyphs
            self._mirror_per_logo = False
        else:
            mirror_arr = np.asarray(mirror_glyphs)
            if mirror_arr.ndim == 0:
                self.mirror_glyphs = bool(mirror_arr)
                self._mirror_per_logo = False
            else:
                if mirror_arr.shape != (self.N,):
                    raise ValueError(
                        f"mirror_glyphs has shape {mirror_arr.shape}, must be scalar "
                        f"or ({self.N},) for per-logo mirroring"
                    )
                self.mirror_glyphs = mirror_arr
                self._mirror_per_logo = True

        self.kwargs = self._get_default_kwargs()
        self.kwargs.update(kwargs)

        # Initialize storage for processed logos
        self.processed_logos = {}

        # Set figure size
        self.figsize = figsize

        # Get font name and weight
        self.font_name = font_name
        self.font_weight = self.kwargs.pop('font_weight', 'normal')

        # Get stack order
        self.stack_order = self.kwargs.pop('stack_order', 'big_on_top')

        # Get color scheme
        color_scheme = self.kwargs.pop('color_scheme', 'classic')

        # Initialize rgb_dict
        self.rgb_dict = {}

        # Handle color scheme
        if isinstance(color_scheme, dict):
            for char in self.alphabet:
                self.rgb_dict[char] = get_rgb(color_scheme.get(char, 'gray'))
        else:
            colors = COLOR_SCHEME_DICT[color_scheme]
            for char in self.alphabet:
                if char in colors:
                    self.rgb_dict[char] = get_rgb(colors[char])
                elif char == 'T' and 'TU' in colors:
                    self.rgb_dict[char] = get_rgb(colors['TU'])
                else:
                    self.rgb_dict[char] = get_rgb('gray')

        self.y_min_max = y_min_max
        self.show_progress = show_progress

    def _get_font_props(self):
        """Get cached font properties with fallback to sans if font not found."""
        cache_key = f"{self.font_name}_{self.font_weight}"
        if cache_key not in self._font_cache:
            try:
                self._font_cache[cache_key] = fm.FontProperties(
                    family=self.font_name, weight=self.font_weight
                )
            except:
                print(f"Warning: Font '{self.font_name}' with weight "
                      f"'{self.font_weight}' not found, falling back to 'sans'")
                self._font_cache[cache_key] = fm.FontProperties(family='sans')
        return self._font_cache[cache_key]

    def process_all(self):
        """
        Process all logos in batches.
        
        Returns
        -------
        self : BatchLogo
            Returns self for method chaining.
        """
        if self.show_progress:
            with tqdm(total=self.N, desc="Processing logos") as pbar:
                for start_idx in range(0, self.N, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, self.N)
                    self._process_batch(start_idx, end_idx)
                    pbar.update(end_idx - start_idx)
        else:
            for start_idx in range(0, self.N, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.N)
                self._process_batch(start_idx, end_idx)
        return self

    def _process_batch(self, start_idx, end_idx):
        """Process a batch of logos."""
        font_props = self._get_font_props()

        # Cache M path first (for width reference)
        if not self._m_path_cache:
            m_path = TextPath((0, 0), 'M', size=1, prop=font_props)
            m_extents = m_path.get_extents()
            self._m_path_cache = {
                'path': m_path,
                'extents': m_extents,
                'width': m_extents.width,
            }

            # Then cache alphabet paths
            for char in self.alphabet:
                if char not in self._path_cache:
                    base_path = TextPath((0, 0), char, size=1, prop=font_props)
                    flipped_path = TextPath((0, 0), char, size=1, prop=font_props)
                    flipped_path = flipped_path.transformed(Affine2D().scale(1, -1))
                    self._path_cache[char] = {
                        'normal': {'path': base_path, 'extents': base_path.get_extents()},
                        'flipped': {'path': flipped_path, 'extents': flipped_path.get_extents()}
                    }

        for idx in range(start_idx, end_idx):
            glyph_data = []
            positions_for_idx = self.positions[idx] if self._positions_per_logo else self.positions
            mirror_for_idx = self.mirror_glyphs[idx] if self._mirror_per_logo else self.mirror_glyphs

            for i, pos in enumerate(positions_for_idx):
                values = self.values[idx, i]
                ordered_indices = self._get_ordered_indices(values)
                values = values[ordered_indices]
                chars = [str(self.alphabet[i]) for i in ordered_indices]

                # Calculate total negative height first
                neg_values = values[values < 0]
                total_neg_height = abs(sum(neg_values)) + (len(neg_values) - 1) * self.kwargs['vsep']

                # Handle positive values (stack up from 0)
                floor = self.kwargs['vsep'] / 2.0
                for value, char in zip(values, chars):
                    if value > 0:
                        ceiling = floor + value

                        path_data = self._path_cache[char]['normal']
                        transformed_path = self._get_transformed_path(
                            path_data, pos, floor, ceiling,
                            self._m_path_cache['extents'].width,
                            mirror=mirror_for_idx
                        )

                        # Apply fade_above and shade_above for positive values (glyphs above x-axis)
                        alpha = self.kwargs['alpha']
                        color = self.rgb_dict[char]
                        fade_above = self.kwargs.get('fade_above', 0)
                        shade_above = self.kwargs.get('shade_above', 0)
                        
                        if fade_above > 0:
                            alpha *= (1 - fade_above)
                        if shade_above > 0:
                            color = tuple(c * (1 - shade_above) for c in self.rgb_dict[char])

                        glyph_data.append({
                            'path': transformed_path,
                            'color': color,
                            'edgecolor': 'none',
                            'edgewidth': 0,
                            'alpha': alpha,
                            'floor': floor,
                            'ceiling': ceiling,
                            'char': char,
                            'pos': pos
                        })
                        floor = ceiling + self.kwargs['vsep']

                # Handle negative values (stack down from -total_height)
                if len(neg_values) > 0:
                    floor = -total_neg_height - self.kwargs['vsep'] / 2.0
                    for value, char in zip(values, chars):
                        if value < 0:
                            ceiling = floor + abs(value)

                            path_data = self._path_cache[char][
                                'flipped' if self.kwargs['flip_below'] else 'normal'
                            ]
                            transformed_path = self._get_transformed_path(
                                path_data, pos, floor, ceiling,
                                self._m_path_cache['extents'].width,
                                mirror=mirror_for_idx
                            )

                            # Apply fade and shade effects for negative values
                            alpha = self.kwargs['alpha']
                            alpha *= (1 - self.kwargs['fade_below'])
                            if self.kwargs['shade_below'] > 0:
                                color = tuple(
                                    c * (1 - self.kwargs['shade_below'])
                                    for c in self.rgb_dict[char]
                                )
                            else:
                                color = self.rgb_dict[char]

                            glyph_data.append({
                                'path': transformed_path,
                                'color': color,
                                'edgecolor': 'none',
                                'edgewidth': 0,
                                'alpha': alpha,
                                'floor': floor,
                                'ceiling': ceiling,
                                'char': char,
                                'pos': pos
                            })
                            floor = ceiling + self.kwargs['vsep']

            self.processed_logos[idx] = {'glyphs': glyph_data}

    def draw_logos(self, indices=None, rows=None, cols=None):
        """
        Draw specific logos in a grid layout.
        
        Parameters
        ----------
        indices : list, optional
            Indices of logos to draw. If None, draws all logos.
        rows : int, optional
            Number of rows in the grid. Auto-determined if None.
        cols : int, optional
            Number of columns in the grid. Auto-determined if None.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : numpy.ndarray
            Array of axes objects.
        """
        if indices is None:
            indices = list(range(self.N))

        N = len(indices)

        # Determine grid layout
        if rows is None and cols is None:
            cols = min(5, N)
            rows = (N + cols - 1) // cols
        elif rows is None:
            rows = (N + cols - 1) // cols
        elif cols is None:
            cols = (N + rows - 1) // rows

        # Create figure with subplots
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(self.figsize[0] * cols, self.figsize[1] * rows),
            squeeze=False
        )

        # Draw requested logos
        for i, idx in enumerate(indices):
            if idx not in self.processed_logos:
                raise ValueError(
                    f"Logo {idx} has not been processed yet. Run process_all() first."
                )

            row = i // cols
            col = i % cols
            ax = axes[row, col]

            logo_data = self.processed_logos[idx]
            self._draw_single_logo(ax, logo_data)

        # Turn off empty subplots
        for i in range(N, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        return fig, axes

    def draw_single(self, idx, fixed_ylim=True, view_window=None, figsize=None,
                    highlight_ranges=None, highlight_colors=None, highlight_alpha=0.5,
                    border=True, baseline=True, ax=None, apply_layout=True):
        """
        Draw a single logo.

        Parameters
        ----------
        idx : int
            Index of logo to draw.
        fixed_ylim : bool, optional
            Whether to use same y-axis limits across all logos. Default is True.
        view_window : list or tuple, optional
            [start, end] positions to view. If None, show entire logo.
        figsize : tuple, optional
            Figure size in inches. If None, use size from initialization.
        highlight_ranges : list of tuple/list, optional
            Either [(start, stop), ...] for continuous ranges
            or [[pos1, pos2, pos3, ...], ...] for specific positions.
        highlight_colors : list of str or str, optional
            Colors for highlighting. Default uses plt.cm.Pastel1.
        highlight_alpha : float, optional
            Alpha transparency for highlights. Default is 0.5.
        border : bool, optional
            Whether to show the axis spines. Default is True.
        baseline : bool, optional
            Whether to draw the horizontal y=0 baseline. Default is True.
        ax : matplotlib.axes.Axes, optional
            If provided, draw the logo on this axes.
        apply_layout : bool, optional
            Whether to call `plt.tight_layout()` after drawing. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object (None if ax was provided).
        ax : matplotlib.axes.Axes
            The axes object.
        """
        if idx not in self.processed_logos:
            raise ValueError(
                f"Logo {idx} has not been processed yet. Run process_all() first."
            )
        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize if figsize is not None else self.figsize
            )
            own_fig = True
        else:
            fig = None
        self._draw_single_logo(
            ax, self.processed_logos[idx], fixed_ylim=fixed_ylim, border=border,
            baseline=baseline, idx=idx
        )
        
        # Add highlighting if specified
        if highlight_ranges is not None:
            if isinstance(highlight_ranges[0], (int, float)):
                highlight_ranges = [highlight_ranges]
            if highlight_colors is None:
                n_ranges = len(highlight_ranges)
                highlight_colors = [plt.cm.Pastel1(i % 9) for i in range(n_ranges)]
            elif isinstance(highlight_colors, str):
                highlight_colors = [highlight_colors]
            for positions, color in zip(highlight_ranges, highlight_colors):
                if len(positions) == 2 and isinstance(positions, tuple):
                    start, end = positions
                    ax.axvspan(start - 0.5, end - 0.5, color=color,
                               alpha=highlight_alpha, zorder=-1)
                else:
                    positions = sorted(positions)
                    start = positions[0]
                    prev = start
                    for curr in positions[1:] + [None]:
                        if curr != prev + 1:
                            end = prev
                            if start == end:
                                ax.axvspan(start - 0.5, start + 0.5, color=color,
                                           alpha=highlight_alpha, zorder=-1)
                            else:
                                ax.axvspan(start - 0.5, end + 0.5, color=color,
                                           alpha=highlight_alpha, zorder=-1)
                            start = curr
                        prev = curr
                        
        # Apply view window last
        if view_window is not None:
            start, end = view_window
            ax.set_xlim(start - 0.5, end - 0.5)
        if apply_layout:
            plt.tight_layout()
        if own_fig:
            return fig, ax
        else:
            return None, ax

    def _draw_single_logo(self, ax, logo_data, fixed_ylim=True, border=True, baseline=True, idx=None):
        """
        Draw a single logo on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        logo_data : dict
            Logo data containing glyphs.
        fixed_ylim : bool, optional
            Whether to use same y-axis limits across all logos. Default is True.
        border : bool, optional
            Whether to show the axis spines. Default is True.
        baseline : bool, optional
            Whether to draw the horizontal y=0 baseline. Default is True.
        idx : int, optional
            Index of the logo being drawn, used to look up its own positions
            when `positions` is per-logo (shape (N, L)). Ignored otherwise.
        """
        patches = []
        for glyph_data in logo_data['glyphs']:
            patch = PathPatch(
                glyph_data['path'],
                facecolor=glyph_data['color'],
                edgecolor=glyph_data['edgecolor'],
                linewidth=glyph_data['edgewidth'],
                alpha=glyph_data['alpha']
            )
            patches.append(patch)

        ax.add_collection(PatchCollection(patches, match_original=True))

        logo_positions = self.positions[idx] if (idx is not None and self._positions_per_logo) else self.positions
        ax.set_xlim(logo_positions.min() - 0.5, logo_positions.max() + 0.5)

        if fixed_ylim and self.y_min_max is not None:
            ax.set_ylim(self.y_min_max[0], self.y_min_max[1])
        else:
            floors = [g['floor'] for g in logo_data['glyphs']]
            ceilings = [g['ceiling'] for g in logo_data['glyphs']]
            ymin = min(floors) if floors else 0
            ymax = max(ceilings) if ceilings else 1
            ymin = min(ymin, 0)
            ax.set_ylim(ymin, ymax)

        if baseline and self.kwargs['baseline_width'] > 0:
            ax.axhline(
                y=0, color='black',
                linewidth=self.kwargs['baseline_width'],
                zorder=-1
            )

        for spine in ax.spines.values():
            spine.set_visible(border)

    def _get_default_kwargs(self):
        """Get default parameters for logo creation."""
        return {
            'baseline_width': 0.5,
            'vsep': 0.0,
            'alpha': 1.0,
            'vpad': 0.0,
            'width': 0.9,
            'flip_below': True,
            'color_scheme': 'classic',
            'fade_below': 0.5,
            'shade_below': 0.5,
            'fade_above': 0,
            'shade_above': 0,
        }

    def _get_ordered_indices(self, values):
        """Get indices ordered according to stack_order."""
        if self.stack_order == 'big_on_top':
            return np.argsort(values)
        elif self.stack_order == 'small_on_top':
            tmp_vs = np.zeros(len(values))
            indices = (values != 0)
            tmp_vs[indices] = 1.0 / values[indices]
            return np.argsort(tmp_vs)
        else:  # fixed
            return np.array(range(len(values)))[::-1]

    def _get_transformed_path(self, path_data, pos, floor, ceiling, m_width, mirror=None):
        """Get transformed path with proper scaling and position."""
        base_path = path_data['path']
        base_extents = path_data['extents']

        bbox_width = self.kwargs['width'] - 2 * self.kwargs['vpad']
        hstretch_char = bbox_width / base_extents.width
        hstretch_m = bbox_width / m_width
        hstretch = min(hstretch_char, hstretch_m)

        char_width = hstretch * base_extents.width
        char_shift = (bbox_width - char_width) / 2.0

        vstretch = (ceiling - floor) / base_extents.height

        if mirror is None:
            mirror = self.mirror_glyphs

        transform = Affine2D()
        transform.translate(tx=-base_extents.xmin, ty=-base_extents.ymin)
        transform.scale(hstretch, vstretch)
        if mirror:
            # Pre-mirror the glyph around its own center. 
            # Useful if reversing the x-axis later with ax.xaxis.set_inverted(True),
            # to keep the glyphs facing the right way even as the overall plot is inverted.
            transform.scale(-1, 1)
            transform.translate(tx=char_width, ty=0)
        transform.translate(
            tx=pos - bbox_width / 2.0 + self.kwargs['vpad'] + char_shift,
            ty=floor
        )

        final_path = transform.transform_path(base_path)
        return final_path

    def _center_matrix(self, values):
        """Center the values in each position (row) of the matrix."""
        return values - values.mean(axis=-1, keepdims=True)

    @staticmethod
    def _sequences_to_onehot(sequences, alphabet):
        """Convert sequences to one-hot encoding array.

        Accepts either a list of strings or a pre-encoded numpy array of
        shape (N, L, A). String sequences are converted using the provided
        alphabet.
        """
        arr = np.asarray(sequences)
        if arr.ndim == 3:
            return arr.astype(np.float64)
        char_to_idx = {c: i for i, c in enumerate(alphabet)}
        A = len(alphabet)
        N = len(sequences)
        L = len(sequences[0])
        oh = np.zeros((N, L, A), dtype=np.float64)
        for n, seq in enumerate(sequences):
            for pos, char in enumerate(seq):
                idx = char_to_idx.get(char)
                if idx is not None:
                    oh[n, pos, idx] = 1.0
        return oh

    def draw_variability_logo(self, view_window=None, figsize=None, border=True):
        """
        Draw a variability logo showing all glyphs from all clusters overlaid.
        
        Parameters
        ----------
        view_window : list or tuple, optional
            [start, end] positions to view. If None, show entire logo.
        figsize : tuple, optional
            Figure size in inches. If None, use size from initialization.
        border : bool, optional
            Whether to show the axis spines. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        if self._positions_per_logo or self._mirror_per_logo:
            raise ValueError(
                "draw_variability_logo does not support per-logo positions/mirror_glyphs."
            )
        logo_data = {'glyphs': []}

        for i, pos in enumerate(self.positions):
            for cluster_idx in range(self.values.shape[0]):
                values = self.values[cluster_idx, i]
                ordered_indices = self._get_ordered_indices(values)
                values = values[ordered_indices]
                chars = [str(self.alphabet[i]) for i in ordered_indices]

                neg_values = values[values < 0]
                total_neg_height = abs(sum(neg_values)) + (len(neg_values) - 1) * self.kwargs['vsep']

                floor = self.kwargs['vsep'] / 2.0
                for value, char in zip(values, chars):
                    if value > 0:
                        ceiling = floor + value

                        path_data = self._path_cache[char]['normal']
                        transformed_path = self._get_transformed_path(
                            path_data, pos, floor, ceiling,
                            self._m_path_cache['extents'].width
                        )

                        logo_data['glyphs'].append({
                            'path': transformed_path,
                            'color': self.rgb_dict[char],
                            'edgecolor': 'none',
                            'edgewidth': 0,
                            'alpha': 1,
                            'floor': floor,
                            'ceiling': ceiling
                        })
                        floor = ceiling + self.kwargs['vsep']

                if len(neg_values) > 0:
                    floor = -total_neg_height - self.kwargs['vsep'] / 2.0
                    for value, char in zip(values, chars):
                        if value < 0:
                            ceiling = floor + abs(value)

                            path_data = self._path_cache[char][
                                'flipped' if self.kwargs['flip_below'] else 'normal'
                            ]
                            transformed_path = self._get_transformed_path(
                                path_data, pos, floor, ceiling,
                                self._m_path_cache['extents'].width
                            )

                            logo_data['glyphs'].append({
                                'path': transformed_path,
                                'color': self.rgb_dict[char],
                                'edgecolor': 'none',
                                'edgewidth': 0,
                                'alpha': 1,
                                'floor': floor,
                                'ceiling': ceiling
                            })
                            floor = ceiling + self.kwargs['vsep']

        fig, ax = plt.subplots(
            figsize=figsize if figsize is not None else self.figsize
        )
        self._draw_single_logo(ax, logo_data, fixed_ylim=True, border=border)

        if view_window is not None:
            start, end = view_window
            ax.set_xlim(start - 0.5, end - 0.5)

        plt.tight_layout()
        return fig, ax

    def style_glyphs_in_sequence(self, sequence, color='darkorange'):
        """
        Style glyphs that match the reference sequence.
        
        Parameters
        ----------
        sequence : str
            Reference sequence to match.
        color : str, optional
            Color for matching glyphs. Default is 'darkorange'.
        """
        if not isinstance(sequence, str):
            raise TypeError('sequence must be a string')
        if len(sequence) != self.L:
            raise ValueError(
                f'sequence length {len(sequence)} must match logo length {self.L}'
            )
        ref_rgb = to_rgb(color)
        dark_gray = (0.4, 0.4, 0.4)
        for pos in range(self.L):
            ref_char = sequence[pos]
            for logo_idx in self.processed_logos:
                for glyph in self.processed_logos[logo_idx]['glyphs']:
                    if glyph['pos'] == pos:
                        glyph['alpha'] = 1.0
                        if glyph['char'] == ref_char:
                            glyph['color'] = ref_rgb
                        else:
                            glyph['color'] = dark_gray
