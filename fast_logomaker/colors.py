"""
Color scheme definitions and utilities for Fast Logomaker.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from .error_handling import check

# Sets default color schemes for specified sets of characters
CHARS_TO_COLORS_DICT = {
    tuple('ACGT'): 'classic',
    tuple('ACGU'): 'classic',
    tuple('ACDEFGHIKLMNPQRSTVWY'): 'weblogo_protein',
}

weblogo_blue = [.02, .09, .74]
weblogo_pink = [.83, .11, .75]
weblogo_green = [.13, .83, .15]
weblogo_red = [.83, .04, .08]
weblogo_black = [0, 0, 0]

# COLOR_SCHEME_DICT provides a default set of logo colorschemes
# that can be passed to the 'color_scheme' argument of BatchLogo()
three_ones = np.ones(3)
COLOR_SCHEME_DICT = {
    'classic': {
        'G': [1, .65, 0],
        'TU': [1, 0, 0],
        'C': [0, 0, 1],
        'A': [0, .5, 0]
    },

    'grays': {
        'A': .2 * three_ones,
        'C': .4 * three_ones,
        'G': .6 * three_ones,
        'TU': .8 * three_ones
    },

    'base_pairing': {
        'TAU': [1, .55, 0],
        'GC': [0, 0, 1]
    },

    'colorblind_safe': {
        'A': '#009980',
        'C': '#59B3E6',
        'G': '#E69B04',
        'TU': '#1A1A1A'
    },

    'weblogo_protein': {
        'RHK': weblogo_blue,
        'DE': weblogo_red,
        'QN': weblogo_pink,
        'GCSTY': weblogo_green,
        'ILMAFVPW': weblogo_black
    },

    'skylign_protein': {
        'F': [.16, .99, .18],
        'Y': [.04, .40, .05],
        'L': [.99, .60, .25],
        'V': [1.0, .80, .27],
        'I': [.80, .60, .24],
        'H': [.40, .02, .20],
        'W': [.42, .79, .42],
        'A': [.99, .60, .42],
        'S': [.04, .14, .98],
        'T': [.17, 1.0, 1.0],
        'M': [.80, .60, .80],
        'N': [.21, .40, .40],
        'Q': [.40, .41, .79],
        'R': [.59, .02, .04],
        'K': [.40, .20, .03],
        'E': [.79, .04, .22],
        'G': [.95, .94, .22],
        'D': [.99, .05, .11],
        'P': [.10, .61, .99],
        'C': [.09, .60, .60]
    },

    'dmslogo_charge': {
        'A': '#000000',
        'C': '#000000',
        'D': '#0000FF',
        'E': '#0000FF',
        'F': '#000000',
        'G': '#000000',
        'H': '#FF0000',
        'I': '#000000',
        'K': '#FF0000',
        'L': '#000000',
        'M': '#000000',
        'N': '#000000',
        'P': '#000000',
        'Q': '#000000',
        'R': '#FF0000',
        'S': '#000000',
        'T': '#000000',
        'V': '#000000',
        'W': '#000000',
        'Y': '#000000'
    },

    'dmslogo_funcgroup': {
        'A': '#f76ab4',
        'C': '#ff7f00',
        'D': '#e41a1c',
        'E': '#e41a1c',
        'F': '#84380b',
        'G': '#f76ab4',
        'H': '#3c58e5',
        'I': '#12ab0d',
        'K': '#3c58e5',
        'L': '#12ab0d',
        'M': '#12ab0d',
        'N': '#972aa8',
        'P': '#12ab0d',
        'Q': '#972aa8',
        'R': '#3c58e5',
        'S': '#ff7f00',
        'T': '#ff7f00',
        'V': '#12ab0d',
        'W': '#84380b',
        'Y': '#84380b'
    },

    'hydrophobicity': {
        'RKDENQ': [0, 0, 1],
        'SGHTAP': [0, .5, 0],
        'YVMCLFIW': [0, 0, 0]
    },

    'chemistry': {
        'GSTYC': [0, .5, 0],
        'QN': [.5, 0, .5],
        'KRH': [0, 0, 1],
        'DE': [1, 0, 0],
        'AVLIPWFM': [0, 0, 0]
    },

    'charge': {
        'KRH': [0, 0, 1],
        'DE': [1, 0, 0],
        'GSTYCQNAVLIPWFM': [.5, .5, .5]
    },

    'NajafabadiEtAl2017': {
        'DEC': [.42, .16, .42],
        'PG': [.47, .47, 0.0],
        'MIWALFV': [.13, .35, .61],
        'NTSQ': [.25, .73, .28],
        'RK': [.74, .18, .12],
        'HY': [.09, .47, .46],
    },
}


def list_color_schemes():
    """
    Provides user with a list of valid color_schemes built into Fast Logomaker.

    Returns
    -------
    colors_df: pandas.DataFrame
        A pandas dataframe listing each color_scheme and the corresponding
        set of characters for which colors are specified.
    """
    names = list(COLOR_SCHEME_DICT.keys())
    colors_df = pd.DataFrame()
    for i, name in enumerate(names):
        color_scheme = COLOR_SCHEME_DICT[name]
        characters = list(''.join(list(color_scheme.keys())))
        characters.sort()
        colors_df.loc[i, 'color_scheme'] = name
        colors_df.loc[i, 'characters'] = ''.join(characters)

    return colors_df


def _expand_color_dict(color_dict):
    """
    Expands the string keys in color_dict, returning new_dict that has
    the same values but whose keys are single characters. These single
    characters are both uppercase and lowercase versions of the characters
    in the color_dict keys.
    """
    new_dict = {}
    for key in color_dict.keys():
        value = color_dict[key]
        for char in key:
            new_dict[char.upper()] = value
            new_dict[char.lower()] = value
    return new_dict


def get_rgb(color_spec):
    """
    Safely returns an RGB np.ndarray given a valid color specification.

    Parameters
    ----------
    color_spec : str, list, tuple, or np.ndarray
        Color specification (name, hex, or RGB values).

    Returns
    -------
    rgb : np.ndarray
        RGB color array.
    """
    rgb = None

    if isinstance(color_spec, str):
        try:
            rgb = np.array(to_rgb(color_spec))
        except:
            check(False, 'invalid choice: color_spec=%s' % color_spec)

    elif isinstance(color_spec, (list, tuple, np.ndarray)):
        check(len(color_spec) == 3,
              'color_scheme, if array, must be of length 3.')
        check(all(0 <= x <= 1 for x in color_spec),
              'Values of color_spec must be between 0 and 1 inclusive.')
        rgb = np.array(color_spec)

    else:
        check(False, 'type(color_spec) = %s is invalid.' % type(color_spec))

    return rgb


def get_color_dict(color_scheme, chars):
    """
    Return a color_dict constructed from a user-specified color_scheme and
    a list of characters.

    Parameters
    ----------
    color_scheme : str, dict, list, tuple, or np.ndarray
        Color scheme specification.
    chars : str, list, tuple, or np.ndarray
        Characters to assign colors to.

    Returns
    -------
    color_dict : dict
        Dictionary mapping characters to RGB colors.
    """
    check(isinstance(chars, (str, list, tuple, np.ndarray)),
          "chars must be a str or be array-like")

    check(len(chars) >= 1, 'chars must have length >= 1')

    chars = list(chars)
    chars.sort()

    for i, c in enumerate(chars):
        c = str(c)
        check(isinstance(c, str) and len(c) == 1,
              'entry number %d in chars is %s; ' % (i, repr(c)) +
              'must instead be a single character')

    if color_scheme is None:
        key = tuple(chars)
        color_scheme = CHARS_TO_COLORS_DICT.get(key, 'gray')
        color_dict = get_color_dict(color_scheme, chars)

    elif isinstance(color_scheme, dict):
        for key in color_scheme.keys():
            check(isinstance(key, str),
                  'color_scheme dict contains a key (%s) ' % repr(key) +
                  'that is not of type str.')
        color_dict = _expand_color_dict(color_scheme)
        for key in color_dict.keys():
            color_dict[key] = to_rgb(color_dict[key])

    elif isinstance(color_scheme, str):
        if color_scheme in COLOR_SCHEME_DICT.keys():
            tmp_dict = COLOR_SCHEME_DICT[color_scheme]
            color_dict = _expand_color_dict(tmp_dict)
            for c in color_dict.keys():
                color = color_dict[c]
                rgb = to_rgb(color)
                color_dict[c] = np.array(rgb)
        else:
            try:
                rgb = to_rgb(color_scheme)
                color_dict = dict([(c, rgb) for c in chars])
            except:
                check(False, 'invalid choice: color_scheme=%s' % color_scheme)

    elif isinstance(color_scheme, (list, tuple, np.ndarray)):
        check(len(color_scheme) == 3,
              'color_scheme, if array, must be of length 3.')
        rgb = np.ndarray(color_scheme)
        color_dict = dict([(c, rgb) for c in chars])

    else:
        check(False,
              'Error: color_scheme has invalid type %s'
              % type(color_scheme))

    if not set(chars) <= set(color_dict.keys()):
        for c in chars:
            if c not in color_dict:
                check(False,
                      " Warning: Character '%s' is not in color_dict. " % c +
                      "Using black.",
                      warn=True)
                color_dict[c] = to_rgb('black')
    return color_dict
