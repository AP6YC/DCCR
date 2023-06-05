"""
    colors.jl

# Description
Defines the color schemes used in the paper results.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using ColorSchemes

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
Yellow-green-9 raw RGB values, range `[0, 1]`.
"""
const ylgn_9_raw = [
    255	255	229;
    247	252	185;
    217	240	163;
    173	221	142;
    120	198	121;
    65	171	93;
    35	132	67;
    0	104	55;
    0	69	41 ;
]/255.0

"""
Purple-blue-9 raw RGB values, range `[0, 1]`.
"""
const pubu_9_raw = [
    255	247	251
    236	231	242
    208	209	230
    166	189	219
    116	169	207
    54	144	192
    5	112	176
    4	90	141
    2	56	88
]/255.0

"""
Inferred number of colors used from the color palettes.
"""
const n_colors = size(ylgn_9_raw)[1]

"""
Yellow-green-9 `ColorScheme`, inferred from the RGB values.
"""
const ylgn_9 = ColorScheme([ColorSchemes.RGB{Float64}(ylgn_9_raw[i, :]...) for i = 1:n_colors])

"""
Purple-blue-9 `ColorScheme`, inferred from the RGB values
"""
const pubu_9 = ColorScheme([ColorSchemes.RGB{Float64}(pubu_9_raw[i, :]...) for i = 1:n_colors])
