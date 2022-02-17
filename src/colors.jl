"""
    colors.jl

Define the color schemes used in the paper results

Authors:
- Sasha Petrenko <sap625@mst.edu>

Timeline:
- 1/15/2022: Created.
- 2/17/2022: Documented.
"""

using ColorSchemes

# Yellow-green-9 raw RGB values, range [0, 1]
ylgn_9_raw = [
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

# Purple-blue-9 raw RGB values, range [0, 1]
pubu_9_raw = [
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

# Infer the number of colors
n_colors = size(ylgn_9_raw)[1]

# Define the colorschemes from the RGB values
ylgn_9 = ColorScheme([RGB{Float64}(ylgn_9_raw[i, :]...) for i = 1:n_colors])
pubu_9 = ColorScheme([RGB{Float64}(pubu_9_raw[i, :]...) for i = 1:n_colors])
