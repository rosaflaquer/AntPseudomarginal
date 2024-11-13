import matplotlib.colors

def hex_to_rgba(hex_color, alpha=1.0):
    """
    Convert a hex color code to RGBA format.

    Parameters:
    - hex_color (str): Hex color code (e.g., '#RRGGBB' or '#RRGGBBAA').
    - alpha (float): Alpha transparency level (default is 1.0 for fully opaque).

    Returns:
    - tuple: RGBA tuple (float values in the range [0.0, 1.0]).
    """
    hex_color = hex_color.lstrip('#')

    # Check if the hex code includes alpha
    if len(hex_color) == 6:
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
        return (r / 255.0, g / 255.0, b / 255.0, alpha)
    elif len(hex_color) == 8:
        r, g, b, a = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:], 16)
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    raise ValueError("Invalid hex color code. It should be in the format '#RRGGBB' or '#RRGGBBAA'.")

"""
cmap_colors = ['8f2d56','a7495e','bc6568','cf8275','e09f87','f0bd9d','ffdbb7','dcbf91','b6a46d','8e8b4c','65742f','3b5c16','004500']
#cmap_colors = [
#'#19626b',
#'#006b88',
#'#2970a2',
#'#5f71b5',
#'#956db8',
#'#c666ab',
#'#ec6290',
#'#ff6b6b',
#]
cmap_colors =["#e27c7c", "#a86464", "#6d4b4b", "#503f3f", "#333333", "#3c4e4b", "#466964", "#599e94", "#6cd4c5"]
"""
cmap_colors = [
'#004500',
'#5c7b30',
'#acb367',
'#ffefa6',
'#e9aa75',
'#c5675f',
'#8f2d56',
]
cmap_colors = [hex_to_rgba(h) for h in cmap_colors]
custom = matplotlib.colors.LinearSegmentedColormap.from_list("custom",cmap_colors)