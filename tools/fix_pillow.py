"""
Fix for Pillow 10+ compatibility with older libraries.
Import this before using kaleido/plotly image export.
"""

import PIL.Image

# Create compatibility aliases for deprecated constants
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR

if not hasattr(PIL.Image, 'CUBIC'):
    PIL.Image.CUBIC = PIL.Image.Resampling.BICUBIC