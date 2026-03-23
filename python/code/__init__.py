"""
Surface Brain Space Processing Toolkit
A Python package for processing and comparing surface brain space data
"""

from .surface_io import SurfaceDataProcessor, load_surface_data, get_surface_data_info

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "SurfaceDataProcessor",
    "load_surface_data",
    "get_surface_data_info"
]