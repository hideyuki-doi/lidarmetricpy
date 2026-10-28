"""lidarmetricpy: lidRmetrics-compatible forest structure metrics from airborne lidar, on GPU or CPU."""
from .api import Grid, Options, compute_tile, compute_tiles, plan_blocks  # noqa: F401
from .backend import get_backend  # noqa: F401
from .canopylazr import canopylazr, canopylazr_file  # noqa: F401

__version__ = "0.2.0"
