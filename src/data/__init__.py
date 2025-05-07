"""
MLbotnew data package for handling data sources and downloads.

This package contains modules for interacting with various data sources
including Coinglass API for cryptocurrency data.
"""

from .coinglass import CoinglassClient

__all__ = ['CoinglassClient']
