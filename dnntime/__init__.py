# -*- coding: utf-8 -*-
# ============================================================================
#     Deep Time-Series Pre-Alpha v0.3.5
#     Python v3.6, v3.7
#     Created by Kevin R. Chen
#     Licensed under Apache License v2
# ============================================================================

# Version
from .__version__ import __version__
# Endpoint functions for users
from .package import run_package as run
from .utils.etl import load_data as load, clean_data as clean


if __name__ == "__main__":
    version_number = __version__
    print(f"Running Deep Time-Series v{version_number}. Execute package by "
          "calling dnntime.run(config) with your provided config YAML file.")
else:
    version_number = __version__
    print(f"Importing Deep Time-Series v{version_number}. Execute package by "
          "calling dnntime.run(config) with your provided config YAML file.")
