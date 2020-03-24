# -*- coding: utf-8 -*-
# ============================================================================
#     Deep Time-Series Early Alpha v0.4.0
#     Python v3.6+
#     Created by Kevin R. Chen
#     Licensed under Apache License v2
# ============================================================================

# Version
from .__version__ import __version__
# Endpoint functions for users
from .run import run_package as run
from .utils import load_data as load, clean_data as clean


if __name__ == "__main__":
    version_number = __version__
    print(f"Running Deep Time-Series v{version_number}. Execute package by "
          "calling dnntime.run(config) with your provided config YAML file.")
else:
    version_number = __version__
    print(f"Importing Deep Time-Series v{version_number}. Execute package by "
          "calling dnntime.run(config) with your provided config YAML file.")
