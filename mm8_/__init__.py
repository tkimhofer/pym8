#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The mm8 package provides a framework to apply machine learning and probabilistic programming algorithms to NMR data.

The package contains the following modules:
        * 'reading' importing 1D and 2D NMR data
        * 'plotting' visualisation functions
        * 'fitting' peak fitting and metabolite quantification with Jres
"""
# import pandas as pd
# import numpy as np
# from plotnine import *
# import nmrglue as ng
# import re
# from scipy import sparse
# from scipy.sparse.linalg import spsolve  
# from scipy.sparse import diags
# from scipy.stats import norm
# import subprocess
# import os
# import nmrglue as ng
# from plotly.offline import plot
# import plotly.graph_objs as go
# import plotly.io as pio
# import plotly.express as px
# from scipy import interpolate
# from skimage.feature import peak_local_max

from . import reading
from . import plotting
from . import preproc
from . import utility