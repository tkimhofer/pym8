###################
<<<<<<< HEAD
# 1D NMR pipeline
=======
# 1D NMR pipeline with pym8
>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060
###################
# V1, 25/01/2022
# Torben Kimhofer
# Murdoch Uni

### import pym8 package
import sys
# add pym8 to python library path variable
d='/path/to/folder/pym8'
sys.path.append(d)
from mm8 import *
import numpy as np

# pym8 package is grouped into 5 modules
# each module contains code for specific set of task: reading, preproc, analyse, plotting, utility
# for example, the module reading contains code to read in 1 or 2D spectra

### read-in 1D spectra

<<<<<<< HEAD
# check directory for 1D NMR experiments
path='/path/to/encolsing/folder'
=======
# define directory of NMR experiments
path='/path/to/enclosing/folder'
# path='/Users/tk2812/Downloads'

# determine experiment type and number in directory
# the following code prints out a summary of experiments into the console
# copy 1D NMR experiment type(s) (e.g. PROF_PLASMA_NOESY and PROF_PLASMA_CPMG)
>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060
exp = reading.list_exp(path, return_results=True, pr=True)
# example output:
#                 exp   n  size_byte  maxdiff_byte               mtime
# 0       CPMGESGP_PN   8  1048576.0             0 2021-08-24 04:19:00
# 1  PROF_URINE_NOESY  16  1048576.0             0 2021-08-24 04:19:00
# see also function documentation `help(reading.list_exp)`

# import experiments
X, ppm, meta =  reading.import1d_procs(flist=exp, exp_type=['PROF_URINE_NOESY'], eretic=True)

# visualise spectrum/spectra
# 1. matplotlib (IDE supported)
# interactive visualisation of up to 1000 spectra should work without much latency on a normal desktop computer
plotting.spec(X, ppm) # full ppm range
plotting.spec(X, ppm, shift=[-0.1, 0.1]) # selected ppm range

# 2. plotly (browser supported)
plotting.ispec(X, ppm, shift=[-0.1, 0.1])



### preprocessing: chemical shift calibration, excision of residual water/capping low and high-field ppm
# calibration using doublet of glucose, alanine or lactate
# select approximate doublet ppm position and J constant, both values with be used to select appropriate signals

# lets calibrate to alanine
plotting.spec(X, ppm, shift=[1.3, 1.45])
cent_ppm = 1.354
j_hz = 8
Xc = preproc.calibrate_doubl(X = X, ppm = ppm, cent_ppm = cent_ppm, j_hz = j_hz, lw=10, tol_ppm=0.015,  niter_bl=5)

# check if calibration was successfull
# if double is not aligned, adjust parameters j_hz and location
# typically j_hz can be relaxed to improve calibration 
plotting.spec(Xc, ppm, shift=[1.3, 1.45])

# keep spectral areas with relevant signals
upfield = [0.25, 4.5]
downfield = [5, 9.5]

Xe, ppe = preproc.excise1d(X, ppm, shifts=[upfield, downfield])
plotting.spec(Xe, ppe)

# basline correction for std. 1D experiment (generally not needed for CPMG)
# enable parallel processing with input argument multiproc=True
Xb = preproc.bline(Xe, multiproc=True)

# compare a spectrum before and after bl correction,
ax1 = plotting.spec(Xe[0], ppe, label='raw')
# pass-on matplotlib axis (ax1)  for overlay plotting
fig, ax1 = plotting.spec(Xb[0], ppe, ax=ax1, label='bline corrected')
fig.legend()


# # PQN normalisation (typically not required for plasma)
# Method description: Dieterle et al (DOI: 10.1021/ac051632c)
# Xn, dilfs = preproc.pqn(Xb, False)
# plotting.spec(Xn, ppe)

# perform PCA
mod=analyse.pca(Xe, ppe)

# PCA scores
mod.plot_scores(an=meta, hue='SFO1')

# PCA loadings
mod.plot_load(pc=0)
mod.plot_load(pc=1)

