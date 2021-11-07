#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions - not intended to be used by the regular user
"""
from skimage.feature import peak_local_max
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve  
from scipy.sparse import diags
import tensorflow as tf
import tensorflow_probability as tfp

tfd=tfp.distributions




def _rho(x, y):
    # rank correlation
    xle=len(x)
    if xle != len(y):
        raise ValueError('Shape mismatch input')

    xu = np.unique(x, return_inverse=True)
    yu = np.unique(y,return_inverse=True)
    xu_rank = np.argsort(xu[0])
    yu_rank = np.argsort(yu[0])

    xut =xu_rank[xu[1]].astype(float)
    yut = yu_rank[yu[1]].astype(float)

    cv=np.cov(xut, yut)[0,1]
    if cv != 0 :
        return (cv, np.cov(xut, yut)[0,1] /(np.std(xut) * np.std(yut)))
    else:
        return (0, 0)

    return (None, 1- ((np.sum((xut-yut)**2) * 6) / (xle*(xle**2-1))))




def _cov_cor(X, Y):
    # x is pca scores matrix
    # y is colmean centered matrix
    if X.ndim == 1:
        X = np.reshape(X, (len(X), 1))
    if Y.ndim == 1:
        Y = np.reshape(Y, (len(Y), 1))
    if np.mean(Y[:, 0]) > 1.0e-10:
        Y = (Y - np.mean(Y, 0))
        X = (X - np.mean(X, 0))
    xy = np.matmul(X.T, Y)
    cov = xy / (X.shape[0] - 1)
    a = np.sum(X ** 2, 0)[..., np.newaxis]
    b = np.sum(Y ** 2, 0)[np.newaxis, ...]
    cor = xy / np.sqrt(a * b)
    return (cov, cor)



def binning1d(Xbl2, pp2, nbins=None, bwidth=0.01):

    if isinstance(nbins, int):
        idx_step=np.arange(0, len(pp2), int(len(pp2)/(nbins-1)))
    else:
        if isinstance(bwidth, float):
            step = np.median(np.diff(pp2))
            bwidth
            stepsize = np.abs(int(np.round(bwidth / step)))
            idx_step = np.arange(0, Xbl2.shape[1], stepsize)

    Xbin=np.zeros((Xbl2.shape[0], len(idx_step)))
    ppm_bin = np.zeros(len(idx_step))
    for i in range(1, len(idx_step)):
        Xbin[:, i] = np.nansum(Xbl2[:, idx_step[i - 1]:idx_step[i]], 1)
        ppm_bin[i]=np.mean(pp2[idx_step[(i-1)]:idx_step[i]])
    return (Xbin, ppm_bin)



def tsp_sym_single(x, hwlen=100):
    #hwlen = 100
    x = x / np.max(x)
    pimax = np.argmax(x)
    #q_le = [0]
    q_ri = [0]
    for i in range(1, hwlen):
        #q_le.append(x[pimax - i] - x[pimax + i])
        q_ri.append(x[pimax + i] - x[pimax - i])

    a=np.sum(np.abs(q_ri))
    b=np.sum(np.abs(x[(pimax+1):((pimax+1)+(hwlen))]))
    return np.round((a/b)*100)

def tsp_sym(Xp, hwlen=100):
    ps = []
    for i in range(Xp.shape[0]):
        #print(i)
        ps.append(tsp_sym_single(Xp[i], hwlen))
    return ps

def tsp_lw_sym(x, ppm, hwlen=100):
    """
        TSP line width and peak symmetry
        This function is included in preproc.excl_doublets

        Args:
            x: single spectrum (rank 1) or NMR matrix (rank 2)
            ppm: chemical shift vector
        Returns:
            Tuple len 1: lw in Hz (<1 Hz is good) and symmetry estimate (the smaller the better, <0.0033 is good)
    """
    from scipy.signal import find_peaks, peak_widths
    idx = get_idx(ppm, [-0.4, 0.4])

    if x.ndim == 1:
        #x = X[1, idx] / np.max(X[1, idx])
        peaks, _ = find_peaks(x, height=0.4)
        if (len(peaks) != 1): return None; #raise Exception('Check signals in TSP area')
        pw=peak_widths(x, peaks)
        sym=tsp_sym_single(x, hwlen)
        lw_idx=pw[0][0]
        lw_ppm=lw_idx * (ppm[0] - ppm[1]) * 600
    if x.ndim == 2:
        lw_ppm=[]
        sym=[]
        for i in range(x.shape[0]):
            xs = x[i, idx] / np.max(x[i, idx])
            peaks, _ = find_peaks(xs, height=0.8)
            if(len(peaks)!=1): print(i); lw_ppm.append(float("nan")); sym.append(float("nan")); continue#raise Exception('Check signals in TSP area')
            pw = peak_widths(xs, peaks, rel_height=0.5)
            sym.append(tsp_sym_single(xs, hwlen))
            lw_idx=pw[0][0]
            lw_ppm.append(lw_idx * (ppm[0] - ppm[1]) * 600)

    return [lw_ppm, sym]



def doub(cent_ppm=1.35, jconst_hz=6, lw=1.35, mag=100., sf=600, out='ppm', shift=[1.2, 1.5]):
    """
    Create a paramteric 1D doublet (Cauchy)
    
    Args:
        cent_ppm: Center position
        jconst_hz: J constant
        lw: Line widht in Hz
        mag: Amplitude
        sf: Spectrometer frequency (important for conversion Hz-ppm)
        out: Chemical shift unit (ppm or hz)
        shift: Chemical shift interval (list)
    Returns:
        Array or rank 1
    """
    
    cent_hz=cent_ppm*sf
    x=shift*sf
    j=jconst_hz/2
    comp=tfd.Cauchy([cent_hz+j, cent_hz-j], lw).prob(x [..., tf.newaxis])
    comp=(comp/tf.reduce_max(comp, 0, keepdims=True))*mag
    out=tf.reduce_sum(comp, 1)
    
    return out.numpy()

def baseline_als(y, lam, p, niter=10):
    """
   Performs 1D basline correcation using asymmetric least squares
   
    Args:
        y: Spectrum (array rank 1)
        lam: Lambda value
        p: Probablity value
        n_inter: Max number of interations
    Returns:
        baseline corrected spectrum (rank 1)
    """
    L = len(y)
    D = diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
      W = sparse.spdiags(w, 0, L, L)
      Z = W + lam * D.dot(D.transpose())
      z = spsolve(Z, w*y)
      w = p * (y > z) + (1-p) * (y < z)
    return z

# def doub(x, meta):
#     """
#     Create a doublet
#     Args:
#         x: Chemical shift array (rank 1)
#         shift: Chemical shift interval (list)
#     Returns:
#         1D array of ppm indices
#     """
#     sf=meta.get('procs').get('SF')
#     J=np.mean(abs(x.f1))
#     csh=np.mean(abs(x.f2))
#     out={'type':'d', 'ppm':csh, 'J':J, 'Int_max':np.mean(abs(x['max'])), 'peak_pos': [csh-(J/sf), csh+(J/sf)]}
#     return out

def sing(x):
    #out={'type':'s', 'ppm':x.loc['f2'], 'J':0, 'Int_max':x.loc['max'], 'peak_pos': x.loc['f2']}
    out={'type':'s', 'ppm':x['f2'].values, 'J':0, 'Int_max':x['max'].values, 'peak_pos': x['f2'].values}
    return out


def get_idx(ppm, shift):
    """
    Returns chemical shift index for a given interval
    
    Args:
        ppm: Chemical shift array (rank 1)
        shift: Chemical shift interval (list)
    Returns:
        1D array of ppm indices
    """
    shift=np.sort(shift)
    out=(ppm>shift[0]) & (ppm <shift[1])
    return np.where(out==True)[0]


def pp2d(X3, ppm1, ppm2, thres_abs='auto_40', mdist=3):
    """
    2D peak picking using skimage
    
    Args:
        X3: NMR data tensor (rank 3)
        ppm1: Chemical shift array for f1 (rank 1)
        ppm2: Chemical shift array for f2 (rank 1)
        thres_abs: Noise intensity threshold
        mdist: Minimum distance between two consecutive peaks
    Returns:
        List of DataFrames with peaks per spectrum
    """
    
    
    if isinstance(thres_abs, str):
        if 'auto' in thres_abs:
            ele=thres_abs.split('_')
            if len(ele)==2:
                thr_abs=np.max(X3, (0,1))/float(ele[1])
            else:
                thr_abs=np.max(X3, (0,1))/40
    else:
        if(len(thres_abs)==1):
            thr_abs=np.tile(thres_abs, X3.shape[0])
        else:
            thr_abs=thres_abs
            

    pl=list()
    for i in range(X3.shape[2]):
        print(i)
        x=X3[:,:,i]
        pco = peak_local_max(x, min_distance=mdist, threshold_abs=thr_abs[i])
        p1=ppm1[pco[:,0]]
        p2=ppm2[pco[:,1]]
        
        
        I=x[pco[:,0], pco[:,1]]
        Isc=I/np.max(x)

        pl.append(pd.DataFrame({'s':i, 'p1':p1, 'p2':p2, 'Int':I, 'IntSc':Isc, 'p1idx':pco[:,0],  'p2idx': pco[:,1]}))
    
    return pl


def est_noise (X3, pp1, pp2, nseg_p1=4, nseg_p2=10):
    """
    Estimate noise intensity threshold of 2D NMR spectra
    
    Args:
        X3: NMR data tensor (rank 3)
        pp1: Chemical shift array for f1 (rank 1)
        pp2: Chemical shift array for f2 (rank 1)
        nseg_p1: Number of segments in f1 (int)
        nseg_p2: Number of segments in f2 (int)
    Returns:
        Noise tensor
    """

    qp1=tfp.stats.quantiles(pp1, num_quantiles=nseg_p1)
    b1 = tfp.stats.find_bins(pp1, edges=qp1)
    
    qp2=tfp.stats.quantiles(pp2, num_quantiles=nseg_p2)
    b2 = tfp.stats.find_bins(pp2, edges=qp2)
    
    out=np.array(np.meshgrid(np.arange(nseg_p1), np.arange(nseg_p2))).T.reshape(-1,2)
 
    xsds=list()
    for i in range(out.shape[0]):
        idx_row=tf.squeeze(tf.where(b1==out[i,0]))
        idx_col=tf.squeeze(tf.where(b2==out[i,1]))
        
        x31=X3[idx_row, :, :]
        x32=x31[:, idx_col, :]
        xsds.append(tf.math.reduce_std(x32, (0,1)))
        
    ff=tf.reduce_min(tf.convert_to_tensor(xsds), 0)
    return ff


def calib_axis_2d(x, ppm1, ppm2):
    """
    Calibrate chemical shift of a 2D NMR spectrum to TSP
    
    Args:
        x: NMR data array (rank 2)
        ppm1: Chemical shift array for f1 (rank 1)
        ppm2: Chemical shift array for f2 (rank 1)
    Returns:
        Calibrated chemical shift arrays: ppm1, ppm2
    """
    
    id1=get_idx(ppm1, [-10,10])
    id2=get_idx(ppm2, [-0.1, 0.1])
    
    xs=x[id1, :]
    xs=xs[:, id2]
    
    px1=ppm1[id1]
    px2=ppm2[id2]
    
    ids=np.where(xs==np.max(xs))

    ppm1c=ppm1-( px1[ids[0][0]])
    ppm2c=ppm2-( px2[ids[1][0]])
    
    return ppm1c, ppm2c


def calibrate(X, ppm, signal='tsp'):
    """
    Calibrate chemical shift of a 1D NMR spectrum to TSP
    
    Args:
        X: NMR data array (rank 2)
        ppm: Chemical shift array (rank 1)
        signal: Signal as string (tsp)
    Returns:
        Calibrated NMR array
    """
    tsp=get_idx(ppm, [-0.1, 0.1])
    shift=ppm[tsp][np.argmax(X[:,tsp], 1)]
    ppm_new=ppm-shift[..., np.newaxis]
    for i in range(X.shape[0]):
        X[i,:]=np.flip(np.interp(np.flip(ppm), np.flip(ppm_new[i,:]), np.flip(X[i,:])))
    
    return X


def importJres(p1, acqus, wf2=tf.signal.hann_window, wf1=tf.signal.hann_window, pad='none'):
    """
    Import a single 2D raw NMR spectrum
    
    Args:
        p1: Location of Bruker ser file
        acqus: Files acqus and acqu2s as dic of dic
        w2f: Window function f2
        w1f: Window function f1
        pad: Should FID be padded or extended: mirror or zero padded at first time point
    Returns:
        2D NMR data as tensor
    """
    # read binary file and rm filter
   
    dat = np.fromfile(p1, dtype=np.dtype('<i4'))*(2**acqus['acqus']['NC'])
    td1=dat.shape[0]
    td2=dat.shape[1]
    re= tf.reshape(dat[0::2], (int(td1), int(td2/2)))
    clx=tf.reshape(dat[1::2], (int(td1), int(td2/2)))
    dc=tf.dtypes.complex(re, clx)
    n_filter=int(acqus['acqus']['GRPDLY'])
    ddc=dc[:,n_filter:]
    
    # f2 apodisation, padd
    win=wf2(ddc.shape[1], dtype=tf.float64)
    win=tf.dtypes.complex(win, win)
    dcw=ddc*win
    dcw=_pad_toPow2(dcw, 1, tf.float64)
    dcwp=tf.pad(dcw, [[0, 0], [0, dcw.shape[1]*3]] )
    # freq 
    ss1=tf.signal.fft(dcwp)
    ss1=tf.signal.fftshift(ss1, axes=1)

    # f1: apodisation, padd
    # add zeros as first entry, otherwise first fid will be set to zero by wfun1
    # ss1=tf.pad(ss1, [[1, 0], [0, 0]] )
    mf=3
    if pad == 'mirror':
        mf=1
        ss1i=ss1[ss1.shape[0]::-1,:]
        ss1=tf.concat([ss1i, ss1],  0)
        # test.shape
    if pad == 'zero':
        ss1=tf.pad(ss1, [[1, 0], [0, 0]] )
    # if pad is 'zero':
    
    win=wf1(ss1.shape[0], dtype=tf.float64)
    win=tf.dtypes.complex(win, tf.zeros(win.shape, dtype=tf.float64))
    ss1=ss1*win[..., tf.newaxis]
    ss1=_pad_toPow2(ss1, 0, tf.float64)
    ss1=tf.pad(ss1, [[0, ss1.shape[0]*mf], [0, 0]] )
    # freq
    ss1=tf.transpose(ss1)
    ss1=tf.signal.fft(ss1)
    ss1=tf.signal.fftshift(ss1, axes=1)
    ss1=tf.transpose(ss1)

    smag=tf.math.sqrt(tf.math.pow(tf.math.imag(ss1), 2)+tf.math.pow(tf.math.real(ss1), 2))
    
    return smag


def cppm(acqus, size):  
    """
    Calculate chemical shift
    
    Args:
        acqus:  Bruker acqus and acqu2s file data as dic
    Returns:
        ppm vector
    """
    afile='acqus'
    offset = (float(acqus[afile]['SW']) / 2) - (float(acqus[afile]['O1']) / float(acqus[afile]['BF1']))
    start = float(acqus[afile]['SW']) - offset
    end = -offset
    step = float(acqus[afile]['SW']) / size
    ppms = np.arange(start, end, -step)[:size]
    return np.flip(ppms)

def _log2(x, dtype=tf.float32):
    """
    Log2 transformation for TensorFlow
    
    Args:
        x:  Untransformed tensor
        dtype:  TensorFlow data type (floatxx)
    Returns:
        Log2 transformed tensor
    """
    return tf.math.log(x)/tf.math.log(tf.constant(2, x.dtype))


def _pad_toPow2(X, axis=0, dtype=tf.float64):
    """
    Zero padding of tensor
    
    Args:
        X:  Tensor of rank 2
        axis: Padding axis
        dtype:  TensorFlow data type (floatxx)
    Returns:
        Padded tensor
    """
    opt2=tf.math.ceil(_log2(tf.cast(X.shape[axis], tf.float64)))
    p1=tf.abs(tf.math.pow(2, opt2)-X.shape[axis])
    if axis == 1:
        p=[[0, 0], [0, p1]]
    if axis == 0:
        p=[[0, p1], [0, 0]]
    Xp=tf.pad(X, p )
    return Xp


def exp_win(n=100, dtype=tf.float32, lb=0.3):
    """
    Exponential window function
    
    Args:
        n:  Length FID
        dtype:  TensorFlow data type (floatxx)
        lb: Line broadening factor
    Returns:
        Window function scaled between 0 and 1
    """
    x=tf.linspace(0, 1, n)
    win=tf.cast(tf.exp((np.pi*-1)*x*lb), dtype=dtype)
    win=win+tf.reduce_max(-win)
    win = tf.cast(win/tf.reduce_max(win), dtype=dtype)
    
    return win
    

def sin_win(n, dtype):
    """
    Sine window function
    
    Args:
        n:  Length FID
        dtype:  TensorFlow data type (floatxx)
    Returns:
        Window function scaled between 0 and 1
    """
    x=tf.linspace(1, n, n)
    y=np.sin((np.pi*x)/n)
    return tf.cast(y, dtype)

# x=tf.linspace(0, 1, 100)
# lb=0.8
# win=tf.cast(tf.exp((np.pi*-1)*x*lb), tf.float32)
# plt.plot(win)

# lb=2.8
# win=tf.cast(tf.exp((np.pi*-1)*x*lb), tf.float32)
# sem=win*tf.signal.hann_window(100)
# sem=sem/tf.reduce_max(sem)
# plt.plot(sem)

# win=apod_norm(100, dtype=tf.float32, sc=5, shift=0)
# plt.plot(win[0,:])

# win=apod_norm(100, dtype=tf.float32, sc=5, shift=0.2)
# plt.plot(win[0,:])

# win=apod_norm(100, dtype=tf.float32, sc=5, shift=-0.2)
# plt.plot(win[0,:])

# tfd=tfp.distributions



# x=np.arange(1000)
# y=tfd.Pareto(50, 10).prob(x)
# plt.plot(y)


def kaiser_bessel(n, dtype, par=5):
    """
    Kaiser-Bessel window function
    
    Args:
        n:  Length FID
        dtype:  TensorFlow data type (floatxx)
        par: Plateau parameter
    Returns:
        Window function scaled between 0 and 1
    """
    if n % 2 == 0:
        out=tf.signal.kaiser_bessel_derived_window(n-2, par, dtype=dtype).numpy()
        out=np.append(np.append(0, out), 0)
    else:
        out=tf.signal.kaiser_bessel_derived_window(n-1, par, dtype=dtype).numpy()
        out[0]=0
        out=np.append(out, 0)
            
    return out
        


def apod_norm(n, dtype, sc=0.5, shift=0):
    """
    (Shifted) Gaussian window function
    
    Args:
        n:  Length FID
        dtype:  TensorFlow data type (floatxx)
        sc: Standard deviation of Normal
        shift: x-positional shift parameter (as fraction)
    Returns:
        Window function scaled between 0 and 1
    """
    shp=(n/2)+shift*(n/2)
    x=np.arange(0, n)
    y=tfd.Normal(shp, sc).prob(x)
    y=y/tf.reduce_max(y)
    y=tf.cast(y, dtype)[tf.newaxis, ...]
    return y


# def apod_sem(n, dtype, lb=0.0051):
#     x=np.arange(0, n)
#     win=tf.cast(tf.exp((np.pi*-1)*x*lb), dtype)
#     sem=win*tf.squeeze(tf.signal.hann_window(n, dtype=dtype))
#     sem=sem/tf.reduce_max(sem)
#     return sem

# y=apod_sem(1000, dtype, lb=1)
# plt.plot(y)

def apod_horse(n, dtype):
    """
    Horeseshoe window function
    
    Args:
        n:  Length FID
        dtype:  TensorFlow data type (floatxx)
    Returns:
        Window function scaled between 0 and 1
    """
    x=tf.cast(tf.linspace(0.001, 0.999, n-2), dtype)
    y=1/tfd.Beta(0.5, 0.5).prob(tf.cast(x, tf.float32))
    y=np.append(np.append(0, y), 0)
    y=y/tf.reduce_max(y)
    
    return tf.cast(y, dtype)

# import plotly.express as px
# import pandas as pd
# import plotly.graph_objects as go


# idx=np.where((ppm>3) & (ppm <4.5))[0]
# idx=np.where((ppm>-0) & (ppm <1))[0]
# idx=np.where((ppm>6.6) & (ppm <7.6))[0]import plotly.io as pio
# pio.renderers.default = "browser"

# x, y = np.linspace(0, 1, smag.shape[0]), ppm[idx]
# fig = go.Figure(data=[go.Surface(z=np.log(smag.numpy()[:,idx]), x=y, y=x)])
# fig.show()



# win=tf.signal.hann_window(100)
# plt.plot(win)

# win=tf.signal.kaiser_window(100)
# plt.plot(win)

# win=tf.signal.vorbis_window(100)
# plt.plot(win)

# win=tf.signal.hamming_window(100)
# plt.plot(win)

# win=tf.signal.hamming_window(100, periodic=False)
# plt.plot(win)

# win=tf.signal.kaiser_bessel_derived_window(100, 5)
# plt.plot(win)
