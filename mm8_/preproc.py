#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processing fucntions for 1D and 2D NMR data
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors



# calibrate to double defined in this function
def calibrate_doubl(X, ppm, cent_ppm, j_hz, lw=0.5, tol_ppm=0.03,  niter_bl=5):
    """
    Calibrate spectra to a doublet (e.g. plasma to glucose at 5.23 ppm with j_hz=6, alanine at 1.491 ppm with j_hz=6, lactate at 1.35 ppm with j_hz=6)
    Args:
        X: NMR data array (rank 1 or 2)
        ppm: Chemical shift array (rank 1)
        cent_pos: Expected chemical shift of doublet
        j_hz: Expected J constant of doublet
        lw: Expected line width of doublet
        tol_ppm: Chemical shift tolerance for doublet
        niter_bl: Maximum of iterations for basline correction (asymmetric least squares)
    Returns:
        X - aligned NMR data
    """
    # alanine at 1.491 ppm with j_hz=6
    # lactate at 1.35 ppm with j_hz=6
    # glucose at 5.23 ppm with j_hz=6
    
    dtype=X.dtype
    #tol_ppm=tol_ppm.astype( dtype)
    #lac=[1.32, 1.38]
    bound=[cent_ppm-tol_ppm, cent_ppm+tol_ppm]
    idx_ra=get_idx(ppm, bound)[0]
    dppm=ppm[1]-ppm[0]
    
    Xs=X[:,idx_ra]
    sub=np.zeros((X.shape[0], len(idx_ra)))
    for i in range(X.shape[0]):
        tt=baseline_als(Xs[i,:], lam=10000, p=0.0001, niter=niter_bl)
        sub[i,:]=tt
    sub=Xs-sub
    sub=sub/np.max(sub, 1,  keepdims=True)*100

    pa=tf.cast(doub(cent_ppm=np.max(bound)-(j_hz/600), jconst_hz=j_hz, lw=lw, mag=100., sf=600, out='ppm', shift=ppm[idx_ra]), dtype)
    #plt.plot(ppm[idx_ra], pa)
    # cross correlation
    # shifting steps
    #((np.max(bound)-j_hz/600) - (np.min(bound)+j_hz/600)) / dppm
    n=np.abs(tf.cast(np.round(((np.max(bound)-j_hz/600) - (np.min(bound)+j_hz/600)) / dppm), tf.int32))
    # calc correlation
    x_cent=sub-tf.reduce_mean(sub, 1)[..., tf.newaxis]
    x_var=tf.math.sqrt(tf.reduce_sum(x_cent**2, 1))/sub.shape[1]
    temp_cent=pa[tf.newaxis, ...]-tf.reduce_mean(pa)
    temp_var=tf.math.sqrt(tf.reduce_sum(temp_cent**2))/sub.shape[1]
    tot_var=tf.math.sqrt((x_var*temp_var[..., tf.newaxis]))
    
    res=np.ones((X.shape[0], n))
    for i in range(n):
        temp_shift=tf.roll(pa, (i+1), 0)
        temp_cent=(temp_shift-tf.reduce_mean(temp_shift))[tf.newaxis, ...]
        res[:,i]=((tf.reduce_sum((temp_cent*x_cent), 1))/sub.shape[1]) / tot_var[tf.newaxis, ...]


    idx_shift=tf.math.argmax(res, 1)    
    # adjust shift for starting ccor at 1.37 and not at 1.35 
    offs=tf.cast(np.round(((cent_ppm-(np.max(bound)-j_hz/600))/(ppm[1]-ppm[0]))), tf.int64)
    idx_shift=idx_shift-(offs)
    # padding on both sides
    add=np.max(np.abs(idx_shift))+1
    paddings = tf.constant([[0, 0], [add, add]])
    Xpad=tf.pad(X, paddings).numpy()
    for i in range(len(idx_shift)):
        Xpad[i,:]=tf.roll(Xpad[i,:], -(idx_shift[i].numpy()+1), 0)
    
    # remove pads
    Xnw=tf.slice(Xpad, [0, add+1], [Xpad.shape[0], X.shape[1]])
    
    return Xnw.numpy()
    

# excision of spectral areas
def excise1d(X, ppm):
    """
    Excist spectral intervals using pre-defined limits: Kept is 0.25- 4.5 ppm and 5-10 ppm
    Args:
        X: NMR data array (rank 1 or 2)
        ppm: Chemical shift array (rank 1)
    Returns:
        Tuple of two: X, ppm
    """
    idx_upf=get_idx(ppm, [0.25, 4.5])[0]
    idx_downf=get_idx(ppm, [5, 10])[0]
    idx_keep= np.concatenate([idx_downf, idx_upf])
   
    Xc = X[:, idx_keep]
    ppc = ppm[idx_keep]
    
    return (Xc, ppc)
    

def pqn(x_raw, log=False):
    """
    Probabilistic Quotient Normalisation using median spectrum as reference and median quotient as dilution coefficient
    Args:
        x_raw: NMR data array (rank 1 or 2)
        log: log transform prior to dilution coefficient calculation
    Returns:
        Tuple of two: Xn, DataFrame of dilution coefficients
    """
    x_raw[x_raw<1]=None
    if log:
        lev=3
        x_raw=np.log10(x_raw)
    else:
        lev=10**3
        
    #idx_keep=x_raw>lev
    #plt.plot(x_raw[0,:])
    ref=np.median(x_raw, 0)[np.newaxis, ...]
    #idx_keep=ref<lev
    ref[ref<lev]=None

    quot=x_raw/ref
   
    xna=list()
    for i in range(x_raw.shape[0]):
        xna.append(quot[i,~np.isnan(quot[i,:])])
    
    emp=tfp.distributions.Empirical(xna)
    mea=emp.mean()
    mo=emp.mode()
    me=np.median(xna, 1)
   
    x_norm=np.ones(x_raw.shape)
    for i in range(x_raw.shape[0]):
        x_norm[i,:]=x_raw[i,:]/me[i][..., np.newaxis]
    
    if log:
        x_norm=10**x_norm
        #x_n1=10**x_norm
    
    dilfs = pd.DataFrame({'med':me, 'mod':mo, 'me':mea})
    
    
    return (x_norm, dilfs)