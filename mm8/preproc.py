#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processing fucntions for 1D and 2D NMR data
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import chi2
from matplotlib import cm
import matplotlib.collections as mcoll
import plotnine as pn
import tensorflow_probability as tfp

import mm8.utility

tfd=tfp.distributions
tfb=tfp.bijectors

# identify and exclude re-runs

def excl_doublets(X, ppm, meta):
    """
        Exclude re-runs from spectral matrix.
        Kept is spectrum with lowest TSP line width

        Args:
            S: Spectral matrix (np array rank 2)
            ppm: Chemical shift vector
            meta: Metadata dataframe
        Returns:
            Tupel of len 2: X and meta with re-runs excluded
    """

    ### define lw
    if not 'lw_hz' in meta.columns:
        lws=mm8.utility.tsp_lw_sym(X, ppm, hwlen=100)
        meta = pd.concat(
            [pd.DataFrame.from_records(lws, index=['lw_hz', 'asym_perc'], columns=meta.index).transpose(), meta],
            axis=1)

    ### map spectrometer ids
    sf = np.round(meta.SFO1, 2)
    meta['spec'] = None
    #specs = sf.unique()
    maps = {'600.31': 'ivdr02', '600.21': 'ivdr04', '600.10': 'ivdr05', '600.40': 'ivdr03'}

    # add to maps
    not_in= set(np.unique(sf)) - set(maps.keys())

    if len(not_in) > 0 :
        meta['spec'] = sf.astype(str)
    else:
        meta['spec'] = [maps[freq] for freq in sf.astype(str).to_list()]

    ## extract plate/exp folder and experiment ids
    exp_path=meta.id.str.split('/', expand=True)
    def sear(x):
        xyes=x.str.contains('^[0-9][0-9].*')
        id=x.loc[xyes]
        id_dir=x.loc[np.where(xyes)[0]-1]
        if (len(id) != 1) | (len(id_dir) != 1):
            raise Exception('Naming convention error: Check path system!')
        return [id.values[0], id_dir.values[0]]
        #id=x.str.findall('(^\d*)').str[0]
        return

    ids=exp_path.apply(sear, axis=1)
    add=pd.DataFrame.from_records(ids, columns=['exp_id', 'folder'], index=meta.index)
    meta=pd.concat([add, meta], axis=1)

    # for each plate and spectrometer find ids with same numbers and filter for lw
    # keep USERA2 for all spectra to
    def cutlast(x):
        print(x.folder.iloc[0])
        ids = x.exp_id.astype(str).str[:-1]
        cs = ids.value_counts()
        csre = cs.loc[cs > 1]
        indic = np.ones(x.shape[0])
        ids1=x.USERA2.values
        #u2ids=[]
        #u2ids_idx=[]
        for i in range(csre.shape[0]):
            iresp = np.where((ids.values == csre.index[i]))[0]
            u2=np.unique(x.USERA2[iresp].values)
            u2u=u2[u2 != '']
            if len(u2u) > 1 :
                raise Exception('ID in USERA2 not unique.')
            if len(u2u) ==0 :
                u2u='uknwn_rerun'+str(i)
            ids1[iresp]=u2u
            #x.iloc[iresp]['USERA2']=u2u[0]
            #u2ids.append(u2u[0])
            keep = iresp[np.nanargmin(x.lw_hz.iloc[iresp])]
            indic[iresp] = 0
            indic[keep] = 1
            #u2ids_idx.append(keep)

        #iid=x['USERA2'].values
        #iid[u2ids_idx]=u2ids
        x['USERA2']=np.array(ids1)
        x['keep_rerun'] = indic.astype(bool)
        return x

    # mms=meta.groupby(['spec', 'folder'])
    # mms.groups.keys()
    # x = mms.get_group(('600.59', 'Autism_Urine_Rack02_RFT_080817'))
    # test=cutlast(x)
    # test.keep_rerun.value_counts()

    meta = meta.groupby(['spec', 'folder']).apply(cutlast)
    #meta.groups.keys()
    #x = meta.get_group(('ivdr02', 'barwin20_IVDR02_BARWINp08_111120'))
    # remove doublets from X and meta
    idx = np.where(meta.keep_rerun.values)[0]
    X = X[idx,:]
    meta = meta.iloc[idx,:]

    return (X, meta)


def noise(x, ppm, sh=[10, 10.1]):

    idx = mm8.utility.get_idx(ppm, sh)

    ilen=int(round(len(idx)/2))
    cent = idx[ilen]
    a=x[cent:(cent+ilen)]
    b = x[(cent - ilen):cent]

    t1=((np.sum((a-b)*(np.arange(ilen)+1))**2)*3) / ((ilen*2)**2 -1)
    t2=np.sum(a+b)**2

    noi=np.sqrt(((np.sum(a**2)+np.sum(b**2))-((t1+t2)/(ilen*2))) / ((ilen*2)-1))

    return noi



def preproc_qc(X, ppm, meta, multiproc=True, thres_lw=1.2):
    """
        Estimation of baseline and residual water signal

        Args:
            x: single spectrum (rank 1) or NMR matrix (rank 2)
            ppm: chemical shift vector
            meta: dataframe with metadata
        Returns:
            X, ppm, meta
    """

    ### define lw
    if not 'lw_hz' in meta.columns:
        lws = mm8.utility.tsp_lw_sym(X, ppm, hwlen=100)
        meta = pd.concat(
            [pd.DataFrame.from_records(lws, index=['lw_hz', 'asym_perc'], columns=meta.index).transpose(), meta],
            axis=1)

    idx=np.where(meta.lw_hz <= thres_lw)[0]
    X=X[idx]
    meta=meta.iloc[idx]


    idx = mm8.utility.get_idx(ppm, [4.72, 4.87])
    h2o = np.sum(np.abs(X[:, idx]), 1)

    idx3 = mm8.utility.get_idx(ppm, [-0.1, 0.1])
    tsp = np.sum(X[:, idx3], 1)
    #tsp_max = np.max(X[:, idx3], 1)

    idx1 = mm8.utility.get_idx(ppm, [0.25, 4.5])
    idx2 = mm8.utility.get_idx(ppm, [5, 9.5])

    # baseline
    xcut = np.concatenate([idx2, idx1])
    if multiproc:
        import multiprocessing
        from joblib import Parallel, delayed
        from tqdm import tqdm
        def bl_par(i, X, xcut):
            out = mm8.utility.baseline_als(X[i, xcut], lam=1e6, p=1e-4, niter=10)
            return out

        ncore = multiprocessing.cpu_count() - 2
        spec_seq = tqdm(np.arange(X.shape[0]))
        bls = Parallel(n_jobs=ncore)(delayed(bl_par)(i, X, xcut) for i in spec_seq)
        bls = np.array(bls)
    else:
        bls = []
        for i in range(X.shape[0]):
            bls.append(baseline_als(X[i, xcut], lam=1e6, p=1e-4, niter=10))
    bls = np.array(bls)
    b_aliph = np.sum(bls[:, 0:len(idx1)], axis=1)
    b_arom = np.sum(bls[:, (len(idx1) + 1):], axis=1)

    Xbl = X[:, xcut] - bls
    ppbl = ppm[xcut]

    # signal estimation: tsp and total integral
    signal = np.sum(Xbl, 1)
    h2o_signal_perc = (h2o / signal) * 100
    h2o_tsp_perc = (h2o / tsp) * 100

    signal_tsp_perc = (signal / tsp) * 100
    signal_bl_per = (signal / (b_aliph + b_arom)) * 100

    # signal to noise estimation
    noi = list()
    for i in range(X.shape[0]):
        noi.append(np.round(noise(X[i], ppm, sh=[10, 10.1])))

    sino_tsp = tsp / noi
    signal_med = np.quantile(Xbl, 0.25, axis=1)
    sino_medsignal = signal_med / noi

    cids = ['rWater', 'bl_aliphatic', 'bl_aromatic', 'signal', 'tsp', 'rWater_tsp_perc', 'rWater_signal_perc', \
            'signa_tsp_perc', 'signal_bl_perc', 'sino_tsp', 'signalQ1', 'sino_signalQ1']
    schars = pd.DataFrame(
        [h2o, b_aliph, b_arom, signal, tsp, h2o_tsp_perc, h2o_signal_perc, signal_tsp_perc, signal_bl_per, sino_tsp, \
         signal_med, sino_medsignal], index=cids).transpose()

    schars.index=meta.index
    meta=pd.concat([meta, schars], axis=1)
    return (Xbl, ppbl, meta)



def get_idx(ppm, shift):
    """
    Returns chemical shift index for a given interval

    Args:
        ppm: Chemical shift array (rank 1)
        shift: Chemical shift interval (list)
    Returns:
        1D array of ppm indices
    """
    shift = np.sort(shift)
    out = (ppm > shift[0]) & (ppm < shift[1])
    return np.where(out == True)[0]


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

    if log:
        x_raw[x_raw < 1] = None
        lev=3
        x_raw=np.log10(x_raw)
    else:
        q=np.nanquantile(x_raw, 0.1, axis=1)
        #lev=10**3
        lev=q
        
    #idx_keep=x_raw>lev
    #plt.plot(x_raw[0,:])
    ref=np.nanmedian(x_raw, 0)[np.newaxis, ...]
    #idx_keep=ref<lev
    ref[ref<np.nanquantile(ref, 0.1)]=None

    quot=x_raw/ref
   
    xna=list()
    for i in range(x_raw.shape[0]):
        xna.append(quot[i,~np.isnan(quot[i,:])])

    tt = [len(x) for x in xna]
    idx_empty = np.where(np.array(tt) == 0)
    print('Removing spectrum/spectra '+str(idx_empty))
    idx_keep = np.where(np.array(tt) > 0)[0]
    x_raw=x_raw[idx_keep]
    quot=quot[idx_keep]
    xna=np.array(xna)[idx_keep]

    #emp=tfp.distributions.Empirical(xna.tolist())
    #mea=emp.mean()
    mea=np.nanmean(xna, 1)
    #mo=emp.mode()
    me=np.nanmedian(xna, 1)
   
    x_norm=np.ones(x_raw.shape)
    for i in range(x_raw.shape[0]):
        x_norm[i,:]=x_raw[i,:]/me[i][..., np.newaxis]
    
    if log:
        x_norm=10**x_norm
        #x_n1=10**x_norm
    
    dilfs = pd.DataFrame({'median':me, 'mean':mea})
    #dilfs = pd.DataFrame({'med':me, 'mod':mo, 'me':mea})

    
    
    return (x_norm, dilfs)