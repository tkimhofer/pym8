#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module imports 1D and 2D NMR spectra
"""

import subprocess
import pandas as pd
import numpy as np
import os
import nmrglue as ng
from scipy import interpolate
import tensorflow as tf
 #import subprocess

    


def list_exp(path, return_results=False, pr=True):
     """
    List all NMR experiment files in directory
    Args:
        path:  Directory path (string)
        return_results:     return DataFrame containing file information (logic)
        pr: Print overview table in console
    Returns:
        DataFrame of NMR experiment information - input for importing functions
    """
    cmd = 'find '+path+' -type f -iname "acqus" ! -name ".*" -print0' + ' | xargs -0 grep EXP=' 
    sp = subprocess.getoutput(cmd)
    out=sp.split('\n')
    df=pd.DataFrame({'id': out})
    out=df.id.str.split(':', n = 1, expand = True)
    df['exp']=out[1]
    df['exp']=df.exp.str.replace('.*<|>', '', regex=True)
    df['fid']=df.id.str.replace('/acqus.*', '', regex=True)
    
    
    fsize=list()
    mtime=list()
    # check if procs exists
    for i in range(df.shape[0]):
        fname=df.fid[i]+'/pdata'
        inf=os.stat(fname)
        try:
            inf
        except NameError:
            mtime.append(None)
            fsize.append(None)
            continue

        mtime.append(inf.st_mtime)
        fsize.append(inf.st_size)
    
    df['size']=fsize
    df['mtime']=mtime
    
    summary=df.groupby(['exp']).agg( n=('size','count'), size_byte= ('size', 'mean'), maxdiff_byte=('size', lambda x: max(x)-min(x)), mtime=('mtime','max')).reset_index()
    summary.sort_values(by ='n', ascending = False)
    summary.mtime=pd.to_datetime(summary.mtime, unit='s').dt.floor('T')
    
    if pr:
        print(summary)
    
    if return_results:
        return df



# load 1d and 2d in same go, order rows to match spectra
def read1d2d_raw(ll, exps=['PROF_URINE_NOESY', 'PROF_URINE_JRES'], n_max=100):
     """
    Import matching 1D processed and 2D raw NMR data
    Args:
        ll:  DataFrame of experiment information (see list_exp())
        exps:   Ordered list of two: 1d and 2D experiment names (used for filtering)
        n_max: Maximum number of experiment read-ins (int)
    Returns:
        Tuple of six: X, ppm, met, X2, ppm1, ppm2
    """
    out=ll.id.str.rsplit('/',2)
    ll['path']=out.str[0]
    ll['eid']=out.str[1]
    ll['eiid']=ll.eid.str.replace('[0-9]$', 'u', regex=True)
    ll['uid']=ll.path +'/'+ ll.eiid
    
    exp1=ll.loc[ll.exp==exps[0]]
    exp2=ll.loc[ll.exp==exps[1]]
   
    # remove doubles from exp1
    ct=exp1.uid.value_counts()>1
    doubl=ct.loc[ct].index
    
    if len(doubl) > 0:
        idx_rm=[]
        for i in range(len(doubl)):
            idx=np.where(exp1.uid==doubl[i])[0]
            idx_rm.append(idx[0::(len(idx))])
        exp1=exp1.drop(exp1.index[[np.array(idx_rm).ravel()]])
    
    
    # remove doubles from exp2
    ct=exp2.uid.value_counts()>1
    doubl=ct.loc[ct].index
    
    if len(doubl) > 0:
        idx_rm=[]
        for i in range(len(doubl)):
            idx=np.where(exp2.uid==doubl[i])[0]
            idx_rm.append(idx[0::(len(idx))])
        exp2=exp2.drop(exp2.index[[np.array(idx_rm).ravel()]])
     
    # establish mapping for exp1 and exp2
    sxp = pd.merge(exp1, exp2, how='inner', on=['uid'])
    
    if n_max<sxp.shape[0]:
        sxp=sxp.iloc[0:n_max]
    
    exp1=sxp.filter(regex=('_x$'))
    exp1.columns=exp1.columns.str.replace('_x', '')

    # read in both data sets using 1D and 2D read functoins
    X, ppm, met= import1d_procs(flist=exp1, exp_type=exps[0])
    
    exp2=sxp.filter(regex=('_y$'))
    exp2.columns=exp2.columns.str.replace('_y', '')
    X2, ppm1, ppm2 =import2d_raw(exp2, exp_type=exps[1], n_max=10000)
    
    return (X, ppm, met, X2, ppm1, ppm2)
    


def import2d_raw(flist, exp_type, calib=True, n_max=10000):
    """
    Import 2D raw NMR data
    Args:
        flist:  DataFrame of experiment information (see list_exp())
        exp_type:   2D experiment name, string (used for filtering)
        calib: Chemical shift calibration to TSP, logic
        n_max: Maximum number of experiment read-ins (int)
    Returns:
        Tuple of three: X, ppm1, ppm2
    """
    
    idx=np.where(flist.exp==exp_type)[0]
    # print('Experiments found: ' + str(len(idx)))
    
    fexp=flist.iloc[idx,:].reset_index()
    if n_max < fexp.shape[0]:
        fexp=fexp.iloc[0:n_max,:]
    
    for i in range(fexp.shape[0]):
        # print(i)
        
        fexp.loc[i, 'fid']
        spec, ppm1, ppm2= import2dJres(fexp.loc[i, 'fid'])
        
        if i==0:
            ppm_ord1=ppm1
            ppm_ord2=ppm2
            
            smat=np.ones((len(ppm_ord1), len(ppm_ord2), fexp.shape[0]))
            smat[:,:, i]=spec.numpy()
        else:    
         imat = interpolate.interp2d(ppm1, ppm2, spec.numpy().T, kind='cubic')
         xout=imat(ppm_ord1, ppm_ord2).T
         smat[:,:, i]=xout

    
    return (smat, ppm_ord1, ppm_ord2)


def import2dJres(path):
    """
    Imports a single 2D processed NMR spectrum
    Args:
        path:  Path to experiment directory
    Returns:
        Tuple of three: X, ppm1, ppm2
    """
    acqus=ng.bruker.read_acqus_file(path)
    p1=path+('/ser')
    jres=importJres(p1, acqus, pad='none')
    ppm=cppm(acqus, jres.shape[1])
    # align ppm
    idx=np.where((ppm>-0.2) & (ppm <0.2))[0]
    sub=jres.numpy()[:,idx]
    # plt.imshow(sub)
    idx_tsp=np.unravel_index(sub.argmax(), sub.shape)
    ppm2=ppm-ppm[idx][idx_tsp[1]]

    le=acqus['acqu2s']['SW']/2
    ppm1=np.linspace(-le, le, jres.shape[0])
    
    return (jres, ppm1, ppm2)


def import1d_procs(flist, exp_type):
    #import subprocess
    import pandas as pd
    import numpy as np
    import os
    import nmrglue as ng
    """
    Imports 1D processed NMR spectra
    Args:
        flist:  DataFrame of experiment information (see list_exp())
        exp_type:   2D experiment name, string (used for filtering)
    Returns:
        Tuple of three: X, ppm, meta
    """
    
    #idx=np.where(flist.exp==exp_type)[0]
    #fexp=flist.iloc[idx,:].reset_index(drop=True)
    #print('Experiments found: ' + str(len(idx)))
    fexp=flist.loc[flist.exp.isin(exp_type)].reset_index(drop=True)
    
    lacqus=[]
    lprocs=[]
    idx_filter=[]
    c=0
    for i in range(fexp.shape[0]):
        #print(i)
        
        f_path= os.path.join(fexp.loc[i, 'fid'], '')+'pdata/1'
        
        p1=f_path+'/1r'
        if not os.path.isfile(p1):
            continue
        
        meta, spec=ng.bruker.read_pdata(f_path)
        SF01 = meta['procs']['OFFSET']  
        SF = meta['procs']['SF']                                      
        SW = meta['procs']['SW_p']/SF                            
        FTsize = meta['procs']['FTSIZE']     
        ppm=np.linspace(SF01, SF01-SW, FTsize)
        
        if c==0:
            ppm_ord=ppm
            smat=np.ones((fexp.shape[0], len(ppm_ord)))
            smat[0,:]=spec
           
        else:
            # interpolate spec to same ppm values across experiments
            s_interp=np.flip(np.interp(np.flip(ppm_ord), np.flip(ppm), np.flip(spec)))
            smat[i,:]=s_interp
        lacqus.append(meta['acqus'])
        lprocs.append(meta['procs'])
        idx_filter.append(c)
        c=c+1
        #exp.append((spec, ppm, meta))
    
    smat=smat[np.array(idx_filter),:]
    procs=pd.DataFrame(lprocs)
    acqus=pd.DataFrame(lacqus)
    
    meta=pd.concat([acqus, procs], axis=1)
    meta['id']=fexp.id.iloc[np.array(idx_filter)]
    
    meta.index=["s" + str(x) for x in meta.index]
    
    #meta['id']=fexp.id.iloc(np.array(idx_filter))
    
    ab=np.split(meta._comments.values, '')[0]
    dtime=list()
    for i in range(len(ab)):
        dtime.append(pd.to_datetime(re.sub('\$\$ |\+.*', '', ab[i][0][0])))
    meta['datetime']=dtime
        
    print('Experiments read-in: ' + str(meta.shape[0]))
    
    return (smat, ppm_ord, meta)
  


def import2d_procs(flist, exp_type, calib=True, n_max=10000):
   """
    Imports 2D processed NMR spectra
    Args:
        flist:  DataFrame of experiment information (see list_exp())
        exp_type:   2D experiment name, string (used for filtering)
        calib:   Chemical shift calibration to TSP (log)
        n_max:   Max number of spectra to read-in (int)
    Returns:
        Tuple of four: X, ppm1, opm2, meta
    """
    
    idx=np.where(flist.exp==exp_type)[0]
    #print('Experiments found: ' + str(len(idx)))
    
    fexp=flist.iloc[idx,:].reset_index()
    if n_max < fexp.shape[0]:
        fexp=fexp.iloc[0:n_max,:]
    
    lacqus=[]
    lprocs=[]
    for i in range(fexp.shape[0]):
        #print(i)
        
        f_path= os.path.join(fexp.loc[i, 'fid'], '')+'pdata/1'
        meta, spec=ng.bruker.read_pdata(f_path)

        #f2 dimension
        SF01 = meta['procs']['OFFSET']  
        SF = meta['procs']['SF']                                      
        SW = meta['procs']['SW_p']/SF                            
        FTsize = meta['procs']['FTSIZE']     
        ppm2=np.linspace(SF01, SF01-SW, FTsize)
        
        #f2 dimension
        SF01 = meta['proc2s']['OFFSET']  
        SF = meta['proc2s']['SF']                                      
        SW = meta['proc2s']['SW_p']/SF                            
        FTsize = meta['proc2s']['FTSIZE']     
        ppm1=np.linspace(SF01, SF01-SW, FTsize)
      
        if calib:
            ppm1, ppm2 = calib_axis_2d(spec, ppm1, ppm2)
        
        if i==0:
            ppm_ord1=np.flip(ppm1)
            ppm_ord2=np.flip(ppm2)
            
            smat=np.ones((len(ppm_ord1), len(ppm_ord2), fexp.shape[0]))
            smat[:,:, i]=np.flip(spec)
        else:    
         imat = interpolate.interp2d(np.flip(ppm1), np.flip(ppm2), np.flip(spec,1).T, kind='cubic')
         xout=imat(ppm_ord1, ppm_ord2)
         smat[:,:, i]=xout.T
        lprocs.append(meta['procs'])
        lacqus.append(meta['acqus'])
        #exp.append(meta)
    
    procs=pd.DataFrame(lprocs)
    acqus=pd.DataFrame(lacqus)
    
    meta=pd.concat([acqus, procs], axis=1)
    meta['id']=fexp.id.values
    
    meta.index=["s" + str(x) for x in meta.index]
    
    meta['id']=fexp.id.values
    ab=np.split(meta._comments.values, '')[0]
    dtime=list()
    for i in range(len(ab)):
        dtime.append(pd.to_datetime(re.sub('\$\$ |\+.*', '', ab[i][0][0])))
    meta['datetime']=dtime
        
    print('Experiments read-in: ' + str(meta.shape[0]))
    
    return (smat, ppm_ord1, ppm_ord2, meta)


# load 1d and 2d in same go, order rows to match spectra
def read1d2d(ll, exps=['PROF_URINE_NOESY', 'PROF_URINE_JRES'], n_max=100):
    
    """
    Imports 1D and 2D processed NMR spectra
    Args:
        ll:  DataFrame of experiment information (see list_exp())
        exps:   Ordered list of two: 1d and 2D experiment names (used for filtering)
        n_max: Maximum number of experiment read-ins (int)
    Returns:
        Tuple of seven: X, ppm, meta, X2, ppm1, ppm2, meta2
    """

    out=ll.id.str.rsplit('/',2)
    ll['path']=out.str[0]
    ll['eid']=out.str[1]
    ll['eiid']=ll.eid.str.replace('[0-9]$', 'u', regex=True)
    ll['uid']=ll.path +'/'+ ll.eiid
    
    exp1=ll.loc[ll.exp==exps[0]]
    exp2=ll.loc[ll.exp==exps[1]]
   
    # remove doubles from exp1
    ct=exp1.uid.value_counts()>1
    doubl=ct.loc[ct].index
    
    if len(doubl) > 0:
        idx_rm=[]
        for i in range(len(doubl)):
            idx=np.where(exp1.uid==doubl[i])[0]
            idx_rm.append(idx[0::(len(idx))])
        exp1=exp1.drop(exp1.index[[np.array(idx_rm).ravel()]])
    
    
    # remove doubles from exp2
    ct=exp2.uid.value_counts()>1
    doubl=ct.loc[ct].index
    
    if len(doubl) > 0:
        idx_rm=[]
        for i in range(len(doubl)):
            idx=np.where(exp2.uid==doubl[i])[0]
            idx_rm.append(idx[0::(len(idx))])
        exp2=exp2.drop(exp2.index[[np.array(idx_rm).ravel()]])
     
    # establish mapping for exp1 and exp2
    sxp = pd.merge(exp1, exp2, how='inner', on=['uid'])
    
    if n_max<sxp.shape[0]:
        sxp=sxp.iloc[0:n_max]
    
    exp1=sxp.filter(regex=('_x$'))
    exp1.columns=exp1.columns.str.replace('_x', '')

    # read in both data sets using 1D and 2D read functoins
    X, ppm, met= import1d_procs(flist=exp1, exp_type=exps[0])
    
    exp2=sxp.filter(regex=('_y$'))
    exp2.columns=exp2.columns.str.replace('_y', '')
    X2, ppm1, ppm2, meta2 =import2d_procs(exp2, exp_type=exps[1], n_max=10000)
    
    return (X, ppm, met, X2, ppm1, ppm2, meta2)
    
path='/Users/torbenkimhofer/Desktop/glycStds/GlycStandards_Haptoglobin_310K_Pn_IVDR01_280421/1/'
# read bruker fid from 1d
def read1dFID(path, ret='fid', win=exp_win, zf=2):
    acqus=ng.bruker.read_acqus_file(path)
    p1=path+'fid'
    acqus.keys()
    nc=acqus['acqus']['NS']
    bp=acqus['acqus']['BYTORDA']
    if bp==0:
        bo='<i4'
    else:
        bo='>i4'
        
    dat = tf.cast(np.fromfile(p1, dtype=np.dtype(bo))*(2**nc), tf.float32)
    
    # conv to comples
    fid=tf.complex(real=dat[0::2], imag=dat[1::2])
    
    # remove digital filter
    gd=tf.cast(tf.math.ceil(tf.cast(acqus['acqus']['GRPDLY'], tf.float32)), tf.int32)
    
    fid=fid[(gd+1):,]
    
    if ret=='fid':
        return fid
    
    wf=win(n=fid.shape[0], dtype=dat.dtype, lb=0.3)
    win_cpl=tf.complex(real=wf, imag=wf)
    
    fwin=fid*win_cpl
    
    if ret=='win':
        return fwin
    
    opt2=tf.math.ceil(np.log2(tf.cast(fwin.shape[0], tf.float64)))
    p1=tf.abs(tf.math.pow(2, opt2)-fwin.shape[0])
    p=[[0, p1], [0, 0]]
    fwinp=tf.pad(fwin[..., tf.newaxis], p )
    
    p=[[0, fwinp.shape[0]], [0, 0]]
    fwinpp=tf.squeeze(tf.pad(fwinp, p ))
    
    if ret=='zf':
        return fwinpp
    
    sp=tf.signal.fft(fwinpp)
    sps=tf.signal.fftshift(sp)
    
    
    if ret='fft':
        return sps
    
    # phasing
    h=sps[1::]-sps[0:-1]
    
# The objective function
def sqrt_quadratic(x):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=-1))

start = tf.constant([6.0, -21.0])  # Starting point for the search.
optim_results = tfp.optimizer.nelder_mead_minimize(
    sqrt_quadratic, initial_vertex=start, func_tolerance=1e-8,
    batch_evaluate_objective=True)    
  plt.plot((sps[70000:]))
  tf.cast(fwin.shape, tf.float32) % 2
    
    
    # plt.plot(tf.math.real(test), tf.math.imag(test))
    
    # plt.plot(tf.math.real(test))
    # test.shape
    
    # plt.plot(tf.math.real(fid))

def init():
    A = tf.Variable(1.0) 
    B = tf.Variable(1.0) 
    C = tf.Variable(1.0) 
    D = tf.Variable(1.0) 
    
    return A, B, C, D
A, B, C, D = init()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

y=tf.math.real(fwin).numpy()
ymax=np.max(np.abs(y))
y=y/ymax
x=tf.cast(np.arange(len(y)), tf.float32)



for epoch in range(1000):
    opt.minimize(lambda: sum(abs(y - (A*x**3+ B*x**2+C*x+D))), var_list=[A, B, C, D])

y_hat=(A*x**3+ B*x**2+C*x+D)
print(sum(abs(y - y_hat)).numpy(), A.numpy(), B.numpy(), C.numpy(), D.numpy(), end="\r")
print()
print(A.numpy())
print(B.numpy())
print(C.numpy())
print(D.numpy())
    
    plt.plot(tf.math.real(fid), tf.math.imag(fid))
    
    fig, axs = plt.subplots(2)
    #fig.suptitle('CWT')
    axs[0].plot(tf.math.real(fid))
    axs[1].plot(tf.math.imag(fid))
    #plt.plot(tf.math.real(fid), tf.math.imag(fid))

sc= np.arange(1, 3021)
cA, cD =pywt.cwt(fid.numpy(), sc,   'morl', 1)

tt=cA/np.sqrt(sc)[..., np.newaxis]

plt.imshow(np.log(np.abs(tt.real[:,0:1000])).T)
plt.imshow(np.log(np.abs(tt.real[:,1000:2000])).T)
plt.imshow(np.log(np.abs(tt.real[:,3000:4000])).T)


plt.imshow(np.abs(cA.real[:,0:1000]))

plt.imshow(np.log(np.abs(cA.imag[:,1000:2000])))
plt.imshow(np.log(np.abs(cA.imag[:,5000:6000])))



fig, axs = plt.subplots(2)
fig.suptitle('CWT')
axs[0].plot(sig)
axs[1].imshow(cA)

    







