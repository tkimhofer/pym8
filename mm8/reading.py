#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module imports 1D and 2D NMR spectra
"""

import subprocess
import sys

import pandas as pd
import numpy as np
import os
import nmrglue as ng
from scipy import interpolate
import tensorflow as tf
import re
import mm8.utility
import xml.etree.ElementTree as xl

def eretic_factor(mc):
    mf = mc.id.str.replace('acqus:.*', '', regex=True)
    ere = []
    ere_len=[]
    for i in range(mc.shape[0]):
        #print(i)
        try:
            ff = xl.parse(mf.iloc[i] + 'QuantFactorSample.xml')
            fr = ff.getroot()
            inf={
                'file': mf.iloc[i],
                fr[0][0][1].tag: float(fr[0][0][1].text),
                fr[0][0][2].tag: float(fr[0][0][2].text),
                fr[0][0][3].tag: float(fr[0][0][3].text),
                fr[0][0][4].tag: fr[0][0][4].text,
                fr[0][0][6][2].tag: list(fr[0][0][6][2].items())[0][1],
                fr[0][1][0][0].tag: float(fr[0][1][0][0].text),
                fr[0][0][6][5][10].tag: fr[0][0][6][5][10].text,
            }
            ere.append(inf)
        except:
            print("Not all exp contain file QuantFactorSample.xml ")
            inf=''
            ere.append(inf)
        ere_len.append(len(inf))
    # fill up where no eretic xml
    idx_repl=np.where(np.array(ere_len) != np.max(np.array(ere_len)))[0]
    rep=ere[np.where(np.array(ere_len) == np.max(np.array(ere_len)))[0][0]]
    rep = dict.fromkeys(rep, None)
    for i in range(len(idx_repl)):
        ere[idx_repl[i]]=rep
    return pd.DataFrame(ere)


def list_exp(path, return_results=True, pr=True, bruker_fits=True):
    """
    List all NMR experiment files in directory

    Args:
        path:  Directory path (string)
        return_results:     return DataFrame containing file information (logic)
        pr: Print overview table in console
    Returns:
        DataFrame of NMR experiment information - input for importing functions
    """
    ospl = sys.platform

    # if ospl == 'darwin' or ospl == 'linux':
    #     cmd = 'find ' + path + ' -type f -iname "acqus" ! -name ".*" -print0' + ' | xargs -0 grep -H EXP='
    #     sp = subprocess.getoutput(cmd)
    #     out = sp.split('\n')
    # else:

    out = []
    for dp, dn, fn in os.walk(path):
        for fid in [f for f in fn if f == 'acqus']:
            id = os.path.join(dp, 'acqus')
            with open(id, 'r') as fhand:
                for l in fhand:
                    if re.search('EXP=', l):
                        out.append(id + ':' + l.split('\n')[0])
                        break

    if len(out) ==0: raise ValueError('No experiments found')
    df = pd.DataFrame({'fid': out})
    out = df.fid.str.split(':', n=1, expand=True)
    df['exp'] = out.iloc[:, 1]
    df['exp'] = df.exp.str.replace('.*<|>', '', regex=True)
    # df['fid'] = df.id.str.replace('/acqus.*', '', regex=True)
    df['fid'] = df.fid.str.replace('acqus.*', '', regex=True)

    # if bruker_fits:
    # cmd = 'find ' + path + ' -iname "*quant_report*.xml"  -print0 | xargs -0 grep "QUANTIFICATION version="'
    # sp = subprocess.getoutput(cmd)
    # out = sp.split('\n')
    # bf=pd.DataFrame({'id': out})
    # outs=bf.id.str.split(':| *<QUANTIFICATION version="|">', expand=True)
    # bf['file']=outs.iloc[:,0]
    # bf['v'] = outs.iloc[:, 2]
    # bf['path']=outs.iloc[:,0].str.replace('/pdata/.*', '', regex=True)
    # bf['exp'] = bf.path.str.extract('/([0-9]{2,})$', expand=True)
    # out = bf.id.str.split(':|<QUANTIFICATION version="', expand=True)
    # check if bruker fits and qc's are included

    fsize = list()
    mtime = list()
    # check if procs exists
    for i in range(df.shape[0]):
        fname = os.path.join(df.fid[i], 'pdata')
        inf = os.stat(fname)
        try:
            inf
        except NameError:
            mtime.append(None)
            fsize.append(None)
            continue

        mtime.append(inf.st_mtime)
        fsize.append(inf.st_size)

    df['size'] = fsize
    df['mtime'] = mtime

    # summary=df.groupby(['exp']).agg( n=('size','count'), size_byte= ('size', 'mean'), maxdiff_byte=('size', lambda x: max(x)-min(x)), mtime=('mtime','max')).reset_index()
    summary = df.groupby(['exp']).agg(n=('size', 'count'), size_byte=('size', 'mean'),
                                      mtime=('mtime', 'max')).reset_index()

    summary.sort_values(by='n', ascending=False)
    summary.mtime = pd.to_datetime(summary.mtime, unit='s').dt.floor('T')
    summary = summary.iloc[summary.n.argsort().values]

    if pr:
        print(summary.iloc[::-1, :])

    if return_results:
        return df

def import1d_procs(flist, exp_type, eretic=True):
    """
    Imports 1D processed NMR spectra

    Args:
        flist:  DataFrame of experiment information (see list_exp())
        exp_type:   1D or 2D experiment name(s), list of string (used for filtering)
        eretic: bool, indicates if ERETIC signal information should be imported into meta (this errors if ERETIC is not calibrated)
    Returns:
        Tuple of three: X, ppm, meta
    """

    fexp = flist.loc[flist.exp.isin(exp_type)].reset_index(drop=True)

    lacqus = []
    lprocs = []
    idx_filter = []
    c = 0
    for i in range(fexp.shape[0]):
        f_path = os.path.join(fexp.loc[i, 'fid'], '') + 'pdata/1'

        p1 = f_path + '/1r'
        if not os.path.isfile(p1):
            continue

        tpath = os.path.join(fexp.loc[i, 'fid'], 'pdata', '1', 'title')
        if os.path.isfile(tpath):
            title = open(tpath, 'r').read()
        else:
            title = 'NA'

        meta, spec = ng.bruker.read_pdata(f_path)
        SF01 = meta['procs']['OFFSET']
        SF = meta['procs']['SF']
        SW = meta['procs']['SW_p'] / SF
        FTsize = meta['procs']['FTSIZE']
        ppm = np.linspace(SF01, SF01 - SW, FTsize)

        if c == 0:
            ppm_ord = ppm
            smat = np.ones((fexp.shape[0], len(ppm_ord)))
            smat[0, :] = spec

        else:
            # interpolate spec to same ppm values across experiments
            s_interp = np.flip(np.interp(np.flip(ppm_ord), np.flip(ppm), np.flip(spec)))
            smat[i, :] = s_interp

        meta['procs'].update({'title': title})
        lacqus.append(meta['acqus'])
        lprocs.append(meta['procs'])
        idx_filter.append(c)
        c = c + 1

    smat = smat[:c, :]
    procs = pd.DataFrame(lprocs)
    acqus = pd.DataFrame(lacqus)

    meta = pd.concat([acqus, procs], axis=1)
    meta['id'] = fexp.fid.iloc[idx_filter].values

    meta.index = ["s" + str(x) for x in meta.index]
    ab = np.split(meta._comments.values, '')[0]
    dtime = list()
    for i in range(len(ab)):
        dtime.append(pd.to_datetime(re.sub('\$\$ |\+.*', '', ab[i][0][0])))
    meta['datetime'] = dtime
    if eretic:
        ere = eretic_factor(meta)

        tsp_pos=ere.Artificial_Eretic_Position.dropna().unique()
        if len(tsp_pos)==1:
            idx = mm8.utility.get_idx(ppm_ord, [tsp_pos[0]-0.1, tsp_pos[0]+1])
            ere['eretic_integral']=np.sum(smat[:,idx], 1)
        ere.columns=ere.columns.str.lower()
        ere.index = meta.index
        meta = pd.concat([meta, ere], axis=1)
        eres = meta.eretic_factor.values[..., np.newaxis]
        smat = smat / eres
    print('Experiments read-in: ' + str(meta.shape[0]))
    return (smat, ppm_ord, meta)


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
    X, ppm, met= import1d_procs(flist=exp1, exp_type=exps[0], eretic=False)
    
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
    fexp=flist.iloc[idx,:].reset_index()
    if n_max < fexp.shape[0]:
        fexp=fexp.iloc[0:n_max,:]
    
    for i in range(fexp.shape[0]):
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
    jres=mm8.utility.importJres(p1, acqus, pad='none')
    ppm=mm8.utility.cppm(acqus, jres.shape[1])
    # align ppm
    idx=np.where((ppm>-0.2) & (ppm <0.2))[0]
    sub=jres.numpy()[:,idx]
    # plt.imshow(sub)
    idx_tsp=np.unravel_index(sub.argmax(), sub.shape)
    ppm2=ppm-ppm[idx][idx_tsp[1]]

    le=acqus['acqu2s']['SW']/2
    ppm1=np.linspace(-le, le, jres.shape[0])
    
    return (jres, ppm1, ppm2)




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
    
    fexp=flist.iloc[idx,:].reset_index()
    if n_max < fexp.shape[0]:
        fexp=fexp.iloc[0:n_max,:]
    
    lacqus=[]
    lprocs=[]
    for i in range(fexp.shape[0]):
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
            ppm1, ppm2 = mm8.utility.calib_axis_2d(spec, ppm1, ppm2)
        
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
    #ll = list_exp(path, return_results=True, pr=True, bruker_fits=True)
    #exps=[['PROF_URINE_NOESY','PROF_MeOD_NOESY_ECD'], 'PROF_URINE_JRES']
    """
    Imports 1D and 2D processed NMR spectra
    
    Args:
        ll:  DataFrame of experiment information (see list_exp())
        exps:   Ordered list of two: names of 1d and 2D experiments ids used for filtering, e.g., ['PROF_URINE_NOESY', 'PROF_URINE_JRES']
        n_max: Maximum number of experiment read-ins (int)
    Returns:
        Tuple of seven: X, ppm, meta, X2, ppm1, ppm2, meta2
    """

    out=ll.id.str.rsplit('/',2)
    ll['path']=out.str[0]
    ll['eid']=out.str[1]
    ll['eiid']=ll.eid.str.replace('[0-9]$', 'u', regex=True)
    ll['uid']=ll.path +'/'+ ll.eiid

    if not isinstance(exps[0], list):
        exps[0]=[exps[0]]

    if not isinstance(exps[1], list):
        exps[1] = [exps[1]]

    exp1=ll.loc[ll.exp.isin(exps[0])]
    exp2=ll.loc[ll.exp.isin(exps[1])]
   
    # remove doubles from exp1
    ct=exp1.uid.value_counts()>1
    doubl=ct.loc[ct].index
    
    if len(doubl) > 0:
        idx_rm=[]
        for i in range(len(doubl)):
            idx=np.where(exp1.uid==doubl[i])[0]
            idx_rm.append(idx[0::(len(idx))])
        exp1=exp1.drop(exp1.index[[np.array(idx_rm).ravel()]])
    ct=exp2.uid.value_counts()>1
    doubl=ct.loc[ct].index
    
    if len(doubl) > 0:
        idx_rm=[]
        for i in range(len(doubl)):
            idx=np.where(exp2.uid==doubl[i])[0]
            idx_rm.append(idx[0::(len(idx))])
        exp2=exp2.drop(exp2.index[[np.array(idx_rm).ravel()]])
    sxp = pd.merge(exp1, exp2, how='inner', on=['uid'])
    if n_max<sxp.shape[0]:
        sxp=sxp.iloc[0:n_max]
    exp1=sxp.filter(regex=('_x$'))
    exp1.columns=exp1.columns.str.replace('_x', '')
    X, ppm, met= import1d_procs(flist=exp1, exp_type=exps[0], eretic=False)
    exp2=sxp.filter(regex=('_y$'))
    exp2.columns=exp2.columns.str.replace('_y', '')
    X2, ppm1, ppm2, meta2 =import2d_procs(exp2, exp_type=exps[1][0], n_max=10000)
    
    return (X, ppm, met, X2, ppm1, ppm2, meta2)
    

def read1dFID(path, win, ret='fid', zf=2):
    """
    Import raw FID
    Args:
        path:  Directory to acqus file
        win:   Window function (see utility module)
        ret:   String indicating which data to return  (fid, win, zf, fft)
        zf:    Int zero-filling factor (e.g., 2: double the number data points)
    Returns:
        Data array depending on ret argument
    """
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
    fid=tf.complex(real=dat[0::2], imag=dat[1::2])
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
    if ret=='fft':
        return sps

# The objective function
def sqrt_quadratic(x):
    """
    L2 norm
    
    Args:
        x:  numeric array of rank 1
    Returns:
        Skalar
    """
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=-1))


