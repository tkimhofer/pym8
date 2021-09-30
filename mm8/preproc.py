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

tfd=tfp.distributions
tfb=tfp.bijectors



class stocsy:
    # mm8 stocsy V1
    # (c) torben kimhofer, 21/08/21
    def __init__(self, X, ppm):
        self.X = X
        self.ppm = ppm
    import numpy as np
    import pandas as pd

    def trace(self, d, shift=[0,10], interactive=True):
        shift=np.sort(shift)
       
        def cov_cor(X, Y):
            #x is pca scores matrix
            #y is colmean centered matrix
            
            if X.ndim == 1:
                X=np.reshape(X, (len(x), 1))
            
            if np.mean(Y[:,1])>1.0e-10:
                Y =(Y-np.mean(Y, 0)) 
                X =(X-np.mean(X, 0)) 
            
            xy=np.matmul(X.T, Y)
            cov = xy/(X.shape[0]-1)
            a=np.sum(X**2, 0)[..., np.newaxis]
            b=np.sum(Y**2, 0)[np.newaxis, ...]
            cor= xy / np.sqrt(a*b)
            
            return (cov, cor)
    
        idx=np.argmin(np.abs(self.ppm-d))
        y=np.reshape(self.X[:,idx], (np.shape(self.X)[0], 1))
        xcov, xcor = cov_cor(y, self.X)  
        
        if interactive:
            import plotly.graph_objects as go
            import plotly.io as pio
            pio.renderers.default = "browser"
            idx_ppm=np.where((self.ppm>=shift[0]) & (self.ppm <= shift[1]))[0]
            t = xcor[0][idx_ppm]
            x, y = self.ppm[idx_ppm], xcov[0][idx_ppm]
           
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers+lines', marker={'color': t, 'colorscale': 'Rainbow', 'size': 5, 'colorbar':dict(title="|r|")}, line={'color': 'black'}) )
            fig.update_xaxes(autorange="reversed")
            fig.show()
            return fig
            
            
            
        else:
            import plotnine as pn
            from mizani.formatters import scientific_format
            dd=pd.DataFrame({'ppm':np.squeeze(self.ppm), 'cov':np.squeeze(xcov), 'cor':np.abs(np.squeeze(xcor))})
            idx_ppm=np.where((dd.ppm>=shift[0]) & (dd.ppm <= shift[1]))[0]
            dd=dd.iloc[idx_ppm]
            dd['fac']='STOCSY: d='+str(d)+' ppm'+', n='+ str(self.X.shape[0])
            
            rainbow=["#0066FF",  "#00FF66",  "#CCFF00", "#FF0000", "#CC00FF"]
            
            g=pn.ggplot(dd, pn.aes(x='ppm', y='cov', color='cor'))+pn.geom_line()+pn.scale_x_reverse()+pn.scale_colour_gradientn(colors=rainbow, limits=[0,1])+pn.theme_bw()+pn.labs(color='r(X,d)')+ pn.scale_y_continuous(labels=scientific_format(digits=2))+pn.facet_wrap('~fac')
            
        return(g)
        
        

class pca:
       # mm8 pca class
       # methods: plot_scores, plot_load
      """
      Principla Components Analysis class and plotting fcts
      TODO: Annotation
      """
      def __init__(self, X, ppm, nc=2, center=True, scale='uv'):
        #from matplotlib.pyplot import plt
        self.X = X
        self.ppm = ppm
        self.nc = nc
        self.center=center
        self.scale=scale
        self.means=np.mean(X, 0)
        self.std=np.std(X, 0)
        self.Xsc=(self.X-self.means) / self.std
       
    
        if self.center and (self.scale=='uv'):
            X=self.Xsc
        else:
            if center:
                X=self.X
            if (scale=='uv'):
                X=X/self.std
        self.ss_tot=np.sum((X)**2)
        self.pca_mod = PCA(n_components=nc).fit(X)
        self.t = self.pca_mod.transform(X)
        self.p = self.pca_mod.components_
        
        tvar=np.sum(X**2)
        r2=[]
        for i in range(self.t.shape[1]):
            xc=np.matmul(self.t[:,i][np.newaxis].T, self.p[i,:][np.newaxis])
            r2.append((np.sum(xc**2)/tvar)*100)
        self.r2=r2
       
       
        def cov_cor(X, Y):
            #x is pca scores matrix
            #y is colmean centered matrix
           
            if X.ndim == 1:
                X=np.reshape(X, (len(x), 1))
           
            if np.mean(Y[:,1])>1.0e-10:
                Y =(Y-np.mean(Y, 0))
                X =(X-np.mean(X, 0))
           
            xy=np.matmul(X.T, Y)
            cov = xy/(X.shape[0]-1)
            a=np.sum(X**2, 0)[..., np.newaxis]
            b=np.sum(Y**2, 0)[np.newaxis, ...]
            cor= xy / np.sqrt(a*b)
           
            return (cov, cor)
       
        xcov, xcor = cov_cor(self.t, self.X)  
           
        self.Xcov=xcov
        self.Xcor=xcor
       
       
      def plot_scores(self, an , pc=[1, 2], hue=None, legend_loc='right'):
        self.an=an
        pc=np.array(pc)
        cc=['t' + str(sub) for sub in np.arange(self.t.shape[1])+1]
        df=pd.DataFrame(self.t, columns=cc)
       
        if self.an.shape[0]!= df.shape[0]:
            raise ValueError('Dimensions of PCA scores and annotation dataframe don\'t match.')
            #return Null
       
        ds=pd.concat([df.reset_index(drop=True), an.reset_index(drop=True)], axis=1)
        print(ds)
        #ds=ds.melt(id_vars=an.columns.values)
        #ds=ds.loc[ds.variable.str.contains('t'+str(pc[0])+"|t"+str(pc[1]))]
       
        # calculate confidence ellipse
       
        x=ds.loc[:, 't'+str(pc[0])]
        y=ds.loc[:,'t'+str(pc[1])]
        theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
        circle = np.array((np.cos(theta), np.sin(theta)))
        cov = np.cov(x,y)
        ed = np.sqrt(chi2.ppf(0.95, 2))
        ell = np.transpose(circle).dot(np.linalg.cholesky(cov) * ed)
        a, b = np.max(ell[: ,0]), np.max(ell[: ,1]) #95% ellipse bounds
        t = np.linspace(0, 2 * np.pi, 100)
       
        el_x=a * np.cos(t)
        el_y=b * np.sin(t)
       
        fg = sns.FacetGrid(ds, hue=hue)
        fg.axes[0][0].axvline(0, color='black', linewidth=0.5, zorder=0)
        fg.axes[0][0].axhline(0, color='black', linewidth=0.5, zorder=0)
       
    
        ax = fg.facet_axis(0,0)
    
        # fg.xlabel('t'+str(pc[0])+' ('+str(self.r2[pc[0]-1])+'%)')
        # fg.ylabel('t'+str(pc[1])+' ('+str(self.r2[pc[1]-1])+'%)')
        ax.plot(el_x, el_y, color = 'gray', linewidth=0.5,)
        fg.map(sns.scatterplot, 't'+str(pc[0]), 't'+str(pc[1]), palette="tab10")
       
        fg.axes[0][0].set_xlabel('t'+str(pc[0])+' ('+str(np.round(self.r2[pc[0]-1],1))+'%)')
        fg.axes[0,0].set_ylabel('t'+str(pc[1])+' ('+str(np.round(self.r2[pc[1]-1],1))+'%)')
       
        fg.add_legend(loc=legend_loc)
    
        return fg
    
      def plot_load(self, pc=1, shift=[0, 10]):
    
       
         # print(shift)
        shift=np.sort(shift)
           
        # print(x)
        # print(self.Xcor)
        # print(self.Xcov)
       
        x=self.ppm
        y=self.Xcov[pc,:]
        z=self.Xcor[pc,:]
        idx=np.where((x>=shift[0]) & (x <= shift[1]))[0]
        x=x[idx]
        y=y[idx]
        z=z[idx]
       
        df=pd.DataFrame({'ppm':x, 'cov':y, 'cor':np.abs(z)})
        df['fac']='PCA: p'+str(pc)
       
        rainbow=["#0066FF",  "#00FF66",  "#CCFF00", "#FF0000", "#CC00FF"]
        g=pn.ggplot(aes(x='ppm', y='cov', color='cor'), data=df)+geom_line()+scale_colour_gradientn(colors=rainbow, limits=[0,1])+theme_bw()+scale_x_reverse()+ scale_y_continuous(labels=scientific_format(digits=2))+facet_wrap('~fac')+labs(color='|r|')
        return(g)



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