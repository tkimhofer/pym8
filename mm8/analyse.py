#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis functions
@author: torbenkimhofer
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
# import plotnine as pn
from mizani.formatters import scientific_format
import mm8.utility
from scipy.stats import chi2

def cor_heatmap(Xc, n_max=600, val_excl=0, ct='rho', title=''):
    """
        Correlation Heatmap
        Args:
            Xc: Pandas or Numpy object of rank 2 with features in columns
            n_max: Maximum number of allowed missing/excluded observations
            val_excl: Observations values that are excluded from analysis (0 for Bruker fits)
            ct: Correlation type (rho for rank correlation or r for Pearson's correlation)
        Returns:
            Tuple of two: 1. tuple of ax, fig, 2: correlation matrix (numpy rank 2)
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.cluster.hierarchy import linkage

    if isinstance(Xc, pd.DataFrame):
        labs=Xc.columns.values
        Xc=Xc.to_numpy()
    else:
        if not isinstance(Xc, np.ndarray):
            raise ValueError('Provide pandas DataFrame or numpy ndarray')
        labs = np.array(['Feat ' + str(x) for x in range(Xc.shape[0])])

    if not Xc.dtype.kind in set('buifc'):
        raise ValueError('Provide numeric values')

    idx_keep=np.where(np.sum((Xc==val_excl) | (Xc == np.nan), 0) <= n_max)[0]
    if len(idx_keep) < 2:
        raise ValueError('Number of features with missing/excluded values below n_min')
    Xc=Xc[:,idx_keep]
    labs=labs[idx_keep]

    cc = np.zeros((Xc.shape[1], Xc.shape[1]))
    tups = itertools.combinations(range(Xc.shape[1]), r=2)

    if ct == 'rho':
        cfun=mm8.utility._rho
    else:
        cfun=mm8.utility._cov_cor

    for i in tups:
        x=Xc[:,i[0]]
        y=Xc[:,i[1]]
        idx = np.where(~(x == val_excl) & ~(y == val_excl))[0]
        xcov, xcor = cfun(np.log(x[idx]), np.log(y[idx]))
        cc[i[0], i[1]]=xcor

    cs=cc + cc.T
    ft=fthresh*cs.shape[0]
    idx_keep=np.where(np.nansum(np.isnan(cs),1)<ft)[0]
    if len(idx_keep) < 2:
        raise ValueError('Number of selected features < 2: Decrease ftrehs parameter')
    ps=cs[np.ix_(idx_keep, idx_keep)]
    np.fill_diagonal(ps,1)

    # reorder based on ward linkage
    Z = linkage(1-ps, method='ward')
    cord = Z[:, 0:2].ravel()
    cord = cord[cord < ps.shape[0]].astype(int)
    labs=labs[cord]
    ps = ps[np.ix_(cord, cord)]


    fig, ax = plt.subplots(tight_layout=True)
    heatmap = ax.pcolor(ps, cmap=plt.cm.rainbow, norm=Normalize(vmin=-1, vmax=1))
    fig.colorbar(heatmap)
    ax.set_title(title)

    ax.set_xticks(np.arange(ps.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(ps.shape[1])+0.5, minor=False)
    ax.set_xticklabels(labs[idx_keep], minor=False, rotation=90)
    ax.set_yticklabels(labs[idx_keep], minor=False)

    plt.show()

    return (fig, ax), ps


class stocsy:
    """
    Create STOCSY class
    
    Args:
        X: NMR matrix rank 2
        ppm: chemical shift vector rank 1
    Returns:
        class stocsy
    """
    def __init__(self, X, ppm):
        self.X = X
        self.ppm = ppm

    def trace(self, d, shift=[0,10], interactive=False, spectra=True):
        """
        Perform STOCSY analysis
        Args:
            d: Driver peak position (ppm)
            shift: Chemical shift range as list of length two
            interactive: boolean, True for plotly, False for plotnine
        Returns:
            graphics object
        """
        shift=np.sort(shift)
        idx=np.argmin(np.abs(self.ppm-d))
        y=np.reshape(self.X[:,idx], (np.shape(self.X)[0], 1))
        xcov, xcor = mm8.utility._cov_cor(y, self.X)
        
        if interactive:

            pio.renderers.default = "browser"
            idx_ppm=np.where((self.ppm>=shift[0]) & (self.ppm <= shift[1]))[0]
            t = xcor[0][idx_ppm]
            x, y = self.ppm[idx_ppm], xcov[0][idx_ppm]
           
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers+lines', marker={'color': t, 'colorscale': 'Rainbow', 'size': 5, 'colorbar':dict(title="|r|")}, line={'color': 'black'}) )
            fig.update_xaxes(autorange="reversed")
            fig.show()
            return fig

        else:
            from matplotlib.collections import LineCollection
            from matplotlib.colors import ListedColormap, BoundaryNorm
            import matplotlib.pyplot as plt
            x=np.squeeze(self.ppm)
            y=np.squeeze(xcov)
            z=np.abs(np.squeeze(xcor))
            xsub=self.X

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(z.min(), z.max())
            lc = LineCollection(segments, cmap='rainbow', norm=norm)
            # Set the values used for colormapping
            lc.set_array(z)
            lc.set_linewidth(2)

            dd = (x.max() - x.min()) / 30
            if spectra:
                fig, axs = plt.subplots(2, 1, sharex=True)
                line = axs[0].add_collection(lc)
                fig.colorbar(line, ax=axs)
                axs[0].set_xlim(x.max() + dd, (x.min() - dd))
                axs[0].set_ylim(y.min() * 1.1, y.max() * 1.1)
                axs[0].vlines(d, ymin=(y.min() * 1.1), ymax=(y.max() * 1.1), linestyles='dotted', label='driver')
                axs[1].plot(x, xsub.T, c='black', linewidth=0.3)
                axs[1].vlines(d, ymin=(xsub.min() * 1.1), ymax=(xsub.max() * 1.1), linestyles='dotted', label='driver',
                              colors='red')
            else:
                fig, axs = plt.subplots(1, 1)
                line = axs.add_collection(lc)
                fig.colorbar(line, ax=axs)
                axs.set_xlim(x.max() +dd, (x.min() - dd))
                axs.set_ylim(y.min() * 1.1, y.max() * 1.1)
                axs.vlines(d, ymin=(y.min() * 1.1), ymax=(y.max() * 1.1), linestyles='dotted', label='driver')

            return (axs, fig)
            #
            # dd=pd.DataFrame({'ppm':np.squeeze(self.ppm), 'cov':np.squeeze(xcov), 'cor':np.abs(np.squeeze(xcor))})
            # idx_ppm=np.where((dd.ppm>=shift[0]) & (dd.ppm <= shift[1]))[0]
            # dd=dd.iloc[idx_ppm]
            # dd['fac']='STOCSY: d='+str(d)+' ppm'+', n='+ str(self.X.shape[0])
            #
            # rainbow=["#0066FF",  "#00FF66",  "#CCFF00", "#FF0000", "#CC00FF"]
            #
            # g=pn.ggplot(dd, pn.aes(x='ppm', y='cov', color='cor'))+pn.geom_line()+pn.scale_x_reverse()+pn.scale_colour_gradientn(colors=rainbow, limits=[0,1])+pn.theme_bw()+pn.labs(color='r(X,d)')+ pn.scale_y_continuous(labels=scientific_format(digits=2))+pn.facet_wrap('~fac')
            #

        
        

class pca:
       # mm8 pca class
       # methods: plot_scores, plot_load
      """
      Create PCA class
        
      Args:
            X: NMR matrix rank 2
            ppm: chemical shift vector rank 1
            pc: Number of desired principal components
            center: boolean, mean centering
            scale: 'uv'
      Returns:
            pca class
      """
      
      def __init__(self, X, ppm, nc=2, center=True, scale='uv'):
        from sklearn.decomposition import PCA
        #from matplotlib.pyplot import plt
        self.X = X
        self.ppm = ppm
        self.nc = nc
        self.center=center
        self.scale=scale
        self.means=np.mean(X, 0)
        self.std=np.std(X, 0)

        if any(self.std == 0):
            print('Matrix contains columns with zero standard deviation - replacing these with eps = 1e-7')

        self.std[self.std == 0]=1e-7

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


        xcov, xcor = mm8.utility._cov_cor(self.t, self.X)
           
        self.Xcov=xcov
        self.Xcor=xcor
       
       
      def plot_scores(self, an , pc=[1, 2], hue=None, labs=None, legend_loc='right'):
           # methods: plot_scores, plot_load
        import seaborn as sns
        """
        Plot PCA scores (2D)
          
        Args:
              an: Pandas DataFrame containig colouring variable as column
              pc: List of indices of principal components, starting at 1, length of two
              hue: Column name in an of colouring variable
              labs: None or list of strings containing scores plot labels
              legend_loc: Legend locatoin given as string ('right', 'left', 
        Returns:
              plotting object
        """
        self.an=an
        pc=np.array(pc)
        cc=['t' + str(sub) for sub in np.arange(self.t.shape[1])+1]
        df=pd.DataFrame(self.t, columns=cc)
       
        if self.an.shape[0]!= df.shape[0]:
            raise ValueError('Dimensions of PCA scores and annotation dataframe don\'t match.')
            #return Null
       
        ds=pd.concat([df.reset_index(drop=True), an.reset_index(drop=True)], axis=1)
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

        if labs is not None:
            if (len(labs) != len(x)):  raise ValueError('Check length of labs')
            for i in range(len(labs)):
                fg.axes[0,0].annotate(labs[i], (x[i], y[i]))
        fg.add_legend(loc=legend_loc)
    
        return fg
    
      def plot_load(self, pc=1, shift=[0, 10]):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.pyplot as plt
        """
        Plot statistical reconstruction of PCA loadings 
          
        Args:
              pc: Index of principal components, starting at 1
              shift: Chemical shift range (list of 2)
        Returns:
              plotting object
        """
       
         # print(shift)
        shift=np.sort(shift)
        x=self.ppm
        y=self.Xcov[pc,:]
        z=self.Xcor[pc,:]
        idx=np.where((x>=shift[0]) & (x <= shift[1]))[0]
        x=x[idx]
        y=y[idx]
        z=np.abs(z[idx])
        xsub=self.X[:,idx]

        fig, axs = plt.subplots(2,1, sharex=True)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(z.min(), z.max())
        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        lc.set_linewidth(2)
        line = axs[0].add_collection(lc)
        fig.colorbar(line, ax=axs)
        dd = (x.max() - x.min()) / 30
        axs[0].set_xlim(x.max() +dd, (x.min() - dd))
        axs[0].set_ylim(y.min() * 1.1, y.max() * 1.1)

        axs[1].plot(x, xsub.T, c='black', linewidth=0.3)

        return (axs, fig)

        #
        # df=pd.DataFrame({'ppm':x, 'cov':y, 'cor':np.abs(z)})
        # df['fac']='PCA: p'+str(pc)
        #
        # rainbow=["#0066FF",  "#00FF66",  "#CCFF00", "#FF0000", "#CC00FF"]
        # g=pn.ggplot(pn.aes(x='ppm', y='cov', color='cor'), data=df)+pn.geom_line()+pn.scale_colour_gradientn(colors=rainbow, limits=[0,1])+pn.theme_bw()+pn.scale_x_reverse()+ pn.scale_y_continuous(labels=scientific_format(digits=2))+pn.facet_wrap('~fac')+pn.labs(color='|r|')
        # return(g)

