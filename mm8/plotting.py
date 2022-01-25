#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 10:39:29 2021

@author: torbenkimhofer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from skimage.feature import peak_local_max
from plotly.subplots import make_subplots



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


def sub2d1d(x, ppm, x2, ppm1, ppm2, f1, f2, mdist=5, thr_abs='auto_100', log10=True):
    """
    Combined 1D and 2D NMR plot (two panels)

    Args:
        x: Single 1D NMR spectrum as array (rank 1)
        ppm: Chemical shift array for 1D (rank 1)
        x2: Single 2D NMR spectrum as array (rank 2, f1 in first dimension)
        ppm1: Chemical shift array f1 for 2D (rank 1)
        ppm2: Chemical shift array f2 for 2D (rank 1)
        f1: Chemical shift interval f1 (list)
        f2: Chemical shift interval f2 (list)
        mdist: Peak picking parameter - min distance
        thr_abs: Peak picking parameter - noise intensity threshold
        log10: Log transformation of x
    Returns:
        Null
    """


    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    idx=get_idx(ppm, f2)
    xs=x[idx]
    ppx=ppm[idx]


    fig.append_trace(go.Scatter(x=ppx, y=xs,
                    mode='lines',
                    name='lines'), row=1, col=1)

    idx_1=get_idx(ppm1, f1)
    idx_2=get_idx(ppm2, f2)

    xs2=x2[idx_1, :]
    xs2=xs2[:, idx_2]
    ppx1=ppm1[idx_1]
    ppx2=ppm2[idx_2]

    if isinstance(thr_abs, str):
        if 'auto' in thr_abs:
            ele=thr_abs.split('_')
            if len(ele)==2:
                thr_abs=np.max(xs2, (0,1))/float(ele[1])
            else:
                thr_abs=np.max(xs2, (0,1))/40
    else:
        if(len(thr_abs)==1):
            thr_abs=np.tile(thr_abs, xs2.shape[0])
        else:
            thr_abs=thr_abs


    if thr_abs == 'auto':
        thr_abs=np.max(xs2)/100

    if log10:
        xs2=np.log10(xs2)
        thr_abs=np.log10(thr_abs)

    pco = peak_local_max(xs2, min_distance=mdist, threshold_abs=thr_abs)
    p1=ppx1[pco[:,0]]
    p2=ppx2[pco[:,1]]
    idx_p=np.where(((p1<np.max(ppx1)) &( p1>np.min(ppx1))) & ((p2<np.max(ppx2)) & ( p2>np.min(ppx2))))[0]

    fig.append_trace(go.Contour(z=xs2, y=ppx1, x=ppx2, coloraxis = "coloraxis"), row=2, col=1)
    fig.append_trace(go.Scatter(x=ppx2[pco[idx_p,1]], y=ppx1[pco[idx_p,0]], mode='markers', marker_color='red', marker_size=10), row=2, col=1)

    fig.update_layout(coloraxis = {'colorscale':'viridis'})
    fig.update_xaxes(autorange='reversed')
    fig.update_yaxes(autorange='reversed', row=2, col=1)
    # fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=1
    # ))
    fig.show()




def sub2d_max(X1, ppm1, ppm2, f1, f2, mdist=5, thr_abs=2000, log10=True, lab=None, plist=None):

    """
    Contour plot of multiple 2D NMR spectra in horizontal panels

    Args:
        X1: 2D NMR data array (rank 2, f1 in first dimension)
        ppm1: Chemical shift array f1 (rank 1)
        ppm2: Chemical shift array f2 (rank 1)
        f1: Chemical shift interval f1 (list)
        f2: Chemical shift interval f2 (list)
        mdist: Peak picking parameter - min distance
        thr_abs: Peak picking parameter - noise intensity threshold
        lab: Plot title
        plist: Provide peak list
        log10: Log transformation of x
    Returns:
        Null
    """

    # Comparison between image_max and im to find the coordinates of local maxima

    fig = make_subplots(rows=X1.shape[2], cols=1, shared_xaxes=True, vertical_spacing=0.02)

    if isinstance(thr_abs, str):
        if 'auto' in thr_abs:
            ele=thr_abs.split('_')
            if len(ele)==2:
                thr_abs=np.max(X1, (0,1))/float(ele[1])
            else:
                thr_abs=np.max(X1, (0,1))/40
    else:
        if(len(thr_abs)==1):
            thr_abs=np.tile(thr_abs, X1.shape[0])
        else:
            thr_abs=thr_abs

    # if thr_abs == 'auto':
    #     thr_abs=np.max(X1, (0,1))/20

    if log10:
        X1=np.log10(X1)
        thr_abs=np.log10(thr_abs)

    for i in range(X1.shape[2]):
        x=X1[:,:,i]

        if plist is None:
            pco = peak_local_max(x, min_distance=mdist, threshold_abs=thr_abs[i])
        else:
            pco=np.array(plist[i].loc[:,['p1idx', 'p2idx']])
            #print(pco)

        p1=ppm1[pco[:,0]]
        p2=ppm2[pco[:,1]]



        idx_1=get_idx(ppm1, f1)
        idx_2=get_idx(ppm2, f2)

        x=x[idx_1, :]
        x=x[:, idx_2]

        idx_p=np.where(((p1<np.max(ppm1[idx_1])) &( p1>np.min(ppm1[idx_1]))) & ((p2<np.max(ppm2[idx_2])) & ( p2>np.min(ppm2[idx_2]))))[0]

        fig.append_trace(go.Contour(z=x, y=ppm1[idx_1], x=ppm2[idx_2], coloraxis = "coloraxis"), row=i+1, col=1)
        # if i ==0:
        #     fig.update_layout(showlegend=True)
        # else:
        #     fig.update_layout(showlegend=False)
        if lab is not None:
            fig.append_trace(go.Scatter(x=ppm2[pco[idx_p,1]], y=ppm1[pco[idx_p,0]], mode='markers', marker_color='red', marker_size=10, name=lab[i]), row=i+1, col=1)
        else:
            fig.append_trace(go.Scatter(x=ppm2[pco[idx_p,1]], y=ppm1[pco[idx_p,0]], mode='markers', marker_color='red', marker_size=10), row=i+1, col=1)




    fig.update_layout(coloraxis = {'colorscale':'viridis'})
    #fig['layout']['xaxis']['autorange'] = "reversed"
    fig.update_xaxes(autorange='reversed')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.show(renderer="chrome")




def ispec2d(x, ppm1, ppm2, shift1=[-10,10], shift2=[3,3.1], theme='gridon', ptype='surface', log10=False, return_fig=False):
    """
    Interactive plotting a single 2D NMR spectrum

    Args:
        x: NMR data array (rank 2, f1 in first dimension)
        ppm1: Chemical shift array f1 (rank 1)
        ppm2: Chemical shift array f2 (rank 1)
        shift1: Chemical shift interval f1 (list)
        shift2: Chemical shift interval f2 (list)
        theme: Plotly theme (str)
        ptype: Plot type as str: surface or contour
        log10: Log transformation of x
        return_fig: Logic, plotly obj return
    Returns:
        Null or plotly obj
    """
    pio.renderers.default = "browser"

    if x.ndim == 2:
        if (shift1 is not None) | (shift2 is not None):
            idx1=get_idx(ppm1, shift1)
            ppm1=ppm1[idx1]
            idx2=get_idx(ppm2, shift2)
            ppm2=ppm2[idx2]
            x=x[:,idx2]
            x=x[idx1,:]

            if log10:
                x=np.log10(x)

            if ptype == 'surface':
                fig = go.Figure(data=[go.Surface(z=x, x=ppm2, y=ppm1)])
                fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
                fig.update_layout(template=theme)


            if ptype == 'contour':

                fig = go.Figure(data=[go.Contour(z=x, x=ppm2, y=ppm1,
                                                 contours=dict(
                        coloring ='heatmap',
                        showlabels = True, # show labels on contours
                        labelfont = dict( # label font properties
                            size = 12,
                            color = 'white',
                        )
                    ))])

            if return_fig:
                return fig
            else:
                fig.show()


def ispec(x, ppm, shift=None, theme='gridon', lab=None):
    """
    Interactive plotting a single or multiple 1D NMR spectra

    Args:
        x: NMR data array (rank 1 or 2)
        ppm: Chemical shift array (rank 1)
        shift: Chemical shift interval (list)
        theme: plotly theme (str)
        lab: Line labels for spectrum/spectra
    Returns:
        NULL
    """
    pio.renderers.default = "browser"

    if x.ndim > 1:
        if shift is not None:
            idx=get_idx(ppm, shift)
            ppm=ppm[idx]
            x=x[:,idx]
        df=pd.DataFrame(x).reset_index().melt('index')
        df.variable=ppm[df.variable.astype(int)]
        df.columns=['Spectrum', 'ppm', 'Intensity']
        if lab is not None:
            df['Spectrum']=lab[df.Spectrum.values]

    else:
        if shift is not None:
            idx=get_idx(ppm, shift)
            x=x[idx]
            ppm=ppm[idx]
        if lab is not None:
            df=pd.DataFrame({'Spectrum':lab, 'ppm': ppm, 'Intensity':x})
        else:
            df=pd.DataFrame({'Spectrum':1, 'ppm': ppm, 'Intensity':x})

    fig= px.line(df, x="ppm", y="Intensity", color='Spectrum', template=theme)

    fig.update_layout(
        hovermode="closest",
        #hoverlabel = 'Spectrum: %{index}<extra></extra>',
        xaxis_title="ppm",
        yaxis_title="Intensity",
        legend_title="Spectrum",
        font=dict(
            #family="Courier New, monospace",
            size=18,
            #color="RebeccaPurple"
            )
        )
    fig['layout']['xaxis']['autorange'] = "reversed"
    plot(fig)
    return None


def spec(x, ppm, shift=None, ax=None, xlab='ppm', ylab='Int', title=None, **kwargs):
    """
    Plotting a single or multiple 1D NMR spectra

    Args:
        x: NMR data array (rank 1 or 2)
        ppm: Chemical shift array (rank 1)
        shift: Chemical shift interval (list)
        ax: Pyplot axis object for adding to exisiting plot
        xlab: X-axis label (str)
        ylab: Y-axis lable (str)
        title: Plot title
        **kwargs**: Additional arguments passed on to matplotlib.pyplot
    Returns:
        Calibrated NMR array
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = ax

    if shift is not None:
        idx=get_idx(ppm, shift)
        ppm=ppm[idx]
        if x.ndim > 1:
            x=x[:,idx]
        else:
            x=x[idx]


    if x.ndim > 1:
        ax.plot(ppm, x.T, **kwargs)

    else:
        ax.plot(ppm, x, **kwargs)

    if ax is None:
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    ax.set_xlim(np.nanmax(ppm), np.nanmin(ppm))

    if title is not None:
        ax.set_title(title)
    # ax.grid(True)
    #plt.show()
    return (fig, ax)