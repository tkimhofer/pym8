# Implementation of Binary Search Tree in Python
# Define a Tree Structure:

# x=Xbl[4]
# ppm=pp
import numpy as np


class BinarySearchTreeNode:

    def __init__(self, x, ppm, depth=4, minle=1000):
        # stop criteria parameters
        self.len = len(ppm)
        self.minle=minle
        self.depth = depth

        # stop criteria
        if (len(ppm) < minle):
            # self.right = self.wmean
            # self.left = self.wmean
            #print('Minlen at depth'+str(depth))
            return
        # stop criteria
        if (depth < 1):
            #print('Depth reached' + str(depth))
            # self.right = self.wmean
            # self.left = self.wmean
            return

        def weighted_mean(x, pp):
            wm = np.sum((x / np.sum(x)) * pp)
            #wm = np.sum(x * pp)
            iwm = np.argmin(np.abs(wm - pp))
            return [wm, iwm]

        wml = weighted_mean(x, ppm)
        self.ppm=ppm
        self.x=x
        self.wmean = wml[0]
        self.icent = wml[1]

        spec_le = x[self.icent:]
        ppm_le = ppm[self.icent:]

        spec_ri = x[:(self.icent-1)]
        ppm_ri = ppm[:(self.icent-1)]


        # recursion
        self.left = BinarySearchTreeNode(x=spec_le, ppm=ppm_le, depth=depth-1)
        self.right = BinarySearchTreeNode(x=spec_ri, ppm=ppm_ri, depth=depth-1)


class bst_search:
    def __init__(self, ss):
        self.bt=ss
        self.out=[]
        self.wmean=[]

    def collect(self, bt='', dir='r'):

        if not isinstance(bt, BinarySearchTreeNode):
            bt=self.bt
        #self.count=self.count+1
        if dir=='r': dir=str(0);  self.count=bt.depth+1

        if hasattr(bt, 'wmean'):
            self.out.append([dir, bt.wmean, self.count-bt.depth])
            if hasattr(bt, 'left'):
                if isinstance(bt.left, BinarySearchTreeNode):
                    self.collect(bt.left, dir=dir+str(self.count-bt.depth)+'1')
            else:
                return

            if hasattr(bt, 'right'):
                if isinstance(bt.right, BinarySearchTreeNode):
                    #self.out.append([bt.depth, 'right', bt.wmean])
                    self.collect(bt.right, dir=dir+str(self.count-bt.depth)+'2')
            else:
                return
        else: return

    def adma(self):
        if len(self.out)==0: self.collect(self.bt)
        self.N = [x[0] for x in self.out]
        # print(self.N)
        self.P = [x[1] for x in self.out]
        D = -np.array([x[2] for x in self.out])
        #D = np.max(D)- np.array(D)

        def minmax(y, start, end):
           return start+ (((y - min(y)) / (max(y) - min(y)))/ (end-start))

        self.D = minmax(D, 1, 2)

        N_le = len(self.N)
        self.coord = {np.round(self.P[0], 2): np.array([self.P[0], self.D[0]])}
        A = np.zeros((N_le, N_le))
        for i in range(1, N_le):
            n = self.N[i]
            n_par = n[:-2]
            A[self.N.index(n_par), i] = 1
            self.coord[np.round(self.P[i], 2)] = np.array([self.P[i], self.D[i]])
        self.AS = np.maximum(A, A.transpose())
        self.tips=(np.sum(ll.AS,0)==1.)

    def plot_graph(self, **kwargs):

        if not hasattr(self, 'AS'):
            self.adma()
        import networkx as nx
        #print(self.P)
        G = nx.from_numpy_matrix(self.AS)
        nmap = dict(zip(sorted(G), np.round(self.P, 2)))
        self.G = nx.relabel_nodes(G, nmap)

        vlines = np.insert(np.sort(np.array(self.P)[np.array(self.tips)]), 0, np.min(self.bt.ppm))
        vlines = np.insert(vlines, len(vlines), np.max(self.bt.ppm))

        if 'edge_color' in kwargs:
            ecol=kwargs['edge_color']
            print('found col')
        else:
            ecol='blue'

        nx.draw(self.G, pos=self.coord, with_labels=True,\
                node_size=110, node_color='white', font_size=6, \
                linewidths=1, node_shape='o', **kwargs)
        #ll.plot_graph(node_size=110, node_color='white', font_size=6, edgecolors="white", linewidths=1, node_shape='o'
        plt.vlines(vlines, ymin=0, ymax=0.99, alpha=0.4, colors=ecol)
        #y=np.ones(len(vlines))*0.8
        #plt.plot(x=vlines, y=y, marker='o')
        #self.bt.ppm = ppm
        xs=self.bt.x
        plt.plot(self.bt.ppm, ((xs) / np.max(xs)), color=ecol)
#


sp=[]
le=[]
for i in range(0, Xn.shape[0]):
        xs = Xn[i]
        ss = BinarySearchTreeNode(xs, pp, depth=4, minle=1)
        ll = bst_search(ss)
        ll.adma()
        le.append(len(ll.P))
        sp.append(ll.P)

np.where(np.array(le) != 15)[0]


iid=np.where(np.array(le) == 15)[0]
iid=iid.astype(np.int64)
out=sp[iid]

spf=[sp[i] for i in iid]
out=np.array(spf)

# perform clustering analysis
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram

td=pairwise_distances(out)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage="ward")
model.fit(td)

plot_dendrogram(model, truncate_mode='level', p=0)

model1 = AgglomerativeClustering(distance_threshold=None, n_clusters=7,linkage="ward")
model1.fit(td)
#model.fit(X)
oo=pd.value_counts(model1.labels_)
idx=np.where(model1.labels_==2)[0][0:20]

import matplotlib._color_data as mcd
fig, axs = plt.subplots(oo.shape[0], sharex=True, sharey=True)

sn=[]
for i in range(oo.shape[0]):
    idx = np.where(model1.labels_ == oo.index[i])[0]
    xq1=np.quantile(Xn[idx]/np.max(Xn[idx], 1)[..., np.newaxis], 0.25, axis=0)
    xq2 = np.quantile(Xn[idx] / np.max(Xn[idx], 1)[..., np.newaxis], 0.5, axis=0)
    xq3 = np.quantile(Xn[idx] / np.max(Xn[idx], 1)[..., np.newaxis], 0.75, axis=0)
    axs[i].plot(pp, xq2, c='black')
    axs[i].fill_between(pp, xq1, xq3, color=mcd.BASE_COLORS[list(mcd.BASE_COLORS.keys())[i]], alpha=0.2)
    sn.append(xm) #/np.max(Xbl[idx[i]])

plotting.ispec(np.array(sn), pp , theme='plotly_dark')

#





def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)




    xs=Xbl[7]
    #xs=xs/np.max(xs)
    ss = BinarySearchTreeNode(xs, pp, depth=5)
    ll = bst_search(ss)
    ll.collect()
    ll.plot_graph(edge_color='blue')

    xs = Xbl[8]
    # xs=xs/np.max(xs)
    ss = BinarySearchTreeNode(xs, pp, depth=5)
    ll = bst_search(ss)
    ll.collect()
    ll.plot_graph(edge_color='green')

    xs = Xbl[99]
    # xs=xs/np.max(xs)
    ss = BinarySearchTreeNode(xs, pp, depth=5)
    ll = bst_search(ss)
    ll.collect()
    ll.plot_graph(edge_color='red')


    # ll.out
    # ll.tips


    vlines= np.insert(np.sort(np.array(ll.P)[np.array(ll.tips)]),0,0)
    vlines=np.insert(vlines, len(vlines), np.max(pp))

    ll.plot_graph(node_size=110, node_color='white', font_size=6, edgecolors="white", linewidths=1, node_shape='o'
    plt.vlines(vlines,ymin=-0.05, ymax=0.98, color='red',alpha=0.4)
    plt.plot(pp, ((xs) / np.max(xs)), c='black')
    for i in range(1, len(vlines)):
        plt.axvspan(xmin=vlines[i-1], xmax=vlines[i], ymin=-0.05, ymax=0.48, color='red', alpha=0.1, fill='white')
    plt.plot(pp, ((xs) / np.max(xs)), c='black')
    text=nx.draw_networkx_labels(ll.G, ll.coord)
    for _, t in text.items():
        t.set_rotation('vertical')
    #plt.axvspan()
#     plt.plot(ll.G)
# #
# ss=BinarySearchTreeNode(Xbl[4], pp)
# ll=bst_search(ss)
# # ll.collect()
# # ll.adma()
# ll.plot_graph()
# ll.out
#
#
# N=np.array([x[0] for x in ll.out])
# N=[x[0] for x in ll.out]
# P=[x[1] for x in ll.out]
# D=[x[2] for x in ll.out]
# D=1/np.array(D)
#
# def minmax(y):
#     return (y-min(y)) / (max(y)-min(y))
# D=minmax(D)
#
#
# N_le=len(N)
# coord={np.round(P[0],2): np.array([P[0], float(D[0])])}
# A=np.zeros((N_le, N_le))
# for i in range(1, N_le):
#     n=N[i]
#     n_par=n[:-2]
#     A[N.index(n_par), i]=1
#     coord[np.round(P[i],2)]=np.array([P[i], float(D[i])])
# As=np.maximum(A, A.transpose())
#
#
# import networkx as nx
# G = nx.from_numpy_matrix(As)
# #nx.draw(G, with_labels=True)
#
# nmap=dict(zip(sorted(G), np.round(P, 2)))
# H = nx.relabel_nodes(G, nmap)
# nx.draw(H, pos=coord, with_labels=True)
#
# sorted(H)
# sorted(G)
#
# pos = coord
# x_pos=P
# y_pos=D
#
#
# P
# (1-D)/np.max(1-D)
# nx.draw(G, pos=coord, with_labels=True)
# plt.plot(ppm, x)
#
#
#
