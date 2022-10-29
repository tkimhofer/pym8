<<<<<<< HEAD
# Hierarchical clustering based on a binary tree similarity
import matplotlib.pyplot as plt
cc = [0,1,2,3,5,6,7,8,9,10,11, 12,13, 14, 15]
Xn=Xn[cc]
f, x = plt.subplots(1, 1)
c=0
for i in range(Xn.shape[0]):
    plotting.spec((Xn[[i], :]/np.max(Xn[i]))+(c*0.3), ppe, c='black', ax=[f,x], shift=[0, 9])
    plt.annotate('s'+str(c), (9, (c*0.3+0.05)))
    c += 1
=======
# Implementation of Binary Search Tree in Python
# Define a Tree Structure:

# x=Xbl[4]
# ppm=pp
import matplotlib.pyplot as plt
import numpy as np
import pickle
>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060

tics, st  = pickle.load( open('/Volumes/Backup Plus/Cambridge_RP_POS/first20_tics_msmate.p', "rb" ) )
# binary tree distance for 1D NMR spectra / LC chromatograms

<<<<<<< HEAD
t1 = BstSearch(ppe, Xn[0]/np.max(Xn[0]))
ax = t1.vis_graph(xlab='ppm', ylab='Int', ax=None, col='cyan', invertx=True)
t2 = BstSearch(ppe, Xn[1]/np.max(Xn[1]))
t2.vis_graph(xlab='ppm', ylab='Int', ax=ax, col='magenta', invertx=False)

# aa = spec_distance(Xn, ppe, metric='euclidean', linkage='ward', minle=4, color_threshold=10, above_threshold_color='black')

cl1 = [13, 5, 11, 7, 1,2,10]
cl2 = [8, 14, 0, 6]
cl3 = [4, 12, 3, 9]

cl=cl2+cl3+cl1
f, x = plt.subplots(1, 1)
c=0
for i in cl:
    if i in cl1:
        col ='orange'
    elif i in cl2:
        col ='green'
    else:
        col='red'
    plotting.spec((Xn[[i], :]/np.max(Xn[i]))+(c*0.3), ppe, c=col, ax=[f,x], shift=[0, 9])
    plt.annotate('s'+str(i), (9, (c*0.3+0.05)))
    c += 1

a=plotting.spec(Xn[cl2], ppe)

fig, axs = plt.subplots(3,1)
plotting.spec(Xn[cl1], ppe, ax=[fig, axs[0]], c='orange')
plotting.spec(Xn[cl2], ppe, ax=[fig, axs[1]], c='lightgreen')
plotting.spec(Xn[cl3], ppe, ax=[fig, axs[2]], c='darkgreen')

Xs=Xn / np.max(Xn[:, np.where((ppe>3 ) &(ppe<3.1))[0]], 1)[..., np.newaxis]


axs[0] = a[1]
axs[1] = plotting.spec(Xn[cl2], ppe)[1]

axs[2] = plotting.spec(Xn[cl3], ppe)[1]



class Node:
    # defines cutpoints based on y-weighted mean of x values
    def wMean(self, x, y):
        import numpy as np
=======
x=tics[0]
y=st

out=bst_search(x, y)
out.
class Node:
    def wMean(self, x, y):
>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060
        wm = np.sum((y / np.sum(y)) * x)
        iwm = np.argmin(np.abs(wm - x))
        return (wm, iwm)

    def __init__(self, x, y, depth=4, minle=100):
<<<<<<< HEAD
        self.len = len(x)
        self.depth = depth

        # check stopping criteria
=======
        # stop criteria parameters
        self.len = len(x)
        self.depth = depth

        # stop criteria
>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060
        if (len(x) < minle) | (depth < 1):
            return

        self.y = y
        self.x = x
        self.wmean, self.icent = self.wMean(self.x, self.y)

        y_le = y[self.icent:]
        x_le = x[self.icent:]

        y_ri = y[:(self.icent-1)]
        x_ri = x[:(self.icent-1)]

        # recursion
        self.left = Node(x=x_le, y=y_le, depth=depth-1, minle=minle)
        self.right = Node(x=x_ri, y=y_ri, depth=depth-1, minle=minle)

class BstSearch():
    # calculates binary tree (btree) for a single spectrum
    def __init__(self, x, y, depth=4, minle=100):
        # depth: recursion depth to define x-cutpoints (recursion stopping criterion)
        # minle: minimum length of a segment defined by right and left cutpoint (recursion stopping criterion)
        import pandas as pd
        import numpy as np
        self.x = x
        self.y = y / np.max(y)
        self.depth = depth
        self.nData = []
        self.btree = Node(self.x, self.y, self.depth, minle)
        self.collect(self.btree)
        self.nodes = pd.DataFrame(self.nData)

<<<<<<< HEAD
    # class Node:
    #     def callNode(self, x, y, depth, minle):
    #         self.Node(x=x, y=y, depth=depth - 1, minle=minle)
    #
    #     def __init__(self, x, y, depth=4, minle=1000):
    #         import numpy as np
    #
    #         def wMean(x, y):
    #             wm = np.sum((y / np.sum(y)) * x)
    #             iwm = np.argmin(np.abs(wm - x))
    #             return (wm, iwm)
    #
    #         # stop criteria parameters
    #         self.len = len(x)
    #         self.depth = depth
    #
    #         self.y = y
    #         self.x = x
    #         self.wmean, self.icent = wMean(self.x, self.y)
    #
    #         y_le = y[self.icent:]
    #         x_le = x[self.icent:]
    #
    #         y_ri = y[:(self.icent - 1)]
    #         x_ri = x[:(self.icent - 1)]
    #
    #         # recursion
    #         if depth > 1:
    #             if len(y_le) > minle:
    #                 self.left = self.callNode(x=x_le, y=y_le, depth=depth - 1, minle=minle)
    #             if len(y_ri) > minle:
    #                 self.right = self.callNode(x=x_ri, y=y_ri, depth=depth - 1, minle=minle)
    def collect(self, btree, dir_par='root', c='0'):
        # traverse btree to collect weighted means
        # collect depth value, right or left and wmean (cutpoints)
        nData = []

        if dir_par == 'root':
            nid = '0'
        elif dir_par =='l':
            nid = str(c) + str(0)
        elif dir_par == 'r':
            nid = str(c) + str(1)

        if hasattr(btree, 'wmean'):
            data = {'id': nid, 'wean': btree.wmean}
            self.nData.append(data)
            nData.append(data)

=======
test=bst_search(st, tics[0])
ax0 = test.vis_graph(col='cyan')

test1=bst_search(st, tics[3])
ax1 = test1.vis_graph(ax=ax0, col='blue')

test2=bst_search(st, tics[14])
ax2 = test2.vis_graph(ax=ax1, col='blue')

test3=bst_search(st, tics[17])
ax3 = test3.vis_graph(ax=ax2, col='blue')


test3=bst_search(st, tics[11])
ax3 = test3.vis_graph(ax=ax3, col='blue')

test4=bst_search(st, tics[2])
test4.vis_graph(ax=ax3, col='cyan')

out=test.collect1(test)
test.nData

np.array(test.nData.items())

class spec_distance():
    def __init__(self, specs, x, depth=4, minle=100, linkage='ward', **kwargs):
        import numpy as np
        self.linkage = linkage
        self.rr = []
        self.n = specs.shape[0]
        self.labels = np.array(['s'+str(i) for i in range(self.n)])
        for i in range(self.n):
            df = bst_search(x, specs[i], depth, minle).nodes
            wmean = df.wean
            wmean.index=df.id
            self.rr.append(wmean)
        self.wm = pd.concat(self.rr, axis=1)

        # calc distances

        D = np.zeros((self.n, self.n))
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                D[i,j] = np.linalg.norm(self.wm.iloc[:,i].values-self.wm.iloc[:,j].values)

        self.D=np.maximum(D, d.transpose())

        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        Z = hierarchy.linkage(pdist(self.D), self.linkage)

        plt.figure()
        dn = hierarchy.dendrogram(Z, labels=self.labels, **kwargs)
        plt.show()


tt =spec_distance(specs=tics, x=st)
self=tt
pd.concat(tt.rr, axis=1)

class bst_search():
    # binary seach tree
    def __init__(self, x, y, depth=4, minle=100):
        import pandas as pd
        import numpy as np
        self.x = x
        self.y = y / np.max(y)
        self.depth = depth
        self.nData = []
        self.btree = self.Node(self.x, self.y, self.depth, minle)
        self.collect(self.btree)
        self.nodes = pd.DataFrame(self.nData)

    class Node:
        def __init__(self, x, y, depth=4, minle=1000):
            def wMean(x, y):
                wm = np.sum((y / np.sum(y)) * x)
                iwm = np.argmin(np.abs(wm - x))
                return (wm, iwm)

            # stop criteria parameters
            self.len = len(x)
            self.depth = depth

            # # stop criteria
            # if (len(x) < minle) | (depth < 1):
            #     return

            self.y = y
            self.x = x
            self.wmean, self.icent = wMean(self.x, self.y)

            y_le = y[self.icent:]
            x_le = x[self.icent:]

            y_ri = y[:(self.icent - 1)]
            x_ri = x[:(self.icent - 1)]

            # recursion
            if depth > 1:
                if len(y_le) > minle:
                    self.left = Node(x=x_le, y=y_le, depth=depth - 1, minle=minle)
                if len(y_ri) > minle:
                    self.right = Node(x=x_ri, y=y_ri, depth=depth - 1, minle=minle)

    def collect(self, btree, dir_par='root', c='0'):

        # traverse btree to collect weighted means
        # collect depth id, right or left and wmean
        # node ids coded as deptth + 0/1 for left/right, resp., so 0 (root), 10 (left child root), 11 (right child root), 100 (left child of left child root), 101 (right child of left child root, ...)
        nData = []

        if dir_par == 'root':
            nid = '0'
        elif dir_par =='l':
            nid = str(c) + str(0)
        elif dir_par == 'r':
            nid = str(c) + str(1)

        print(c)
        if hasattr(btree, 'wmean'):
            data = {'id': nid, 'wean': btree.wmean}
            self.nData.append(data)
            nData.append(data)

>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060
        if hasattr(btree, 'left'):
            nData = nData + [self.collect(btree.left, dir_par='l', c=nid)]
        else:
            return

        if hasattr(btree, 'right'):
            nData = nData + [self.collect(btree.right, dir_par='r', c=nid)]
        else:
            return
<<<<<<< HEAD
=======

        return nData

    def vis_graph(self, xlab='Scantime (s)', ylab='Total Count', ax=None, col='green'):
        import networkx as nx

        def srange(t, tdiff, tmin, x_diff, x_min):
            return ((x_diff * (t - tmin)) / tdiff) + x_min
        y_max = np.max(self.y) * 2
        y_min = np.max(self.y) * 0.8
        y_diff = y_max - y_min

        tmax=np.log(self.depth+1)

        A = np.zeros((len(self.nodes), len(self.nodes)))
        N = self.nodes.id.values
        U = self.nodes[1:].id.values
        xy = {'0': np.array([self.nodes.wean[0], srange(t=np.log(self.depth + 1), tdiff=tmax, tmin=0, x_diff=y_diff, x_min=y_min)])}
        adj_list = []
        c = 1
        while len(U) > 0:
            ix = np.where(N == U[0][0:-1])[0][0]
            adj_list.append((N[ix], U[0]))
            A[ix, c] = 1
            xy.update({U[0]: np.array([self.nodes.wean.iloc[c], srange(t=np.log((self.depth+1)-len(U[0])), tdiff=tmax, tmin=0, x_diff=y_diff, x_min=y_min) ])})
            c=c+1
            U = U[1:]

        # adjacency order is test.nodes.id
        self.tips = (np.sum(np.maximum(A, A.transpose()), 0) == 1.)

        #vlines=np.array([(xy[x][0], 0, xy[x][1]) for x in self.nodes.id.iloc[np.array(self.tips)]])
        vlines = np.array([xy[x][0] for x in self.nodes.id.iloc[np.array(self.tips)]])

        G = nx.from_numpy_matrix(A)
        nmap = dict(zip(sorted(G), self.nodes.id.values))
        G = nx.relabel_nodes(G, nmap)

        if isinstance(ax, type(None)):
            fig, ax = plt.subplots()
        ax.plot(self.x, self.y, color=col, linewidth=0.7)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        plt.vlines(vlines, ymin=0, ymax=y_min, color=col, linestyles='dotted')
        nx.draw(G, pos=xy, with_labels=False,
                node_size=3, node_color=col, font_size=6, \
                linewidths=3, node_shape='o', ax=ax, edge_color=col)
        limits = plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        return ax
>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060

        return nData

<<<<<<< HEAD
    def vis_graph(self, xlab='Scantime (s)', ylab='Total Count', ax=None, col='green', invertx=False):
        #  binary tree visualised over input data
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt

        def srange(t, tdiff, tmin, x_diff, x_min):
            return ((x_diff * (t - tmin)) / tdiff) + x_min
        y_max = np.max(self.y) * 2
        y_min = np.max(self.y) * 0.8
        y_diff = y_max - y_min

        tmax=np.log(self.depth+1)

        A = np.zeros((len(self.nodes), len(self.nodes)))
        N = self.nodes.id.values
        U = self.nodes[1:].id.values
        xy = {'0': np.array([self.nodes.wean[0], srange(t=np.log(self.depth + 1), tdiff=tmax, tmin=0, x_diff=y_diff, x_min=y_min)])}
        adj_list = []
        c = 1
        while len(U) > 0:
            ix = np.where(N == U[0][0:-1])[0][0]
            adj_list.append((N[ix], U[0]))
            A[ix, c] = 1
            xy.update({U[0]: np.array([self.nodes.wean.iloc[c], srange(t=np.log((self.depth+1)-len(U[0])), tdiff=tmax, tmin=0, x_diff=y_diff, x_min=y_min) ])})
            c=c+1
            U = U[1:]

        # adjacency order is test.nodes.id
        self.tips = (np.sum(np.maximum(A, A.transpose()), 0) == 1.)

        #vlines=np.array([(xy[x][0], 0, xy[x][1]) for x in self.nodes.id.iloc[np.array(self.tips)]])
        vlines = np.array([xy[x][0] for x in self.nodes.id.iloc[np.array(self.tips)]])

        G = nx.from_numpy_matrix(A)
        nmap = dict(zip(sorted(G), self.nodes.id.values))
        G = nx.relabel_nodes(G, nmap)

        if isinstance(ax, type(None)):
            fig, ax = plt.subplots()
        ax.plot(self.x, self.y, color=col, linewidth=0.7)
        if invertx:
            ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        plt.vlines(vlines, ymin=0, ymax=y_min, color=col, linestyles='dotted')
        nx.draw(G, pos=xy, with_labels=False,
                node_size=3, node_color=col, font_size=6, \
                linewidths=3, node_shape='o', ax=ax, edge_color=col)
        limits = plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        return ax

class SpecDistance():
    # hierarchical clustering based on binary tree similarity
    def __init__(self, specs, x, depth=4, minle=4, metric='euclidean', linkage='ward', **kwargs):
        # specs: input data with rows representing individual spectra
        # x: x variables, len(x) == specs.shape[1]
        # depth: recursion depth to define x-axis cutpoints (recursion stopping criterion)
        # minle: minimum length of a segment defined by right and left cutpoint (recursion stopping criterion)
        # metric: distance metric (see scipy.spatial.distance, most useful ones are ‘cosine’ and ‘euclidean’)
        # linkage: linkage function for hierarchical agglomerative clustering
        # **kwargs: additional parameters for scipy function dendrogram (hierarchy module)

        # returns cluster dendrogram
        import numpy as np
        import pandas as pd
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        import matplotlib.pyplot as plt

        self.linkage = linkage
        self.rr = []
        self.n = specs.shape[0]
        self.labels = np.array(['s'+str(i) for i in range(self.n)])
        for i in range(self.n):
            df = BstSearch(x, specs[i], depth, minle).nodes
            wmean = df.wean
            wmean.index=df.id
            self.rr.append(wmean)
        self.wm = pd.concat(self.rr, axis=1)

        D = np.zeros((self.n, self.n))
        for i in range(self.n-1):
            for j in range(i+1, self.n):
                D[i,j] = np.linalg.norm(self.wm.iloc[:,i].values-self.wm.iloc[:,j].values)

        self.D=np.maximum(D, D.transpose())

        Z = hierarchy.linkage(pdist(self.D, metric=metric), self.linkage)

        plt.figure()
        hierarchy.dendrogram(Z, labels=self.labels, **kwargs)
        plt.show()
=======

#
#
# def adma(self):
#     import numpy as np
#     if len(self.out)==0: self.collect(self.bt)
#     self.N = [x[0] for x in self.out]
#     # print(self.N)
#     self.P = [x[1] for x in self.out]
#     D = -np.array([x[2] for x in self.out])
#
#
#     def minmax(y, start, end):
#        return start+ (((y - min(y)) / (max(y) - min(y)))/ (end-start))
#
#     self.D = minmax(D, 1, 2)
#
#     N_le = len(self.N)
#     self.coord = {np.round(self.P[0], 2): np.array([self.P[0], self.D[0]])}
#     A = np.zeros((N_le, N_le))
#     for i in range(1, N_le):
#         print(i)
#         n = self.N[i]
#         n_par = n[:-2]
#         A[self.N.index(n_par), i] = 1
#         self.coord[np.round(self.P[i], 2)] = np.array([self.P[i], self.D[i]])
#     self.AS = np.maximum(A, A.transpose())
#     self.tips=(np.sum(ll.AS,0)==1.)
#
# def plot_graph(self, **kwargs):
#
#     if not hasattr(self, 'AS'):
#         self.adma()
#     import networkx as nx
#     #print(self.P)
#     G = nx.from_numpy_matrix(self.AS)
#     nmap = dict(zip(sorted(G), np.round(self.P, 2)))
#     self.G = nx.relabel_nodes(G, nmap)
#
#     vlines = np.insert(np.sort(np.array(self.P)[np.array(self.tips)]), 0, np.min(self.bt.ppm))
#     vlines = np.insert(vlines, len(vlines), np.max(self.bt.ppm))
#
#     if 'edge_color' in kwargs:
#         ecol=kwargs['edge_color']
#         print('found col')
#     else:
#         ecol='blue'
#
#     nx.draw(self.G, pos=self.coord, with_labels=True,\
#             node_size=110, node_color='white', font_size=6, \
#             linewidths=1, node_shape='o', **kwargs)
#     #ll.plot_graph(node_size=110, node_color='white', font_size=6, edgecolors="white", linewidths=1, node_shape='o'
#     plt.vlines(vlines, ymin=0, ymax=0.99, alpha=0.4, colors=ecol)
#     #y=np.ones(len(vlines))*0.8
#     #plt.plot(x=vlines, y=y, marker='o')
#     #self.bt.ppm = ppm
#     xs=self.bt.x
#     plt.plot(self.bt.ppm, ((xs) / np.max(xs)), color=ecol)
#
#
# def collect(self, bt='', dir='r'):
# # traverse btree to collect weighted means
#     if not isinstance(bt, Node):
#         bt=self.bt
#     #self.count=self.count+1
#     if dir=='r': dir=str(0);  self.count=bt.depth+1
#
#     if hasattr(bt, 'wmean'):
#         self.out.append([dir, bt.wmean, self.count-bt.depth])
#         if hasattr(bt, 'left'):
#             if isinstance(bt.left, BinarySearchTreeNode):
#                 self.collect(bt.left, dir=dir+str(self.count-bt.depth)+'1')
#         else:
#             return
#
#         if hasattr(bt, 'right'):
#             if isinstance(bt.right, BinarySearchTreeNode):
#                 #self.out.append([bt.depth, 'right', bt.wmean])
#                 self.collect(bt.right, dir=dir+str(self.count-bt.depth)+'2')
#         else:
#             return
#     else: return
#
# #
# #
# #
# # sp=[]
# # le=[]
# # for i in range(0, Xn.shape[0]):
# #         xs = Xn[i]
# #         ss = BinarySearchTreeNode(xs, pp, depth=4, minle=1)
# #         ll = bst_search(ss)
# #         ll.adma()
# #         le.append(len(ll.P))
# #         sp.append(ll.P)
# #
# # np.where(np.array(le) != 15)[0]
# #
# #
# # iid=np.where(np.array(le) == 15)[0]
# # iid=iid.astype(np.int64)
# # out=sp[iid]
# #
# # spf=[sp[i] for i in iid]
# # out=np.array(spf)
# #
# # # perform clustering analysis
# # from sklearn.cluster import AgglomerativeClustering
# # from sklearn.metrics import pairwise_distances
# # from scipy.cluster.hierarchy import dendrogram
# #
# # td=pairwise_distances(out)
# # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage="ward")
# # model.fit(td)
# #
# # plot_dendrogram(model, truncate_mode='level', p=0)
# #
# # model1 = AgglomerativeClustering(distance_threshold=None, n_clusters=7,linkage="ward")
# # model1.fit(td)
# # #model.fit(X)
# # oo=pd.value_counts(model1.labels_)
# # idx=np.where(model1.labels_==2)[0][0:20]
# #
# # import matplotlib._color_data as mcd
# # fig, axs = plt.subplots(oo.shape[0], sharex=True, sharey=True)
# #
# # sn=[]
# # for i in range(oo.shape[0]):
# #     idx = np.where(model1.labels_ == oo.index[i])[0]
# #     xq1=np.quantile(Xn[idx]/np.max(Xn[idx], 1)[..., np.newaxis], 0.25, axis=0)
# #     xq2 = np.quantile(Xn[idx] / np.max(Xn[idx], 1)[..., np.newaxis], 0.5, axis=0)
# #     xq3 = np.quantile(Xn[idx] / np.max(Xn[idx], 1)[..., np.newaxis], 0.75, axis=0)
# #     axs[i].plot(pp, xq2, c='black')
# #     axs[i].fill_between(pp, xq1, xq3, color=mcd.BASE_COLORS[list(mcd.BASE_COLORS.keys())[i]], alpha=0.2)
# #     sn.append(xm) #/np.max(Xbl[idx[i]])
# #
# # plotting.ispec(np.array(sn), pp , theme='plotly_dark')
# #
# # #
# #
#
#
#
#
# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram
#
#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count
# #
# # linkage_matrix = np.column_stack([model.children_, model.distances_,
# #                                       counts]).astype(float)
# #
# # # Plot the corresponding dendrogram
# # dendrogram(linkage_matrix, **kwargs)
# #
# #
# #
# #
# #     xs=Xbl[7]
# #     #xs=xs/np.max(xs)
# #     ss = BinarySearchTreeNode(xs, pp, depth=5)
# #     ll = bst_search(ss)
# #     ll.collect()
# #     ll.plot_graph(edge_color='blue')
# #
# #     xs = Xbl[8]
# #     # xs=xs/np.max(xs)
# #     ss = BinarySearchTreeNode(xs, pp, depth=5)
# #     ll = bst_search(ss)
# #     ll.collect()
# #     ll.plot_graph(edge_color='green')
# #
# #     xs = Xbl[99]
# #     # xs=xs/np.max(xs)
# #     ss = BinarySearchTreeNode(xs, pp, depth=5)
# #     ll = bst_search(ss)
# #     ll.collect()
# #     ll.plot_graph(edge_color='red')
# #
# #
# #     # ll.out
# #     # ll.tips
# #
# #
# #     vlines= np.insert(np.sort(np.array(ll.P)[np.array(ll.tips)]),0,0)
# #     vlines=np.insert(vlines, len(vlines), np.max(pp))
# #
# #     ll.plot_graph(node_size=110, node_color='white', font_size=6, edgecolors="white", linewidths=1, node_shape='o'
# #     plt.vlines(vlines,ymin=-0.05, ymax=0.98, color='red',alpha=0.4)
# #     plt.plot(pp, ((xs) / np.max(xs)), c='black')
# #     for i in range(1, len(vlines)):
# #         plt.axvspan(xmin=vlines[i-1], xmax=vlines[i], ymin=-0.05, ymax=0.48, color='red', alpha=0.1, fill='white')
# #     plt.plot(pp, ((xs) / np.max(xs)), c='black')
# #     text=nx.draw_networkx_labels(ll.G, ll.coord)
# #     for _, t in text.items():
# #         t.set_rotation('vertical')
#     #plt.axvspan()
# #     plt.plot(ll.G)
# # #
# # ss=BinarySearchTreeNode(Xbl[4], pp)
# # ll=bst_search(ss)
# # # ll.collect()
# # # ll.adma()
# # ll.plot_graph()
# # ll.out
# #
# #
# # N=np.array([x[0] for x in ll.out])
# # N=[x[0] for x in ll.out]
# # P=[x[1] for x in ll.out]
# # D=[x[2] for x in ll.out]
# # D=1/np.array(D)
# #
# # def minmax(y):
# #     return (y-min(y)) / (max(y)-min(y))
# # D=minmax(D)
# #
# #
# # N_le=len(N)
# # coord={np.round(P[0],2): np.array([P[0], float(D[0])])}
# # A=np.zeros((N_le, N_le))
# # for i in range(1, N_le):
# #     n=N[i]
# #     n_par=n[:-2]
# #     A[N.index(n_par), i]=1
# #     coord[np.round(P[i],2)]=np.array([P[i], float(D[i])])
# # As=np.maximum(A, A.transpose())
# #
# #
# # import networkx as nx
# # G = nx.from_numpy_matrix(As)
# # #nx.draw(G, with_labels=True)
# #
# # nmap=dict(zip(sorted(G), np.round(P, 2)))
# # H = nx.relabel_nodes(G, nmap)
# # nx.draw(H, pos=coord, with_labels=True)
# #
# # sorted(H)
# # sorted(G)
# #
# # pos = coord
# # x_pos=P
# # y_pos=D
# #
# #
# # P
# # (1-D)/np.max(1-D)
# # nx.draw(G, pos=coord, with_labels=True)
# # plt.plot(ppm, x)
# #
# #
# #
>>>>>>> 5661dcc71cac544d9fe55654144610a90af74060
