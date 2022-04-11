# cv - opls da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cmd = 'find ' + path + ' -iname "*quant_report*.xml"  -print0 | xargs -0 grep "QUANTIFICATION version="'
sp = subprocess.getoutput(cmd)
out = sp.split('\n')


find /Volumes/Torben_2/Sam_urine_021019 -name "*quant_report.xml"

find /Volumes/Torben_2/Sam_urine_021019 -iname "*.xml"  -print0 | xargs -0 grep "QUANTIFICATION version="

from importlib import reload
reload(utility)

def _scale(X):
    xm = np.mean(X, 0)
    xsd = np.std(X, 0)
    return (X - xm / xsd)


# case single col Y
y = np.max(X, 1)[..., np.newaxis]

Xs = _scale(X)
ys = _scale(y)


from sklearn.datasets import load_iris
iris = load_iris()

X=iris['data']
Y=iris['target']
idx=np.where((Y==1) | (Y==2))[0]

X=X[idx]
y=Y[idx]
Xs=X
Ys=Y

mod=opls(Xs, Ys,  cv={'type': 'mc', 'f': 2 / 3, 'k': 20, 'repl': False})

mod._nipals_comp(orth=True)
mod._nipals_comp(orth=False)

len(mod.cvv_t_mean)

plt.scatter(mod.cvv_t_mean[0], mod.cvv_t_sd[0])

cols=['red', 'blue']
oo=[cols[Ys[i]==1] for i in range(len(Ys))]
plt.errorbar(x=mod.cvv_t_mean[1][:,0], y=mod.cvv_t_orth_mean[0][:,0], xerr=mod.cvv_t_sd[1][:,0], yerr=mod.cvv_t_orth_sd[0][:,0], fmt='o', ecolor=oo, ms=3.5, mfc='black', mec='white')

mod.cvt_p_mean[0]
mod.cvt_p_sd[0][0]
# x=np.arange(len(mod.cvt_p_sd[0][0]))
ix=np.argsort(-mod.cvt_p_mean[1][0])
x=np.array(iris['feature_names'])[ix.tolist()]
plt.errorbar(x=x, y=mod.cvt_p_mean[1][0][ix], yerr=mod.cvt_p_sd[1][0][ix], fmt='o',ms=3.5, mfc='black', mec='white')

self=out
out._nipals_comp()

mod.cvt_t_mean
mod.cvt_t_mean[0][:,0]
mod.cvt_t_orth_mean[0][:,0]

def _scale(X):
    xm = np.mean(X, 0)
    xsd = np.std(X, 0)
    return (X - xm / xsd)


dX, dy = datasets.load_diabetes(return_X_y=True)
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

X=dX
Y=dy



from sklearn.datasets import load_iris
iris = load_iris()

X=iris['data']
Y=iris['target']
idx=np.where((Y==1) | (Y==2))[0]

X=X[idx]
Y=Y[idx]
Xs=X
Ys=np.int64(Y)
Y=Ys
_check_Y(Ys)



auc=1; c=1
while auc > 0.7:
    print(c)
    mod._nipals_comp(orth=True,  verbose=1)
    mod._nipals_comp(orth=False, verbose=1)
    auc=mod.auroc()
    print('done')
    c+=1


len(mod.cvt_t_orth_mean)
len(mod.cvt_p_orth_mean)
mod.cvt_p_orth_mean[0].shape

len(mod.cvv_t_mean)
mod.cvv_t_mean[0].shape
plt.plot(mod.cvv_t_mean[0])

plt.scatter(mod.cvv_t_mean[0], mod.cvv_t_orth_mean[0], c=Y)


mod._nipals_comp(orth=True)
len(mod.cvt_t_sd)
len(mod.cvt_t_orth_sd)

mf=300
iid=np.where(np.squeeze(np.array(mod.cvt_t_sd[0])) > np.squeeze(np.array(mod.cvt_t_orth_sd[0])))[0].tolist()
iis=np.where(~(np.squeeze(np.array(mod.cvt_t_sd[0])) > np.squeeze(np.array(mod.cvt_t_orth_sd[0]))))[0].tolist()
plt.scatter(mod.cvv_t_mean[0][iid], mod.cvt_t_orth_mean[0][iid], s=np.array(mod.cvt_t_sd[0][iid])*mf, c='red', label='sd pred')
plt.scatter(mod.cvv_t_mean[0], mod.cvt_t_orth_mean[0], s=np.array(mod.cvt_t_orth_sd[0])*mf, c='green', label='sd orth')
plt.scatter(mod.cvv_t_mean[0][iis], mod.cvt_t_orth_mean[0][iis], s=np.array(mod.cvt_t_sd[0][iis])*mf, c='red')
plt.legend()

plt.scatter(mod.cvv_t_mean, mod.cvt_t_sd)
plt.scatter(mod.cvv_t_mean, mod.cvt_t_sd)

mod._nipals_comp(orth=False)

mod.cvt_xres

len(mod.cvt_t_sd)
mf=300
x=np.squeeze(mod.cvv_t_mean[:,0])
y=np.squeeze(mod.cvt_t_orth_mean[0][:,0])
iid=np.where(np.squeeze(np.array(mod.cvt_t_sd[0])) > np.squeeze(np.array(mod.cvt_t_orth_sd[0])))[0].tolist()
iis=np.where(~(np.squeeze(np.array(mod.cvt_t_sd[0])) > np.squeeze(np.array(mod.cvt_t_orth_sd[0]))))[0].tolist()
plt.scatter(x[iid], y[iid], s=np.array(mod.cvt_t_sd[0][iid])*mf, c='red', label='sd pred')
plt.scatter(x, y, s=np.array(mod.cvt_t_orth_sd[0])*mf, c='green', label='sd orth')
plt.scatter(x[iis], y[iis], s=np.array(mod.cvt_t_sd[0][iis])*mf, c='red')
plt.legend()

mod._nipals_comp(orth=True)
mod._nipals_comp(orth=False)
mod.auroc()

mod._nipals_comp(orth=True)
mod._nipals_comp(orth=False)
mod.auroc()

len(mod.cvt_t_orth_mean)
len(mod.cvt_t_mean)

len(mod.cvt_t_sd)
mf=300
i_orth=0
x=np.squeeze(mod.cvv_t_mean[:,0])
y=np.squeeze(mod.cvv_t_orth_mean[i_orth][:,0])
iid=np.where(np.squeeze(np.array(mod.cvt_t_sd[:,0])) > np.squeeze(np.array(mod.cvt_t_orth_sd[i_orth][:,0])))[0].tolist()
iis=np.where(~(np.squeeze(np.array(mod.cvt_t_sd[:,0])) > np.squeeze(np.array(mod.cvt_t_orth_sd[i_orth][:,0]))))[0].tolist()
plt.scatter(x[iid], y[iid], s=np.array(mod.cvt_t_sd[iid])*mf, edgecolors='red', label='sd pred', c='none')
plt.scatter(x, y, s=np.array(mod.cvt_t_orth_sd[i_orth])*mf, edgecolors='green', label='sd orth', c='none')
plt.scatter(x[iis], y[iis], s=np.array(mod.cvt_t_sd[iis])*mf, edgecolors='red',  c='none')
plt.scatter(x, y,  c=Y)

plt.legend()


self=mod





i_orth=1
x=np.squeeze(mod.cvv_t_mean[:,0])
y=np.squeeze(mod.cvt_t_orth_mean[i_orth][:,0])
plt.scatter(x, y, c=Ys)

ymap={1:'high', 0:'low'}
YY=[ymap[x] for x in Y]

cv={'type': 'mc', 'f': 2 / 3, 'k': 50, 'repl': False}
mod=opls(X, YY, cv=cv)
mod.run_da()
mod.plot_scores(mf=5)



class opls:
    def __init__(self, X, Y, cv={'type': 'mc', 'f': 2 / 3, 'k': 2, 'repl': False}, eps=10e-10):
        self.yinfo=_check_Y(Y)
        # NIPALS stop crit
        self.eps = 10e-10
        # define ori data and scaled data
        self.ppm = ppm
        self.xmean = np.mean(X, 0)
        self.xsd = np.std(X, 0)
        self.Xs = (X - self.xmean) / self.xsd
        self.Y=self.yinfo[5]
        # if self.Y.ndim ==1:
        #     self.Y = self.Y[..., np.newaxis]
        # else: self.Y = Y
        self.Ymean = np.mean(self.yinfo[0])
        self.Ysd = np.std(self.yinfo[0])
        self.Ys = ((self.yinfo[0] - self.Ymean) / self.Ysd)
        self.ymap = self.yinfo[4]
        # define cv sets
        cv['n']= Xs.shape[0]
        self.cv = cv
        self.sets_t = np.array([np.random.choice(self.cv['n'], size=int(np.round(self.cv['f'] * self.cv['n'])), \
                                                 replace=self.cv['repl']) for i in range(self.cv['k'])])
        self.sets_v = np.array([list(set(np.arange(self.Xs.shape[0])) - set(self.sets_t[i, :])) for i in
                                range(self.sets_t.shape[0])])
        mask = np.zeros((self.cv['n'], self.cv['k']))
        mask[:] = np.NaN
        self.mask = mask
        self.cvt_xres = self.Xs[self.sets_t];
        self.cvv_xres = self.Xs[self.sets_v]
        self.full_xres = self.Xs ;
        # define model parameters (predictive and orthogonal component parameters, residuals)
        # full model
        self.nc=0; self.full_p = []; self.full_w = []; self.full_t = []; self.full_c = []
        self.full_p_orth = [];  self.full_w_orth = []; self.full_t_orth = []
        self.R = self.Xs
        self.b=[]

        # cv model (tr and test)
        # scores cv training set
        self.cvt_t = []; self.cvt_t_mean = []; self.cvt_t_sd = []
        # loadings cv training set
        self.cvt_p = [];  self.cvt_p_mean = []; self.cvt_p_sd = []
        # orth loadings cv training set
        self.cvt_p_orth = []; self.cvt_p_orth_mean = [];  self.cvt_p_orth_sd = []
        # orth scores cv training set
        self.cvt_t_orth= []; self.cvt_t_orth_mean = [];  self.cvt_t_orth_sd = []
        # orth scores cv validation set
        self.cvv_t_orth = []; self.cvv_t_orth_mean = []; self.cvv_t_orth_sd = []

        self.cvv_t_pred_eval=None
        # scores cv validation set
        self.cvv_t= []; self.cvv_t_mean = [];  self.cvv_t_sd = []
        # orth filtered data
        # self.full_xres = []; self.cvt_xres = []; self.cvv_xres = []

        self.r2=[]
        self.q2 = []
        self.auc=[]
        self.pc=0
        self.oc=0
    def _nipals_comp(self, orth=True, verbose=0):
        #u = ys
        if verbose==1: print('Assigning X to cv sets')
        se=self.sets_t
        u_old = self.Ys[se]
        #Xsub = self.R[se]
        Xsub = self.cvt_xres
        eps = np.ones((se.shape[1], 1))
        if verbose == 1: print('Iteration u/w/t, then calc p')
        while all(eps > self.eps):
            #print(eps)
            # X block, w, t
            u_old_t = np.transpose(u_old, axes=[0, 2, 1])
            w_t = (u_old_t @ Xsub) / (u_old_t @ u_old)
            w_t = w_t / np.linalg.norm(w_t, axis=2)[..., np.newaxis]
            w = np.transpose(w_t, axes=[0, 2, 1])
            t = (Xsub @ w)/ (w_t @ w)
            t_t = np.transpose(t, axes=[0, 2, 1])

            # Y block c (q in Kowalski et al), u
            c_t = (t_t @ self.Ys[se]) / (t_t @ t)
            c = np.transpose(c_t,  axes=[0, 2, 1])
            u = (self.Ys[se] @ c) /(c_t @ c)
            u_t = np.transpose(u, axes=[0, 2, 1])

            eps = np.linalg.norm(u - u_old, axis=1) / np.linalg.norm(u, axis=1)
            u_old = u

        p_t = (t_t @ Xsub) / (t_t @ t)
        p = np.transpose(p_t,  axes=[0, 2, 1])

        # cv loadings
        # plt.scatter(p[0, :, 0], p[1, :, :])
        # # inner relation
        # cc=[]
        # for i in range(t.shape[0]):
        #     cc.append(_cov_cor(t[i,:,0], u[i,:,0])[1][0].tolist())
        #     plt.scatter(t[i,:,0], u[i,:,0])
        # plt.plot(cc)
        #

        if orth:
            # orth comp:w_orth, t_orth and p_orth, Xr for traning and prediction set
            if verbose == 1: print('orth: calc comp - w/t/p')
            self.oc = self.oc + 1
            w_orth = p - (((w_t @ p) /( w_t @ w))*w)
            w_orth = w_orth / np.linalg.norm(w_orth, axis=1)[..., np.newaxis]
            w_orth_t =  np.transpose(w_orth,  axes=[0, 2, 1])
            t_orth = (Xsub @ w_orth) / (w_orth_t @ w_orth)
            t_orth_t= np.transpose(t_orth,  axes=[0, 2, 1])
            p_orth_t = t_orth_t @ Xsub / (t_orth_t @ t_orth)
            p_orth = np.transpose(p_orth_t,  axes=[0, 2, 1])
            R = Xsub - (t_orth @ p_orth_t)

            # Xre and t_orth for validation set
            if verbose == 1: print('orth: prediction validation set')
            #xn=self.Xs[self.sets_v]
            xn=self.cvv_xres.copy()
            # self.cvv_xres[0:2, 0:2, 0:2]
            # self.cvt_xres[0:2, 0:2, 0:2]
            tn_orth = xn @ w_orth / (w_orth_t @ w_orth)
            Rn = xn - tn_orth @ p_orth_t
            tn = Rn @ w
            if verbose == 1: print('orth: iteration u/w/t tr & val')
            val_t = self.mask.copy(); val_t_orth = self.mask.copy();  tr_t = self.mask.copy(); tr_t_orth = self.mask.copy();
            tr_t_orth1 = np.zeros(self.mask.shape)
            tr_t_orth1.fill(np.nan)
            for i in range(self.mask.shape[1]):
                tr_t_orth[se[i,:],i]=np.squeeze(t_orth)[i,:]
                tr_t_orth1[se[i, :], i] = np.squeeze(t_orth)[i, :]
                #tr_t[se[i, :], i] = np.squeeze(t)[i, :]
                val_t_orth[self.sets_v[i,:],i]=np.squeeze(tn_orth)[i,:]
                #val_t[self.sets_v[i,:],i] =np.squeeze(tn)[i,:]

            if verbose == 1: print('orth: storing cv train and test results')
            #self.cvt_t.append(tr_t.copy());
            #self.cvt_t_mean.append(np.nanmean(tr_t, 1, keepdims=True));
           # self.cvt_t_sd.append(np.nanstd(tr_t, 1, keepdims=True))
            #self.cvv_t.append(val_t.copy());
            #self.cvv_t_mean.append(np.nanmean(val_t, 1, keepdims=True));
            #self.cvv_t_sd.append(np.nanstd(val_t, 1, keepdims=True))
           # self.cvv_t_pred_eval = np.nanmean(val_t, 1, keepdims=True);

            self.cvt_p_orth.append(p_orth)
            self.cvt_p_orth_mean.append(np.nanmean(p_orth_t, 0))
            self.cvt_p_orth_sd.append(np.nanstd(p_orth_t, 0))
            # orth scores cv training set

            # self.cvt_t.append(tr_t); self.cvt_t_mean.append(np.nanmean(tr_t, 1, keepdims=True)); self.cvt_t_sd.append(np.nanstd(tr_t, 1, keepdims=True))
            # self.cvv_t.append(val_t); self.cvv_t_mean.append(np.nanmean(val_t, 1, keepdims=True)); self.cvv_t_sd.append(np.nanstd(val_t, 1, keepdims=True))

            self.cvt_t_orth.append(tr_t_orth);  self.cvt_t_orth_mean.append(np.nanmean(tr_t_orth, 1, keepdims=True));self.cvt_t_orth_sd.append(np.nanstd(tr_t_orth, 1, keepdims=True))
            self.cvv_t_orth.append(val_t_orth); self.cvv_t_orth_mean.append(np.nanmean(val_t_orth, 1, keepdims=True)); self.cvv_t_orth_sd.append(np.nanstd(val_t_orth, 1, keepdims=True))

            #self.cvt_p.append(p_t); self.cvt_p_mean.append(np.nanmean(p_t, 0)); self.cvt_p_sd.append(np.nanstd(p_t, 0))
            self.cvt_xres = R; self.cvv_xres = Rn

            if verbose == 1: print('orth: done')
            # plt.scatter(self.cvt_t_mean[0][:,0], self.cvt_t_orth_mean[0][:,0], c=Y)
            # plt.scatter(self.cvt_t_mean[0][:,0], self.cvv_t_mean[0][:,0], c=Y)
            # plt.scatter(self.cvt_t_mean[0][:,0], self.Ys[:,0], c=Y)

        else:
            if verbose == 1: print('pred: calc comp')
            self.pc = self.pc + 1
            self.cvt_xres = Xsub - t @ p_t
            xn=self.cvv_xres
            tn = xn @ w
            Rn=  xn - tn @  p_t
            self.cvv_xres = Rn
            self.b = (u_t @ t) / (t_t @ t)

            val_t = self.mask.copy(); tr_t = self.mask.copy();  # val_t
            for i in range(self.mask.shape[1]):
                tr_t[se[i, :], i] = np.squeeze(t)[i, :]
                val_t[self.sets_v[i, :], i] = np.squeeze(tn)[i, :]
            if verbose == 1: print('pred: assigning vars')
            self.cvt_t.append(tr_t);
            self.cvt_t_mean.append(np.nanmean(tr_t, 1, keepdims=True));
            self.cvt_t_sd.append(np.nanstd(tr_t, 1, keepdims=True));

            # this is predictive component
            self.cvv_t.append(val_t.copy());
            self.cvv_t_mean.append(np.nanmean(val_t, 1, keepdims=True));
            self.cvv_t_sd.append(np.nanstd(val_t, 1, keepdims=True));

            #print('assigning next comp')
            self.cvv_t_pred_eval=np.nanmean(val_t, 1, keepdims=True);

            if verbose == 1: print('pred: assigning vars 1')
            #self.cvt_t.append(tr_t); self.cvt_t_mean.append(np.nanmean(tr_t, 1, keepdims=True)); self.cvt_t_sd.append(np.nanstd(tr_t, 1, keepdims=True))
            #self.cvv_t.append(val_t); self.cvv_t_mean.append(np.nanmean(val_t, 1, keepdims=True)); self.cvv_t_sd.append(np.nanstd(val_t, 1, keepdims=True))
            self.cvt_p.append(p_t); self.cvt_p_mean.append(np.nanmean(p_t, 0)); self.cvt_p_sd.append(np.nanstd(p_t, 0))
            self.cvv_xres=Rn
    # def _press(y, y_hat):
    #     return np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    def _r2(y, y_hat):
        return 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
    def auroc(self):
        # predictive component
        from sklearn import metrics
        idx=np.where(~np.isnan(self.cvv_t_pred_eval[:, 0]))[0]
        fpr, tpr, thresholds = metrics.roc_curve(self.yinfo[0][idx, 0], self.cvv_t_pred_eval[idx, 0],
                                                 pos_label=self.ymap.columns[1])
        self.auc.append(metrics.auc(fpr, tpr))
        return self.auc[len(self.auc)-1]
    def run_da(self):
        auc = 1;
        c = 1
        while auc > 0.7:
            print(c)
            mod._nipals_comp(orth=True, verbose=1)
            mod._nipals_comp(orth=False, verbose=1)
            auc = mod.auroc()
            print('done')
            c += 1
    def plot_scores(self, torth=1, tpred=1, mean=True, cv=True, mf=300):
        import matplotlib._color_data as mcd
        cdict = dict(zip(self.yinfo[3], list(mcd.TABLEAU_COLORS)[0:len(self.yinfo[3])]))
        x = np.squeeze(self.cvv_t_mean[tpred - 1][:, 0])
        x_sd = np.squeeze(np.array(self.cvt_t_sd[tpred - 1][:, 0]))
        y = np.squeeze(self.cvv_t_orth_mean[torth-1][:, 0])
        y_sd = np.squeeze(np.array(self.cvt_t_orth_sd[torth-1][:, 0]))

        # calculate ellipse
        el = self.ellipse(x, y, alpha=0.95)

        ax = plt.subplot()
        ax.axhline(0, color='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=0.3)
        ax.plot(el[0], el[1], color='gray', linewidth=0.5, linestyle='dotted')
        ax.set_xlabel('t_pred' + str(tpred))  # +' ('+str(self.r2[pc[0]-1])+'%)')
        ax.set_ylabel('t_orth' + str(torth))  # +' ('+str(self.r2[pc[1]-1])+'%)')

        for i in self.yinfo[3]:
            ix = np.where(self.Y[:,0] == i)
            ax.scatter(x[ix], y[ix], c=cdict[i], label=i)


        iid = np.where(x_sd > y_sd)[0].tolist()
        iis = np.where(~(x_sd > y_sd))[0].tolist()
        ax.scatter(x[iid], y[iid], s=np.array(x_sd[iid]) * mf, edgecolors='red', label='sd pred', c='none')
        ax.scatter(x, y, s=np.array(y_sd) * mf, edgecolors='green', label='sd orth', c='none')
        ax.scatter(x[iis], y[iis], s=np.array(x_sd[iis]) * mf, edgecolors='red', c='none')
        ax.legend()
        n = self.cvv_t[0].shape[0]

        for i in range(n):
            print(i)
            x1 = self.cvv_t[0][i, :]
            x2 = self.cvv_t_orth[0][i, :]
            idx = ~np.isnan(x1)
            x2 = x2[idx]
            x1 = x1[idx]

            el = self.ellipse(x2, x1, alpha=0.95)
            #plt.plot(el[0], el[1], c='black')
            #plt.plot(el[0]+np.mean(x1), el[1]+np.mean(x2), color='gray', linewidth=0.5, linestyle='dotted')
            #plt.plot(el[0]+np.mean(x1), el[1]+np.mean(x2))
            ax.plot(el[0]+np.mean(x1), el[1]+np.mean(x2), color='gray', linewidth=0.5, linestyle='dotted')

        return ax

    def ellipse(self, x, y, alpha=0.95):
        from scipy.stats import chi2
        theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
        circle = np.array((np.cos(theta), np.sin(theta)))
        cov = np.cov(x, y)
        ed = np.sqrt(chi2.ppf(alpha, 2))
        ell = circle.T.dot(np.linalg.cholesky(cov).T * ed)
        a, b = np.max(ell[:, 0]), np.max(ell[:, 1])  # 95% ellipse bounds
        t = np.linspace(0, 2 * np.pi, len(x))

        el_x = a * np.cos(t)
        el_y = b * np.sin(t)
        #plt.scatter(el_x, el_y)
        return el_x, el_y


n=self.cvv_t[0].shape[0]

for i in range(n):
    print(i)
    x1 = self.cvv_t[0][i,:]
    x2 = self.cvv_t_orth[0][i,:]
    idx = ~np.isnan(x1)
    x2 = x2[idx]
    x1 = x1[idx]

    el = ellipse(x1, x2, alpha=0.95)
    plt.plot(el[0], el[1], color='gray', linewidth=0.5, linestyle='dotted')



plt.scatter(x1, x2)


theta = np.arange(0, 2*np.pi, 0.01)
a = 1
b = 2

xpos = a*np.cos(theta)
ypos = b*np.cos(theta)
new_xpos = xpos*np.cos(np.pi/2)+ypos*np.sin(np.pi/2)
new_ypos = -xpos*np.sin(np.pi/2)+ypos*np.cos(np.pi/2)
plt.plot(xpos, ypos, 'b-')
plt.plot(new_xpos, new_ypos, 'r-')
plt.show()

# calculate auroc for cv ev evaluation
if self.yinfo[1] == 'DA':

    plt.plot(fpr, tpr)
    plt.plot(sens, 1 - spec)
    outc = metrics.auc(fpr, tpr)
else:
    y_hat =
    1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))

def _autostop(self):
    nc=2
    r2=1
    _nipals_comp(self, orth=False)
    while r2 > 0.1:
        _nipals_comp(self, orth=True)

def auroc_youden(self, ycont=self.cvv_t_mean[0][:,self.cvv_t_mean[0].shape[1]-1])  :
    tt = opt_cut(y=self.Ys[:, 0], ycont)
    self.optcut = pd.DataFrame(tt, columns=['cp', 'n', 'tp', 'tn', 'fp', 'fn'])
    sens = self.optcut.tp / self.optcut.fn
    spec = (self.optcut.tn / self.optcut.fp)
    youd = sens + spec - 1
    plt.plot(youd)
    cutp_idx = np.argmin(youd)
    df.cp.iloc[cutp_idx]


            if self.yinfo[1] == 'DA':
                tt = opt_cut(y=self.Ys[:, 0], ycont=self.cvv_t_mean[0][:, 0])
                self.optcut = pd.DataFrame(tt, columns=['cp', 'n', 'tp', 'tn', 'fp', 'fn', 'recall', 'prec', 'f1'])
                sens =  self.optcut.tp /  self.optcut.fn
                spec = ( self.optcut.tn /  self.optcut.fp)
                youd = sens + spec - 1
                plt.plot(youd)
                cutp_idx=np.argmin(youd)
                df.cp.iloc[cutp_idx]


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(self.Ys[:, 0], self.cvv_t_mean[0][:, 0], pos_label=self.ymap.columns[1])
plt.plot(fpr, tpr)
plt.plot(sens, 1-spec)
metrics.auc(fpr, tpr)

            _r2(self.Ys[:,0], self.cvv_t_mean[0][:,0])

            _press(self.Ys[:,0], self.cvv_t_mean[0][:,0])
            tt=opt_cut(y=self.Ys[:,0], ycont=self.cvv_t_mean[0][:,0])
            df = pd.DataFrame(tt, columns=['cp', 'n', 'tp', 'tn', 'fp', 'fn', 'recall', 'prec', 'f1'])
            sens=df.tp/df.fn
            spec=(df.tn/df.fp)
            len(spec)
            #youd=np.sum(sens)/len(spec)
            sens
            youd=sens+spec
            plt.plot(youd)

            df.iloc[np.argmin(youd)]

            idx=np.argsort(spec1.values)
            spec1 = 1 - (df.tn / df.fp)
            spec1=spec1.values[idx]
            sens=sens.values[idx]
            plt.plot(spec1, sens)
            ars=[]
            for i in range(1, len(sens)):
                h=spec1[i]-spec1[i - 1]
                ars.append(trap(sens[i-1], sens[i], h))

            plt.scatter(df.tp/df.fn, 1-(df.tn/df.fp))
            plt.scatter(df.tn/df.fp, df.tn)

plt.scatter(sens, 1-spec)

xr=sens.values
yr=1-spec.values

trap(xr, yr)

def trap(xr, yr):
    idx=np.argsort(xr)
    xr=xr[idx]; yr=yr[idx]
    ss=[]
    for i in range(1, len(xr)):
        n=(xr[i-1]-xr[i])
        ys=np.min([xr[i-1], xr[i]])
        yt_min = np.min([yr[i - 1], yr[i]])
        yt_max=np.max([yr[i-1], yr[i]])
        ss.append(ys*(yt_min) +( 1/2 * ys * yt_max-yt_min))
    return np.sum(ss)

# calculate sensitivity and specificity for each cutpoint
def sens_spec(y, yh, cp):
    # make sure that y is two-valued (0 and 1) with 0 being negative (control) and 1 being positive (case)
    ct=pd.crosstab(y, yh)
    tn=ct.iloc[0,0]
    tp = ct.iloc[1, 1]
    fn = ct.iloc[0, 1]
    fp = ct.iloc[1, 0]
    # recall = tp / (tp+fn)
    # precision = tp / (tp + fp)
    # f1 = 2 * ((precision * recall) / (precision + recall))
    # return {'cp':cp, 'n':len(y), 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'recall':recall, 'prec':precision, 'f1':f1}
    return [cp, len(y), tp, tn, fp, fn] # , recall, precision, f1



ycont=self.cvv_t_mean[0][:, 0]
y = Y - 1
def opt_cut(y, ycont):
    yc_un = np.unique(ycont)
    return [sens_spec(y, ycut, cut) for cut in yc_un]


y = Y - 1
yh = (self.cvv_t_mean[0][:, 0] > 0).astype(int)
sens_spec(y, yh)

_check_Y(Y)



def _check_Y(y):

    yori=y
    ymap=None
    if isinstance(y, list):
        y=np.array(y)
        yori=np.array(yori)

    if (y.ndim > 1 and y.shape[1] > 1):
        raise ValueError('Multi Y not allowed')
    else:
        if y.ndim == 1:
            if isinstance(y, np.ndarray):
                y = y[..., np.newaxis]
                yori=yori[..., np.newaxis]

            if isinstance(y, list):
                y = np.array(y)[..., np.newaxis]
                yori = np.array(yori)[..., np.newaxis]
            else:
                if isinstance(y, pd.Series):
                    y = np.array(y)[..., np.newaxis]
                    yori = np.array(yori)[..., np.newaxis]

            if not isinstance(y, np.ndarray):
                raise ValueError('Proved numpy array or list.')

        dkind = y.dtype.kind
        uy=np.unique(y)
        yle=len(uy)

        if yle == 1:
            raise ValueError('Y has only a single level.')

        if (dkind in ['i', 'f', 'u']):
            if (yle > 2):
                kind = 'R'
                y = y.astype(float)
            if (yle == 2):
                kind = 'DA'
                y = y.astype(float)


        else:
            if (dkind in ['b', 'S', 'U']) | (yle == 2):
                kind = 'DA'

                y = (y==uy[0]).astype(int)
                ymap = pd.crosstab(np.squeeze(yori), np.squeeze(y))
            else:
                raise ValueError('Check data type of Y')

    return (y, kind, y.shape, uy, ymap, yori)

_check_Y(YY)

def pred(self):


    tn_orth = xn @ w_orth /( w_orth_t @ w_orth)
     En = xn - tn_orth @ p_orth_t
    t_cor=[]
    for i in len(self.t_orth):
    # remove orth
    t_cor.append((vs[i] @ self.w_orth[i]) /  (np.transpose(self.w_orth[i], axes=[0,2,1]) @ self.w_orth[i]))

    np.transpose(self.p_orth[i], axes=[0,2,1])

    self.t_orth[i] @ self.p_orth[i]



        cs=self.sets[..., np.newaxis]
        Xs = self.R[cs]
        u_old.shape
        Xs.shape
        u_old.shape
        (u_old @ Xs.T).shape

        cs.shape
        Xs[cs].shape
        eps = 1
        while eps > ct:
            print(eps)
            w=[u_old[:,i][..., np.newaxis].T @ Xs[:,i,:] / u_old[:,i].dot(u_old[:,i]) for i in range(cs.shape[1])]
            #w = (u_old.T @ Xs / u_old.T.dot(y)).T
            #w = w / np.linalg.norm(w)
            w = [(u_old[:, i][..., np.newaxis].T @ Xs[:, i, :] / u_old[:, i].dot(u_old[:, i])).T for i in
                 range(cs.shape[1])]
            w=[w[i] / np.linalg.norm(w[i]) for i in range(len(w))]

            # t = Xs @ w / w.T.dot(w)
            t = [Xs[:,i,:] @ w[i] / w[i] .T.dot(w[i] ) for i in range(len(w))]
            # c = (t.T @ ys / t.T.dot(t)).T
            c=[t[i].T @ u_old[:,i][..., np.newaxis] / t[i].T.dot(t[i]) for i in range(len(w))]
            u = ys @ c / c.dot(c)
            eps = np.linalg.norm(u - u_old)
            eps = eps / np.linalg.norm(u)
            u_old = u

        p = (t.T @ Xs / t.T.dot(t)).T

        if orth:
            w_orth = p - (w.T @ p / w.T.dot(w))
            w_orth = w_orth / np.linalg.norm(w_orth)
            t_orth = Xs @ w_orth / w_orth.T.dot(w_orth)
            p_orth = t_orth.T @ Xs / t_orth.T.dot(t_orth)
            R = Xs - t_orth @ p_orth

            self.p_orth.append(p_orth)
            self.w_orth.append(w_orth)
            self.t_orth.append(w_orth)
            self.R=R
        else:
            R = Xs - t @ p.T
            self.R = R
            self.p.append(p)
            self.w.append(w)
            self.t.append(t)
            self.c.append(c)

mod=opls(Xs, Ys)
mod.cv
mod.mccv_train()
mod.sets.shape

mod.nipals_comp()


# input is either pd series or np array

_

y=meta[['EXP', 'SFO1']]
_check_Y(y)

from sklearn.datasets import load_iris
iris = load_iris()

X=iris['data']
Y=iris['target']
idx=np.where((Y==1) | (Y==2))[0]

X=X[idx]
y=Y[idx]
Xs=X
Ys=Y

_check_Y(Ys)

Xs=(Xs-np.mean(Xs, 0)) / np.std(Xs, 0)
ys=((Ys-np.mean(Ys)) / np.std(Ys))[..., np.newaxis]
(p, w, t, c, R, w_orth, t_orth, p_orth) = nipals_comp(Xs, ys, ct=10e-10, orth=True)

(p1, w1, t1, c1, R1) = nipals_comp(R, ys, ct=10e-10, orth=False)
plt.scatter(t, t_orth, c=Ys)

_predict_y()



# estimator roc



(p, w, t, c, R) = nipals_comp(Xs, ys, ct=10e-10, orth=False)

(p1, w1, t1, c1, R1) = nipals_comp(R, ys, ct=10e-10, orth=False)

plt.scatter(t, t_orth, c=Ys)





def nipals_comp(Xs, ys, ct=10e-10, orth=True):
    #u = ys
    u_old = ys
    eps = 1
    while eps > ct:
        print(eps)
        w = (u_old.T @ Xs / u_old.T.dot(y)).T
        w = w / np.linalg.norm(w)

        t = Xs @ w / w.T.dot(w)
        c = (t.T @ ys / t.T.dot(t)).T
        u = ys @ c / c.dot(c)
        eps = np.linalg.norm(u - u_old)
        eps = eps / np.linalg.norm(u)
        u_old = u

    p = (t.T @ Xs / t.T.dot(t)).T

    if orth:
        w_orth = p - (w.T @ p / w.T.dot(w))
        w_orth = w_orth / np.linalg.norm(w_orth)
        t_orth = Xs @ w_orth / w_orth.T.dot(w_orth)
        p_orth = t_orth.T @ Xs / t_orth.T.dot(t_orth)
        R = Xs - t_orth @ p_orth
        return (p, w, t, c, R, w_orth, t_orth, p_orth)
    else:
        R = Xs - t @ p.T

    return (p, w, t, c, R)


def orthproj(X, t, p, w):