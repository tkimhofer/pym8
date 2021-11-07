# cv - opls da
import sys
sys.path.insert(0, "/Users/torbenkimhofer/py/mm8")
import mm8
import numpy as np
import pandas as pd



def _scale(X):
    xm = np.mean(X, 0)
    xsd = np.std(X, 0)
    return (X - xm / xsd)


# case single col Y
y = np.max(X, 1)[..., np.newaxis]

Xs = _scale(X)
ys = _scale(y)

class opls:
    def __init__(self, X, Y, cv={'type': 'mc', 'f': 2 / 3, 'k': 2, 'repl': False}, ct=10e-10):

        # NIPALS stop crit
        self.ct = ct

        # define ori data and scaled data
        self.ppm = ppm
        self.xmean = np.mean(X, 0)
        self.xsd = np.std(X, 0)
        self.Xs = (X - self.xmean) / self.xsd
        self.Y = Y
        self.Ymean = np.mean(Y)
        self.Ysd = np.std(Y)
        self.Ys = ((self.Y - self.Ymean) / self.Ysd)[..., np.newaxis]

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

        # define model parameters (predictive and orthogonal component parameters, residuals)
        # full model
        self.nc=0; self.full_p = []; self.full_w = []; self.full_t = []; self.full_c = []
        self.full_p_orth = [];  self.full_w_orth = []; self.full_t_orth = []
        self.R = self.Xs

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
        # scores cv validation set
        self.cvv_t= []; self.cvv_t_mean = [];  self.cvv_t_sd = []
        # orth filtered data
        self.full_xres = []; self.cvt_xres = []; self.cvv_xres = []

        self.cvt_xres=self.Xs[self.sets_t]; self.cvv_xres=self.Xs[self.sets_v]


    def _nipals_comp(self, orth=True):
        #u = ys
        se=self.sets_t
        u_old = self.Ys[se]
        #Xsub = self.R[se]
        Xsub = self.cvt_xres
        eps = np.ones((se.shape[1], 1))

        while all(eps > self.ct):
            # print(eps)
            u_old_t = np.transpose(u_old, axes=[0, 2, 1])
            w_t = (u_old_t @ Xsub) / (u_old_t @ u_old)
            w_t = w_t / np.linalg.norm(w_t, axis=2)[..., np.newaxis]
            w = np.transpose(w_t, axes=[0, 2, 1])


            t = (Xsub @ w)/ (w_t @ w)
            t_t = np.transpose(t, axes=[0, 2, 1])
            c_t = (t_t @ u_old) / (t_t @ t)
            c = np.transpose(c_t,  axes=[0, 2, 1])
            u = (u_old @ c) /(c_t @ c)
            eps = np.linalg.norm(u - u_old, axis=1) / np.linalg.norm(u, axis=1)
            u_old = u

        p_t = (t_t @ Xsub) / (t_t @ t)
        p = np.transpose(p_t,  axes=[0, 2, 1])

        if orth:
            w_orth = p - (((w_t @ p) /( w_t @ w))*w)
            w_orth = w_orth / np.linalg.norm(w_orth, axis=1)[..., np.newaxis]
            w_orth_t =  np.transpose(w_orth,  axes=[0, 2, 1])
            t_orth = (Xsub @ w_orth) / (w_orth_t @ w_orth)
            t_orth_t= np.transpose(t_orth,  axes=[0, 2, 1])
            p_orth_t = t_orth_t @ Xsub / (t_orth_t @ t_orth)
            p_orth = np.transpose(p_orth_t,  axes=[0, 2, 1])
            R = Xsub - (t_orth @ p_orth_t)
            # valset_idx = np.array([list(set(np.arange(self.Xs.shape[0])) - set(self.sets[i, :])) for i in
            #               range(self.sets.shape[0])])

            #xn=self.Xs[self.sets_v]
            xn=self.cvv_xres
            tn_orth = xn @ w_orth / (w_orth_t @ w_orth)
            Rn = xn - tn_orth @ p_orth_t
            tn = Rn @ w

            val_t_orth = self.mask.copy(); # val_t_orth
            val_t = self.mask.copy(); # val_t
            tr_t_orth = self.mask.copy();  # val_t_orth
            tr_t = self.mask.copy();  # val_t

            for i in range(self.mask.shape[1]):
                tr_t_orth[se[i,:],i]=np.squeeze(t_orth)[i,:] #TODO: DOUBLE CHECK COL ROW INDEX
                tr_t[se[i, :], i] = np.squeeze(t)[i, :]  # TODO: DOUBLE CHECK COL ROW INDEX
                val_t_orth[self.sets_v[i,:],i]=np.squeeze(tn_orth)[i,:]
                val_t[self.sets_v[i,:],i] =np.squeeze(tn)[i,:]

            self.cvt_t.append(tr_t)
            self.cvt_t_mean.append(np.nanmean(tr_t, 1, keepdims=True))
            self.cvt_t_sd.append(np.nanstd(tr_t, 1, keepdims=True))

            self.cvt_t_orth.append(tr_t_orth)
            self.cvt_t_orth_mean.append(np.nanmean(tr_t_orth, 1, keepdims=True))
            self.cvt_t_orth_sd.append(np.nanstd(tr_t_orth, 1, keepdims=True))

            self.cvv_t.append(val_t)
            self.cvv_t_mean.append(np.nanmean(val_t, 1, keepdims=True))
            self.cvv_t_sd.append(np.nanstd(val_t, 1, keepdims=True))

            self.cvv_t_orth.append(val_t_orth)
            self.cvv_t_orth_mean.append(np.nanmean(val_t_orth, 1, keepdims=True))
            self.cvv_t_orth_sd.append(np.nanstd(val_t_orth, 1, keepdims=True))

            self.cvt_p.append(p_t);  # train_p
            self.cvt_p_mean.append(np.nanmean(p_t, 0))
            self.cvt_p_sd.append(np.nanstd(p_t, 0))

            self.cvt_p_orth.append(p_orth)  # train_p_orth
            self.cvt_p_orth_mean.append(np.nanmean(p_orth, 0))
            self.cvt_p_orth_sd.append(np.nanstd(p_orth, 0))

            self.cvt_xres=R
            self.cvv_xres=Rn

        else:
            R = Xsub - t @ p_t
            xn=self.cvv_xres
            tn = xn  @ w
            Rn=  xn - tn @  p_t
            # self.R = R
            # self.p.append(p)
            # self.w.append(w)
            # self.t.append(t)
            # self.c.append(c)

            val_t = self.mask.copy();  # val_t
            tr_t = self.mask.copy();  # val_t
            for i in range(self.mask.shape[1]):
                tr_t[se[i, :], i] = np.squeeze(t)[i, :]
                val_t[self.sets_v[i, :], i] = np.squeeze(tn)[i, :]

            self.cvt_t.append(tr_t)
            self.cvt_t_mean.append(np.nanmean(tr_t, 1, keepdims=True))
            self.cvt_t_sd.append(np.nanstd(tr_t, 1, keepdims=True))

            self.cvv_t.append(val_t)
            self.cvv_t_mean.append(np.nanmean(val_t, 1, keepdims=True))
            self.cvv_t_sd.append(np.nanstd(val_t, 1, keepdims=True))

            self.cvt_p.append(p_t);  # train_p
            self.cvt_p_mean.append(np.nanmean(p_t, 0))
            self.cvt_p_sd.append(np.nanstd(p_t, 0))

            self.cvt_xres=R
            self.cvv_xres=Rn


    def _r2(self, y, y_hat):
        return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def _autostop(self):

        while xxx:
            _nipals_comp(self, orth=True)



def pred(self):
    valset=[self.Xs[list(set(np.arange(self.Xs.shape[0]))-set(self.sets[i,:])),:] for i in range(self.sets.shape[0])]
    xn=np.array(valset)

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


def _check_Y(y):
    if (y.ndim > 1 and y.shape[1]>1):
        raise ValueError('Multi Y not allowed')
    # if y.ndim == 2:
    #     if isinstance(y, pd.DataFrame):
    #         dtypes = y.dtypes
    #         y = np.array(y)
    #     else:
    #         raise ValueError('Provide pandas dataframe')
    else:
        if y.ndim == 1 :
            if isinstance(y, list):
                y=np.array(y)[..., np.newaxis]
            else:
                if isinstance(y, pd.Series):
                    y = np.array(y)[..., np.newaxis]

            if not isinstance(y, np.ndarray):
                raise ValueError('Proved numpy array or list.')

        dkind=y.dtype.kind
        if dkind in ['i', 'f', 'u']:
            kind = 'R'
            y =y.astype(float)

        else:
            if dkind in ['b', 'S', 'U']:
                kind='DA'
                y=y.astype(str)
            else:
                raise ValueError('Check data type of Y')

    return (y, kind, y.shape)



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


