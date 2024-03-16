import sys
from scipy import linalg
from sklearn.base import BaseEstimator
sys.path.append(r'../')
from utils import *
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import scipy

# Numbers of disease labels and cognitive scores
N_dis = 3
N_score = 12

class SMJFS(BaseEstimator):
    def __init__(self, alpha1=1, alpha2=1, beta=1, gamma=1):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.gamma = gamma

    def get_params(self, deep=True):
        return {'alpha1': self.alpha1,
                'alpha2': self.alpha2,
                'beta': self.beta,
                'gamma': self.gamma
                }

    def fit(self, X, Y):
        # Y_cls: disease label, Y_score: cognitive scores, X: sMRI feature matrix
        Y_cls = Y[:, :N_dis]
        Y_score = Y[:, N_dis:]

        alpha1 = self.alpha1
        alpha2 = self.alpha2
        beta = self.beta
        gamma = self.gamma

        niumax = 1e6
        FRA = 1.1
        niu = 1e0
        SMALL = 1e-8
        ITE = 200
        m = 2

        X = X.transpose()
        Y_score = Y_score.transpose()
        Y_cc = np.argmax(Y_cls, axis=1)

        n_samples = X.shape[1]
        n_features = X.shape[0]
        n_scores = Y_score.shape[0]
        n_classes = np.unique(Y_cc)

        Xlist = list()
        Y_scorelist = list()
        Y_cclist = list()

        # Reorder input data according to category order
        for ite_c in n_classes:
            idxc = np.where(Y_cc == ite_c)[0]
            Xlist.append(X[:, idxc])
            Y_scorelist.append(Y_score[:, idxc])
            Y_cclist.append(Y_cc[idxc])
        X = np.concatenate(Xlist, axis=1)
        Y_score = np.concatenate(Y_scorelist, axis=1)
        Y_cc = np.concatenate(Y_cclist)

        idxcList = list()
        for ite_c in n_classes:
            idxcList.append(np.where(Y_cc == ite_c)[0])

        # Calculate S
        S = squareform(pdist(X.transpose()))
        S = np.exp(-1 * S * S)
        L_A = np.diag(np.sum(S, axis=1)) - S

        # Calculate St
        St2 = np.zeros((n_features, n_features))
        ave = np.sum(X, axis=1) / n_samples
        for ite_i in range(n_samples):
            tepmv = X[:, ite_i] - ave
            tempv = tepmv.reshape((n_features, 1))
            St2 += np.matmul(tempv, tempv.transpose())
        St2_INV = np.linalg.inv(St2)

        W = np.random.randn(n_features, m)
        W = np.matmul(W, np.diag(1. / np.sqrt(np.diagonal(np.matmul(np.matmul(W.transpose(), St2), W)) + SMALL)))
        R = 1 * Y_score
        P = np.zeros((n_features, n_scores))
        Z = np.zeros_like(R)
        F = np.zeros_like(R)

        for ite in range(ITE):
            lossInf = np.linalg.norm(R - F, ord=np.inf)
            if lossInf < 1e-7 and ite > 5:
                break

            # Update P
            P1 = np.matmul(X, X.transpose()) + alpha2 * np.diag(1.0 / (2 * np.linalg.norm(P, axis=1) + SMALL))
            P2 = np.matmul(X, R.transpose())
            P = np.matmul(np.linalg.inv(P1), P2)

            # Update R
            R1 = np.matmul(P.transpose(), X) + niu / 2 * F - Z / 2
            R2 = 2 * gamma * L_A + (1 + niu / 2) * np.eye(L_A.shape[0])
            R = np.matmul(R1, np.linalg.inv(R2))

            # Update F
            F1 = R + Z / niu
            F2 = beta * Y_score + niu / 2 * R + Z / 2
            F = (Y_score >= 0) * F2 / (beta + niu / 2) + (Y_score < 0) * F1

            # Update W
            W_1 = 2 * np.matmul(X, np.matmul(L_A, X.transpose()))
            W = np.diag(1.0 / (2 * np.linalg.norm(W, axis=1) + SMALL))
            W = np.matmul(St2_INV, W_1 + alpha1 * W)
            eigValues, eigVectors = scipy.linalg.eig(W)

            smallests = eigValues.argsort()[:m]
            W = eigVectors[:, smallests]
            W = np.matmul(W, np.diag(1. / np.sqrt(np.diagonal(np.matmul(np.matmul(W.transpose(), St2), W)) + SMALL)))

            # Update S
            for ite_c in n_classes:
                idxc = idxcList[ite_c]
                Xd = np.matmul(W.transpose(), X[:, idxc]).transpose()
                Sd = R[:, idxc].transpose()
                Xd = squareform(pdist(Xd))
                Sd = squareform(pdist(Sd))
                Sd = 1 / (Xd * Xd + gamma * Sd * Sd + SMALL)

                Sd = Sd * (np.ones((Sd.shape[0], Sd.shape[0])) - np.eye(Sd.shape[0]))
                for d, ite_j in enumerate(idxc):
                    S[ite_j, idxc] = Sd[d, :] / (np.sum(Sd[d, :]) - Sd[d, d])
            S = S * S
            S = (S + S.transpose()) * 0.5
            L_A = np.diag(np.sum(S, axis=1)) - S

            Z = Z + niu * (R - F)
            niu = min(niumax, FRA * niu)

        return W, P
