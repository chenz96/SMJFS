import sys
sys.path.append(r'../')
from model import *
import numpy as np

if __name__=='__main__':
    dataName = 'ADNI2'
    X = np.array( load_obj('data_'+dataName))
    Y_score  = np.array  (load_obj('scores_'+dataName))
    Y_cls = load_obj('dis_'+dataName)

    model = SMJFS()
    W, P = model.fit(X, np.concatenate((Y_cls, Y_score), axis=1))

    normv_cls = np.linalg.norm(W,axis=1)
    featureImportance_cls = np.argsort(normv_cls)

    normv_reg = np.linalg.norm(P,axis=1)
    featureImportance_reg = np.argsort(normv_reg)






