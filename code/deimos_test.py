# test function for deimos results evaluation

def deimos_1D_test(centroid_cluster, var_all, regions, X_test, y_test, lREG, Wrf, Br, REG):

    import pandas as pd
    import numpy as np  
    from scipy.spatial import distance
    
    samples_test = y_test.count() 
    
    dst_test = []
    for re in range(regions):
        dst_flag = [[] for i in range(len(X_test.iloc[:, var_all]))]
        for i in range(len(X_test)):
            dst_flag[i] = distance.euclidean(X_test.iloc[:,var_all].values.tolist()[i], centroid_cluster[re])
        dst_test.append(dst_flag) 
        
    test_idx = [[0 for i in range(samples_test)] for re in range(regions)]
    for i in range(len(X_test)):
        row_flag = list(dst_test[re][i] for re in range(regions))
        position = row_flag.index(min(row_flag))
        test_idx[position][i] = 1
        
    Pred_test = pd.DataFrame(0, index=range(regions), columns=range(samples_test))

    for re in range(regions):
        for i in range(samples_test):
            Pred_test.iloc[re, i] = sum(X_test.values.tolist()[i][fe]*Wrf[re][fe] for fe in range(len(X_test.columns))) + Br[re]

    Ps_test = pd.DataFrame(test_idx) * Pred_test
    y_hat_test = pd.DataFrame(sum(Ps_test.iloc))

    E = []
    for i in range(samples_test):
        E.append(float(abs(y_test.iloc[i]-y_hat_test.iloc[i])))

    MAE_test = sum(E)/samples_test

    z_test = MAE_test + lREG * REG

    return z_test, MAE_test, test_idx, Ps_test, y_hat_test