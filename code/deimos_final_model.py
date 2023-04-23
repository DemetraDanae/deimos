# function for deimos prediction on untested samples

def deimos_1D_model(centroid_cluster, var_all, regions, X_untested, Wrf, Br):

    import pandas as pd
    import numpy as np  
    from scipy.spatial import distance
    
    samples_untested = X_untested.iloc[:,0].count()
    
    dst_untested = []
    for re in range(regions):
        dst_flag = [[] for i in range(len(X_untested.iloc[:, var_all]))]
        for i in range(len(X_untested)):
            dst_flag[i] = distance.euclidean(X_untested.iloc[:,var_all].values.tolist()[i], centroid_cluster[re])
        dst_untested.append(dst_flag) 
        
    untested_idx = [[0 for i in range(samples_untested)] for re in range(regions)]
    for i in range(len(X_untested)):
        row_flag = list(dst_untested[re][i] for re in range(regions))
        position = row_flag.index(min(row_flag))
        untested_idx[position][i] = 1
        
    Pred_untested = pd.DataFrame(0, index=range(regions), columns=range(samples_untested))

    for re in range(regions):
        for i in range(samples_untested):
            Pred_untested.iloc[re, i] = sum(X_untested.values.tolist()[i][fe]*Wrf[re][fe] for fe in range(len(X_untested.columns))) + Br[re]

    Ps_untested = pd.DataFrame(untested_idx) * Pred_untested
    y_hat_untested = pd.DataFrame(sum(Ps_untested.iloc))

    return y_hat_untested, untested_idx