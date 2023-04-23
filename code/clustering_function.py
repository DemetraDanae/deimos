# clustering function after OPLRA training

def clust(regions, Frs, Wrf, X_train, y_train):
    
    import pandas as pd
    import numpy as np
    from scipy.spatial import distance
    
    samples_train = y_train.count()
    
    # All selected variables
    var = []
    for re in range(regions):
        var.append(list(np.where(Wrf[re]))[0])
    
    var_all = []
    for re in range(regions):
        var_all = list(set().union(var_all, var[re]))
        
    # Training clusters
    X_cluster = []
    y_cluster = []
    for re in range(regions):
        X_flag = X_train.iloc[np.where(Frs[re])]
        X_cluster.append(X_flag.iloc[:, var_all])
        y_cluster.append(y_train.iloc[np.where(Frs[re])])   
        
    # Centroids calculation
    centroid_cluster = []
    for re in range(regions):
        centroid_cluster.append(np.mean(X_cluster[re]))
        
    # Check if training samples are correctly classified
    dst_train = []
    for re in range(regions):
        dst_flag = [[] for i in range(len(X_train.iloc[:, var_all]))]
        for i in range(len(X_train)):
            dst_flag[i] = distance.euclidean(X_train.iloc[:,var_all].values.tolist()[i], centroid_cluster[re])
        dst_train.append(dst_flag)
    
    train_idx = [[0 for i in range(samples_train)] for re in range(regions)]
    for i in range(len(X_train)):
        row_flag = list(dst_train[re][i] for re in range(regions))
        position = row_flag.index(min(row_flag))
        train_idx[position][i] = 1
    
    correctly_assigned = pd.DataFrame(Frs)== pd.DataFrame(train_idx)
    print("Correctly assigned training samples", sum(correctly_assigned.iloc[0]), "out of", samples_train)
    
    return centroid_cluster, var_all, correctly_assigned