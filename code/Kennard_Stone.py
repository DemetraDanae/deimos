# Kennard and Stone partition method same as R kenStone


def kenStone(X, Y, train_ratio):
    import pandas as pd
    import numpy as np

    #X = (X - X.min()) / (X.max() - X.min())

    # Initialization
    k = round(train_ratio*len(X))
    originalX = X

    train_idx = list()

    from scipy.spatial import distance_matrix

    # Select first two objects (farthest apart)
    distA = pd.DataFrame(distance_matrix(X, X), index=X.index, columns=X.index)
    obj1, obj2 = distA.stack().index[np.argmax(distA.values)]
    train_idx.append(obj1)  # add 1st object to list
    train_idx.append(obj2)  # add 2nd object to list
    X = X.drop(obj1) # remove 1st object from initial dataset
    X = X.drop(obj2) # remove 2nd object from initial dataset

    # Rest of objects:
    # For each remaining object find the "training" ones that are neighbors (min distance)
    # Select the most distant one
    for i in range(0, k-2):
        distB = pd.DataFrame(distance_matrix(X, pd.DataFrame(originalX.loc[train_idx,:])), index=X.index, columns=train_idx)
        #closer = list()
        #for j in range(0, len(X)):
        #    closer.append(min(distB.iloc[j,])) #(pd.DataFrame([min(distB.iloc[j,])]))
        #obj = X.index[closer.index(max(closer))]

        obj = pd.DataFrame(distB.min(axis=1)).index[np.argmax(distB.min(axis=1).values)] #alternative to lines 31-34

        train_idx.append(obj)
        X = X.drop(obj)

    trainX = originalX.loc[train_idx,]
    trainY = Y.loc[train_idx,]

    test_idx = X.index
    testX = originalX.loc[test_idx,]
    testY = Y.loc[test_idx,]

    return trainX, testX, trainY, testY