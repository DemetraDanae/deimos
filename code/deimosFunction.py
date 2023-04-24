import numpy as np
import pandas as pd
from core_function import deimos_core
from deimos_test import deimos_1D_test
from clustering_function import clust
from math import sqrt
from validation import q2ext
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def deimos_regions(lREG, MAE_in_test, REG_in, epsilon, beta, U, U2, features, samples_train, X_train, y_train, X_test, y_test, y_hat_base, y_hat_test_base):
    
    bestYr = []
    bestWrf = []  
    bestBr = []
    bestFrs = [] 
    bestREG = [] 
    bestMAE = [] 
    bestPred = [] 
    bestEs = []
    bestcentroids = []
    bestvar = []
    y_hat = []
    best_z = []
    best_correctly_assigned = []
    Ps_test = []
    RMSE_test = [] 
    y_hat_test = []
    test_idx = []
    
    z_in_test = MAE_in_test + lREG * REG_in

    ERRORcurrent_test = z_in_test
    ERRORold_test = float('inf')
    ERRORtmp_test = float('inf')

    regions = 1
    finalregions = 1
    status_sol = "OptimizationStatus.OPTIMAL"

    while ERRORcurrent_test < (1-beta) * ERRORold_test and ERRORcurrent_test != 0 and status_sol == "OptimizationStatus.OPTIMAL":

        regions = regions + 1

        # MIP Optimization
        Yr_sol, Wrf_sol, Br_sol, Frs_sol, REG_sol, MAE_sol, Pred_sol, Es_sol, status_sol = deimos_core(lREG, epsilon, regions, U, U2, features, samples_train, X_train, y_train)
    
        if status_sol == "OptimizationStatus.OPTIMAL":
               
            # Perform clustering
            centroids, var_sol, correctly_assigned_sol = clust(regions, Frs_sol, Wrf_sol, X_train, y_train)
               
            # Check optimized parameters in test set
            z_test_sol, MAE_test_sol, test_idx_sol, Ps_test_sol, y_hat_test_sol = deimos_1D_test(centroids, var_sol, regions, X_test, y_test, lREG, Wrf_sol, Br_sol, REG_sol)

            # Check if the solutions are improving
            ERRORold_test = ERRORcurrent_test
            ERRORcurrent_test = z_test_sol

            if ERRORcurrent_test < (1-beta) * ERRORold_test and ERRORcurrent_test != 0:
                finalregions = regions
                bestYr = Yr_sol
                bestWrf = Wrf_sol
                bestBr = Br_sol
                bestFrs = Frs_sol
                bestREG = REG_sol
                bestMAE = MAE_sol
                bestPred = Pred_sol
                best_z = bestMAE + lREG * bestREG
                bestEs = Es_sol
            
            del Yr_sol, Wrf_sol, Br_sol, Frs_sol, REG_sol, MAE_sol, Pred_sol, Es_sol, z_test_sol, MAE_test_sol, test_idx_sol, Ps_test_sol, y_hat_test_sol, centroids, correctly_assigned_sol
            
        else:
            ERRORcurrent_test = 0
    
    # Train test
    if finalregions >= 2:
        Ps = pd.DataFrame(bestPred)*pd.DataFrame(bestFrs)
        y_hat = np.array(sum(Ps.iloc))
    else:
        y_hat = y_hat_base
   
    if finalregions >= 2:
        bestcentroids, bestvar, best_correctly_assigned = clust(finalregions, bestFrs, bestWrf, X_train, y_train)
        
    # Classify test samples
    if finalregions >= 2:
    
        z_test, MAE_test, test_idx, Ps_test, y_hat_test = deimos_1D_test(bestcentroids, bestvar, finalregions, X_test, y_test, lREG, bestWrf, bestBr, bestREG)
    
    else:
        y_hat_test = y_hat_test_base

    # RMSE
    RMSE_test = sqrt(mean_squared_error(y_test, y_hat_test))
    
    return finalregions, bestYr, bestWrf, bestBr, bestFrs, bestREG, bestMAE, bestPred, bestEs, y_hat, best_z, best_correctly_assigned, Ps_test, RMSE_test, y_hat_test, test_idx, bestcentroids, bestvar
