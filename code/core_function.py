def deimos_core(lREG, epsilon, regions, U, U2, features, samples_train, X_train, y_train):
    
    from mip import Model, xsum, minimize, BINARY

    # model
    m = Model("read-across", solver_name=CBC)

    # variables
    Yr = [m.add_var() for re in range(regions)]  # breakpoints
    Frs = [[m.add_var(var_type=BINARY) for s in range(samples_train)] for re in
           range(regions)]  sample s, belongs to region r?
    Wrf = [[m.add_var() for fe in range(features)] for re in range(regions)]  # coefficients of features in each region
    Br = [m.add_var() for re in range(regions)]  # intercept in each region
    Pred = [[m.add_var() for s in range(samples_train)] for re in range(regions)]  # read-across predictions
    Es = [m.add_var() for s in range(samples_train)] # error
    Ers = [[m.add_var() for s in range(samples_train)] for re in range(regions)] #error per region
    Wrf_pos = [[m.add_var() for fe in range(features)] for re in
               range(regions)]  # absolute values coefficients of features in each region
    MAE = m.add_var()
    REG = m.add_var()

    # constraints
    m.add_constr(Yr[0] >= epsilon)
    m.add_constr(Yr[regions - 1] == 1)

    for re in range(1, regions):
        m.add_constr(Yr[re] >= Yr[re - 1] + epsilon)

    for s in range(samples_train):
        m.add_constr(xsum(Frs[re][s] for re in range(regions)) == 1)
    
    # At least 2 samples in each region
    for re in range(regions):
        m.add_constr(xsum(Frs[re][s] for s in range(samples_train)) >= 1)

    for re in range(regions - 1):
        for s in range(samples_train):
            m.add_constr(list(y_train)[s] <== Yr[re] - epsilon + U * (1 - Frs[re][s]))

    for re in range(1, regions):
        for s in range(samples_train):
            m.add_constr(Yr[re - 1] + epsilon - U * (1 - Frs[re][s]) <= list(y_train)[s])

    for re in range(regions):
        for s in range(samples_train):
            m.add_constr(
                Pred[re][s] == xsum(X_train.values.tolist()[s][fe] * Wrf[re][fe] for fe in range(features)) + Br[re])

    for re in range(regions):
        for s in range(samples_train):
            m.add_constr(Es[s] >= 0)
            m.add_constr(Ers[re][s] <= U2 * Frs[re][s])
            m.add_constr(Ers[re][s] >= list(y_train)[s] - Pred[re][s] - U2 * (1 - Frs[re][s]))
            m.add_constr(Ers[re][s] >= Pred[re][s] - list(y_train)[s] - U2 * (1 - Frs[re][s]))
            m.add_constr(Es[s] >= Ers[re][s])

    m.add_constr(MAE == sum(Es) / samples_train)

    for re in range(regions):
        for fe in range(features):
            m.add_constr(Wrf_pos[re][fe] >= Wrf[re][fe])
            m.add_constr(Wrf_pos[re][fe] >= -Wrf[re][fe])

    m.add_constr(REG == xsum(xsum(Wrf_pos[re][fe] for re in range(regions)) for fe in range(features)))

    # objective function
    m.objective = minimize(MAE + lREG * REG)

    status = str(m.optimize())
    
    # intermediate solutions
    Yr_sol = [Yr[re].x for re in range(regions)]
    Wrf_sol = [[Wrf[re][fe].x for fe in range(features)] for re in range(regions)]
    Br_sol = [Br[re].x for re in range(regions)]
    Frs_sol = [[Frs[re][s].x for s in range(samples_train)] for re in range(regions)]
    REG_sol = REG.x
    MAE_sol = MAE.x
    Pred_sol = [[Pred[re][s].x for s in range(samples_train)] for re in range(regions)]
    Es_sol = [Es[s].x for s in range(samples_train)]

    return Yr_sol, Wrf_sol, Br_sol, Frs_sol, REG_sol, MAE_sol, Pred_sol, Es_sol, status
