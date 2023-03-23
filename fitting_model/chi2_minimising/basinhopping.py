import numpy as np
import chi2 as X2
from scipy.optimize import minimize

def TPA(data: np.dtype, p0: np.dtype, L: float, ALPHA0: float, I0: float, Z_R: float, MODEL_PARAMETERS: list):
    """Returns optimal fitting parameters and optimal chi-squared following a Monte-Carlo minimisation algorithm
        for the 2PA model
    
    PARAMETERS
    data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    p0 : Initial guess
        [0]: z-position of focal point - [1]: first-order saturation intensity - [2]: second-order saturation intensity - [3]: nonlinear absorption coefficient
    L : Sample thickness
    ALPHA0 : Linear absorption coefficient
    I0 : Beam intensity at the focal point
    Z_R : Rayleigh length
    MODEL_PARAMETERS : List of parameters that determine basinhopping conditions
        MAX_PERTURBATION : highest multiplication factor
        MAX_AGE : stop-condition : maximum age of current optimal point
        MAX_ITER: stop-condition : maxumum number of iterations
        BOUNDS : bounds on the fitting parameters
        T : basin hopping temperature
        MAX_JUMP : number of consecutive jumps allowed
        MAX_REJECT : number of rejects before jumping
    """
    ## Basin hopping parameters
    MAX_PERTURBATION, MAX_AGE, MAX_ITER, BOUNDS, T, MAX_JUMP, MAX_REJECT = MODEL_PARAMETERS
    
    fitting_model = lambda x: X2.TPA(data, x[0], x[1], x[2], x[3], L, ALPHA0, I0, Z_R)
    
    ## Set initial chi2
    popt = minimize(fitting_model, x0=p0, bounds=BOUNDS)
    pBest = pMin = popt.x
    chi2Prev = chi2Best = fitting_model(pMin)
    
    ## Start minimalisation process
    ### prepare iteration process
    niter = 0
    rejection = 0
    jump = 0
    bestAge = -1

    ### Iteration process
    while niter < MAX_ITER and bestAge < MAX_AGE:
        niter += 1
        bestAge += 1
        
        #### Monte Carlo Move
        pPerturbation = list(pMin)
        for i in [1,2]:
            perturbation = np.random.uniform(1, MAX_PERTURBATION)    # Determine perturbation strength
            direction = np.random.choice(['decrease', 'increase'])    # Determine perturbation direction
            if direction == 'decrease':
                pPerturbation[i] = pMin[i] / perturbation
            else:
                pPerturbation[i] = pMin[i] * perturbation

        #### Minimise Chi2
        popt = minimize(fitting_model, x0=pPerturbation, bounds=BOUNDS)
        p_newMin = popt.x
        chi2 = fitting_model(p_newMin)
        
        #### Metropolis criterion
        ##### Accept perturbation
        ###### Jumping -> accept perturbation regardless of conditions
        if jump != 0:
            chi2Prev = chi2
            pMin = p_newMin
            jump += 1
            ## Check if result is overall best
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
            ## End condition for jumping
            if jump == MAX_JUMP:    
                jump = 0
        ###### Better chi2 than previous chi2
        elif chi2 < chi2Prev:    
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            ## Check if result is overall best            
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
        ###### Metropolis
        elif np.random.uniform(0,1) < np.exp(-(chi2 - chi2Prev)/T):
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0

        ##### Reject perturbation
        else:
            rejection += 1

        ##### Max Rejections achieved? -> Start jumping
        if rejection == MAX_REJECT:
            # Accept perturbation
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            jump += 1
    
    return pBest, chi2Best
##########################################################################################
def TPA_no_Is1(data: np.dtype, p0: np.dtype, L: float, ALPHA0: float, I0: float, Z_R: float, MODEL_PARAMETERS: list):
    """Returns optimal fitting parameters and optimal chi-squared following a Monte-Carlo minimisation algorithm
        for the 2PA model without Is1
    
    PARAMETERS
    data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    p0 : Initial guess
        [0]: z-position of focal point - [1]: second-order saturation intensity - [2]: nonlinear absorption coefficient
    L : Sample thickness
    ALPHA0 : Linear absorption coefficient
    I0 : Beam intensity at the focal point
    Z_R : Rayleigh length
    MODEL_PARAMETERS : List of parameters that determine basinhopping conditions
        MAX_PERTURBATION : highest multiplication factor
        MAX_AGE : stop-condition : maximum age of current optimal point
        MAX_ITER: stop-condition : maxumum number of iterations
        BOUNDS : bounds on the fitting parameters
        T : basin hopping temperature
        MAX_JUMP : number of consecutive jumps allowed
        MAX_REJECT : number of rejects before jumping
    """
    ## Basin hopping parameters
    MAX_PERTURBATION, MAX_AGE, MAX_ITER, BOUNDS, T, MAX_JUMP, MAX_REJECT = MODEL_PARAMETERS
    
    fitting_model = lambda x: X2.TPA_no_Is1(data, x[0], x[1], x[2], L, ALPHA0, I0, Z_R)
    
    ## Set initial chi2
    popt = minimize(fitting_model, x0=p0, bounds=BOUNDS)
    pBest = pMin = popt.x
    chi2Prev = chi2Best = fitting_model(pMin)
    
    ## Start minimalisation process
    ### prepare iteration process
    niter = 0
    rejection = 0
    jump = 0
    bestAge = -1

    ### Iteration process
    while niter < MAX_ITER and bestAge < MAX_AGE:
        niter += 1
        bestAge += 1
        
        #### Monte Carlo Move
        pPerturbation = list(pMin)
        for i in [1,2]:
            perturbation = np.random.uniform(1, MAX_PERTURBATION)    # Determine perturbation strength
            direction = np.random.choice(['decrease', 'increase'])    # Determine perturbation direction
            if direction == 'decrease':
                pPerturbation[i] = pMin[i] / perturbation
            else:
                pPerturbation[i] = pMin[i] * perturbation

        #### Minimise Chi2
        popt = minimize(fitting_model, x0=pPerturbation, bounds=BOUNDS)
        p_newMin = popt.x
        chi2 = fitting_model(p_newMin)
        
        #### Metropolis criterion
        ##### Accept perturbation
        ###### Jumping -> accept perturbation regardless of conditions
        if jump != 0:
            chi2Prev = chi2
            pMin = p_newMin
            jump += 1
            ## Check if result is overall best
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
            ## End condition for jumping
            if jump == MAX_JUMP:    
                jump = 0
        ###### Better chi2 than previous chi2
        elif chi2 < chi2Prev:    
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            ## Check if result is overall best            
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
        ###### Metropolis
        elif np.random.uniform(0,1) < np.exp(-(chi2 - chi2Prev)/T):
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0

        ##### Reject perturbation
        else:
            rejection += 1

        ##### Max Rejections achieved? -> Start jumping
        if rejection == MAX_REJECT:
            # Accept perturbation
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            jump += 1
    
    return pBest, chi2Best
############################################################################################
def TPA_no_Is2(data: np.dtype, p0: np.dtype, L: float, ALPHA0: float, I0: float, Z_R: float, MODEL_PARAMETERS: list):
    """Returns optimal fitting parameters and optimal chi-squared following a Monte-Carlo minimisation algorithm
        for the 2PA model without Is2
    
    PARAMETERS
    data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    p0 : Initial guess
        [0]: z-position of focal point - [1]: first-order saturation intensity - [2]: nonlinear absorption coefficient
    L : Sample thickness
    ALPHA0 : Linear absorption coefficient
    I0 : Beam intensity at the focal point
    Z_R : Rayleigh length
    MODEL_PARAMETERS : List of parameters that determine basinhopping conditions
        MAX_PERTURBATION : highest multiplication factor
        MAX_AGE : stop-condition : maximum age of current optimal point
        MAX_ITER: stop-condition : maxumum number of iterations
        BOUNDS : bounds on the fitting parameters
        T : basin hopping temperature
        MAX_JUMP : number of consecutive jumps allowed
        MAX_REJECT : number of rejects before jumping
    """
    ## Basin hopping parameters
    MAX_PERTURBATION, MAX_AGE, MAX_ITER, BOUNDS, T, MAX_JUMP, MAX_REJECT = MODEL_PARAMETERS
    
    fitting_model = lambda x: X2.TPA_no_Is2(data, x[0], x[1], x[2], L, ALPHA0, I0, Z_R)
    
    ## Set initial chi2
    popt = minimize(fitting_model, x0=p0, bounds=BOUNDS)
    pBest = pMin = popt.x
    chi2Prev = chi2Best = fitting_model(pMin)
    
    ## Start minimalisation process
    ### prepare iteration process
    niter = 0
    rejection = 0
    jump = 0
    bestAge = -1

    ### Iteration process
    while niter < MAX_ITER and bestAge < MAX_AGE:
        niter += 1
        bestAge += 1
        
        #### Monte Carlo Move
        pPerturbation = list(pMin)
        for i in [1,2]:
            perturbation = np.random.uniform(1, MAX_PERTURBATION)    # Determine perturbation strength
            direction = np.random.choice(['decrease', 'increase'])    # Determine perturbation direction
            if direction == 'decrease':
                pPerturbation[i] = pMin[i] / perturbation
            else:
                pPerturbation[i] = pMin[i] * perturbation

        #### Minimise Chi2
        popt = minimize(fitting_model, x0=pPerturbation, bounds=BOUNDS)
        p_newMin = popt.x
        chi2 = fitting_model(p_newMin)
        
        #### Metropolis criterion
        ##### Accept perturbation
        ###### Jumping -> accept perturbation regardless of conditions
        if jump != 0:
            chi2Prev = chi2
            pMin = p_newMin
            jump += 1
            ## Check if result is overall best
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
            ## End condition for jumping
            if jump == MAX_JUMP:    
                jump = 0
        ###### Better chi2 than previous chi2
        elif chi2 < chi2Prev:    
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            ## Check if result is overall best            
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
        ###### Metropolis
        elif np.random.uniform(0,1) < np.exp(-(chi2 - chi2Prev)/T):
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0

        ##### Reject perturbation
        else:
            rejection += 1

        ##### Max Rejections achieved? -> Start jumping
        if rejection == MAX_REJECT:
            # Accept perturbation
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            jump += 1
    
    return pBest, chi2Best
######################################################################################
def TPA_no_Is2(data: np.dtype, p0: np.dtype, L: float, ALPHA0: float, I0: float, Z_R: float, MODEL_PARAMETERS: list):
    """Returns optimal fitting parameters and optimal chi-squared following a Monte-Carlo minimisation algorithm
        for the 2PA model without saturation
    
    PARAMETERS
    data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    p0 : Initial guess
        [0]: z-position of focal point - [1]: nonlinear absorption coefficient
    L : Sample thickness
    ALPHA0 : Linear absorption coefficient
    I0 : Beam intensity at the focal point
    Z_R : Rayleigh length
    MODEL_PARAMETERS : List of parameters that determine basinhopping conditions
        MAX_PERTURBATION : highest multiplication factor
        MAX_AGE : stop-condition : maximum age of current optimal point
        MAX_ITER: stop-condition : maxumum number of iterations
        BOUNDS : bounds on the fitting parameters
        T : basin hopping temperature
        MAX_JUMP : number of consecutive jumps allowed
        MAX_REJECT : number of rejects before jumping
    """
    ## Basin hopping parameters
    MAX_PERTURBATION, MAX_AGE, MAX_ITER, BOUNDS, T, MAX_JUMP, MAX_REJECT = MODEL_PARAMETERS
    
    fitting_model = lambda x: X2.TPA_no_sat(data, x[0], x[1], L, ALPHA0, I0, Z_R)
    
    ## Set initial chi2
    popt = minimize(fitting_model, x0=p0, bounds=BOUNDS)
    pBest = pMin = popt.x
    chi2Prev = chi2Best = fitting_model(pMin)
    
    ## Start minimalisation process
    ### prepare iteration process
    niter = 0
    rejection = 0
    jump = 0
    bestAge = -1

    ### Iteration process
    while niter < MAX_ITER and bestAge < MAX_AGE:
        niter += 1
        bestAge += 1
        
        #### Monte Carlo Move
        pPerturbation = list(pMin)
        for i in [1,2]:
            perturbation = np.random.uniform(1, MAX_PERTURBATION)    # Determine perturbation strength
            direction = np.random.choice(['decrease', 'increase'])    # Determine perturbation direction
            if direction == 'decrease':
                pPerturbation[i] = pMin[i] / perturbation
            else:
                pPerturbation[i] = pMin[i] * perturbation

        #### Minimise Chi2
        popt = minimize(fitting_model, x0=pPerturbation, bounds=BOUNDS)
        p_newMin = popt.x
        chi2 = fitting_model(p_newMin)
        
        #### Metropolis criterion
        ##### Accept perturbation
        ###### Jumping -> accept perturbation regardless of conditions
        if jump != 0:
            chi2Prev = chi2
            pMin = p_newMin
            jump += 1
            ## Check if result is overall best
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
            ## End condition for jumping
            if jump == MAX_JUMP:    
                jump = 0
        ###### Better chi2 than previous chi2
        elif chi2 < chi2Prev:    
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            ## Check if result is overall best            
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
        ###### Metropolis
        elif np.random.uniform(0,1) < np.exp(-(chi2 - chi2Prev)/T):
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0

        ##### Reject perturbation
        else:
            rejection += 1

        ##### Max Rejections achieved? -> Start jumping
        if rejection == MAX_REJECT:
            # Accept perturbation
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            jump += 1
    
    return pBest, chi2Best
##################################################################
def TPA_no_Is2(data: np.dtype, p0: np.dtype, L: float, ALPHA0: float, I0: float, Z_R: float, MODEL_PARAMETERS: list):
    """Returns optimal fitting parameters and optimal chi-squared following a Monte-Carlo minimisation algorithm
        for the 2PA model without Is2
    
    PARAMETERS
    data : Data structure, of size (N, 3) where N is the number of datapoints.
        [:,0]: z-data - [:, 1]: intensity-data - [:, 2]: uncertainty on intensity
    p0 : Initial guess
        [0]: z-position of focal point - [1]: first-order saturation intensity
    L : Sample thickness
    ALPHA0 : Linear absorption coefficient
    I0 : Beam intensity at the focal point
    Z_R : Rayleigh length
    MODEL_PARAMETERS : List of parameters that determine basinhopping conditions
        MAX_PERTURBATION : highest multiplication factor
        MAX_AGE : stop-condition : maximum age of current optimal point
        MAX_ITER: stop-condition : maxumum number of iterations
        BOUNDS : bounds on the fitting parameters
        T : basin hopping temperature
        MAX_JUMP : number of consecutive jumps allowed
        MAX_REJECT : number of rejects before jumping
    """
    ## Basin hopping parameters
    MAX_PERTURBATION, MAX_AGE, MAX_ITER, BOUNDS, T, MAX_JUMP, MAX_REJECT = MODEL_PARAMETERS
    
    fitting_model = lambda x: X2.TPA_no_Is2(data, x[0], x[1], L, ALPHA0, I0, Z_R)
    
    ## Set initial chi2
    popt = minimize(fitting_model, x0=p0, bounds=BOUNDS)
    pBest = pMin = popt.x
    chi2Prev = chi2Best = fitting_model(pMin)
    
    ## Start minimalisation process
    ### prepare iteration process
    niter = 0
    rejection = 0
    jump = 0
    bestAge = -1

    ### Iteration process
    while niter < MAX_ITER and bestAge < MAX_AGE:
        niter += 1
        bestAge += 1
        
        #### Monte Carlo Move
        pPerturbation = list(pMin)
        for i in [1,2]:
            perturbation = np.random.uniform(1, MAX_PERTURBATION)    # Determine perturbation strength
            direction = np.random.choice(['decrease', 'increase'])    # Determine perturbation direction
            if direction == 'decrease':
                pPerturbation[i] = pMin[i] / perturbation
            else:
                pPerturbation[i] = pMin[i] * perturbation

        #### Minimise Chi2
        popt = minimize(fitting_model, x0=pPerturbation, bounds=BOUNDS)
        p_newMin = popt.x
        chi2 = fitting_model(p_newMin)
        
        #### Metropolis criterion
        ##### Accept perturbation
        ###### Jumping -> accept perturbation regardless of conditions
        if jump != 0:
            chi2Prev = chi2
            pMin = p_newMin
            jump += 1
            ## Check if result is overall best
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
            ## End condition for jumping
            if jump == MAX_JUMP:    
                jump = 0
        ###### Better chi2 than previous chi2
        elif chi2 < chi2Prev:    
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            ## Check if result is overall best            
            if chi2 < chi2Best:    
                pBest = p_newMin
                chi2Best = chi2
                bestAge = 0
        ###### Metropolis
        elif np.random.uniform(0,1) < np.exp(-(chi2 - chi2Prev)/T):
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0

        ##### Reject perturbation
        else:
            rejection += 1

        ##### Max Rejections achieved? -> Start jumping
        if rejection == MAX_REJECT:
            # Accept perturbation
            chi2Prev = chi2
            pMin = p_newMin
            rejection = 0
            jump += 1
    
    return pBest, chi2Best