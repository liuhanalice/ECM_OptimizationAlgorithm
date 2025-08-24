import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import matplotlib.pyplot as plt
from functools import partial
from ECM_impedance_v3 import *


# Initial Guess Mapping
INITIAL_GUESS = {
    "v3CM1": [0.005, 0.01, 0.05, 0.9], # v3CM1: R0, R1, C1, n1
    "v3CM2":[0.005, 0.01, 0.01, 0.05, 0.9, 0.1, 0.9], # v3CM2: R0, R1, R2, C1, n1, C2, n2
    "v3CM3":[0.005, 0.01, 0.01, 0.01, 0.05, 0.9, 0.1, 0.9, 0.5, 0.9], # v3CM3: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3
    "v3CM4":[0.005, 0.01, 0.01, 0.01, 0.01, 0.05, 0.9, 0.1, 0.9, 0.5, 0.9, 0.5, 0.9], # v3CM4: R0, R1, R2, R3, R4, C1, n1, C2, n2, C3, n3, C4, n4
    "v3CM5":[0.005, 0.01, 0.05, 0.9, 0.001], # v3CM5: R0, R1, C1, n1, Aw
    "v3CM6":[0.005, 0.01, 0.01, 0.05, 0.9, 0.1, 0.9, 0.001], # v3CM6: R0, R1, R2, C1, n1, C2, n2, Aw
    "v3CM7":[0.005, 0.005, 0.01, 0.01, 0.05, 0.9, 0.1, 0.9, 0.005, 0.9, 0.001], # v3CM7: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3, Aw
    "v3CM8":[0.005, 0.005, 0.01, 0.01, 0.01, 0.05, 0.9, 0.1, 0.9, 0.5, 0.9, 0.005, 0.9, 0.001], # v3CM8: R0, R1, R2, R3, R4, C1, n1, C2, n2, C3, n3, C4, n4, Aw
    "v3CM9":[0.005, 0.005, 0.01, 0.01, 0.05, 0.9, 0.1, 0.9, 0.5, 0.9, 0.001], # v3CM9: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3, Aw
}

# predefined ECM Parameter Names Mapping
PARAMS_NAMES = {
    "v3CM1": ["R0", "R1", "C1", "n1"], # v3CM1: R0, R1, C1, n1
    "v3CM2": ["R0", "R1", "R2", "C1", "n1", "C2", "n2"], # v3CM2: R0, R1, R2, C1, n1, C2, n2
    "v3CM3": ["R0", "R1", "R2", "R3", "C1", "n1", "C2", "n2", "C3", "n3"], # v3CM3: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3
    "v3CM4": ["R0", "R1", "R2", "R3", "R4", "C1", "n1", "C2", "n2", "C3", "n3", "C4", "n4"], # v3CM4: R0, R1, R2, R3, R4, C1, n1, C2, n2, C3, n3, C4, n4
    "v3CM5": ["R0", "R1", "C1", "n1", "Aw"], # v3CM5: R0, R1, C1, n1, Aw
    "v3CM6": ["R0", "R1", "R2", "C1", "n1", "C2", "n2", "Aw"], # v3CM6: R0, R1, R2, C1, n1, C2, n2, Aw
    "v3CM7": ["R0", "R1", "R2", "R3", "C1", "n1", "C2", "n2", "C3", "n3", "Aw"], # v3CM7: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3, Aw
    "v3CM8": ["R0", "R1", "R2", "R3", "R4", "C1", "n1", "C2", "n2", "C3", "n3", "C4", "n4", "Aw"], # v3CM8: R0, R1, R2, R3, R4, C1, n1, C2, n2, C3, n3, C4, n4, Aw
    "v3CM9": ["R0", "R1", "R2", "R3", "C1", "n1", "C2", "n2", "C3", "n3", "Aw"], # v3CM9: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3, Aw
}

# Bounds Mapping
eps = 1e-7
BOUNDS = {
    "v3CM1": [(eps, 100), (eps, 100), (eps, 100), (0.8, 1)], # v3CM1: R0, R1, C1, n1
    "v3CM2": [(eps, 100), (eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1)], # v3CM2: R0, R1, R2, C1, n1, n2, C2
    "v3CM3": [(eps, 100), (eps, 100), (eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1)], # v3CM3: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3
    "v3CM4": [(eps, 100), (eps, 100), (eps, 100), (eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1)], # v3CM4: R0, R1, R2, R3, R4, C1, n1, C2, n2, C3, n3, C4, n4
    "v3CM5": [(eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100)], # v3CM5: R0, R1, C1, n1, Aw
    "v3CM6": [(eps, 100), (eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100)], # v3CM6: R0, R1, R2, C1, n1, C2, n2, Aw
    "v3CM7": [(eps, 100), (eps, 100), (eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100)], # v3CM7: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3, Aw
    "v3CM8": [(eps, 100), (eps, 100), (eps, 100), (eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100)], # v3CM8: R0, R1, R2, R3, R4, C1, n1, C2, n2, C3, n3, C4, n4, Aw
    "v3CM9": [(eps, 100), (eps, 100), (eps, 100), (eps, 100), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100), (0.8, 1), (eps, 100)], # v3CM9: R0, R1, R2, R3, C1, n1, C2, n2, C3, n3, Aw
}

ECM_NAMES = ["v3CM1","v3CM2","v3CM3","v3CM4","v3CM5","v3CM6", "v3CM7", "v3CM8", "v3CM9"]
ECM_IMPEDANCE_FUNCS = [compute_v3CM1_impedance, compute_v3CM2_impedance, compute_v3CM3_impedance, compute_v3CM4_impedance,
                       compute_v3CM5_impedance, compute_v3CM6_impedance, compute_v3CM7_impedance, compute_v3CM8_impedance, compute_v3CM9_impedance,]

ECM_NUM_RCS = {
    "v3CM1": 1,
    "v3CM2": 2,
    "v3CM3": 3,
    "v3CM4": 4,
    "v3CM5": 0,
    "v3CM6": 1,
    "v3CM7": 2,
    "v3CM8": 3,
    "v3CM9": 3
}


# Define Nonlinear Constrained Optimization (TC) ECM estimation
def time_constant_constraints(params, ECM_name):
    """
    Define nonlinear constraints for time constant ordering.
    This function returns a list of constraints that ensure the time constants of RC components are in ascending order.
    """
    if ECM_name == "v3CM2":
        R0_val, R1_val, R2_val, C1_val, n1_val, C2_val, n2_val = params
        return [compute_time_constant(R2_val, C2_val, n2_val) - compute_time_constant(R1_val, C1_val, n1_val)]
    
    elif ECM_name == "v3CM3":
        R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val = params
        return [compute_time_constant(R2_val, C2_val, n2_val) - compute_time_constant(R1_val, C1_val, n1_val),
                compute_time_constant(R3_val, C3_val, n3_val) - compute_time_constant(R2_val, C2_val, n2_val)]
    
    elif ECM_name == "v3CM4":
        R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val = params
        return [compute_time_constant(R2_val, C2_val, n2_val) - compute_time_constant(R1_val, C1_val, n1_val),
                compute_time_constant(R3_val, C3_val, n3_val) - compute_time_constant(R2_val, C2_val, n2_val),
                compute_time_constant(R4_val, C4_val, n4_val) - compute_time_constant(R3_val, C3_val, n3_val)]
    
    elif ECM_name == "v3CM7":
        R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, sigma_val = params
        return [compute_time_constant(R2_val, C2_val, n2_val) - compute_time_constant(R1_val, C1_val, n1_val)]
    
    elif ECM_name == "v3CM8":
        R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val, sigma_val = params
        return [compute_time_constant(R2_val, C2_val, n2_val) - compute_time_constant(R1_val, C1_val, n1_val),
                compute_time_constant(R3_val, C3_val, n3_val) - compute_time_constant(R2_val, C2_val, n2_val)]
    #TODO: CM9
    elif ECM_name not in ECM_NAMES:
        raise ValueError(f"Unknown ECM name: {ECM_name}. Cannot define time constant constraints.")
    else:
        return [] # No constraints



### Pre-define cost functions
# Default: Sum of Squares
def cost_RMSE_abs(params, Z_exp, angular_freq, impedance_func):
    Z_model = impedance_func(params, angular_freq)
    error = np.abs(Z_model - Z_exp)
    rmse_abs =  np.sqrt(np.mean(error **2))
    
    max_Zexp = np.max(np.abs(Z_exp)) # max Z magnitude
    normalized_rmse_abs = rmse_abs / max_Zexp
    return normalized_rmse_abs

# def cost_RMSE_abs(params, Z_exp, angular_freq, impedance_func):
#     Z_model = impedance_func(params, angular_freq)
#     error = np.abs(Z_model - Z_exp)
#     rmse_abs =  np.sqrt(np.mean(error **2))
    
#     max_Zexp = np.max(np.abs(Z_exp)) # max Z magnitude
#     normalized_rmse_abs = rmse_abs / max_Zexp 

#     # regularization (#C8 only)
#     # R0, R1, R2, R3, R4, C1, n1, C2, n2, C3, n3, C4, n4, Aw
#     tau1 = compute_time_constant(params[1], params[5], params[6])
#     tau2 = compute_time_constant(params[2], params[7], params[8])
#     tau3 = compute_time_constant(params[3], params[9], params[10])

#     # Regularization to enforce tau1 < tau2 < tau3
#     penalty12 = np.maximum(0, tau1 - tau2)
#     penalty23 = np.maximum(0, tau2 - tau3)

#     return normalized_rmse_abs + 0 * (penalty12 + penalty23)

 
# Normalized mean
def cost_RMSE_rel(params, Z_exp, angular_freq, impedance_func, epsilon=1e-8):
    Z_model = impedance_func(params, angular_freq)
    abs_error = np.abs(Z_model - Z_exp)
    denom = np.abs(Z_exp)
    denom[denom < epsilon] = epsilon # avoid division by near-zero
    rel_error = abs_error / denom
    rmsre = np.sqrt(np.mean(rel_error**2))

    max_Zexp = np.max(np.abs(Z_exp)) # max Z magnitude
    normalized_rmsre_rel = rmsre / max_Zexp
    return normalized_rmsre_rel


def cost_R2_flatten(params, Z_exp, angular_freq, impedance_func):
    # Z is complex number, so flatten it into real and imaginary part
    Z_model = impedance_func(params, angular_freq)
    # Flatten real and imaginary parts into one vector
    Z_exp_combined = np.concatenate([Z_exp.real, Z_exp.imag])
    Z_model_combined = np.concatenate([Z_model.real, Z_model.imag])

    rss = np.sum((Z_exp_combined - Z_model_combined) ** 2)
    exp_mean = np.mean(Z_exp_combined)
    tss = np.sum((Z_exp_combined - exp_mean) ** 2)
    return 1 - (rss / tss)


def cost_R2_magnitude(params, Z_exp, angular_freq, impedance_func):
    # Compute model impedance
    Z_model = impedance_func(params, angular_freq)
    
    # Magnitudes
    Z_exp_mag = np.abs(Z_exp)
    Z_model_mag = np.abs(Z_model)
    
    # R2 calculation
    rss = np.sum((Z_exp_mag - Z_model_mag) ** 2)
    exp_mean = np.mean(Z_exp_mag)
    tss = np.sum((Z_exp_mag - exp_mean) ** 2)
    
    return 1 - (rss / tss)


COST_FUNCTION_MAP = {
    "RMSE_abs": cost_RMSE_abs,
    "RMSE_rel": cost_RMSE_rel,
    "R2_flatten": cost_R2_flatten,
    "R2_magnitude": cost_R2_magnitude,
}

def compute_time_constant(R, Q, n):
    try:
        if R <= 0 or Q <= 0 or n <= 0:
            return np.inf
        return (R * Q) ** (1 / n)
    except (ZeroDivisionError, FloatingPointError):
        return np.inf


# Sort parameter estimation by time constant such that rc1 <= rc2 <= rc3 <=...
# NOTE: CPE's time constant should take sqrt of n
def sort_by_tau(params, ECM_name):
    """
    Sort parameters by their time constant. 
    Given ECM_name and its corresponding parameters, sort the parameters based on their time constants in ascending order.
    """
    if ECM_name == "v3CM2":
        R0_val, R1_val, R2_val, C1_val, n1_val, C2_val, n2_val = params
        if compute_time_constant(R1_val, C1_val, n1_val) > compute_time_constant(R2_val, C2_val, n2_val):
            temp = R1_val
            R1_val = R2_val
            R2_val = temp

            temp = n1_val
            n1_val = n2_val
            n2_val = temp

            temp = C1_val
            C1_val = C2_val
            C2_val = temp
        params = [R0_val, R1_val, R2_val, C1_val, n1_val, C2_val, n2_val]
        params = [float(v) for v in params]
    
    elif ECM_name == "v3CM3":
        R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val = params

        RC_products = [compute_time_constant(R1_val, C1_val, n1_val), compute_time_constant(R2_val, C2_val, n2_val), compute_time_constant(R3_val, C3_val, n3_val)]
        sorted_indices = sorted(range(3), key=lambda i: RC_products[i])

        R_vals = [R1_val, R2_val, R3_val]
        C_vals = [C1_val, C2_val, C3_val]
        n_vals = [n1_val, n2_val, n3_val]

        sorted_R = [R_vals[i] for i in sorted_indices]
        sorted_C = [C_vals[i] for i in sorted_indices]
        sorted_n = [n_vals[i] for i in sorted_indices]

        R1_val, R2_val, R3_val = sorted_R
        C1_val, C2_val, C3_val = sorted_C
        n1_val, n2_val, n3_val = sorted_n

        params = [R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val]
        params = [float(v) for v in params]
    
    elif ECM_name == "v3CM4":
        R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val = params

        RC_products = [compute_time_constant(R1_val, C1_val, n1_val), compute_time_constant(R2_val, C2_val, n2_val), compute_time_constant(R3_val, C3_val, n3_val), compute_time_constant(R4_val, C4_val, n4_val)]
        sorted_indices = sorted(range(4), key=lambda i: RC_products[i])

        R_vals = [R1_val, R2_val, R3_val, R4_val]
        C_vals = [C1_val, C2_val, C3_val, C4_val]
        n_vals = [n1_val, n2_val, n3_val, n4_val]

        sorted_R = [R_vals[i] for i in sorted_indices]
        sorted_C = [C_vals[i] for i in sorted_indices]
        sorted_n = [n_vals[i] for i in sorted_indices]

        R1_val, R2_val, R3_val, R4_val = sorted_R
        C1_val, C2_val, C3_val, C4_val = sorted_C
        n1_val, n2_val, n3_val, n4_val = sorted_n

        params = [R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val]
        params = [float(v) for v in params]
    

    elif ECM_name == "v3CM7":
        R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, sigma_val = params
        if compute_time_constant(R1_val, C1_val, n1_val) > compute_time_constant(R2_val, C2_val, n2_val):
            temp = R1_val
            R1_val = R2_val
            R2_val = temp

            temp = n1_val
            n1_val = n2_val
            n2_val = temp

            temp = C1_val
            C1_val = C2_val
            C2_val = temp
        params = [R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, sigma_val]
        params = [float(v) for v in params]

    elif ECM_name == "v3CM8":
        R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val, sigma_val = params

        RC_products = [compute_time_constant(R1_val, C1_val, n1_val), compute_time_constant(R2_val, C2_val, n2_val), compute_time_constant(R3_val, C3_val, n3_val)]
        sorted_indices = sorted(range(3), key=lambda i: RC_products[i])

        R_vals = [R1_val, R2_val, R3_val]
        C_vals = [C1_val, C2_val, C3_val]
        n_vals = [n1_val, n2_val, n3_val]

        sorted_R = [R_vals[i] for i in sorted_indices]
        sorted_C = [C_vals[i] for i in sorted_indices]
        sorted_n = [n_vals[i] for i in sorted_indices]

        R1_val, R2_val, R3_val = sorted_R
        C1_val, C2_val, C3_val = sorted_C
        n1_val, n2_val, n3_val = sorted_n

        params = [R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val, sigma_val]
        params = [float(v) for v in params]
    
    elif ECM_name == "v3CM9":
        R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, sigma_val = params

        RC_products = [compute_time_constant(R1_val, C1_val, n1_val), compute_time_constant(R2_val, C2_val, n2_val), compute_time_constant(R3_val, C3_val, n3_val)]
        sorted_indices = sorted(range(3), key=lambda i: RC_products[i])

        R_vals = [R1_val, R2_val, R3_val]
        C_vals = [C1_val, C2_val, C3_val]
        n_vals = [n1_val, n2_val, n3_val]

        sorted_R = [R_vals[i] for i in sorted_indices]
        sorted_C = [C_vals[i] for i in sorted_indices]
        sorted_n = [n_vals[i] for i in sorted_indices]

        R1_val, R2_val, R3_val = sorted_R
        C1_val, C2_val, C3_val = sorted_C
        n1_val, n2_val, n3_val = sorted_n

        params = [R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, sigma_val]
        params = [float(v) for v in params]
    
    return params
    

def Powell_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=None, cost_func_name = None, optimizer_options=None, verbose=True):
    # Suppose EIS data has m points, then Z_exp and initial_guess should be array of size m
    # ECM_name is the name of ECM use to fit EIS, call impedance_func to calculate the impedance given (parameters, angular_freq); 
    #       initial_guess is the initial guess of parameters in ECM (should match length and order)

    if optimizer_options is None:
        optimizer_options = {
            'maxiter': 1000,
            'xtol': 1e-8, # parameter change tolerance
            'ftol': 1e-8,
            'disp': False,
            'return_all': False
        }

    # Select Cost Function
    if cost_func_name in COST_FUNCTION_MAP:
        # print(f"Using Cost Func: {cost_func_name}")
        raw_cost_func = COST_FUNCTION_MAP[cost_func_name]
        cost_function = partial(raw_cost_func, Z_exp=Z_exp, angular_freq=angular_freq, impedance_func=impedance_func)
    else:
        if cost_func_name is not None:
            print(f"Warning: Unknown cost function '{cost_func_name}', using 'Root Mean Sum of Squares' as default.")
        cost_function = partial(cost_RMSE_abs, Z_exp=Z_exp, angular_freq=angular_freq, impedance_func=impedance_func)
    
    if verbose:
        print(f"Estimating input EIS with ECM {ECM_name}")
        print("bounds", bounds)
    # Perform optimization
    result = minimize(
        cost_function,
        initial_guess,
        method='Powell',
        options=optimizer_options,
        bounds = bounds # Note: the bounds for Powell is soft bounds?
    )

    fitted_params = result.x
    sorted_params = sort_by_tau(fitted_params, ECM_name)
    if verbose:
        print("Optimization success:", result.success)
        print("Estimated parameter values (sorted):", sorted_params)
        print("Final cost:", result.fun)
    if not result.success:
        print("Powell Optimization not success:")
        print(result)
    return sorted_params, result.fun, result


def Bounded_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=None, cost_func_name = None, optimizer_options=None, verbose=True):
    if optimizer_options is None:
        optimizer_options = {
            'maxiter': 3000,
            'ftol': 1e-8,
            'eps': 1e-8, # step size
            'disp': False,
        }

   # Select Cost Function
    if cost_func_name in COST_FUNCTION_MAP:
        # print(f"Using Cost Func: {cost_func_name}")
        raw_cost_func = COST_FUNCTION_MAP[cost_func_name]
        cost_function = partial(raw_cost_func, Z_exp=Z_exp, angular_freq=angular_freq, impedance_func=impedance_func)
    else:
        if cost_func_name is not None:
            print(f"Warning: Unknown cost function '{cost_func_name}', using 'Root Mean Sum of Squares' as default.")
        cost_function = partial(cost_RMSE_abs, Z_exp=Z_exp, angular_freq=angular_freq, impedance_func=impedance_func)

    # Default Bounds: non-negative for each parameter
    if bounds is None:
        bounds = [(eps, None)] * len(initial_guess) # epsilon = 1e-5 to avoid divide by zero when calculating impedance

    if verbose:
        print(f"Estimating input EIS with ECM {ECM_name} (bounded, non-negative params)")

    result = minimize(
        cost_function,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options=optimizer_options
    )

    fitted_params = result.x
    sorted_params = sort_by_tau(fitted_params, ECM_name)
    if verbose:
        print("Optimization success:", result.success)
        print("Estimated parameter values (sorted):", sorted_params)
        print("Final cost:", result.fun)
    if not result.success:
        print("L-BFGS-B Optimization not success:")
        print(result)
    return sorted_params, result.fun, result


# Define Nonlinear Constrained Optimization (TC) ECM estimation


def TC_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=None, cost_func_name = None, optimizer_options=None, verbose=True):
    """
    ECM estimation using Scipy-"trust-constr" to add ordering contraints on time constants.
    This function is used to estimate ECM parameters
    """
    if optimizer_options is None:
        optimizer_options = {
            'maxiter': 3000,
            'disp': False,
            'finite_diff_rel_step': 1e-8  # More stable than default for small-valued params
        }

    # Select Cost Function
    if cost_func_name in COST_FUNCTION_MAP:
        raw_cost_func = COST_FUNCTION_MAP[cost_func_name]
        cost_function = partial(raw_cost_func, Z_exp=Z_exp, angular_freq=angular_freq, impedance_func=impedance_func)
    else:
        if cost_func_name is not None:
            print(f"Warning: Unknown cost function '{cost_func_name}', using 'Root Mean Sum of Squares' as default.")
        cost_function = partial(cost_RMSE_abs, Z_exp=Z_exp, angular_freq=angular_freq, impedance_func=impedance_func)

    if verbose:
        print(f"Estimating input EIS with ECM {ECM_name} (trust-constr)")
    
    # Define Nonlinear Constraints for time constant ordering
    constraint_func = lambda params: time_constant_constraints(params, ECM_name)
    num_constraints = len(constraint_func(initial_guess))  # Number of constraints

    constraint_vals = constraint_func(initial_guess)
    if len(constraint_vals) > 0:
        nonlinear_constraints = NonlinearConstraint(
            fun=constraint_func,
            lb=[1e-7] * len(constraint_vals),
            ub=[np.inf] * len(constraint_vals)
        )
        constraints = [nonlinear_constraints]
    else:
        constraints = []
    

    result = minimize(
        fun=cost_function,
        x0=initial_guess,
        method='trust-constr',
        bounds=bounds,
        constraints=constraints,
        options=optimizer_options
    )

    fitted_params = result.x
    # sorted_params = sort_by_tau(fitted_params, ECM_name)
    if verbose:
        print("Optimization success:", result.success)
        print("Estimated parameter values (sorted):", fitted_params)
        print("Final cost:", result.fun)
    
    if not result.success:
        print("Trust-Constr Optimization not success:")
        print(result)
    
    return fitted_params, result.fun, result



# Use either Powell or BFGS algorithm, default is Powell.
# Find best fitting among all ECM_candidates
def ECM_Search_and_Estimate(Z_exp, angular_freq, ECM_cadidates_names, ECM_candidates_impedance_funcs, ECM_initial_guesses, ECM_bounds=None, cost_func_name = None, input_config="input battery", visualization=True, verbose=True, algo="Powell"):
    res_ECM = None
    res_fitted_params = None
    res_fitted_cost = None
    
    for i in range(len(ECM_cadidates_names)):
        ECM_name = ECM_cadidates_names[i]
        impedance_func = ECM_candidates_impedance_funcs[i]
        initial_guess = ECM_initial_guesses[i]
        bounds = ECM_bounds[i]

        if algo == "Powell":
            fitted_params, cost, _ = Powell_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=verbose)
        elif algo == "BFGS":
            fitted_params, cost, _ = Bounded_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=verbose)
        
        # visualize
        if visualization:
            print(f"Reconstruction for ECM {ECM_name}")
            Z_fit = impedance_func(fitted_params, angular_freq)
            plt.figure(figsize=(8, 6))
            plt.plot(np.real(Z_exp), -np.imag(Z_exp), 'bo-', label="Experimental Data")
            if algo == "Powell":
                plt.plot(np.real(Z_fit), -np.imag(Z_fit), 'ro-', label=f"{ECM_name} Powell Reconstructed (Fitted Model)")
            elif algo == "BFGS":
                plt.plot(np.real(Z_fit), -np.imag(Z_fit), 'go-', label=f"{ECM_name} L-BFGS-B Reconstructed (Fitted Model)")
            plt.xlabel("Real Z (Ohms)")
            plt.ylabel("-Imaginary Z (Ohms)")
            plt.title(f"Nyquist Plot: {input_config}")
            plt.legend()
            plt.grid(True)

        if(res_fitted_cost is None or res_fitted_cost > cost):
            res_fitted_cost = cost
            res_fitted_params = fitted_params
            res_ECM = ECM_name
        
        if verbose:
            print(" ------------ ")

    
    print(f"--- Searched all predefined ECMS, find best fitting ECM: {res_ECM}, estimated parameter values: {res_fitted_params}, cost (sum of squares): {res_fitted_cost}.")
    return res_ECM, res_fitted_params, res_fitted_cost


# Version 2: compare Powell and L-BFGS-B algorithm and find the best one
# visualize the best-fitting plot if visualization = true
def ECM_Search_and_Estimate_v2(Z_exp, angular_freq, ECM_candidates_names, ECM_candidates_impedance_funcs, ECM_initial_guesses, ECM_bounds=None, cost_func_name = None, input_config="input battery", visualization=True, verbose=True):
    res_ECM = None
    res_fitted_params = None
    res_fitted_cost = None

    powell_res_ECM = None
    powell_res_fitted_params = None
    powell_res_fitted_cost = None
    powell_i = None
    
    BFGS_res_ECM = None
    BFGS_res_fitted_params = None
    BFGS_res_fitted_cost = None
    BFGS_i = None
    
    for i in range(len(ECM_candidates_names)):
        ECM_name = ECM_candidates_names[i]
        impedance_func = ECM_candidates_impedance_funcs[i]
        initial_guess = ECM_initial_guesses[i]
        bounds = ECM_bounds[i]

        powell_fitted_params, powell_cost, _ = Powell_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=verbose)
        BFGS_fitted_params, BFGS_cost, _ = Bounded_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=verbose)
        
        
        if(powell_res_fitted_cost is None or powell_res_fitted_cost > powell_cost):
            powell_res_fitted_cost = powell_cost
            powell_res_fitted_params = powell_fitted_params
            powell_res_ECM = ECM_name
            powell_i = i
        
        if(BFGS_res_fitted_cost is None or BFGS_res_fitted_cost > BFGS_cost):
            BFGS_res_fitted_cost = BFGS_cost
            BFGS_res_fitted_params = BFGS_fitted_params
            BFGS_res_ECM = ECM_name
            BFGS_i = i
        
        if verbose:
            print(" ------------ ")

    # Decide which algo to use
    if (powell_res_fitted_cost > BFGS_res_fitted_cost) or any(p < 0 for p in powell_res_fitted_params):
        # incorrect for one case
        if(powell_res_fitted_cost < BFGS_res_fitted_cost and any(p < 0 for p in powell_res_fitted_params)):
            print("Powell better but gives negative estimation, do not use it")
        res_fitted_cost = BFGS_res_fitted_cost
        res_fitted_params = BFGS_res_fitted_params
        res_ECM = BFGS_res_ECM
        print("L-BFGS-B wins, use L-BFGS-B fitting result")
    elif powell_res_fitted_cost < BFGS_res_fitted_cost:
        res_fitted_cost = powell_res_fitted_cost
        res_fitted_params = powell_res_fitted_params
        res_ECM = powell_res_ECM
        print("Powell wins, use Powell fitting result")
    else: 
        res_fitted_cost = powell_res_fitted_cost
        res_fitted_params = powell_res_fitted_params
        res_ECM = powell_res_ECM
        print("Ties, use Powell fitting result")


    # Visualize
    if visualization:
        print(f"Visualizing Reconstruction")
        Z_fit_powell = ECM_candidates_impedance_funcs[powell_i](powell_res_fitted_params, angular_freq)
        Z_fit_BFGS = ECM_candidates_impedance_funcs[BFGS_i](BFGS_res_fitted_params, angular_freq)

        plt.figure(figsize=(8, 6))
        plt.plot(np.real(Z_exp), -np.imag(Z_exp), 'bo-', label="Experimental Data")
        plt.plot(np.real(Z_fit_powell), -np.imag(Z_fit_powell), 'ro-', label=f"{powell_res_ECM} Powell Reconstructed (Fitted Model)")
        plt.plot(np.real(Z_fit_BFGS), -np.imag(Z_fit_BFGS), 'go-', label=f"{BFGS_res_ECM} L-BFGS-B Reconstructed (Fitted Model)")

        plt.xlabel("Real Z (Ohms)")
        plt.ylabel("-Imaginary Z (Ohms)")
        plt.title(f"Nyquist Plot: {input_config}")
        plt.legend()
        plt.grid(True)

    print(f"--- Searched all predefined ECMS, find best fitting ECM: {res_ECM}, estimated parameter values: {res_fitted_params}, cost (sum of squares): {res_fitted_cost}.")
    return res_ECM, res_fitted_params, res_fitted_cost



# Version 3: compare Powell, L-BFGS-B and Trust-Constr algorithm and find the best one; support list of ECM_candidates and their bounds
# visualize the best-fitting plot if visualization = true
def ECM_Search_and_Estimate_v3(Z_exp, angular_freq, ECM_candidates_names, ECM_candidates_impedance_funcs, ECM_initial_guesses, ECM_bounds=None, cost_func_name=None, input_config="input battery", visualization=True, verbose=True):
    res_ECM, res_fitted_params, res_fitted_cost = None, None, None

    # Store best results for each algorithm
    powell_best = {"cost": None}
    bfgs_best = {"cost": None}
    trust_best = {"cost": None}


    for i in range(len(ECM_candidates_names)):
        ECM_name = ECM_candidates_names[i]
        impedance_func = ECM_candidates_impedance_funcs[i]
        
        if ECM_initial_guesses is None:
            initial_guess = INITIAL_GUESS[ECM_name]
        else:
            initial_guess = ECM_initial_guesses[i]
        
        if ECM_bounds is None:
            bounds = BOUNDS[ECM_name]
        else:
            bounds = ECM_bounds[i]

        # Powell
        powell_params, powell_cost, _ = Powell_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name=cost_func_name, verbose=verbose)
        if powell_best["cost"] is None or powell_cost < powell_best["cost"]:
            powell_best.update({"ECM": ECM_name, "params": powell_params, "cost": powell_cost, "i": i})

        # L-BFGS-B
        bfgs_params, bfgs_cost, _ = Bounded_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name=cost_func_name, verbose=verbose)
        if bfgs_best["cost"] is None or bfgs_cost < bfgs_best["cost"]:
            bfgs_best.update({"ECM": ECM_name, "params": bfgs_params, "cost": bfgs_cost, "i": i})

        # Trust-Constr (with TC ordering)
        trust_params, trust_cost, _ = TC_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name=cost_func_name, verbose=verbose)
        if trust_best["cost"] is None or trust_cost < trust_best["cost"]:
            trust_best.update({"ECM": ECM_name, "params": trust_params, "cost": trust_cost, "i": i})

        if verbose:
            print(" ------------ ")

    # Decide best method overall
    candidates = [powell_best, bfgs_best, trust_best]
    valid_candidates = [c for c in candidates if c["params"] is not None and all(p >= 0 for p in c["params"])]
    best = min(valid_candidates, key=lambda x: x["cost"]) if valid_candidates else None

    if best is not None:
        res_ECM, res_fitted_params, res_fitted_cost = best["ECM"], best["params"], best["cost"]
        print(f"{res_ECM} wins, use {res_ECM} ({'Trust-Constr' if best == trust_best else 'L-BFGS-B' if best == bfgs_best else 'Powell'}) fitting result")
    else:
        print("No valid fitting result found.")
        return None, None, None

    # Visualization
    if visualization:
        print(f"Visualizing Reconstruction")

        Z_fit_powell = ECM_candidates_impedance_funcs[powell_best["i"]](powell_best["params"], angular_freq)
        Z_fit_BFGS = ECM_candidates_impedance_funcs[bfgs_best["i"]](bfgs_best["params"], angular_freq)
        Z_fit_trust = ECM_candidates_impedance_funcs[trust_best["i"]](trust_best["params"], angular_freq)

        plt.figure(figsize=(8, 6))
        plt.plot(np.real(Z_exp), -np.imag(Z_exp), 'bo-', label="Experimental Data")
        plt.plot(np.real(Z_fit_powell), -np.imag(Z_fit_powell), 'ro-', label=f"{powell_best['ECM']} Powell")
        plt.plot(np.real(Z_fit_BFGS), -np.imag(Z_fit_BFGS), 'go-', label=f"{bfgs_best['ECM']} L-BFGS-B")
        plt.plot(np.real(Z_fit_trust), -np.imag(Z_fit_trust), 'mo-', label=f"{trust_best['ECM']} Trust-Constr")

        plt.xlabel("Real Z (Ohms)")
        plt.ylabel("-Imaginary Z (Ohms)")
        plt.title(f"Nyquist Plot: {input_config}")
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"--- Searched all predefined ECMs. Best ECM: {res_ECM}, Estimated parameters: {res_fitted_params}, Cost: {res_fitted_cost}.")
    return res_ECM, res_fitted_params, res_fitted_cost



# Given input EIS/impedance Z_exp and one ECM_candidate, return parameter estimation value for one ECM_candidate.
# the return parameter estimation is given by the best result of Powell or L-BFGS-B
# hardcode ECM6 time_constant sort that RC1 , RC2
def ECM_result_single_candidate(Z_exp, angular_freq, ECM_candidate_name, ECM_candidate_impedance_func, ECM_initial_guess, ECM_bounds, cost_func_name = None, verbose=True, optimizer_option_Powell=None, optimizer_option_BFGS=None):
    params = None
    sse = None
    opt_result = None
    ECM_name = ECM_candidate_name
    impedance_func = ECM_candidate_impedance_func
    initial_guess = ECM_initial_guess

    # base estimation algo already sort by tau
    powell_fitted_params, powell_cost, powell_optresult = Powell_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, ECM_bounds, cost_func_name = cost_func_name, verbose=verbose, optimizer_options=optimizer_option_Powell)
    BFGS_fitted_params, BFGS_cost, BFGS_optresult = Bounded_ECM_estimation(Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, ECM_bounds, cost_func_name = cost_func_name, verbose=verbose, optimizer_options=optimizer_option_BFGS)
    
    if verbose:
        print(" ------------ ")

    # Decide which algo to use
    if (powell_cost > BFGS_cost) or any(p < 0 for p in powell_fitted_params):
        # I know there is only one incorrect case (131 lab data - 21EOL cycle 5)
        if(powell_cost < BFGS_cost and any(p < 0 for p in powell_fitted_params) and verbose):
            print("Powell better but gives negative estimation, do not use it")
        params = BFGS_fitted_params
        sse = BFGS_cost
        opt_result = BFGS_optresult
        if verbose:
            print("L-BFGS-B wins, use L-BFGS-B fitting result")
    elif powell_cost < BFGS_cost:
        params = powell_fitted_params
        sse = powell_cost
        opt_result = powell_optresult
        if verbose:
            print("Powell wins, use Powell fitting result")
    else: 
        params = powell_fitted_params
        sse = powell_cost
        opt_result = powell_optresult
        if verbose:
            print("Ties, use Powell fitting result")
    
    if verbose:
        print("params:", params)
    
    return params, sse, opt_result


def perturb_initial_guess(base_guess, scale=0.2):
    """
    Slightly perturb each parameter in the base_guess by ±scale percentage.
    """
    return [max(eps, p * (1 + np.random.uniform(-scale, scale))) for p in base_guess]



def perturb_initial_guess_elementwise(base_guess, params_name=None, eps=1e-7,
                          scale_map={'R': 0.5, 'C': 0.5, 'n': 0.1, 'w': 0.1}):
    """
    Perturb each parameter in base_guess based on its name/type using a specific scale.

    Parameters:
    - base_guess: list of base parameter values
    - params_name: list of parameter names (e.g., ["R1", "C1", "n1", "Aw"])
    - eps: minimum allowed parameter value
    - scale_map: dict specifying perturbation scales for R, C, n, w(Warburg)

    Returns:
    - List of perturbed parameters
    """
    perturbed = []

    for i, p in enumerate(base_guess):
        param_scale = scale_map.get('R', 0.2)  # fallback default
        if params_name:
            name = params_name[i].lower()
            if name.startswith('r'):
                param_scale = scale_map.get('R', 0.2)
            elif name.startswith('c'):
                param_scale = scale_map.get('C', 0.2)
            elif name.startswith('n'):
                param_scale = scale_map.get('n', 0.1)
            elif 'w' in name:  # for 'Aw'
                param_scale = scale_map.get('w', 0.1)
        factor = 1 + np.random.uniform(-param_scale, param_scale)
        perturbed.append(max(eps, p * factor))
    
    return perturbed



# Version 3: Run multiple times with designed different intial guess, compare Powell and L-BFGS-B algorithm and find the best fitting ECM & its cost
# Return: params, cost
def ECM_result_wrapper(Z_exp, angular_freq, ECM_candidate_name, ECM_candidate_impedance_func, ECM_initial_guess=None, ECM_bounds=None, cost_func_name = None, verbose=True, optimizer_option_Powell=None, optimizer_option_BFGS=None):
    best_params = None
    best_err = float("inf")
    np.random.seed(1) # for reproducibility

    if ECM_bounds is None:
        ECM_bounds = BOUNDS[ECM_candidate_name]


    # 1. Step 1: Start with predefined initial guess
    if ECM_initial_guess is None:
        base_guess = INITIAL_GUESS[ECM_candidate_name]
    else:
        base_guess = ECM_initial_guess
    


    if verbose:
        print(f"Trying base initial guess: {base_guess}")

    params, err = ECM_result_single_candidate(
        Z_exp,
        angular_freq,
        ECM_candidate_name,
        ECM_candidate_impedance_func,
        base_guess,
        ECM_bounds,
        cost_func_name = cost_func_name,
        verbose=False
    )

    if err < best_err:
        best_err = err
        best_params = params

    # 2. Step 2: Try using last optimal value (5 times)
    # params_reuse = best_params
    # for i in range(3):
    #     within_bounds = all(lower <= p <= upper for p, (lower, upper) in zip(params_reuse, ECM_bounds))
    #     if not within_bounds:
    #         if verbose:
    #             print(f"Perturbed guess {params_reuse} is out of bounds, skipping...")
    #         continue
    #     if verbose:
    #         print(f"Trying reuse of last optimal parameters, trial {i+1}, initial guess: {params_reuse}")
    #     params_reuse, err_reuse = ECM_result_single_candidate(
    #         Z_exp,
    #         angular_freq,
    #         ECM_candidate_name,
    #         ECM_candidate_impedance_func,
    #         best_params,
    #         ECM_bounds,
    #         cost_func_name = cost_func_name,
    #         verbose=False
    #     )
    #     if err_reuse < best_err:
    #         best_err = err_reuse
    #         best_params = params_reuse
    #     else:
    #         break

    # 3. Step 3: Try 10 perturbed initial guesses
    for i in range(10):
        perturbed_guess = perturb_initial_guess(base_guess, scale=0.5)
        if verbose:
            print(f"Trying perturbed initial guess, trial {i+1}, initial guess: {perturbed_guess}...")
        within_bounds = all(lower <= p <= upper for p, (lower, upper) in zip(perturbed_guess, ECM_bounds))
        if not within_bounds:
            if verbose:
                print(f"Perturbed guess {perturbed_guess} is out of bounds, skipping...")
                i = i - 1
            continue
        if verbose:
            print(f"Trying perturbed guess, trial {i+1}, initial guess: {perturbed_guess}...")
        params_perturbed, err_perturbed, _ = ECM_result_single_candidate(
            Z_exp,
            angular_freq,
            ECM_candidate_name,
            ECM_candidate_impedance_func,
            perturbed_guess,
            ECM_bounds,
            cost_func_name = cost_func_name,
            verbose=False
        )
        if err_perturbed < best_err:
            best_err = err_perturbed
            best_params = params_perturbed


    print("Best Estimation Err:", best_err, "; Best Estimation Params:", best_params)
    all_costs = evaluate_all_costs(best_params, Z_exp, angular_freq, ECM_candidate_impedance_func)
    err_str = " | ".join(
        f"{name}: {val:.5e}" if isinstance(val, (float, int)) else f"{name}: {val}"
        for name, val in all_costs.items()
    )
    return best_params, best_err, err_str



##### NEW Wrapper Function with Detailed Information #####
def ECM_result_wrapper_new(Z_exp, angular_freq, ECM_candidate_name, ECM_candidate_impedance_func, trial_num=10, ECM_initial_guess=None, ECM_bounds=None, cost_func_name = None, verbose=True, optimizer_option_Powell=None, optimizer_option_BFGS=None):
    trial_results = []
    trial_id = 1
    np.random.seed(1) # for reproducibility

    best_params = None
    best_err = float("inf")
    best_trial_id = None

    # Setup default bounds and initial guess
    if ECM_bounds is None:
        ECM_bounds = BOUNDS.get(ECM_candidate_name, None)

    # 1. Step 1: Start with predefined initial guess
    if ECM_initial_guess is None:
        base_guess = INITIAL_GUESS.get(ECM_candidate_name, None)
    else:
        base_guess = ECM_initial_guess
    
    if ECM_bounds is None or base_guess is None:
        raise ValueError(f"ECM bounds or initial guess not defined for ECM '{ECM_candidate_name}'")
    
    
    # Loop through trials until we find trial_num of successful trials
    while trial_id <= trial_num:
        perturbed_guess = perturb_initial_guess_elementwise(base_guess, params_name=PARAMS_NAMES.get(ECM_candidate_name, None))
        within_bounds = all(lower <= p <= upper for p, (lower, upper) in zip(perturbed_guess, ECM_bounds))
        if not within_bounds:
            if verbose:
                print(f"Perturbed guess {perturbed_guess} is out of bounds, skipping...")
            continue
        if verbose:
            print(f"[Trial {trial_id}] Trying initial guess: {perturbed_guess}...")
        params_perturbed, err_perturbed, opt_result = ECM_result_single_candidate(
            Z_exp,
            angular_freq,
            ECM_candidate_name,
            ECM_candidate_impedance_func,
            perturbed_guess,
            ECM_bounds,
            cost_func_name = cost_func_name,
            verbose=False
        )
        if opt_result.success:
            # Record only successful trial
            all_costs = evaluate_all_costs(params_perturbed, Z_exp, angular_freq, ECM_candidate_impedance_func)
            # err_str = " | ".join(
            #     f"{name}: {val:.5e}" if isinstance(val, (float, int)) else f"{name}: {val}"
            #     for name, val in all_costs.items()
            # )
            trial_results.append({
                'trial_id': trial_id,
                'initial_guess': perturbed_guess,
                'estimated_params': params_perturbed,
                **all_costs   # merge cost metrics into the same dict
            })
            # Update best parameters if necessary
            if err_perturbed < best_err:
                best_err = err_perturbed
                best_params = params_perturbed
                best_trial_id = trial_id
        
        trial_id += 1


    print("Best Trial ID:", best_trial_id,"; Estimation Err:", best_err, "; Best Estimation Params:", best_params)
    all_costs = evaluate_all_costs(best_params, Z_exp, angular_freq, ECM_candidate_impedance_func)
    best_err_str = " | ".join(
        f"{name}: {val:.5e}" if isinstance(val, (float, int)) else f"{name}: {val}"
        for name, val in all_costs.items()
    )

    for trial in trial_results:
        trial['is_best'] = (trial['trial_id'] == best_trial_id)
    
    return best_params, best_err, best_err_str, trial_results

# Version 3: compare Powell and L-BFGS-B algorithm and find the best fitting ECM & its cost, second best-fitting ECM & its cost, third best-fitting ECM & its cost
# Return a list of tuples: [top1=("Fitting Algo", "ECM_name", cost, [list of params estimation]), top2=(...), top3=(...)]
def ECM_top3_result(Z_exp, angular_freq, ECM_candidates_names, ECM_candidates_impedance_funcs, ECM_initial_guesses, ECM_bounds=None, cost_func_name = None, input_config="input battery", visualization=True, verbose=True):
    results = []

    for i in range(len(ECM_candidates_names)):
        ECM_name = ECM_candidates_names[i]
        impedance_func = ECM_candidates_impedance_funcs[i]
        initial_guess = ECM_initial_guesses[i]
        bounds = ECM_bounds[i]

        # Powell
        powell_params, powell_cost, _ = Powell_ECM_estimation(
            Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=verbose
        )
        if all(p >= 0 for p in powell_params):  # ignore negative fits
            results.append(("Powell", ECM_name, powell_cost, powell_params))

        # L-BFGS-B
        bfgs_params, bfgs_cost, _ = Bounded_ECM_estimation(
            Z_exp, angular_freq, ECM_name, impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=verbose
        )
        if all(p >= 0 for p in bfgs_params):  # ignore negative fits
            results.append(("L-BFGS-B", ECM_name, bfgs_cost, bfgs_params))

        if verbose:
            print(" ------------ ")

    # Sort results by cost (ascending)
    sorted_results = sorted(results, key=lambda x: x[2])

    # Return top 3
    top3 = sorted_results[:3]

    # Optional verbose print
    if verbose:
        print("\nTop 3 fitting results:")
        for rank, (algo, name, cost, params) in enumerate(top3, start=1):
            print(f"Top {rank}: Algo={algo}, ECM={name}, Cost={cost:.6f}, Params={params}")


    # Visualization of top 3
    if visualization:
        top1_cost = top3[0][2]
        rank_to_color = {
            1: "#FF0000", 
            2: "#FF93F1", 
            3: "#FFA143",
        }
        plt.figure(figsize=(8, 6))

        # Plot experimental data
        plt.plot(np.real(Z_exp), -np.imag(Z_exp), 'bo-', label="Experimental Data")

        # Plot each of the top 3 fits
        for rank, (algo, name, cost, params) in enumerate(top3, start=1):
            i = ECM_candidates_names.index(name)
            impedance_func = ECM_candidates_impedance_funcs[i]
            Z_fit = impedance_func(params, angular_freq)

            color = rank_to_color.get(rank, "gray")
            # Calculate percentage deviation from top1 cost
            if rank == 1:
                deviation_str = "0.00%"
            else:
                deviation = (cost - top1_cost) / top1_cost * 100
                deviation_str = f"{deviation:.2f}%"
            plt.plot(np.real(Z_fit), -np.imag(Z_fit), 'o-', color=color, label=f"Top {rank}: {algo} + {name}\nCost={cost:.2e}, Relative Error to top1={deviation_str}")
        

        # Include (ECM8 if not already in top3)
        ecm8_in_top3 = any(name == "v3CM8" for (_, name, _, _) in top3)
        if not ecm8_in_top3:
            i = ECM_candidates_names.index("v3CM8")
            impedance_func = ECM_candidates_impedance_funcs[i]
            initial_guess = ECM_initial_guesses[i]
            bounds = ECM_bounds[i]

            # Powell fit
            powell_params, powell_cost, _ = Powell_ECM_estimation(Z_exp, angular_freq, "v3CM8", impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=False)
            Z_powell = impedance_func(powell_params, angular_freq)
            deviation = (powell_cost - top1_cost) / top1_cost * 100
            deviation_str = f"{deviation:.2f}%"
            plt.plot(np.real(Z_powell), -np.imag(Z_powell), '--', color="#5900FF",
                    label=f"Powell + v3CM8\nCost={powell_cost:.2e}, Relative Error to top1={deviation_str}")

            # L-BFGS-B fit
            lbfgs_params, lbfgs_cost, _ = Bounded_ECM_estimation(Z_exp, angular_freq, "v3CM8", impedance_func, initial_guess, bounds=bounds, cost_func_name = cost_func_name, verbose=False)
            Z_lbfgs = impedance_func(lbfgs_params, angular_freq)
            deviation = (lbfgs_cost - top1_cost) / top1_cost * 100
            deviation_str = f"{deviation:.2f}%"
            plt.plot(np.real(Z_lbfgs), -np.imag(Z_lbfgs), ':', color="#00FF55",
                    label=f"L-BFGS-B + v3CM8\nCost={lbfgs_cost:.2e}, Relative Error to top1={deviation_str}")
        

        plt.xlabel("Re(Z) [Ω]")
        plt.ylabel("-Im(Z) [Ω]")
        plt.title(f"Top 3 ECM Fits ({input_config})")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return top3


def evaluate_all_costs(params, Z_exp, angular_freq, impedance_func):
    results = {}
    for name, func in COST_FUNCTION_MAP.items():
        # Use partial to bind shared arguments
        try:
            cost = func(params, Z_exp=Z_exp, angular_freq=angular_freq, impedance_func=impedance_func)
            results[name] = cost
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    return results