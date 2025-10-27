"""
Configuration module for optimization project parameters.
This module defines default parameters for various components of the optimization system,
including prediction types, projection methods, problem types, and neural network configurations.
"""

def config():
    """
    Returns a dictionary of default configuration parameters for the optimization system.
    Returns:
        dict: A dictionary containing all default parameters organized by component
    """
    defaults = {}
    
    # Basic configuration parameters
    defaults['predType'] = ['NN', 'NN_Eq'][1]  # Neural Network prediction type
    defaults['projType'] = ['None', 'WS', 'Proj', 'D_Proj', 'H_Proj', 'B_Proj'][5]  # Projection method
    defaults['probType'] = ['qp', 'convex_qcqp', 'socp', 'sdp',            # Convex benchmarks
                           'acopf', 'jccim'][0]  #
    defaults['SA'] = False
    # Problem size configurations for different problem types
    defaults['probSize'] = {
        'qp': [400, 100, 100, 10000],
        'convex_qcqp': [400, 100, 100, 10000],
        'socp': [400, 100, 100, 10000],
        'sdp': [1600, 40, 40, 10000],
        'acopf': [793, 10000],
        'jccim': [400, 100, 100, 10000, 100],
    }[defaults['probType']]
    
    # General testing parameters
    defaults['testSize'] = 1024
    defaults['saveAllStats'] = False
    defaults['seed'] = 2023

    # Neural Network parameters
    defaults['nn_para'] = {
        'training': True,         # Training mode flag
        'testing': True,          # Testing mode flag
        'approach': 'supervise',  # Training approach
        'total_iteration': 10000, # Total training iterations
        'batch_size': 64,         # Batch size for training
        'pre_training': 10000,    # Pre-training iterations (MSE loss only)
        'num_layer': 3,           # Number of network layers
        'lr': 1e-4,               # Learning rate
        'lr_decay': 0.9,          # Learning rate decay factor
        'lr_decay_step': 1000,    # Steps between learning rate decay
        'w_obj': 0.01,            # Weight for objective term
        'w_ineq': 0.001,            # Weight for inequality constraints
        'w_eq': 0.0,              # Weight for equality constraints
        'resultsSaveFreq': 1000   # Frequency of saving results
    }


    # IPNN (Interior Point Neural Network) parameters
    defaults['ipnn_para'] = {
        'training': True,         # Training mode flag
        'training_sample': 10000,       # Number of constraint samples
        'total_iteration': 10000, # Total training iterations
        'pre_training': 10000,    # Pre-training iterations
        'cal_boudanry_dist': False, # Calculate boundary distance flag
        'fixed_margin': False,     # Fixed margin flag
        'gamma': 1e-3,            # Margin parameter
        'batch_size': 64,         # Batch size for training
        'n_layer': 3,             # Number of network layers
        'n_ip': 1,                # Number of interior points
        'sa_pred_noise': 0,       # Prediction noise
        'lr': 1e-4,              # Learning rate
        'lr_decay': 0.9,         # Learning rate decay factor
        'lr_decay_step': 1000,   # Steps between learning rate decay
        'resultsSaveFreq': 1000   # Frequency of saving results
    }

    # Projection parameters
    defaults['proj_para'] = {
        'useTestCorr': False,      # Use test correction flag
        'corrEps': 1e-5,  # Correction tolerance
        'corrMode': 'partial',    # Correction mode
        'corrTestMaxSteps': 30,   # Maximum correction steps
        'corrBis': 0.5,          # Bisection parameter
        'corrLr': 1e-5,           # Correction learning rate
        'corrMomentum': 0.5,      # Correction momentum
    }


    return defaults