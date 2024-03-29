def config():
    defaults = {}
    defaults['predType'] = ['NN', 'NN_Eq'][1]
    defaults['projType'] = ['WS', 'Proj', 'D_Proj', 'H_Proj', 'B_Proj'][4]
    defaults['probType'] = ['qp', 'convex_qcqp', 'socp', 'sdp', 'acopf', 'jccim'][1]
    defaults['probSize'] = [[100, 50, 50, 10000],
                            [400, 100, 100, 10000],
                            [100, 10, 10, 10000],
                            [400, 20, 20, 40000]][1]
    defaults['opfSize'] = [[30, 10000], [57, 10000], [118, 20000],[200, 30000],
                           [500, 30000], [793, 30000], [1354,30000], [2000, 30000]][0]
    defaults['testSize'] = 1024
    defaults['saveAllStats'] = False
    defaults['resultsSaveFreq'] = 1000
    defaults['seed'] = 2023

    defaults['ipnn_para'] = \
        {'training': True,
         'c_samples': 10000, 'total_iteration': 10000, 'batch_size': 64,
         'n_layer': 3, 'n_ip': 1, 'softmin': True, 'softrange': True, 'minimum_ecc': True,
         'n_boundary_point': 10, 'n_bisect_sampling': 5, 'w_penalty': 10, 'w_ecc': 1,
         'lr': 1e-4, 'lr_decay': 0.9, 'lr_decay_step': 1000,}

    defaults['inn_para'] = \
        {'training': False, 'testing': False,
         'n_samples': 10000, 'c_samples': 10000,
         'shape': 'square', 'bound': [0, 1], 'scale_ratio': 1,
         'total_iteration': 10000, 'batch_size': 256,
         'lr': 1e-4, 'lr_decay': 0.9, 'lr_decay_step': 1000,
        'num_layer': 3, 'inv_type': 'made', 'bilip': True, 'L': 2,
        'penalty_coefficient': 10, 'distortion_coefficient': 1, 'transport_coefficient': 0,
        'testing_samples': 1024}

    defaults['nn_para'] = \
        {'training': False, 'testing': True, 'approach': 'supervise',
        'total_iteration': 10000, 'batch_size': 256, 'pre_training': 0,
        'num_layer': 3, 'lr': 1e-3, 'lr_decay': 0.9, 'lr_decay_step': 1000,
        'objWeight': 0.00, 'softWeightInEqFrac': 0.00, 'softWeightEqFrac': 0}

    defaults['proj_para'] = \
        {'useTestCorr': False,    # post-process for infeasible solutions
        'corrMode': 'partial',    # equality completion
        'corrTestMaxSteps': 30,   # steps for D-Proj
        'corrBis': 0.9,           # steps for bisection
        'corrEps': 1e-5,          # tolerance for constraint violation
        'corrLr': 1e-3,           # stepsize for gradient descent in D-Proj
        'corrMomentum': 0.1, }    # momentum parameter in D-Proj

    return defaults

