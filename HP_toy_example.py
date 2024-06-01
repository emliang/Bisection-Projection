from visualization import *
from utils.nn_utils import *
from utils.sampling_utils import *
from utils.toy_utils import *
from utils.training_utils import *

def constraint_learning(paras, instance_path, instance):
    simple_set = paras['shape']
    constraints = paras['constraint']
    distortion = paras['distortion_coefficient']
    #### Input-Output Specification: (t->x)
    n_dim = 2
    c_dim = constraints.c_dim
    #### Flow-based model: sphere -> constraint set
    num_layer = paras['num_layer']
    h_dim = paras['h_dim']
    inv_type = paras['inv_type']
    model = INN(n_dim, h_dim, c_dim, num_layer, inv=inv_type, outact=None).to(device=DEVICE)
    #### Sampling input parameters and output decision
    x_samples = paras['x_samples']
    c_samples = paras['c_samples']
    x_train = sampling_body(x_samples, n_dim, simple_set)  # + bias
    c_train = np.random.uniform(low=constraints.sampling_range[0],
                                high=constraints.sampling_range[1],
                                size=[c_samples, c_dim])
    x_train_tensor = torch.tensor(x_train).view(-1, n_dim).to(device=DEVICE)
    c_train_tensor = torch.tensor(c_train).view(-1, c_dim).to(device=DEVICE)
    #### Unsupervised Training for Hemo Mapping
    model, volume_list, penalty_list, dist_list, trans_list = unsupervised_training_mdh(model, constraints,
                                                                                         x_train_tensor, c_train_tensor, 
                                                                                         paras)
    torch.save(model, instance_path + f'/nns/model_{instance}.pth')
    # np.save(instance_path + f'/nns/records_{instance}.npy', [volume_list, penalty_list, dist_list, trans_list])
    return model

def __main__():
    paras = {'shape': 'sphere',
             'seed': 2002,
             'constraint': Disconnected_Ball(),#Complex_Constraints(),
             'bound': [-1, 1],
             'scale_ratio': 1.0,
             'x_samples': 10000,
             'c_samples': 10000,
             'total_iteration': 10000,
             'batch_size': 256,
             'num_layer': 3,
             'h_dim': 64,
             'inv_type': 'made', 'bilip': False, 'L': 2,
             'lr': 1e-3,
             'lr_decay': 0.9,
             'lr_decay_step': 1000,
             'penalty_coefficient': 10,
             'distortion_coefficient': 0.1,
             'transport_coefficient': 0.0, }
    constraints = paras['constraint']

    instance_path = f'results/toy_example/HP/{str(constraints)}'
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    if not os.path.exists(instance_path + '/pics'):
        os.makedirs(instance_path + '/pics')
    if not os.path.exists(instance_path + '/nns'):
        os.makedirs(instance_path + '/nns')

    constraints.device = DEVICE
    for attr in dir(constraints):
        var = getattr(constraints, attr)
        if torch.is_tensor(var):
            try:
                setattr(constraints, attr, var.to(device = DEVICE))
                # setattr(data, attr, var.to(device = DEVICE, dtype=torch.float64))
            except AttributeError:
                pass

    #### training INN for MDH mapping
    for shape in ['sphere']:
        paras['shape'] = shape
        simple_set = shape
        instance = f'shape_{simple_set}'
        constraint_learning(paras, instance_path, instance)

        #### Ploting results
        n_dim = 2
        x_train = sampling_body(100000, n_dim, simple_set)
        x_train_tensor = torch.tensor(x_train).view(-1, n_dim).to(device=DEVICE)
        model = torch.load(instance_path + f'/nns/model_{instance}.pth', map_location=DEVICE)
        # volume_list, penalty_list, dist_list, trans_list = np.load(instance_path + f'/nns/records_{instance}.npy')
        np.random.seed(paras['seed'])
        constraints.t_test = np.random.uniform(low=constraints.sampling_range[0],
                                               high=constraints.sampling_range[1],
                                               size=[3, constraints.c_dim])
        scatter_constraint_approximation(model, constraints, x_train_tensor, instance_path, paras)
        # scatter_constraint_evolution(model, constraints, x_train_tensor, instance_path, paras)
        # plot_convergence(volume_list, penalty_list, dist_list, trans_list, instance_path, paras)
        visualize_homeo_projection(model, constraints, x_train_tensor, instance_path, paras)


if __name__ == '__main__':
    __main__()