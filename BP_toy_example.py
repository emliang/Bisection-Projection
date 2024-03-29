from utils.nn_utils import ResNet
from utils.toy_utils import *
from utils.training_utils import unsupervised_training_meip
from visualization import *

def __main__():
    paras = {'constraint': Disconnected_Ball(),
             'fix_input': False,
             'softmin': True,
             'softrange': True,
             'minimum_ecc': True,
             'n_samples': 10000,
             'total_iteration': 1000,
             'batch_size': 512,
             'n_layer': 3,
             'n_hid': 64,
             'n_ip': 1,
             'n_boundary_point': 10,
             'n_bisect_sampling': 10,
             'w_penalty': 1, 'w_ecc': 1,
             'lr': 1e-3,
             'lr_decay': 0.9,
             'lr_decay_step': 1000,}
    paras['proj_para'] = {'corrTestMaxSteps': 100, 'corrEps': 1e-5, 'corrBis':0.5}
    data = Disconnected_Ball()
    n_dim = 2
    c_dim = data.c_dim
    err = 0.2
    density = 10
    fix_input = paras['fix_input']
    softmin = paras['softmin']
    softrange = paras['softrange']
    minimum_ecc = paras['minimum_ecc']
    # plot_illustration(paras, instance_path)

    instance_path = f'results/toy_example/BP/{data.__class__.__name__}'
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    if not os.path.exists(instance_path + '/pics'):
        os.makedirs(instance_path + '/pics')
    if not os.path.exists(instance_path + '/nns'):
        os.makedirs(instance_path + '/nns')

    for n_ip in [1,2,4,8]:
        instance = f'fix_{fix_input}_softmin_{softmin}_softrange_{softrange}_ip_{n_ip}_ecc_{minimum_ecc}'
        # for softmin in [True, False]:
        paras['softmin'] = softmin
        paras['fix_input'] = fix_input
        paras['n_ip'] = n_ip
        model = ResNet(data.c_dim, n_dim * paras['n_ip'], paras['n_hid'], paras['n_layer'] , act=None).to(device)
        if paras['fix_input']:
            c = data.fix_c
        else:
            c = np.random.rand(paras['n_samples'], c_dim)
            c = c * (data.sampling_range[1] - data.sampling_range[0]) + data.sampling_range[0]
        c_tensor = torch.tensor(c).to(device)
        model, penalty_list, eccentricity_list = unsupervised_training_meip(model, data, n_dim, c_tensor, paras)
        torch.save(model, instance_path + f'/nns/model_{instance}.pth')
        np.save(instance_path + f'/nns/loss_{instance}.npy', [penalty_list, eccentricity_list])

        plot_bp_loss(data, paras, instance_path, instance, n_dim, c_dim)
        plot_bp_traj_varying_input(data, paras, instance_path, instance, n_dim, c_dim, err, density)
    # plot_bp_traj_varying_ip(data, paras, instance_path, instance, n_dim, c_dim, err, density)
    # plot_ip_bp_loss(data, paras, instance_path, instance, n_dim, c_dim)

if __name__ == '__main__':
    __main__()