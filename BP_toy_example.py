from utils.nn_utils import ResNet
from utils.toy_utils import *
from utils.training_utils import unsupervised_training_meip
from utils.visualization import *





def __main__():
    paras = {'constraint':  Complex_Constraints(), #Disconnected_Ball(), #
             'fix_input': False,
             'fixed_margin': False, 
             'gamma': 1e-3,
             'pre_training': 0,
             'training_sample': 10000,
             'total_iteration': 10000,
             'batch_size': 64,
             'n_layer': 3,
             'n_hid': 64,
             'lr': 1e-3,
             'lr_decay': 0.9,
             'lr_decay_step': 1000,
             'cal_boudanry_dist': True,
             'n_boundary_point': 100,
             'n_bisect_sampling': 20,
             'resultsSaveFreq': 100,}
    paras['proj_para'] = {'corrTestMaxSteps': 100, 'corrEps': 1e-5, 'corrBis':0.9}
    data = paras['constraint']
    data.to_device(DEVICE)
    n_dim = 2
    c_dim = data.c_dim
    fix_input = paras['fix_input']
    fixed_margin = paras['fixed_margin']
    gamma = paras['gamma']

    instance_path = f'results/toy_example/BP/{str(data)}'
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    if not os.path.exists(instance_path + '/pics'):
        os.makedirs(instance_path + '/pics')
    if not os.path.exists(instance_path + '/nns'):
        os.makedirs(instance_path + '/nns')

    instance = f'fix_{fix_input}_ecc_{fixed_margin}_{gamma}'

    for exper_id in range(10):
        paras['fix_input'] = fix_input
        model = NoiseResNet(data.c_dim, n_dim, paras['n_hid'], paras['n_layer'] , act=None, gamma=gamma).to(DEVICE)
        if paras['fix_input']:
            c = data.fix_c
        else:
            c = np.random.rand(paras['training_sample'], c_dim)
            c = c * (data.sampling_range[1] - data.sampling_range[0]) + data.sampling_range[0]
        c_tensor = torch.tensor(c).to(DEVICE)
        model, training_record = unsupervised_training_meip(model, data, paras, instance_path, input_tensor=c_tensor)
        torch.save(model, instance_path + f'/nns/model_{instance}.pth')
        np.save(instance_path + f'/nns/{exper_id}_training_record_{instance}.npy', training_record)

if __name__ == '__main__':
    __main__()