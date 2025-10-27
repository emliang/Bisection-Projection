from configuration.default_args import *
from utils.optimization_utils import *
from utils.training_utils import *

def train_all():
    """
    preds = ['NN', 'NN_Eq']
    probs: ['qp', 'convex_qcqp', 'socp', 'sdp', 'acopf', 'jccim', 'ccacopf']
    methods: ['WS', 'Proj', 'D_Proj', 'H_Proj', 'B_Proj']
    NN: [['NN', 'supervise'],  ['NN_Eq','supervise'],  ['NN_Eq', 'unsupervise']]
    """
    args = config()



def load_instance(args):
    # Load data, and put on GPU if needed
    seed = 2023#args['seed']
    args['algoType'] = args['predType'] + '_' + args['projType']
    test_size = args['testSize']
    prob_type = args['probType']
    prob_size = args['probSize']
    if prob_type in ['acopf', 'ccacopf', 'graph_acopf']:
        filepath = os.path.join('datasets', 'acopf', 'acopf_{}_{}_{}_dataset'.format(
            seed, prob_size[0], prob_size[1]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        if prob_type == 'graph_acopf':
            data = Grpah_ACOPF_Problem(dataset, test_size)
        elif prob_type == 'acopf':
            data = ACOPF_Problem(dataset, test_size)
    elif prob_type in ['qp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, prob_size[0], prob_size[1], prob_size[2], prob_size[3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = QP_Problem(dataset, test_size)
    elif prob_type in ['convex_qcqp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, prob_size[0], prob_size[1], prob_size[2], prob_size[3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = QCQP_Probem(dataset, test_size)
    elif prob_type in ['socp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, prob_size[0], prob_size[1], prob_size[2], prob_size[3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = SOCP_Probem(dataset, test_size)
    elif prob_type in ['sdp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, prob_size[0], prob_size[1], prob_size[2], prob_size[3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = SDP_Probem(dataset, test_size)
    elif prob_type in ['jccim']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_scenario{}_ex{}".format(
            seed, prob_type, prob_size[0], prob_size[1], prob_size[2], prob_size[4], prob_size[3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = JCCIM_Problem(dataset, test_size)
    elif prob_type in ['graph_qp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_node{}_spar{}_fix{}_ex{}".format(
            seed, prob_type, prob_size[0], prob_size[1], prob_size[2], prob_size[4], prob_size[5], prob_size[6], prob_size[3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = GraphQP_Problem(dataset, test_size)
    elif prob_type in ['power_control']:
        filepath = os.path.join('datasets', 'power_control', 'random_{}_power_control_dataset_node{}_ex{}'.format(
            seed, prob_size[0], prob_size[1]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = PowerControl_Problem(dataset, test_size)
    else:
        raise NotImplementedError
    print(f'load {filepath} to {DEVICE}')

    data.device = DEVICE
    for attr in dir(data):
        var = getattr(data, attr)
        if torch.is_tensor(var):
            try:
                # setattr(data, attr, var.to(device = DEVICE))
                setattr(data, attr, var.to(device = DEVICE, dtype=torch.float64))
            except AttributeError:
                pass
    if args['SA']:
        model_save_dir = os.path.join('models', 'SA', prob_type, str(data))
        result_save_dir = os.path.join('results', 'SA', prob_type, str(data))
    else:
        model_save_dir = os.path.join('models', prob_type, str(data))
        result_save_dir = os.path.join('results', prob_type, str(data))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    with open(os.path.join(model_save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    args['proj_para']['useTestCorr'] = False
    return data, result_save_dir, model_save_dir

def run_instance(args):
    """
    Load data
    """
    data, result_save_dir, model_save_dir = load_instance(args)

    """
    Run IPNN
    """
    if args['ipnn_para']['training']:
        train_meip_mapping(data, args, model_save_dir)

    """
    Run neural network solver
    """
    data, result_save_dir, model_save_dir = load_instance(args)
    if args['nn_para']['training']:
        train_nn_solver(data, args, model_save_dir)
    if args['nn_para']['testing']:
        test_nn_solver(data, args, model_save_dir, result_save_dir)

def test_all():
    args = config()
    args['inn_para']['training'] = False
    args['nn_para']['training'] = False
    args['ipnn_para']['training'] = False
    args['proj_para']['useTestCorr'] = True
    """
    preds = ['NN', 'NN_Eq']
    probs: ['qp', 'convex_qcqp', 'socp', 'sdp', 'acopf', 'jccim', 'ccacopf']
    methods: ['None', 'WS', 'Proj', 'D_Proj', 'H_Proj', 'B_Proj']
    """
    for prob in ['jccim']:
        for pred,approach in [['NN_Eq', 'supervise']]:
            args['predType'] = pred
            args['nn_para']['approach'] = approach
            for proj in ['Proj']:
                args['projType'] = proj

                if prob in ['acopf']:
                    for size in [[118,10000], [200,10000]]:
                        args['probType'], args['probSize'] = prob, size
                elif prob == 'sdp':
                    for size in [[1600, 40, 40, 10000]]:
                        args['probType'], args['probSize'] = prob, size
                elif prob == 'jccim':
                    for size in [[400, 100, 100, 10000, 100]]:
                        args['probType'], args['probSize'] = prob, size
                else:
                    for size in [[400, 100, 100, 10000]]:
                        args['probType'], args['probSize'] = prob, size
                test_single(args)

def test_single(args):
    data, result_save_dir, model_save_dir = load_instance(args)
    test_nn_solver(data, args, model_save_dir, result_save_dir)
    test_inf_time(data, args, result_save_dir)



import concurrent.futures
from functools import partial
import multiprocessing


def process_pair(pair):
    args, expr_id, training_sample, index = pair
    local_args = args.copy()
    local_args['seed'] = expr_id + 2025
    local_args['ipnn_para']['training_sample'] = training_sample
    data, result_save_dir, model_save_dir = load_instance(local_args)
    results = train_meip_mapping(data, local_args, model_save_dir)
    return (expr_id, index, results['ip_valid_rate'])



if __name__ == '__main__':
    train_all()
    test_all()
