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

    # args['probType'] = 'qp'
    # args['probSize'] = [400, 100, 100, 10000]
    # run_instance(args)
    #
    # args['probType'] = 'convex_qcqp'
    # args['probSize'] = [400, 100, 100, 10000]
    # run_instance(args)
    #
    # args['probType'] = 'socp'
    # args['probSize'] = [400, 100, 100, 10000]
    # run_instance(args)
    # #
    # args['probType'] = 'sdp'
    # args['probSize'] = [1600, 40, 40, 10000]
    # run_instance(args)
    # #
    args['probType'] = 'acopf'
    # args['probSize'] = [30, 10000]
    # run_instance(args)
    # args['probSize'] = [57, 10000]
    # run_instance(args)
    # args['probSize'] = [118, 10000]
    # run_instance(args)
    # args['probSize'] = [200, 10000]
    # run_instance(args)
    args['probSize'] = [793, 10000]
    run_instance(args)
    # #
    # args['probType'] = 'jccim'
    # # args['probSize'] = [400, 100, 100, 10000, 10]
    # # run_instance(args)
    # args['probSize'] = [400, 100, 100, 10000, 100]
    # run_instance(args)



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
    Run homeomorphic mapping
    """
    if args['inn_para']['training']:
        train_mdh_mapping(data, args, model_save_dir)
    """
    Run IPNN
    """
    if args['ipnn_para']['training']:
        train_meip_mapping(data, args, model_save_dir)

    """
    Run neural network solver
    """
    # for pred,approach in [['NN_Eq', 'supervise']
    #                       ]:
        # ['NN_Eq', 'unsupervise']
        # args['predType']` = pred
        # args['nn_para'][`'approach'] = approach
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


def sensitivity_analysis():
    args = config()
    args['SA'] = True
    """
    sensitivity analysis for gamma and training_sample
    """
    probs = ['qp', 'convex_qcqp', 'socp', 'sdp', 'acopf', 'jccim'] # 'convex_qcqp', 'socp', 'sdp', 'acopf',
    # Map problem types to their sizes
    prob_size_map = {
        'qp': [400, 100, 100, 10000],
        'convex_qcqp': [400, 100, 100, 10000],
        'socp': [400, 100, 100, 10000],
        'sdp': [1600, 40, 40, 10000],
        'acopf': [200, 10000],
        'jccim': [400, 100, 100, 10000, 100]
    }

    # Common parameters
    args['ipnn_para'].update({
        'training': True,
        'pre_training': 10000,
        'total_iteration': 10000,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'resultsSaveFreq': 5000,
    })

    sample_list = [10, 30, 100, 300, 1000, 10000]
    num_test = 2
    # Define scenarios to loop through
    scenarios = [
        {'gamma': 0, 'fixed_margin': True, 'suffix': 'no_gamma'},
        {'gamma': 5e-3, 'fixed_margin': True, 'suffix': 'fix_gamma'},
        {'gamma': 5e-3, 'fixed_margin': False, 'suffix': 'train_gamma'}
    ]
    for prob in ['jccim']:
        args['probType'], args['probSize'] = prob, prob_size_map[prob]
        for scenario in scenarios:
            print(scenario)
            args['ipnn_para']['gamma'] = scenario['gamma']
            args['ipnn_para']['fixed_margin'] = scenario['fixed_margin']
            ip_oos_feasibility = np.zeros((num_test, len(sample_list)))
            for expr_id in range(num_test):
                # Prepare all (expr_id, training_sample) pairs
                tasks = [(args, expr_id, training_sample, i)  for i,training_sample in enumerate(sample_list)]
                with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn')) as executor:
                    results = list(executor.map(process_pair,tasks ))
                # Fill the results into the array
                for expr_id, sample_idx, value in results:
                    ip_oos_feasibility[expr_id, sample_idx] = value
            np.save(f'results/SA/{prob}/{prob}_ipnn_{scenario["suffix"]}_sensitivity_oos_feasibility', ip_oos_feasibility)

    # for prob in probs:
    #     args['probType'], args['probSize'] = prob, prob_size_map[prob]
    #     for scenario in scenarios:
    #         print(scenario)
    #         args['ipnn_para']['gamma'] = scenario['gamma']
    #         args['ipnn_para']['fixed_margin'] = scenario['fixed_margin']
    #         ip_oos_feasibility = []
    #         for expr_id in range(num_test):
    #             args['seed'] = expr_id + 2025
    #             record = []
    #             for training_sample in sample_list:
    #                 args['ipnn_para']['training_sample'] = training_sample
    #                 data, result_save_dir, model_save_dir = load_instance(args)
    #                 results = train_meip_mapping(data, args, model_save_dir)
    #                 record.append(results['ip_valid_rate'])
    #             ip_oos_feasibility.append(record)
    #         np.save(f'results/SA/{prob}_ipnn_{scenario["suffix"]}_sensitivity_oos_feasibility', ip_oos_feasibility)
    print(1/0)



    """
    sensitivity analysis for bisection steps
    """
    args['projType'] = 'B_Proj'
    args['ipnn_para']['gamma'] = 5e-3
    args['ipnn_para']['fixed_margin'] = False
    args['ipnn_para']['corrEps'] = 1e-3
    num_test = 5
    for prob in probs:
        for expr_idx in range(num_test):
            args['probType'], args['probSize'] = prob, prob_size_map[prob]
            record = {'ave_opt_gap':[] , 'std_opt_gap':[], 'run_time':[] }
            for bis_step in [0,0,2,4,6,8,10,12,14,16,18,20]:
                args['ipnn_para']['sa_pred_noise'] = 5e-3
                args['proj_para']['corrTestMaxSteps'] = bis_step
                data, result_save_dir, model_save_dir = load_instance(args)
                epoch_stats = test_nn_solver(data, args, model_save_dir, result_save_dir)
                infeasible_instance = epoch_stats['test_raw_vio_instance']
                ave_opt_gap = np.mean(epoch_stats['test_cor_mape_loss'][infeasible_instance])
                std_opt_gap = np.std(epoch_stats['test_cor_mape_loss'][infeasible_instance])
                run_time = epoch_stats['test_raw_time'] + epoch_stats['test_proj_time']
                record['ave_opt_gap'].append(ave_opt_gap)
                record['std_opt_gap'].append(std_opt_gap)
                record['run_time'].append(run_time)
            np.save(f'results/SA/{expr_idx}_{prob}_ipnn_sensitivity_bis_step_opt', record)
    print(1/0)



    # """
    # sensitivity analysis for pred_noise and fixed_margin
    # """
    # # Define scenarios to loop through
    # scenarios = [
    #     {'gamma': 0, 'fixed_margin': True, 'suffix': 'no_gamma'},
    #     {'gamma': 1e-1, 'fixed_margin': True, 'suffix': 'fix_gamma'},
    #     {'gamma': 1e-1, 'fixed_margin': False, 'suffix': 'train_gamma'}
    # ]s
    probs  = ['jccim']
    for prob in probs:
        args['projType'] = 'B_Proj'
        args['probType'], args['probSize'] = prob, prob_size_map[prob]
        for scenario in scenarios:
            args['ipnn_para']['sa_pred_noise'] = 0
            args['ipnn_para']['gamma'] = scenario['gamma']
            args['ipnn_para']['fixed_margin'] = scenario['fixed_margin']
            data, result_save_dir, model_save_dir = load_instance(args)
            # train_meip_mapping(data, args, model_save_dir)
            # record = {'init_opt':[], 'proj_opt':[], 'proj_dist': [], 'cons_vio': []}
            record = {'proj_dist': [], 'ip_dist': []}
            sa_pred_noise_list = 10 ** (np.linspace(-4,1,30))
            for i, noise in enumerate(sa_pred_noise_list):
                args['seed'] = i + 2025
                args['ipnn_para']['sa_pred_noise'] = noise
                IP_dist, Proj_dist = eval_ipnn_solution(data, args, model_save_dir)
                record['ip_dist'].append(IP_dist)
                record['proj_dist'].append(Proj_dist)
                # data, result_save_dir, model_save_dir = load_instance(args)
                # epoch_stats = test_nn_solver(data, args, model_save_dir, result_save_dir)
                # record['init_opt'].append(epoch_stats['test_raw_mape_loss'])
                # record['cons_vio'].append(epoch_stats['test_raw_ineq_sum'])
                # record['proj_opt'].append(epoch_stats['test_cor_mape_loss'])
                # record['proj_dist'].append(epoch_stats['test_proj_mae_dist'])
            np.save(f'results/SA/{prob}_ipnn_{scenario["suffix"]}_sensitivity_opt_gap', record)




if __name__ == '__main__':
    # train_all()
    # test_all()
    sensitivity_analysis()
