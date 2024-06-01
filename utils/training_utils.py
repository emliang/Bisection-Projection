from .nn_utils import *
from .sampling_utils import *
from .proj_utils import  *
import numpy as np
import torch
import torch.optim as optim
import time
import subprocess
import pickle

def get_least_utilized_gpu():
    # Query nvidia-smi to find the current GPU utilization
    gpu_stats = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    # Convert the output into a list of tuples (free_memory, total_memory) for each GPU
    gpu_memory = [tuple(map(int, x.split(','))) for x in gpu_stats.strip().split('\n')]
    # Calculate the used memory and sort the GPUs based on the free memory (descending)
    gpu_memory = [(total - free, idx) for idx, (free, total) in enumerate(gpu_memory)]
    least_utilized_gpu = sorted(gpu_memory, key=lambda x: x[0])[0][1]
    return least_utilized_gpu
# Use the function to set the CUDA device
least_utilized_gpu = get_least_utilized_gpu()
print(f"Using GPU: {least_utilized_gpu}")
DEVICE = torch.device(least_utilized_gpu if torch.cuda.is_available() else "cpu")




###################################################################
# (Un)supervised Training for NN Solver
###################################################################
def train_nn_solver(data, args, save_dir):
    lr = args['nn_para']['lr']
    nepochs = args['nn_para']['total_iteration']
    batch_size = args['nn_para']['batch_size']
    lr_decay = args['nn_para']['lr_decay']
    lr_decay_step = args['nn_para']['lr_decay_step']
    training_appoach = args['nn_para']['approach']
    pre_training = args['nn_para']['pre_training']

    ### Equality completion
    if 'Eq' in args['algoType']:
        out_dim = len(data.partial_vars_idx)
    else:
        out_dim = data.testY.shape[1]
    in_dim = data.xdim
    n_layer = args['nn_para']['num_layer']
    solver_net = ResNet(in_dim, out_dim, int(data.intrin_dim*1.1), n_layer)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=lr, weight_decay=1e-5)
    solver_shce = optim.lr_scheduler.StepLR(solver_opt, step_size=lr_decay_step, gamma=lr_decay)
    loss = nn.MSELoss()
    stats = {}
    solver_net.train()

    Xtrain = data.trainX.to(DEVICE)
    Ytrain = data.trainY.squeeze().to(DEVICE)
    Xtest = data.testX.to(DEVICE)
    Ytest = data.testY.squeeze().to(DEVICE)
    training_time_list = []
    total_time = 0
    for i in range(nepochs + 1):
        epoch_stats = {}
        if training_appoach == 'supervise':
            if i==0:
                """
                pre-training
                """
                for pn in range(pre_training):
                    batch_index = np.random.choice(np.arange(Xtrain.shape[0]), batch_size, replace=False)
                    # Get train loss
                    Xtrain_batch = Xtrain[batch_index]
                    Ytrain_batch = Ytrain[batch_index]
                    Z_pred_batch = solver_net(Xtrain_batch)
                    Z_pred_scale_batch = data.scale(Xtrain_batch, Z_pred_batch)
                    mse_loss = loss(Z_pred_scale_batch, Ytrain_batch[:, data.partial_vars_idx])
                    mse_loss.backward()
                    if pn%100==0:
                        print('Pre-training loss', pn, mse_loss.mean(), end='\r')
                    solver_opt.step()
                    solver_shce.step()
                    solver_opt.zero_grad()
            """
            formal training
            """
            iter_st = time.time()
            batch_index = np.random.choice(np.arange(Xtrain.shape[0]), batch_size, replace=False)
            # Get train loss
            Xtrain_batch = (Xtrain[batch_index]).view(batch_size,-1)
            Ytrain_batch = (Ytrain[batch_index]).view(batch_size,-1)
            start_time = time.time()
            Z_pred_batch = solver_net(Xtrain_batch)
            Z_pred_scale_batch = data.scale(Xtrain_batch, Z_pred_batch)
            if 'Eq' in args['algoType']:
                Y_pred_scale_batch = data.complete_partial(Xtrain_batch, Z_pred_scale_batch)
            training_obj = data.obj_fn(Y_pred_scale_batch)
            real_obj = data.obj_fn(Ytrain_batch)
            eq_penalty = torch.sum(torch.abs(data.eq_resid(Xtrain_batch, Y_pred_scale_batch)), dim=1)
            ineq_penalty = torch.sum(torch.abs(data.ineq_resid(Xtrain_batch, Y_pred_scale_batch)), dim=1)
            mse_loss = loss(Y_pred_scale_batch, Ytrain_batch)
            # mse_loss = loss(Z_pred_scale_batch, Ytrain_batch[:, data.partial_vars_idx])
            train_loss = mse_loss \
                        + args['nn_para']['softWeightInEqFrac'] * ineq_penalty.mean() \
                        + args['nn_para']['softWeightEqFrac'] * eq_penalty.mean() \
                        + args['nn_para']['objWeight'] * (training_obj-real_obj).abs().mean()
        else:
            Xtrain_batch = torch.rand([batch_size, Xtest.shape[1]]).to(device=DEVICE)
            Xtrain_batch = Xtrain_batch * (data.input_U - data.input_L) + data.input_L
            start_time = time.time()
            Y_pred_batch = solver_net(Xtrain_batch)
            Y_pred_scale_batch = data.scale(Xtrain_batch, Y_pred_batch)
            if 'Eq' in args['algoType']:
                Y_pred_scale_batch = data.complete_partial(Xtrain_batch, Y_pred_scale_batch)
            training_obj = data.obj_fn(Y_pred_scale_batch)
            eq_penalty = torch.sum(torch.abs(data.eq_resid(Xtrain_batch, Y_pred_scale_batch)), dim=1)
            ineq_penalty = torch.sum(torch.abs(data.ineq_resid(Xtrain_batch, Y_pred_scale_batch)), dim=1)
            train_loss = args['nn_para']['softWeightInEqFrac'] * ineq_penalty.mean() \
                        + args['nn_para']['softWeightEqFrac'] * eq_penalty.mean() \
                        + args['nn_para']['objWeight'] * training_obj.mean()

        train_loss.backward()
        solver_opt.step()
        solver_shce.step()
        solver_opt.zero_grad()
        train_time = time.time() - start_time
        iter_et = time.time()
        training_time_list.append(iter_et - iter_st)
        dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
        dict_agg(epoch_stats, 'train_obj', training_obj.detach().cpu().numpy())
        # dict_agg(epoch_stats, 'train_real_obj', real_obj.detach().cpu().numpy())
        dict_agg(epoch_stats, 'train_time', train_time, op='sum')
        total_time+= train_time
        # Print results
        if i % args['resultsSaveFreq'] == 0 and i > 0:
            # print(total_time)
            batch_index = np.random.choice(np.arange(Xtest.shape[0]), batch_size, replace=False)
            solver_net.eval()
            with torch.no_grad():
                eval_solution(data, Xtest[batch_index], Ytest[batch_index], solver_net, args, save_dir, 'test', epoch_stats)
            print('Epoch:{}, Feas_rate:{:.2f}, '
                  'Raw_loss: MSE({:.4f}), MAP({:.4f}), '
                  'Raw_obj: MSE({:.4f}), MAP({:.4f}), '
                  'Raw_Ineq: Max({:.4f}), Raw_eq:  Max({:.4f})'.format(
                i, 1-np.mean(epoch_stats['test_raw_vio_instance']),
                np.mean(epoch_stats['test_raw_mse_loss']), np.mean(epoch_stats['test_raw_mape_loss']),
                np.mean(epoch_stats['test_raw_obj_mse']),  np.mean(epoch_stats['test_raw_obj_mape']),
                np.mean(epoch_stats['test_raw_ineq_max']), np.mean(epoch_stats['test_raw_eq_max'])))
            print(np.mean(training_time_list))
                # np.mean(epoch_stats['test_raw_ineq_num_viol_0']) / data.nineq,
                # np.mean(epoch_stats['test_raw_eq_num_viol_0']) / data.neq))
            with open(os.path.join(save_dir, 'solver_net.pth'), 'wb') as f:
                torch.save(solver_net, f)
        if args['saveAllStats']:
            if i == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats
    with open(os.path.join(save_dir, 'train_stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    return solver_net, stats

def test_nn_solver(data, args, model_save_dir, result_save_dir):
    print(args['probType'], args['projType'])
    args['proj_para']['useTestCorr'] = True
    ## Run pure optimization baselines
    # DEVICE = torch.device("cpu")
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE
    Xtest = data.testX.to(DEVICE)
    Ytest = data.testY.squeeze().to(DEVICE)


    solver_net = torch.load(os.path.join(model_save_dir, 'solver_net.pth'), map_location=DEVICE)
    epoch_stats = {}
    with torch.no_grad():
        eval_solution(data, Xtest, Ytest, solver_net, args, model_save_dir, 'test', epoch_stats)
    print('Raw_loss: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}),     Raw_obj: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}) \n'
          'Cor_loss: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}),     Cor_obj: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}) \n'
          'Raw_Ineq: Max({:.4f}), Sum({:.4f}), Per({:.4f}),     Raw_eq:  Max({:.4f}), Sum({:.4f}), Per({:.4f})\n'
          'Cor_Ineq: Max({:.4f}), Sum({:.4f}), Per({:.4f}),     Cor_eq:  Max({:.4f}), Sum({:.4f}), Per({:.4f})\n'
          'Raw_Rate: FRate({:.4f}), Raw_inf: Batch({:.4f})\n'
          'Cor_Rate: FRate({:.4f}), Cor_inf: Batch({:.4f})'.format(
        np.mean(epoch_stats['test_raw_mse_loss']), np.mean(epoch_stats['test_raw_mae_loss']),
        np.mean(epoch_stats['test_raw_mape_loss']), np.mean(epoch_stats['test_raw_obj_mse']),
        np.mean(epoch_stats['test_raw_obj_mae']), np.mean(epoch_stats['test_raw_obj_mape']),
        np.mean(epoch_stats['test_cor_mse_loss']), np.mean(epoch_stats['test_cor_mae_loss']),
        np.mean(epoch_stats['test_cor_mape_loss']), np.mean(epoch_stats['test_cor_obj_mse']),
        np.mean(epoch_stats['test_cor_obj_mae']), np.mean(epoch_stats['test_cor_obj_mape']),
        np.mean(epoch_stats['test_raw_ineq_max']), np.mean(epoch_stats['test_raw_ineq_sum']),
        np.mean(epoch_stats['test_raw_ineq_num_viol_0']) / data.nineq, np.mean(epoch_stats['test_raw_eq_max']),
        np.mean(epoch_stats['test_raw_eq_sum']), np.mean(epoch_stats['test_raw_eq_num_viol_0']) / data.neq,
        np.mean(epoch_stats['test_cor_ineq_max']), np.mean(epoch_stats['test_cor_ineq_sum']),
        np.mean(epoch_stats['test_cor_ineq_num_viol_0']) / data.nineq, np.mean(epoch_stats['test_cor_eq_max']),
        np.mean(epoch_stats['test_cor_eq_sum']), np.mean(epoch_stats['test_cor_eq_num_viol_0']) / data.neq,
        1 - np.mean(epoch_stats['test_raw_vio_instance']), epoch_stats['test_raw_time'],
        1 - np.mean(epoch_stats['test_cor_vio_instance']), epoch_stats['test_proj_time']))
    with open(os.path.join(result_save_dir, 'test_stats.dict'), 'wb') as f:
        pickle.dump(epoch_stats, f)

def eval_solution(data, X, Ytarget, solver_net, args, save_dir, prefix, stats):
    ### NN solution prediction
    raw_start_time = time.time()
    solver_net.eval()
    with torch.no_grad():
        Y_pred = solver_net(X)
        Y_pred_scale = data.scale(X, Y_pred)
        if 'Eq' in args['predType']:
            Y = data.complete_partial(X, Y_pred_scale)
        else:
            Y = Y_pred_scale
    raw_end_time = time.time()
    NN_pred_time = raw_end_time - raw_start_time
    # print(0.001)
    ### Post-processing for infeasible only
    steps = args['proj_para']['corrTestMaxSteps']
    eps_converge = args['proj_para']['corrEps']
    penalty = data.check_feasibility(X, Y).abs()
    infeasible_index = (penalty.max(-1)[0] > eps_converge).view(-1)
    Y_pred_infeasible = Y[infeasible_index]
    # print(penalty)
    # print(1/0)
    num_infeasible_prediction = Y_pred_infeasible.shape[0]
    Ycorr = Y.detach().clone()
    # print(f'num of infeasible instance {Y_pred_infeasible.shape[0]}')
    ### load post-processing mdoel
    ### star projection
    cor_start_time = time.time()
    if args['proj_para']['useTestCorr'] and 'H_Proj' in args['algoType']:
        homeo_mapping = torch.load(os.path.join(save_dir, 'mdh_mapping.pth'), map_location=DEVICE)
        homeo_mapping.eval()
    elif args['proj_para']['useTestCorr'] and 'B_Proj' in args['algoType']:
        n_ip = args['ipnn_para']['n_ip']
        minimum_ecc = args['ipnn_para']['minimum_ecc']
        ipnn = torch.load(os.path.join(save_dir, f'ipnn_{n_ip}_{minimum_ecc}.pth'), map_location=DEVICE)
        ipnn.eval()
        with torch.no_grad():
            feasible_ip = ipnn(X[infeasible_index])
            feasible_ip = feasible_ip.view(feasible_ip.shape[0], n_ip, -1)
        # feasible_ip = torch.as_tensor(data.opt_ip(X[infeasible_index])).to(X.device)
        # feasible_ip = (feasible_ip[:, data.partial_vars_idx] - data.L ) / (data.U - data.L)
        # feasible_ip = feasible_ip.view(feasible_ip.shape[0], 1, -1)

    if num_infeasible_prediction > 0:
        if args['proj_para']['useTestCorr']:
            if 'H_Proj' in args['algoType']:
                Yproj, steps = homeo_bisection(homeo_mapping, data, args, Y_pred[infeasible_index], X[infeasible_index])
            elif 'G_Proj' in args['algoType']:
                Yproj, steps = gauge_bisection(homeo_mapping, data, args, Y_pred[infeasible_index], X[infeasible_index])
            elif 'B_Proj' in args['algoType']:
                Yproj, _ = ip_bisection(feasible_ip, data, args, Y_pred[infeasible_index], X[infeasible_index])
            elif 'D_Proj' in args['algoType']:
                Yproj, steps = diff_projection(data, X[infeasible_index], Y[infeasible_index], args)
            elif 'Proj' in args['algoType']:
                Yproj = data.opt_proj(X[infeasible_index], Y[infeasible_index]).to(Y.device)
            elif 'WS' in args['algoType']:
                Yproj = data.opt_warmstart(X[infeasible_index], Y[infeasible_index]).to(Y.device)
            else:
                Yproj = Y_pred_infeasible
            Ycorr[infeasible_index] = Yproj
    cor_end_time = time.time()
    Proj_time = cor_end_time - cor_start_time

    make_prefix = lambda x: "{}_{}".format(prefix, x)
    dict_agg(stats, make_prefix('time'), Proj_time + NN_pred_time, op='sum')
    dict_agg(stats, make_prefix('proj_time'), Proj_time, op='sum')
    dict_agg(stats, make_prefix('raw_time'), NN_pred_time, op='sum')
    # dict_agg(stats, make_prefix('steps'), np.array([steps]))

    dict_agg(stats, make_prefix('num_infeasible'), num_infeasible_prediction)
    dict_agg(stats, make_prefix('index_infeasible'), infeasible_index.detach().cpu().numpy())

    Y_obj = data.obj_fn(Y).detach().cpu()
    Ycor_obj = data.obj_fn(Ycorr).detach().cpu()
    Ytarget_obj = data.obj_fn(Ytarget).detach().cpu()
    raw_ineq_vio = torch.abs(data.ineq_resid(X, Y)).detach().cpu()
    raw_eq_vio = torch.abs(data.eq_resid(X, Y)).detach().cpu()
    raw_vio = torch.abs(data.check_feasibility(X, Y)).detach().cpu()
    cor_vio = torch.abs(data.check_feasibility(X, Ycorr)).detach().cpu()
    cor_ineq_vio = torch.abs(data.ineq_resid(X, Ycorr)).detach().cpu()
    cor_eq_vio = torch.abs(data.eq_resid(X, Ycorr)).detach().cpu()
    Y = Y.detach().cpu()
    X = X.detach().cpu()
    Ycorr = Ycorr.detach().cpu()
    Ytarget = Ytarget.detach().cpu()

    solution_res = Y - Ytarget
    proj_solution_res = Ycorr - Ytarget
    target_solution_norm = torch.norm(Ytarget, dim=1, p=1)
    cor_dist = Ycorr - Y

    raw_mae_loss = torch.norm(solution_res, dim=1, p=1)
    raw_mse_loss = torch.norm(solution_res, dim=1, p=2) ** 2
    raw_mape_loss = raw_mae_loss / target_solution_norm
    dict_agg(stats, make_prefix('raw_mae_loss'), raw_mae_loss.numpy())
    dict_agg(stats, make_prefix('raw_mse_loss'), raw_mse_loss.numpy())
    dict_agg(stats, make_prefix('raw_mape_loss'), raw_mape_loss.numpy())

    cor_mae_loss = torch.norm(proj_solution_res, dim=1, p=1)
    cor_mse_loss = torch.norm(proj_solution_res, dim=1, p=2) ** 2
    cor_mape_loss = cor_mae_loss / target_solution_norm
    dict_agg(stats, make_prefix('cor_mae_loss'), cor_mae_loss.numpy())
    dict_agg(stats, make_prefix('cor_mse_loss'), cor_mse_loss.numpy())
    dict_agg(stats, make_prefix('cor_mape_loss'), cor_mape_loss.numpy())

    dict_agg(stats, make_prefix('raw_cor_mae_dist'), torch.norm(cor_dist, dim=1, p=1).numpy())
    dict_agg(stats, make_prefix('raw_cor_mse_dist'), torch.norm(cor_dist, dim=1, p=2).numpy())

    dict_agg(stats, make_prefix('raw_eval'), Y_obj.numpy())
    dict_agg(stats, make_prefix('cor_eval'), Ycor_obj.numpy())
    dict_agg(stats, make_prefix('real_eval'), Ytarget_obj.numpy())
    dict_agg(stats, make_prefix('raw_obj_mae'), (Y_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('raw_obj_mse'), torch.square(Y_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('raw_obj_mape'), ((Y_obj - Ytarget_obj) / torch.abs(Ytarget_obj)).numpy())
    dict_agg(stats, make_prefix('cor_obj_mae'), (Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mse'), torch.square(Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mape'), ((Ycor_obj - Ytarget_obj) / torch.abs(Ytarget_obj)).numpy())

    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(raw_ineq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(raw_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_ineq_sum'), torch.sum(raw_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_0'), torch.sum(raw_ineq_vio > eps_converge, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_eq_max'), torch.max(raw_eq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'), torch.mean(raw_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_eq_sum'), torch.sum(raw_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_0'), torch.sum(raw_eq_vio > eps_converge, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_vio_instance'), (torch.max(raw_vio, dim=1)[0] > eps_converge).numpy())

    dict_agg(stats, make_prefix('cor_ineq_max'), torch.max(cor_ineq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('cor_ineq_mean'), torch.mean(cor_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_ineq_sum'), torch.sum(cor_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_ineq_num_viol_0'), torch.sum(cor_ineq_vio > eps_converge, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_eq_max'), torch.max(cor_eq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('cor_eq_mean'), torch.mean(cor_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_eq_sum'), torch.sum(cor_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_eq_num_viol_0'), torch.sum(cor_eq_vio > eps_converge, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_vio_instance'), (torch.max(cor_vio, dim=1)[0] > eps_converge).numpy())
    return stats

def test_inf_time(data, args, model_save_dir, result_save_dir):
    args['proj_para']['useTestCorr'] = True
    ## Run pure optimization baselines
    if args['probType'] == 'acopf':
        solvers = ['pypower']
    elif args['probType'] == 'nonconvex':
        solvers = ['ipopt']
    else:
        solvers = ['cvxpy']  # 'qpth osqp'

    # DEVICE = torch.device("cpu")
    np.random.seed(args['seed'])


    epoch_stats = np.load(os.path.join(result_save_dir, 'test_stats.dict'), allow_pickle=True)
    ### run instance paralell by multi-processing instance parallel for average opt inf time
    test_index = np.random.choice([i for i in range(data.testX.shape[0])], args['testSize'], replace=False)
    Xtest = data.testX.to(DEVICE)[test_index]
    Ytest = data.testY.squeeze().to(DEVICE)[test_index]
    n_process = 50
    start_time = time.time()
    _ = data.opt_solve(Xtest[:n_process], solver_type=solvers[0], tol=args['proj_para']['corrEps'])
    end_time = time.time()
    opt_time = (end_time - start_time)/n_process
    print(opt_time)
    # opt_time = 1.553299

    ave_inf_time_raw = epoch_stats['test_raw_time'] / Xtest.shape[0]
    ave_inf_time_proj = epoch_stats['test_proj_time'] / epoch_stats['test_num_infeasible']
    ave_inf_time = epoch_stats['test_time'] / Xtest.shape[0]
    # print(epoch_stats['test_raw_time'], epoch_stats['test_proj_time'])
    print('\n')
    print(f'ave_speed_up: {opt_time / ave_inf_time}, ave_proj_speed_up: {opt_time / ave_inf_time_proj}')
    # print(1/0)
    # epoch_stats['batch_solve_raw_time'] = ave_inf_time_raw
    # epoch_stats['batch_raw_speed_up'] = opt_time / ave_inf_time_raw
    # epoch_stats['batch_solve_proj_time'] = ave_inf_time_proj
    # epoch_stats['batch_proj_speed_up'] = opt_time / ave_inf_time_proj
    # epoch_stats['batch_solve_time'] = ave_inf_time
    # epoch_stats['batch_speed_up'] = opt_time / ave_inf_time
    #
    # csv_record(epoch_stats, data, args)
    #
    # with open(os.path.join(result_save_dir, 'test_stats.dict'), 'wb') as f:
    #     pickle.dump(epoch_stats, f)






###################################################################
# (Un)supervised Training for Graph-NN Solver
###################################################################
# Graph QP: node feature, adj matrix
# power_control: edge feature
# ACOPF: node feature, edge feature
def train_graph_nn_solver(data, args, save_dir):
    lr = args['nn_para']['lr']
    nepochs = args['nn_para']['total_iteration']
    batch_size = args['nn_para']['batch_size']
    lr_decay = args['nn_para']['lr_decay']
    lr_decay_step = args['nn_para']['lr_decay_step']
    training_appoach = args['nn_para']['approach']
    ### Equality completion
    if args['probType'] in ['graph_acopf', 'power_control']:
        n_out = data.xdim
    elif args['probType'] in ['graph_qp']:
        n_out = len(data.partial_vars_idx)
    n_in = data.cdim # node context variables
    e_in = data.edim # edge context variables
    n_layer = args['nn_para']['num_layer']
    n_hid = max(n_in+1, 64)
    solver_net = GraphNet(n_in, n_out, n_hid, e_in, n_layer, act='sigmoid')
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=lr, weight_decay=1e-5)
    solver_shce = optim.lr_scheduler.StepLR(solver_opt, step_size=lr_decay_step, gamma=lr_decay)
    stats = {}
    Ctrain = Ctest = Adjtrain = Adjtest = Etrain = Etest = None
    if data.trainAdj is not None:
        Adjtrain = data.trainAdj.to(DEVICE)
        Adjtest = data.testAdj.to(DEVICE)
    if data.trainE is not None:
        Etrain = data.trainE.to(DEVICE)
        Etest = data.testE.to(DEVICE)
    if data.trainC is not None:
        Ctrain = data.trainC.to(DEVICE)
        Ctest = data.testC.to(DEVICE)
    Xtrain = data.trainX.to(DEVICE)
    Xtest = data.testX.to(DEVICE)
    for i in range(nepochs + 1):
        solver_net.train()
        epoch_stats = {}
        batch_index = np.random.choice(np.arange(Ctrain.shape[0]), batch_size, replace=False)
        Ctrain_batch = Adjtrain_batch = Etrain_batch = None
        if Ctrain is not None:
            Ctrain_batch = Ctrain[batch_index]
        if Adjtrain is not None:
            if Adjtrain.shape[0] > 1:
                Adjtrain_batch = Adjtrain[batch_index]
            else:
                Adjtrain_batch = Adjtrain
        if Etrain is not None:
            if Etrain.shape[0] >1:
                Etrain_batch = Etrain[batch_index]
            else:
                Etrain_batch = Etrain

        Z_pred_batch = solver_net(Ctrain_batch, Etrain_batch, Adjtrain_batch)
        if 'acopf' in args['probType']:
            Ctrain_batch, Z_pred_batch = data.extend_prediction(Ctrain_batch, Z_pred_batch)
        Z_pred_scale_batch = data.scale(Ctrain_batch, Z_pred_batch)
        Y_pred_scale_batch = data.complete_partial(Ctrain_batch, Z_pred_scale_batch)
        # Y_pred_scale_batch = Z_pred_scale_batch
        if 'power_control' in args['probType']:
            training_obj = data.obj_fn(Etrain_batch, Y_pred_scale_batch)
            penalty = data.cal_penalty(Etrain_batch, Y_pred_scale_batch, Adjtrain_batch).sum(-1)
        else:
            training_obj = data.obj_fn(Y_pred_scale_batch)
            penalty = data.cal_penalty(Ctrain_batch, Y_pred_scale_batch, Adjtrain_batch).sum(-1)
        if training_appoach == 'supervise':
            mse_loss = ((Y_pred_scale_batch - Xtrain[batch_index]) ** 2).mean()
            train_loss = mse_loss \
                        + args['nn_para']['softWeightInEqFrac'] * penalty.mean() \
                        + args['nn_para']['objWeight'] * training_obj.mean()
        else:
            train_loss = args['nn_para']['softWeightInEqFrac'] * penalty.mean() \
                         + args['nn_para']['objWeight'] * training_obj.mean()
        train_loss.backward()
        solver_opt.step()
        solver_shce.step()
        solver_opt.zero_grad()

        # Print results
        if i % args['resultsSaveFreq'] == 0 and i > 0:
            solver_net.eval()
            with torch.no_grad():
                graph_eval_solution(data, Ctest, Etest, Xtest, Adjtest, solver_net, args, save_dir, 'test', epoch_stats)
            print('\nEpoch:{}, Fea_rate:{:.2f}, '
                  '\nRaw_loss: MSE({:.4f}), MAP({:.4f}), '
                  '\nRaw_obj: MSE({:.4f}), MAP({:.4f}), '
                  '\nRaw_vio: Max({:.4f})'.format(
                i, epoch_stats['test_raw_fea_rate'],
                np.mean(epoch_stats['test_raw_mse_loss']), np.mean(epoch_stats['test_raw_mape_loss']),
                np.mean(epoch_stats['test_raw_obj_mse']),  np.mean(epoch_stats['test_raw_obj_mape']),
                np.mean(epoch_stats['test_raw_vio'])))
            with open(os.path.join(save_dir, 'solver_net.pth'), 'wb') as f:
                torch.save(solver_net, f)
        if args['saveAllStats']:
            if i == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats
    with open(os.path.join(save_dir, 'train_stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    return solver_net, stats

def test_graph_nn_solver(data, args, model_save_dir, result_save_dir):
    print(args['probType'], args['projType'])
    args['proj_para']['useTestCorr'] = True
    ## Run pure optimization baselines
    # DEVICE = torch.device("cpu")
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE
    Xtest = data.testX.squeeze().to(DEVICE)
    if data.Ctest is not None:
        Adjtest = data.testAdj.to(DEVICE)
    if data.testE is not None:
        Etest = data.testE.to(DEVICE)
    if data.testC is not None:
        Ctest = data.testC.to(DEVICE)

    solver_net = torch.load(os.path.join(model_save_dir, 'solver_net.pth'), map_location=DEVICE)
    epoch_stats = {}
    with torch.no_grad():
        graph_eval_solution(data, Ctest, Etest, Xtest, Adjtest, solver_net, args, model_save_dir, 'test', epoch_stats)
    print('Raw_loss: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}),     Raw_obj: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}) \n'
          'Cor_loss: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}),     Cor_obj: MSE({:.4f}), MAE({:.4f}), MAP({:.4f}) \n'
          'Raw_Ineq: Max({:.4f}), Sum({:.4f}), Per({:.4f}),     Raw_eq:  Max({:.4f}), Sum({:.4f}), Per({:.4f})\n'
          'Cor_Ineq: Max({:.4f}), Sum({:.4f}), Per({:.4f}),     Cor_eq:  Max({:.4f}), Sum({:.4f}), Per({:.4f})\n'
          'Raw_Rate: FRate({:.4f}), Raw_inf: Batch({:.4f})\n'
          'Cor_Rate: FRate({:.4f}), Cor_inf: Batch({:.4f})'.format(
        np.mean(epoch_stats['test_raw_mse_loss']), np.mean(epoch_stats['test_raw_mae_loss']),
        np.mean(epoch_stats['test_raw_mape_loss']), np.mean(epoch_stats['test_raw_obj_mse']),
        np.mean(epoch_stats['test_raw_obj_mae']), np.mean(epoch_stats['test_raw_obj_mape']),
        np.mean(epoch_stats['test_cor_mse_loss']), np.mean(epoch_stats['test_cor_mae_loss']),
        np.mean(epoch_stats['test_cor_mape_loss']), np.mean(epoch_stats['test_cor_obj_mse']),
        np.mean(epoch_stats['test_cor_obj_mae']), np.mean(epoch_stats['test_cor_obj_mape']),
        np.mean(epoch_stats['test_raw_edge_max']), np.mean(epoch_stats['test_raw_edge_sum']),
        np.mean(epoch_stats['test_raw_edge_num_viol_0']) / data.num_node, np.mean(epoch_stats['test_raw_node_max']),
        np.mean(epoch_stats['test_raw_node_sum']), np.mean(epoch_stats['test_raw_node_num_viol_0']) / data.num_edge,
        np.mean(epoch_stats['test_cor_edge_max']), np.mean(epoch_stats['test_cor_edge_sum']),
        np.mean(epoch_stats['test_cor_edge_num_viol_0']) / data.num_node, np.mean(epoch_stats['test_cor_node_max']),
        np.mean(epoch_stats['test_cor_node_sum']), np.mean(epoch_stats['test_cor_node_num_viol_0']) / data.num_edge,
        1 - np.mean(epoch_stats['test_raw_vio_instance']), epoch_stats['test_raw_time'],
        1 - np.mean(epoch_stats['test_cor_vio_instance']), epoch_stats['test_proj_time']))
    with open(os.path.join(result_save_dir, 'test_stats.dict'), 'wb') as f:
        pickle.dump(epoch_stats, f)

def graph_eval_solution(data, C, E, Ytarget, Adjtest, solver_net, args, save_dir, prefix, stats):
    ### NN solution prediction
    raw_start_time = time.time()
    solver_net.eval()
    with torch.no_grad():
        Y_pred = solver_net(C, E, Adjtest)
        if 'acopf' in args['probType']:
            C, Y_pred = data.extend_prediction(C, Y_pred)
        Y_pred_scale = data.scale(C, Y_pred)
        Y = data.complete_partial(C, Y_pred_scale)
    raw_end_time = time.time()
    NN_pred_time = raw_end_time - raw_start_time
    ### Post-processing for infeasible only
    eps_converge = args['proj_para']['corrEps']
    if 'power_control' in args['probType']:
        violation = data.check_feasibility(E, Y, Adjtest).abs()
    else:
        violation = data.check_feasibility(C, Y, Adjtest).abs()
    # print(violation.shape)
    infeasible_index = (violation.max(-1)[0] > eps_converge).view(-1)
    Y_pred_infeasible = Y[infeasible_index]
    num_infeasible_prediction = Y_pred_infeasible.shape[0]
    Ycorr = Y.detach().clone()
    cor_start_time = time.time()
    if args['proj_para']['useTestCorr'] and 'H_Proj' in args['algoType']:
        homeo_mapping = torch.load(os.path.join(save_dir, 'mdh_mapping.pth'), map_location=DEVICE)
        homeo_mapping.eval()
    if num_infeasible_prediction > 0:
        if args['proj_para']['useTestCorr']:
            if 'H_Proj' in args['algoType']:
                Yproj, steps = homeo_bisection(homeo_mapping, data, args, Y_pred[infeasible_index], C[infeasible_index])
            # elif 'G_Proj' in args['algoType']:
            #     Yproj, steps = gauge_bisection(homeo_mapping, data, args, Y_pred[infeasible_index], X[infeasible_index])
            # elif 'B_Proj' in args['algoType']:
            #     Yproj, _ = ip_bisection(feasible_ip, data, args, Y_pred[infeasible_index], X[infeasible_index])
            elif 'D_Proj' in args['algoType']:
                Yproj, steps = diff_projection(data, C[infeasible_index], Y[infeasible_index], args)
            elif 'Proj' in args['algoType']:
                Yproj = data.opt_proj(C[infeasible_index], Y[infeasible_index]).to(Y.device)
            elif 'WS' in args['algoType']:
                Yproj = data.opt_warmstart(C[infeasible_index], Y[infeasible_index]).to(Y.device)
            else:
                Yproj = Y_pred_infeasible
            Ycorr[infeasible_index] = Yproj
    cor_end_time = time.time()
    Proj_time = cor_end_time - cor_start_time


    make_prefix = lambda x: "{}_{}".format(prefix, x)
    dict_agg(stats, make_prefix('raw_time'), NN_pred_time, op='sum')
    dict_agg(stats, make_prefix('proj_time'), Proj_time, op='sum')
    # dict_agg(stats, make_prefix('steps'), np.array([steps]))

    dict_agg(stats, make_prefix('num_infeasible'), num_infeasible_prediction)
    dict_agg(stats, make_prefix('index_infeasible'), infeasible_index.detach().cpu().numpy())

    if 'power_control' in args['probType']:
        Y_obj = data.obj_fn(E, Y).detach().cpu()
        Ycor_obj = data.obj_fn(E, Ycorr).detach().cpu()
        Ytarget_obj = data.obj_fn(E, Ytarget).detach().cpu()
        raw_vio = torch.abs(data.check_feasibility(E, Y, Adjtest)).detach().cpu()
        raw_fea_rate = (torch.count_nonzero(raw_vio.max(-1)[0] < eps_converge) / C.shape[0]).item()
        cor_vio = torch.abs(data.check_feasibility(E, Ycorr, Adjtest)).detach().cpu()
        cor_fea_rate = (torch.count_nonzero(cor_vio.max(-1)[0] < eps_converge) / C.shape[0]).item()
    else:
        Y_obj = data.obj_fn(Y).detach().cpu()
        Ycor_obj = data.obj_fn(Ycorr).detach().cpu()
        Ytarget_obj = data.obj_fn(Ytarget).detach().cpu()
        raw_vio = torch.abs(data.check_feasibility(C, Y, Adjtest)).detach().cpu()
        raw_fea_rate = (torch.count_nonzero(raw_vio.max(-1)[0] < eps_converge) / C.shape[0]).item()
        cor_vio = torch.abs(data.check_feasibility(C, Ycorr, Adjtest)).detach().cpu()
        cor_fea_rate = (torch.count_nonzero(cor_vio.max(-1)[0] < eps_converge) / C.shape[0]).item()

    # raw_edge_vio = torch.abs(data.edge_penalty(X, Y, Adjtest)).detach().cpu()
    # raw_edge_vio = raw_edge_vio.view(raw_edge_vio.shape[0], -1)
    # raw_node_vio = torch.abs(data.node_penalty(X, Y)).detach().cpu()
    # raw_node_vio = raw_node_vio.view(raw_node_vio.shape[0], -1)
    # raw_vio = torch.abs(data.check_feasibility(X, Y, Adjtest)).detach().cpu()
    # raw_fea_rate = (torch.count_nonzero(raw_vio.max(-1)[0] < eps_converge) / X.shape[0]).item()

    # cor_edge_vio = torch.abs(data.edge_penalty(X, Ycorr, Adjtest)).detach().cpu()
    # cor_edge_vio = cor_edge_vio.view(cor_edge_vio.shape[0], -1)
    # cor_node_vio = torch.abs(data.node_penalty(X, Ycorr)).detach().cpu()
    # cor_node_vio = cor_node_vio.view(cor_node_vio.shape[0], -1)
    # cor_vio = torch.abs(data.check_feasibility(X, Ycorr, Adjtest)).detach().cpu()
    # cor_fea_rate = (torch.count_nonzero(cor_vio.max(-1)[0] < eps_converge) / X.shape[0]).item()

    Y = Y.detach().cpu()
    C = C.detach().cpu()
    Ycorr = Ycorr.detach().cpu()
    Ytarget = Ytarget.detach().cpu()

    solution_res = (Y - Ytarget).view(Ytarget.shape[0], -1)
    proj_solution_res = (Ycorr - Ytarget).view(Ytarget.shape[0], -1)
    target_solution_norm = torch.norm(Ytarget.view(Ytarget.shape[0], -1), dim=1, p=1)
    cor_dist = (Ycorr - Y).view(Ytarget.shape[0], -1)


    raw_mae_loss = torch.norm(solution_res, dim=1, p=1)
    raw_mse_loss = torch.norm(solution_res, dim=1, p=2) ** 2
    raw_mape_loss = raw_mae_loss / target_solution_norm
    dict_agg(stats, make_prefix('raw_mae_loss'), raw_mae_loss.numpy())
    dict_agg(stats, make_prefix('raw_mse_loss'), raw_mse_loss.numpy())
    dict_agg(stats, make_prefix('raw_mape_loss'), raw_mape_loss.numpy())

    cor_mae_loss = torch.norm(proj_solution_res, dim=1, p=1)
    cor_mse_loss = torch.norm(proj_solution_res, dim=1, p=2) ** 2
    cor_mape_loss = cor_mae_loss / target_solution_norm
    dict_agg(stats, make_prefix('cor_mae_loss'), cor_mae_loss.numpy())
    dict_agg(stats, make_prefix('cor_mse_loss'), cor_mse_loss.numpy())
    dict_agg(stats, make_prefix('cor_mape_loss'), cor_mape_loss.numpy())

    raw_mae_loss = torch.norm(solution_res, dim=1, p=1)
    raw_mse_loss = torch.norm(solution_res, dim=1, p=2) ** 2
    raw_mape_loss = raw_mae_loss / target_solution_norm
    dict_agg(stats, make_prefix('raw_mae_loss'), raw_mae_loss.numpy())
    dict_agg(stats, make_prefix('raw_mse_loss'), raw_mse_loss.numpy())
    dict_agg(stats, make_prefix('raw_mape_loss'), raw_mape_loss.numpy())
    dict_agg(stats, make_prefix('cor_obj_mae'), (Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mse'), torch.square(Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mape'), ((Ycor_obj - Ytarget_obj) / torch.abs(Ytarget_obj)).numpy())


    obj_res = Y_obj-Ytarget_obj
    dict_agg(stats, make_prefix('raw_eval'), Y_obj.numpy())
    dict_agg(stats, make_prefix('real_eval'), Ytarget_obj.numpy())
    dict_agg(stats, make_prefix('raw_obj_mae'), (Y_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('raw_obj_mse'), torch.square(Y_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('raw_obj_mape'), ((Y_obj - Ytarget_obj) / torch.abs(Ytarget_obj)).numpy())
    dict_agg(stats, make_prefix('cor_obj_mae'), (Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mse'), torch.square(Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mape'), ((Ycor_obj - Ytarget_obj) / torch.abs(Ytarget_obj)).numpy())

    # dict_agg(stats, make_prefix('raw_edge_max'), torch.max(raw_edge_vio, dim=1)[0].numpy())
    # dict_agg(stats, make_prefix('raw_edge_mean'), torch.mean(raw_edge_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('raw_edge_sum'), torch.sum(raw_edge_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('raw_edge_num_viol_0'), torch.sum(raw_edge_vio > eps_converge, dim=1).numpy())
    # dict_agg(stats, make_prefix('raw_node_max'), torch.max(raw_node_vio, dim=1)[0].numpy())
    # dict_agg(stats, make_prefix('raw_node_mean'), torch.mean(raw_node_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('raw_node_sum'), torch.sum(raw_node_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('raw_node_num_viol_0'), torch.sum(raw_node_vio > eps_converge, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_vio'), (raw_vio.sum(-1)).numpy())
    dict_agg(stats, make_prefix('raw_vio_instance'), (raw_vio > eps_converge).numpy())
    dict_agg(stats, make_prefix('raw_fea_rate'), raw_fea_rate)

    # dict_agg(stats, make_prefix('cor_edge_max'), torch.max(cor_edge_vio, dim=1)[0].numpy())
    # dict_agg(stats, make_prefix('cor_edge_mean'), torch.mean(cor_edge_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('cor_edge_sum'), torch.sum(cor_edge_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('cor_edge_num_viol_0'), torch.sum(cor_edge_vio > eps_converge, dim=1).numpy())
    # dict_agg(stats, make_prefix('cor_node_max'), torch.max(cor_node_vio, dim=1)[0].numpy())
    # dict_agg(stats, make_prefix('cor_node_mean'), torch.mean(cor_node_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('cor_node_sum'), torch.sum(cor_node_vio, dim=1).numpy())
    # dict_agg(stats, make_prefix('cor_node_num_viol_0'), torch.sum(cor_node_vio > eps_converge, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_vio'), (cor_vio.sum(-1)).numpy())
    dict_agg(stats, make_prefix('cor_vio_instance'), (torch.max(cor_vio, dim=1)[0] > eps_converge).numpy())
    dict_agg(stats, make_prefix('cor_fea_rate'), cor_fea_rate)


    return stats

def train_graph_mdh_mapping(data, args, save_dir):
    paras = args['inn_para']
    probType = args['probType']
    c_dim = data.cdim
    e_dim = data.edim # edge context variables
    if probType in ['graph_acopf', 'power_control']:
        n_out = data.xdim
    elif probType in ['graph_qp']:
        n_out = len(data.partial_vars_idx)
    n_node = data.num_node
    #### Flow-based model: ball -> constraint set
    num_node = data.num_node
    num_layer = paras['num_layer']
    h_dim = max(n_out+1, 64)
    model = GINN(num_node, n_out, h_dim, c_dim, e_dim, num_layer, outact='sigmoid').to(device=DEVICE)
    #### Sampling input parameters and output decision
    initial_shape = paras['shape']
    bound = paras['bound']  # initial shape to [0,1] bounded constraint set
    c_train_tensor = e_train_tensor = x_train_tensor = adj_train_tensor = None
    if probType in ['graph_qp']:
        x_train = sampling_body(paras['n_samples'], n_out*n_node, initial_shape, lu=bound)
        x_train_tensor = torch.tensor(x_train).view(-1, n_node, n_out).to(device=DEVICE).double()
        c_train_tensor = torch.rand([paras['c_samples'], n_node, c_dim]).to(device=DEVICE).double()
        c_train_tensor = c_train_tensor * (data.input_U - data.input_L) + data.input_L
        adj_train_tensor = data.trainAdj.double()
        e_train_tensor = data.trainE

    elif probType in ['power_control']:
        x_train = sampling_body(paras['n_samples'], n_out*n_node, initial_shape, lu=bound)
        x_train_tensor = torch.tensor(x_train).view(-1, n_node, n_out).to(device=DEVICE).double()
        adj_train_tensor = data.trainAdj
        c_train_tensor = data.trainC
        e_train_tensor = data.trainE

    elif probType in ['graph_acopf']:
        x_train = sampling_body(paras['n_samples'], len(data.partial_vars_idx), initial_shape, lu=bound)
        x_train_tensor = torch.tensor(x_train).view(-1, len(data.partial_vars_idx)).to(device=DEVICE).double()
        c_train_tensor = torch.rand([paras['c_samples'], data.nb*2]).to(device=DEVICE).double()
        c_train_tensor = c_train_tensor * (data.input_U - data.input_L) + data.input_L
        adj_train_tensor = data.trainAdj
        e_train_tensor = data.trainE
        c_train_tensor, x_train_tensor = data.graph_prediction(c_train_tensor, x_train_tensor)
    #### Unsupervised Training for Homeo Mapping
    model, volume_list, penalty_list = unsupervised_training_graph_mdh(model, data, probType,
                                                                       x_train_tensor, c_train_tensor,
                                                                       e_train_tensor, adj_train_tensor,
                                                                       paras)
    # plot_convergence(volume_list, penalty_list, dist_list, trans_list, save_dir, args)
    torch.save(model, os.path.join(save_dir, 'mdh_mapping.pth'))

def unsupervised_training_graph_mdh(model, constraints, probType, x_tensor, c_tensor, e_tensor, adj_tensor, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    volume_list = []
    penalty_list = []
    distortion_list = []
    bias_tensor = x_tensor[:batch_size]
    bias_tensor[bias_tensor!=0] = np.mean(args['bound'])
    # torch.ones_like(x_tensor[:batch_size]).to(x_tensor.device) * np.mean(args['bound'])
    optimizer = optim.Adam(model.parameters(),
                           lr=args['lr'],
                           weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=args['lr_decay_step'],
                            gamma=args['lr_decay'])
    for n in range(total_iteration):
        model.train()
        optimizer.zero_grad()
        batch_index = np.random.choice([i for i in range(x_tensor.shape[0])], batch_size, replace=True)
        x_input = x_tensor[batch_index]

        c_input = adj_input = e_input = None
        if c_tensor is not None:
            batch_index = np.random.choice([i for i in range(c_tensor.shape[0])], batch_size, replace=True)
            c_input = c_tensor[batch_index]
        if adj_tensor is not None:
            if adj_tensor.shape[0] > 1:
                batch_index = np.random.choice([i for i in range(adj_tensor.shape[0])], batch_size, replace=True)
                adj_input = adj_tensor[batch_index]
            else:
                adj_input = adj_tensor
        if e_tensor is not None:
            if e_tensor.shape[0] > 1:
                batch_index = np.random.choice([i for i in range(e_tensor.shape[0])], batch_size, replace=True)
                e_input = e_tensor[batch_index]
            else:
                e_input = e_tensor

        n_dim = x_input.shape[1] * x_input.shape[2]
        xt, logdet, logdis = model(x_input, c_input, e_input, adj_input)
        if 'acopf' in probType:
            c_input, xt = constraints.extend_prediction(c_input, xt)
        volume = logdet
        xt_scale = constraints.scale(c_input, xt)
        xt_full = constraints.complete_partial(c_input, xt_scale)
        if 'power_control' in probType:
            penalty = constraints.cal_penalty(e_input, xt_full, adj_input).sum(-1)
        else:
            penalty = constraints.cal_penalty(c_input, xt_full, adj_input).sum(-1)
        loss =  ( torch.mean(logdis) - torch.mean(volume) ) / n_dim \
                + args['penalty_coefficient'] * penalty.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        volume_list.append(torch.mean(logdet).detach().cpu().numpy() / n_dim)
        penalty_list.append(torch.mean(penalty).detach().cpu().numpy())
        distortion_list.append(torch.mean(logdis).detach().cpu().numpy() / n_dim)
        if n % 100 == 0 and n > 0:
            model.eval()
            with torch.no_grad():
                # if 'acopf' in probType:
                #     c_input, x0 = constraints.graph_prediction(c_input, bias_tensor)
                batch_index = np.random.choice([i for i in range(c_tensor.shape[0])], batch_size, replace=True)
                c_input = c_tensor[batch_index]
                x0, _, _ = model(bias_tensor, c_input, e_input, adj_input)
                if 'acopf' in probType:
                    c_input, x0 = constraints.extend_prediction(c_input, x0)
                xt_scale = constraints.scale(c_input, x0)
                xt_full = constraints.complete_partial(c_input, xt_scale)
                if 'power_control' in probType:
                    penalty_0 = constraints.check_feasibility(e_input, xt_full, adj_input)
                else:
                    penalty_0 = constraints.check_feasibility(c_input, xt_full, adj_input)
            print(f'Iteration: {n}/{total_iteration}, '
                  f'Volume: {volume_list[-1]:.4f}, '
                  f'Disotrtion: {distortion_list[-1]:.4f}, '
                  f'Penalty: {penalty_list[-1]:.4f}, '
                  f'Center Penalty: {penalty_0.sum(-1).cpu().numpy().mean():.4f}, '
                  f'Valid rate: {(penalty_0.max(-1)[0] < 1e-5).cpu().numpy().mean():.2f}',
                  end='\n')
    return model, volume_list, penalty_list






###################################################################
# Unsupervised Training for Minimum-Distortion-Homeomoprhic Mapping
###################################################################
def unsupervised_training_mdh(model, constraints, x_tensor, c_tensor, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    penalty_coefficient = args['penalty_coefficient']
    distortion_coefficient = args['distortion_coefficient']
    volume_list = []
    penalty_list = []
    dist_list = []
    trans_list = []
    bias_tensor = torch.ones(batch_size, x_tensor.shape[1]).to(x_tensor.device) * np.mean(args['bound'])
    optimizer = optim.Adam(model.parameters(),
                           lr=args['lr'],
                           weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=args['lr_decay_step'],
                            gamma=args['lr_decay'])
    for n in range(total_iteration):
        model.train()
        optimizer.zero_grad()
        batch_index = np.random.choice([i for i in range(x_tensor.shape[0])],  batch_size, replace=True)
        x_input = x_tensor[batch_index]
        batch_index = np.random.choice([i for i in range(c_tensor.shape[0])], batch_size, replace=True)
        c_input = c_tensor[batch_index]
        n_dim = x_input.shape[1]
        if args['scale_ratio']>1:
            xt, logdet, _ = model(x_input, c_input)
            _, _, logdis = model((x_input-bias_tensor)*args['scale_ratio']+bias_tensor, c_input)
        else:
            xt, logdet, logdis = model(x_input, c_input)
        volume = logdet
        xt_scale = constraints.scale(c_input, xt)
        xt_full = constraints.complete_partial(c_input, xt_scale)
        # eq_resid = constraints.eq_resid(c_input, xt_full).abs()
        # ineq_resid = constraints.ineq_resid(c_input, xt_full)
        penalty = constraints.cal_penalty(c_input, xt_full).sum(-1)
        loss =  ( torch.mean(logdis) - torch.mean(volume) ) / n_dim  \
                + penalty_coefficient * torch.mean(penalty[penalty > 1e-9])
        loss.backward()
        optimizer.step()
        scheduler.step()
        volume_list.append(torch.mean(logdet).detach().cpu().numpy()/n_dim)
        penalty_list.append(torch.mean(penalty).detach().cpu().numpy())
        dist_list.append(torch.mean(logdis).detach().cpu().numpy()/n_dim)

        if n%1000==0 and n>0:
            model.eval()
            with torch.no_grad():
            # bias_tensor.requires_grad = True
                x0,_,_ = model(bias_tensor, c_input)
                x0_scale = constraints.scale(c_input, x0)
                x0_full = constraints.complete_partial(c_input, x0_scale)
                violation_0 = constraints.check_feasibility(c_input, x0_full)
                penalty_0 = torch.abs(violation_0).detach().cpu()
            print(f'Iteration: {n}/{total_iteration}, '
                  f'Volume: {volume_list[-1]:.3f}, '
                  f'Penalty: {penalty_list[-1]:.3f}, '
                  f'Distortion: {dist_list[-1]:.3f}, '
                  f'Center Penalty: {penalty_0.sum(-1).mean().numpy():.4f}, '
                  f'Valid rate: {(penalty_0.max(-1)[0] < 1e-5).numpy().mean():.2f}',
                  end='\n')
    return model, volume_list, penalty_list, dist_list, trans_list

def train_mdh_mapping(data, args, save_dir):
    paras = args['inn_para']
    c_dim = data.xdim
    n_dim = len(data.partial_vars_idx)
    #### Flow-based model: ball -> constraint set
    num_layer = paras['num_layer']
    inv_type = paras['inv_type']
    h_dim = max(n_dim+1, 64)
    model = INN(n_dim, h_dim, c_dim, num_layer, inv=inv_type, outact='sigmoid', bilip=paras['bilip'], lip=paras['L']).to(device=DEVICE)
    #### Sampling input parameters and output decision
    initial_shape = paras['shape']
    bound = paras['bound']  # initial shape to [0,1] bounded constraint set
    # x_train = sampling_body(paras['n_samples'], n_dim, initial_shape, lu=bound)
    # x_train_tensor = torch.tensor(x_train).view(-1, n_dim).to(device=DEVICE).double()
    # c_train_tensor = torch.rand([paras['c_samples'], c_dim]).to(device=DEVICE).double()
    # c_train_tensor = c_train_tensor * (data.input_U - data.input_L) + data.input_L
    x_train_tensor = torch.rand(size=[paras['n_samples'], n_dim]).to(device=DEVICE).double() * (bound[1] - bound[0]) + bound[0]
    c_train_tensor = data.trainX.mean(0) + data.trainX.std(0) * torch.randn(size=[paras['c_samples'], c_dim]).to(device=DEVICE)
    #### Unsupervised Training for Homeo Mapping
    # model = pre_training_mdh(model, data, data.trainY, data.trainX, paras)
    model, volume_list, penalty_list, dist_list, trans_list = unsupervised_training_mdh(model, data,
                                                                       x_train_tensor, c_train_tensor,
                                                                       paras)
    # plot_convergence(volume_list, penalty_list, dist_list, trans_list, save_dir, args)
    torch.save(model, os.path.join(save_dir, 'mdh_mapping.pth'))

def test_mdh_mapping(data, save_dir, args):
    paras = args['inn_para']
    homeo_mapping = torch.load(os.path.join(save_dir, 'mdh_mapping.pth'), map_location=DEVICE)
    ### input pparameters --> output solutions
    c_tensor = data.X.squeeze()
    x_tensor = data.Y[:, data.partial_vars_idx].squeeze()
    c_samples, c_dim = c_tensor.shape
    n_samples, n_dim = x_tensor.shape
    #### Sampling input parameters and output decision
    initial_shape = paras['shape']
    bound = paras['bound']
    test_n_dim = paras['testing_samples']
    c_tensor = torch.rand([test_n_dim, c_dim]).to(device=DEVICE)
    c_tensor = c_tensor * (data.input_U - data.input_L) + data.input_L
    x_surface = (sampling_surface(test_n_dim, n_dim, initial_shape, lu=bound) - np.mean(bound)) \
        * np.random.uniform(0.7, 0.9, size=[test_n_dim, n_dim]) + np.mean(bound)
    x_surface = torch.tensor(x_surface).view(-1, n_dim).to(device=DEVICE)
    with torch.no_grad():
        x_tensor, _, _ = homeo_mapping(x_surface, c_tensor[:x_surface.shape[0]])
    # args['proj_para']['corrTestMaxSteps'] = 50
    # args['proj_para']['corrBis'] = 0.5
    # scatter_projection_error(homeo_mapping, data, x_tensor, c_tensor, save_dir, args)






###################################################################
# Unsupervised Training for Minimum-Eccentricity-IP Mapping
###################################################################
def general_boundary_sampling(data, c_tensor, feasible_ip, num_boundary_point, bisect_step=10):
    batch_size = feasible_ip.shape[0]
    n_ip = feasible_ip.shape[1]
    n_dim = feasible_ip.shape[-1]
    c_dim = c_tensor.shape[-1]
    # direction sampling
    unit_vec = torch.randn(size=[batch_size * n_ip * num_boundary_point, n_dim]).to(feasible_ip.device)
    # unit_vec = unit_vec/torch.norm(unit_vec, dim=-1, keepdim=True)
    # repeat for multiple IPs and extend to 2-dim vectors
    feasible_ip = feasible_ip.view(batch_size, n_ip, 1, n_dim).repeat(1, 1, num_boundary_point, 1)
    feasible_ip_extend = feasible_ip.view(-1, n_dim)
    c_tensor = c_tensor.view(batch_size, n_ip, 1, c_dim).repeat(1, 1, num_boundary_point, 1)
    c_tensor_extend = c_tensor.view(-1, c_dim)
    # lower bound and upper bound for bisection
    al = torch.zeros(size=[batch_size * n_ip * num_boundary_point, 1]).to(feasible_ip.device)
    au = torch.ones(size=[batch_size * n_ip * num_boundary_point, 1]).to(feasible_ip.device)
    for _ in range(bisect_step):
        a = au.clone()
        xt = feasible_ip_extend + unit_vec * a
        xt_scale = data.scale(c_tensor_extend, xt)
        xt_full = data.complete_partial(c_tensor_extend, xt_scale)
        penalty = data.check_feasibility(c_tensor_extend, xt_full).sum(dim=1)
        # penalty = penalty.view(num_feasible_points, num_interior_points, num_boundary_point, 1)
        infeasible_index = penalty > 1e-5
        feasible_index = penalty <= 1e-5
        au[infeasible_index] = (au[infeasible_index] + al[infeasible_index]) / 2
        al[feasible_index] = a[feasible_index]
        au[feasible_index] *= 2
    boundary_samples = feasible_ip_extend + unit_vec * al
    return boundary_samples.view(batch_size, 1, n_ip * num_boundary_point, n_dim)

def logsumexp(x, omega):
    return (torch.logsumexp(x * omega, dim=1))/omega

def unsupervised_training_meip(model, constraints, n_dim, c_tensor, args):
    n_ip = args['n_ip']
    total_iteration = args['total_iteration']
    batch_size = args['batch_size']
    num_boundary_point = args['n_boundary_point']
    num_bisect_step = args['n_bisect_sampling']
    minimum_ecc = args['minimum_ecc']
    c_dim = c_tensor.shape[1]
    optimizer = optim.Adam(model.parameters(),
                           lr=args['lr'],
                           weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=args['lr_decay_step'],
                            gamma=args['lr_decay'])
    penalty_list = []
    eccentricity_list = []
    training_time_list = []
    model.train()
    for n in range(total_iteration):
        iter_st = time.time()
        optimizer.zero_grad()
        batch_index = np.random.choice(np.arange(c_tensor.shape[0]), batch_size)
        c_batch = c_tensor[batch_index]
        xt = model(c_batch)

        c_batch_extend = c_batch.view(-1, 1, c_dim).repeat(1, n_ip, 1)
        c_batch_extend = c_batch_extend.view(-1, c_dim)

        ip_batch = xt.view(-1, n_ip, n_dim)
        ip_batch_extend = ip_batch.view(-1, n_dim)

        xt_scale = constraints.scale(c_batch_extend, ip_batch_extend)
        xt_full = constraints.complete_partial(c_batch_extend, xt_scale)
        violation = constraints.cal_penalty(c_batch_extend, xt_full)
        penalty = torch.sum(torch.abs(violation), dim=-1)
        penalty = penalty.view(-1, n_ip).sum(dim=1)
        ### infeasible loss function
        infeasible_index = penalty > 1e-5
        infeasible_loss = (penalty[infeasible_index]).mean()
        ### feasible loss function
        feasible_index = penalty <= 1e-5
        eccentric_dist = 0
        feasible_loss = 0
        sample_time =  0
        if minimum_ecc and ip_batch[feasible_index].shape[0]>0:
            feasible_ip = ip_batch[feasible_index]
            feasible_input = c_batch[feasible_index]
            ### boudary sampling
            st = time.time()
            ip_index = torch.randint(high=n_ip, size=(feasible_ip.shape[0],1,1)).repeat(1,1,feasible_ip.shape[-1]).to(feasible_ip.device)
            sampled_ip = feasible_ip.gather(1, ip_index).detach()
            sampled_input = feasible_input.view(-1,1,c_dim)
            boundary_point = general_boundary_sampling(constraints,
                                                       sampled_input,
                                                       sampled_ip,
                                                       num_boundary_point,
                                                       num_bisect_step)
            et = time.time()
            point_to_boundary_dist = torch.norm(feasible_ip.view(-1, n_ip, 1, n_dim) - boundary_point, dim=-1, p=2)
            ### cal eccentric dist
            if args['softmin']:
                point_to_boundary_dist_min = logsumexp(point_to_boundary_dist, omega=-(n + 1)/5)
            else:
                point_to_boundary_dist_min = point_to_boundary_dist.min(1)[0]
            if args['softrange']:
                dist_max = logsumexp(point_to_boundary_dist_min, omega=(n + 1)/5)
                dist_min = logsumexp(point_to_boundary_dist_min, omega=-(n + 1)/5)
            else:
                dist_max = point_to_boundary_dist_min.max(-1)[0]
                dist_min = point_to_boundary_dist_min.min(-1)[0]

            ### cal loss function
            feasible_loss = (dist_max - dist_min).mean()
            eccentric_dist = ((point_to_boundary_dist.min(1)[0]).max(1)[0]
                              - (point_to_boundary_dist.min(1)[0]).min(1)[0]).mean()
            eccentricity_list.append(eccentric_dist.detach().cpu())
            sample_time = et-st
        else:
            eccentricity_list.append(np.nan)
        penalty_list.append(infeasible_loss.detach().cpu())
        loss = args['w_penalty'] * infeasible_loss + n_ip*args['w_ecc'] * feasible_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        iter_et = time.time()
        training_time_list.append(iter_et - iter_st)
        if (n + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                batch_index = np.random.choice(np.arange(c_tensor.shape[0]), batch_size)
                c_batch = c_tensor[batch_index]
                xt = model(c_batch)
                c_batch = c_batch.view(-1, 1, c_dim).repeat(1, n_ip, 1)
                ip_batch = xt.view(-1, n_ip, n_dim)
                c_batch_extend = c_batch.view(-1, c_dim)
                ip_batch_extend = ip_batch.view(-1, n_dim)
                xt_scale = constraints.scale(c_batch_extend, ip_batch_extend)
                xt_full = constraints.complete_partial(c_batch_extend, xt_scale)
                violation = constraints.check_feasibility(c_batch_extend, xt_full)
                penalty = torch.sum(torch.abs(violation), dim=-1)
                print(f'n_ip: {n_ip}, iteration: {n}, valid: {(penalty > 1e-5).cpu().numpy().mean():.4f}, ecc: {eccentricity_list[-1]:.4f}, sampling time: {sample_time:.4f}, training_time: {np.mean(training_time_list)}')
    return model, penalty_list, eccentricity_list

def train_meip_mapping(data, args, save_dir):
    paras = args['ipnn_para']
    n_ip = paras['n_ip']
    minimum_ecc = paras['minimum_ecc']
    c_dim = data.xdim
    n_dim = len(data.partial_vars_idx)
    #### IPNN: input to multiple IPs
    model = ResNet(c_dim, n_dim * paras['n_ip'], data.intrin_dim+1, paras['n_layer'], act='sigmoid').to(DEVICE)
    c_tensor = torch.rand([paras['c_samples'], c_dim]).to(DEVICE)
    c_tensor = c_tensor * (data.input_U - data.input_L) + data.input_L
    model, penalty_list, eccentricity_list = unsupervised_training_meip(model, data, n_dim, c_tensor, paras)
    torch.save(model, os.path.join(save_dir, f'ipnn_{n_ip}_{minimum_ecc}.pth'))






###################################################################
# Unsupervised Training for Minimum-Distortion-Gauge Mapping
###################################################################
def unsupervised_training_mdg(model, constraints, x_tensor, xs_tensor, c_tensor, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    penalty_coefficient = args['penalty_coefficient']
    distortion_coefficient = args['distortion_coefficient']
    transport_coefficient = args['transport_coefficient']
    volume_list = []
    penalty_list = []
    dist_list = []
    trans_list = []
    optimizer = optim.Adam(model.parameters(),
                           lr=args['lr'],
                           weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=args['lr_decay_step'],
                            gamma=args['lr_decay'])
    model.train()
    for n in range(total_iteration):
        optimizer.zero_grad()
        batch_index = np.random.choice([i for i in range(x_tensor.shape[0])],  batch_size, replace=True)
        x_input = x_tensor[batch_index]
        xs_input = xs_tensor[batch_index]
        batch_index = np.random.choice([i for i in range(c_tensor.shape[0])], batch_size, replace=True)
        c_input = c_tensor[batch_index]
        n_dim = x_input.shape[1]

        xt = model(x_input, c_input)
        # x0 = model.interior_forward(c_input)
        # xt = torch.cat([xt,x0], dim=-1)
        logdis = model.distortion_forward(x_input, c_input)
        volume = model.scaling_forward(xs_input, c_input).mean(dim=1)
        # trans = torch.mean((x_input - xt) ** 2, dim=1, keepdim=True)

        xt_scale = constraints.scale(c_input, xt)
        xt_full = constraints.complete_partial(c_input, xt_scale)
        violation = constraints.cal_penalty(c_input, xt_full)
        penalty = torch.sum(torch.abs(violation), dim=-1, keepdim=True)
        loss = -  torch.mean(volume)   \
                +  penalty_coefficient * torch.mean(penalty) \
                +  distortion_coefficient * torch.mean(logdis) \
                # +  transport_coefficient * torch.mean(trans)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        scheduler.step()
        volume_list.append(torch.mean(volume).detach().cpu().numpy())
        penalty_list.append(torch.mean(penalty).detach().cpu().numpy())
        dist_list.append(torch.mean(logdis).detach().cpu().numpy())
        # trans_list.append(torch.mean(trans).detach().cpu().numpy())
        if n%1000==0 and n>0:
            model.eval()
            with torch.no_grad():
            # bias_tensor.requires_grad = True
                x0 = model.interior_nn(c_input)
                x0_scale = constraints.scale(c_input, x0)
                x0_full = constraints.complete_partial(c_input, x0_scale)
                violation_0 = constraints.check_feasibility(c_input, x0_full)
                penalty_0 = torch.sum(torch.abs(violation_0), dim=-1, keepdim=True)
            print(f'Iteration: {n}/{total_iteration}, '
                  f'Volume: {volume_list[-1]:.4f}, '
                  f'Penalty: {penalty_list[-1]:.4f}, '
                  f'Distortion: {dist_list[-1]:.4f}, '
                #   f'Transport: {trans_list[-1]:.4f}, '
                  f'test: {torch.mean(penalty_0).detach().cpu().numpy():.8f}',
                  end='\n')
    return model, volume_list, penalty_list, dist_list, trans_list

def train_mdg_mapping(data, args, save_dir):
    paras = args['inn_para']
    ### input pparameters --> output solutions
    c_tensor = data.X.squeeze()
    x_tensor = data.Y[:, data.partial_vars_idx].squeeze()
    c_samples, c_dim = c_tensor.shape
    n_samples, n_dim = x_tensor.shape
    #### Gauge-NN: ball -> constraint set
    model = GaugeNN(n_dim, c_dim, (n_dim+c_dim)).to(device=DEVICE)
    #### Sampling input parameters and output decision
    initial_shape = 'square'
    bound = [-1,1]  # initial shape to [0,1] bounded constraint set
    x_train = sampling_body(paras['n_samples'], n_dim, initial_shape, lu=bound)
    xs_train = sampling_surface(paras['n_samples'], n_dim, initial_shape, lu=bound)
    x_train_tensor = torch.as_tensor(x_train).view(-1, n_dim).to(device=DEVICE)
    xs_train_tensor = torch.as_tensor(xs_train).view(-1, n_dim).to(device=DEVICE)
    c_train_tensor = torch.rand([paras['c_samples'], c_dim]).to(device=DEVICE)
    c_train_tensor = c_train_tensor * (data.input_U - data.input_L) + data.input_L

    #### Unsupervised Training for Gauge Mapping
    model, volume_list, penalty_list, dist_list, trans_list \
                        = unsupervised_training_mdg(model, data, x_train_tensor, xs_train_tensor, c_train_tensor, paras)
    # plot_convergence(volume_list, penalty_list, dist_list, trans_list, save_dir)
    torch.save(model, os.path.join(save_dir, 'mdg_mdh_mapping.pth'))






###################################################################
# Other useful functions
###################################################################
# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value


import pandas as pd
import os
def csv_record(epoch_stats, data, args):
    record_file = '../results/all_bp_record.csv'
    labels = ['Prob', 'Algo',
              'Fea_rate',
              'Ineq_vio', 'Ineq_vio_rate',
              'Eq_vio', 'Eq_vio_rate',
              'Sol_MAE', 'Sol_MAPE', 'Infea_Sol_MAE', 'Infea_Sol_MAPE',
              'Obj_MAE', 'Obj_MAPE', 'Infea_Obj_MAE', 'Infea_Obj_MAPE',
              'Ave_time', 'Ave_speedup',
              'Ave_porj_time', 'Ave_proj_sppedup',
              'Ave_raw_time', 'Ave_raw_speedup']
    if not os.path.exists(record_file):
        data_record = pd.DataFrame(columns=labels)
        data_record.loc[0] = [str(0)]*len(labels)
    else:
        data_record = pd.read_csv(record_file, index_col=False)
    ### Record pure NN prediction & x-Proj post-processing
    infeasible_index = epoch_stats['test_index_infeasible']

    row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == 'NN')
    if not row_index.any():
        data_record.loc[data_record.shape[0]] = {'Prob': str(data), 'Algo': 'NN'}
        row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == 'NN')
    data_record.loc[row_index, 'Fea_rate'] = round((1-np.mean(epoch_stats['test_raw_vio_instance']))*100, 2)
    data_record.loc[row_index, 'Ineq_vio'] = round(np.mean(epoch_stats['test_raw_ineq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Ineq_vio_rate'] = round(np.mean(epoch_stats['test_raw_ineq_num_viol_0'][infeasible_index])/data.nineq*100,2)
    data_record.loc[row_index, 'Eq_vio'] = round(np.mean(epoch_stats['test_raw_eq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Eq_vio_rate'] = round(np.mean(epoch_stats['test_raw_eq_num_viol_0'][infeasible_index])/data.neq*100, 2)

    data_record.loc[row_index, 'Sol_MAE'] = round(np.mean(epoch_stats['test_raw_mae_loss']), 2)
    data_record.loc[row_index, 'Sol_MAPE'] = round(np.mean(epoch_stats['test_raw_mape_loss'])*100, 2)
    data_record.loc[row_index, 'Infea_Sol_MAE'] = round(np.mean(epoch_stats['test_raw_mae_loss'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Sol_MAPE'] = round(np.mean(epoch_stats['test_raw_mape_loss'][infeasible_index])*100, 2)

    data_record.loc[row_index, 'Obj_MAE'] = round(np.mean(epoch_stats['test_raw_obj_mae']), 2)
    data_record.loc[row_index, 'Obj_MAPE'] = round(np.mean(epoch_stats['test_raw_obj_mape'])*100, 2)
    data_record.loc[row_index, 'Infea_Obj_MAE'] = round(np.mean(epoch_stats['test_raw_obj_mae'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Obj_MAPE'] = round(np.mean(epoch_stats['test_raw_obj_mape'][infeasible_index])*100, 2)

    data_record.loc[row_index, 'Ave_time'] =  round(epoch_stats['batch_solve_raw_time'], 4)
    data_record.loc[row_index, 'Ave_speedup'] = round(epoch_stats['batch_raw_speed_up'], 1)
    # data_record.loc[row_index, 'Ave_raw_time'] = round(epoch_stats['batch_solve_raw_time'], 4)
    # data_record.loc[row_index, 'Ave_raw_speedup'] = round(epoch_stats['batch_raw_speed_up'], 1)
    # data_record.loc[row_index, 'Ave_porj_time'] = round(epoch_stats['batch_solve_proj_time'], 4)
    # data_record.loc[row_index, 'Ave_proj_sppedup'] = round(epoch_stats['batch_proj_speed_up'], 1)




    row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == args['projType'])
    if not row_index.any():
        data_record.loc[data_record.shape[0]] = {'Prob': str(data), 'Algo': args['projType']}
        row_index = (data_record['Prob'] == str(data)) & (data_record['Algo'] == args['projType'])
    data_record.loc[row_index, 'Fea_rate'] = round((1-np.mean(epoch_stats['test_cor_vio_instance']))*100, 2)
    data_record.loc[row_index, 'Ineq_vio'] = round(np.mean(epoch_stats['test_cor_ineq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Ineq_vio_rate'] = round(np.mean(epoch_stats['test_cor_ineq_num_viol_0'][infeasible_index])/data.nineq*100,2)
    data_record.loc[row_index, 'Eq_vio'] = round(np.mean(epoch_stats['test_cor_eq_sum'][infeasible_index]), 3)
    data_record.loc[row_index, 'Eq_vio_rate'] = round(np.mean(epoch_stats['test_cor_eq_num_viol_0'][infeasible_index])/data.neq*100, 2)

    data_record.loc[row_index, 'Sol_MAE'] = round(np.mean(epoch_stats['test_cor_mae_loss']), 2)
    data_record.loc[row_index, 'Sol_MAPE'] = round(np.mean(epoch_stats['test_cor_mape_loss'])*100, 2)
    data_record.loc[row_index, 'Infea_Sol_MAE'] = round(np.mean(epoch_stats['test_cor_mae_loss'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Sol_MAPE'] = round(np.mean(epoch_stats['test_cor_mape_loss'][infeasible_index])*100, 2)

    data_record.loc[row_index, 'Obj_MAE'] = round(np.mean(epoch_stats['test_cor_obj_mae']), 2)
    data_record.loc[row_index, 'Obj_MAPE'] = round(np.mean(epoch_stats['test_cor_obj_mape'])*100, 2)
    data_record.loc[row_index, 'Infea_Obj_MAE'] = round(np.mean(epoch_stats['test_cor_obj_mae'][infeasible_index]), 2)
    data_record.loc[row_index, 'Infea_Obj_MAPE'] = round(np.mean(epoch_stats['test_cor_obj_mape'][infeasible_index])*100, 2)


    data_record.loc[row_index, 'Ave_time'] = round(epoch_stats['batch_solve_time'], 4)
    data_record.loc[row_index, 'Ave_speedup'] = round(epoch_stats['batch_speed_up'], 1)
    # data_record.loc[row_index, 'Ave_raw_time'] = round(epoch_stats['batch_solve_raw_time'], 4)
    # data_record.loc[row_index, 'Ave_raw_speedup'] = round(epoch_stats['batch_raw_speed_up'], 1)
    data_record.loc[row_index, 'Ave_porj_time'] = round(epoch_stats['batch_solve_proj_time'], 4)
    data_record.loc[row_index, 'Ave_proj_sppedup'] = round(epoch_stats['batch_proj_speed_up'], 1)


    data_record.to_csv(record_file, index=False)


