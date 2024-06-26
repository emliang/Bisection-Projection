from default_args import *
from visualization import *
from utils.nn_utils import *
from utils.optimization_utils import *
import pickle

defaults = config()
DEVICE = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")



def load_instance(args):
    # Load data, and put on GPU if needed
    seed = args['seed']
    args['algoType'] = args['predType'] + args['projType']
    test_size = args['testSize']
    prob_type = args['probType']
    if prob_type in ['acopf', 'ccacopf']:
        filepath = os.path.join('datasets', 'acopf', 'acopf_{}_{}_{}_dataset'.format(
            seed, args['opfSize'][0], args['opfSize'][1]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        # if prob_type == 'acopf':
        data = ACOPF_Problem(dataset, test_size)
        # elif prob_type == 'ccacopf':
        #     data = CC_ACOPF_Problem(dataset, test_size)
    elif prob_type in ['qp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, args['probSize'][0], args['probSize'][1], args['probSize'][2], args['probSize'][3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = QP_Problem(dataset, test_size)
    elif prob_type in ['convex_qcqp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, args['probSize'][0], args['probSize'][1], args['probSize'][2], args['probSize'][3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = QCQP_Probem(dataset, test_size)
    elif prob_type in ['socp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, args['probSize'][0], args['probSize'][1], args['probSize'][2], args['probSize'][3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = SOCP_Probem(dataset, test_size)
    elif prob_type in ['sdp']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, args['probSize'][0], args['probSize'][1], args['probSize'][2], args['probSize'][3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = SDP_Probem(dataset, test_size)
    elif prob_type in ['jccim']:
        filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
            seed, prob_type, args['probSize'][0], args['probSize'][1], args['probSize'][2], args['probSize'][3]))
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        data = JCCIM_Problem(dataset, test_size)
    else:
        NotImplementedError
        print("Not Implement Instance")
    # elif prob_type in ['nonconvex']:
    #     filepath = os.path.join('datasets', prob_type, "random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(
    #         seed, prob_type, args['probSize'][0], args['probSize'][1], args['probSize'][2], args['probSize'][3]))
    #     with open(filepath, 'rb') as f:
    #         dataset = pickle.load(f)
    #     data = NonconvexProblem(dataset, test_size)
    data.device = DEVICE
    print(DEVICE)
    for attr in dir(data):
        var = getattr(data, attr)
        if torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(device = DEVICE))
                # setattr(data, attr, var.to(device = DEVICE, dtype=torch.float32))
            except AttributeError:
                pass

    result_save_dir = os.path.join('results', prob_type, str(data), args['algoType'])
    model_save_dir = os.path.join('models', prob_type, str(data), args['predType'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    with open(os.path.join(model_save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    print(filepath)
    return data, result_save_dir, model_save_dir


def run_instance(args):
    """
    Load data
    """
    data, result_save_dir, model_save_dir = load_instance(args)
    # print(args['probType'], args['probSize'])

    """
    Run homeomorphic mapping
    """
    if args['inn_para']['training']:
        train_mdh_mapping(data, args, model_save_dir)
    # if args['inn_para']['testing']:
    #     test_mdh_mapping(data, model_save_dir, args)

    """
    Run gauge mapping
    """
    # train_mdg_mapping(data, args, model_save_dir)

    """
    Run MEIPNN
    """
    if args['ipnn_para']['training']:
        train_meip_mapping(data, args, model_save_dir)


    """
    Run neural network solver
    """
    if args['nn_para']['training']:
        train_nn_solver(data, args, model_save_dir)
    if args['nn_para']['testing']:
        test_nn_solver(data, args, model_save_dir, result_save_dir)


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


def train_mdh_mapping(data, args, save_dir):
    paras = args['inn_para']
    c_dim = data.xdim
    n_dim = len(data.partial_vars_idx)
    #### Flow-based model: ball -> constraint set
    num_layer = paras['num_layer']
    inv_type = paras['inv_type']
    h_dim = max(n_dim // 2, 64)
    model = INN(n_dim, h_dim, c_dim, num_layer, inv=inv_type, outact='sigmoid', bilip=paras['bilip'], lip=paras['L']).to(device=DEVICE)
    #### Sampling input parameters and output decision
    initial_shape = paras['shape']
    bound = paras['bound']  # initial shape to [0,1] bounded constraint set
    x_train = sampling_body(paras['n_samples'], n_dim, initial_shape, lu=bound)
    x_train_tensor = torch.tensor(x_train).view(-1, n_dim).to(device=DEVICE)
    c_train_tensor = torch.rand([paras['c_samples'], c_dim]).to(device=DEVICE)
    c_train_tensor = c_train_tensor * (data.input_U - data.input_L) + data.input_L

    #### Unsupervised Training for Homeo Mapping
    # model = pre_training_mdh(model, data, data.trainY, data.trainX, paras)
    model, volume_list, penalty_list, dist_list, trans_list = unsupervised_training_mdh(model, data, 
                                                                       x_train_tensor, c_train_tensor, 
                                                                       paras)
    # plot_convergence(volume_list, penalty_list, dist_list, trans_list, save_dir, args)
    torch.save(model, os.path.join(save_dir, 'mdh_mapping.pth'))


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
    plot_convergence(volume_list, penalty_list, dist_list, trans_list, save_dir)
    torch.save(model, os.path.join(save_dir, 'mdg_mdh_mapping.pth'))


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
    scatter_projection_error(homeo_mapping, data, x_tensor, c_tensor, save_dir, args)


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
    ### Direct
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
            print('Epoch:{}, Vio_rate:{:.2f}, '
                  'Raw_loss: MSE({:.4f}), MAP({:.4f}), '
                  'Raw_obj: MSE({:.4f}), MAP({:.4f}), '
                  'Raw_Ineq: Max({:.4f}), Raw_eq:  Max({:.4f})'.format(
                i, np.mean(epoch_stats['test_raw_vio_instance']),
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

    # homeo_mapping = torch.load(os.path.join(model_save_dir, 'mdh_mapping.pth'), map_location=DEVICE)
    # ipnn = torch.load(os.path.join(model_save_dir, 'ipnn.pth'), map_location=DEVICE)
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
        if 'Eq' in args['algoType']:
            Y = data.complete_partial(X, Y_pred_scale)
        else:
            Y = Y_pred_scale
    raw_end_time = time.time()
    NN_pred_time = raw_end_time - raw_start_time
    # print(0.001)
    ### Post-processing for infeasible only
    steps = args['proj_para']['corrTestMaxSteps']
    eps_converge = args['proj_para']['corrEps']
    violation = data.check_feasibility(X, Y)
    penalty = torch.max(torch.abs(violation), dim=1)[0]
    infeasible_index = (penalty > eps_converge).view(-1)
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
    # n_process = 50
    # start_time = time.time()
    # _ = data.opt_solve(Xtest[:n_process], solver_type=solvers[0], tol=args['proj_para']['corrEps'])
    # end_time = time.time()
    # opt_time = (end_time - start_time)/n_process
    # print(opt_time)
    opt_time = 1.553299

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


if __name__ == '__main__':
    main()
