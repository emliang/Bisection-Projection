from .nn_utils import *
from .sampling_utils import *
from .proj_utils import  *
import numpy as np
import torch
import torch.optim as optim
import time
import subprocess
import pickle
import pandas as pd
import os

def get_least_utilized_gpu():
    # Query nvidia-smi to find the current GPU utilization
    try:
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
    except:
        return 'cpu'
# Use the function to set the CUDA device
least_utilized_gpu = get_least_utilized_gpu()
print(f"Using GPU: {least_utilized_gpu}")
DEVICE = torch.device(least_utilized_gpu if torch.cuda.is_available() else "cpu")


###################################################################
# (Un)supervised Training for NN Solver for Solution Mapping
###################################################################
def train_nn_solver(data, args, save_dir):
    lr = args['nn_para']['lr']
    nepochs = args['nn_para']['total_iteration']
    batch_size = args['nn_para']['batch_size']
    lr_decay = args['nn_para']['lr_decay']
    lr_decay_step = args['nn_para']['lr_decay_step']
    training_approach = args['nn_para']['approach']
    NN_structure = args['predType']
    print(f'NN {training_approach} training with {NN_structure}')
    ### Equality completion
    if 'Eq' in args['algoType']:
        out_dim = len(data.partial_vars_idx)
    else:
        out_dim = data.testY.shape[1]
    in_dim = data.xdim
    n_layer = args['nn_para']['num_layer']
    model = ResNet(in_dim, out_dim, out_dim+1, n_layer)
    model.to(DEVICE)
    solver_opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    solver_shce = optim.lr_scheduler.StepLR(solver_opt,
                                            step_size=lr_decay_step,
                                            gamma=lr_decay)
    loss = nn.MSELoss()
    stats = {}
    model.train()
    Xtrain = data.trainX.to(DEVICE)
    Ytrain = data.trainY.squeeze().to(DEVICE)
    Xtest = data.testX.to(DEVICE)
    Ytest = data.testY.squeeze().to(DEVICE)
    training_time_list = []
    total_time = 0
    # """
    # pre-training
    # """
    # if training_approach == 'supervise':
    #     _pretrain(model, data, Xtrain, Ytrain, args['nn_para'], batch_size)
 
    for i in range(nepochs + 1):
        epoch_stats = {}
        iter_st = time.time()
        if training_approach == 'supervise':
            """
            formal training
            """
            batch_index = np.random.choice(np.arange(Xtrain.shape[0]), batch_size, replace=False)
            Xtrain_batch = (Xtrain[batch_index]).view(batch_size,-1)
            Ytrain_batch = (Ytrain[batch_index]).view(batch_size,-1)
            Z_pred_batch = model(Xtrain_batch)
            Z_pred_scale_batch = data.scale(Xtrain_batch, Z_pred_batch)
            if 'Eq' in args['algoType']:
                Y_pred_scale_batch = data.complete_partial(Xtrain_batch, Z_pred_scale_batch)
            else:
                Y_pred_scale_batch = Z_pred_scale_batch
            training_obj = data.obj_fn(Y_pred_scale_batch)
            eq_vio = data.eq_resid(Xtrain_batch, Y_pred_scale_batch).abs()
            ineq_vio = data.ineq_resid(Xtrain_batch, Y_pred_scale_batch).abs()
            eq_penalty = eq_vio.sum(-1)
            ineq_penalty = ineq_vio.sum(-1)
            if 'Eq' in args['algoType']:
                train_loss = loss(Y_pred_scale_batch, Ytrain_batch).sum(-1).mean() \
                            + args['nn_para']['w_ineq'] * ineq_penalty.mean() \
                            + args['nn_para']['w_obj'] * training_obj.mean()
            else:
                train_loss = loss(Y_pred_scale_batch, Ytrain_batch).sum(-1).mean() \
                            + args['nn_para']['w_ineq'] * ineq_penalty.mean() \
                            + args['nn_para']['w_eq'] * eq_penalty.mean() \
                            + args['nn_para']['w_obj'] * training_obj.mean()
        else:
            Xtrain_batch = torch.rand([batch_size, Xtest.shape[1]]).to(device=DEVICE)
            Xtrain_batch = Xtrain_batch * (data.input_U - data.input_L) + data.input_L
            Z_pred_batch = model(Xtrain_batch)
            Z_pred_scale_batch = data.scale(Xtrain_batch, Z_pred_batch)
            if 'Eq' in args['algoType']:
                Y_pred_scale_batch = data.complete_partial(Xtrain_batch, Z_pred_scale_batch)
            else:
                Y_pred_scale_batch = Z_pred_scale_batch
            training_obj = data.obj_fn(Y_pred_scale_batch)
            eq_vio = data.eq_resid(Xtrain_batch, Y_pred_scale_batch).abs()
            ineq_vio = data.ineq_resid(Xtrain_batch, Y_pred_scale_batch).abs()
            # eq_penalty = (0.5 * eq_vio ** 2 + eq_vio).sum(-1)
            # ineq_penalty = (0.5 * ineq_vio ** 2 + ineq_vio).sum(-1)
            eq_penalty = eq_vio.sum(-1)
            ineq_penalty = ineq_vio.sum(-1)
            if 'Eq' in args['algoType']:
                train_loss = args['nn_para']['w_ineq'] * ineq_penalty.mean() \
                            + args['nn_para']['w_obj'] * training_obj.mean()
            else:
                train_loss = args['nn_para']['w_ineq'] * ineq_penalty.mean() \
                            + args['nn_para']['w_eq'] * eq_penalty.mean() \
                            + args['nn_para']['w_obj'] * training_obj.mean()

        train_loss.backward()
        solver_opt.step()
        solver_shce.step()
        solver_opt.zero_grad()

        iter_et = time.time()
        train_time = iter_et - iter_st
        training_time_list.append(train_time)
        dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
        dict_agg(epoch_stats, 'train_obj', training_obj.detach().cpu().numpy())
        dict_agg(epoch_stats, 'train_time', train_time, op='sum')
        total_time+= train_time
        # Print results
        if i % args['nn_para']['resultsSaveFreq'] == 0:
            # print(total_time)
            batch_index = np.random.choice(np.arange(Xtest.shape[0]), batch_size, replace=False)
            model.eval()
            with torch.no_grad():
                eval_nn_solution(data, Xtest[batch_index], Ytest[batch_index], model, args, save_dir, 'test', epoch_stats)
            print('Epoch:{}, Feas_rate:{:.2f}, '
                  'Raw_loss: MSE({:.4f}), MAP({:.4f}), '
                  'Raw_obj: MSE({:.4f}), MAP({:.4f}), '
                  'Raw_Ineq: Max({:.4f}), Raw_eq:  Max({:.4f})'.format(
                i, 1-np.mean(epoch_stats['test_raw_vio_instance']),
                np.mean(epoch_stats['test_raw_mse_loss']), np.mean(epoch_stats['test_raw_mape_loss']),
                np.mean(epoch_stats['test_raw_obj_mse']),  np.mean(epoch_stats['test_raw_obj_mape']),
                np.mean(epoch_stats['test_raw_ineq_max']), np.mean(epoch_stats['test_raw_eq_max'])))

            with open(os.path.join(save_dir, f'{training_approach}_{NN_structure}.pth'), 'wb') as f:
                torch.save(model, f)
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
    return model, stats

def test_nn_solver(data, args, model_save_dir, result_save_dir):
    """
    Test neural network solver performance on test dataset.

    Args:
        data: Data handler object containing test data
        args: Configuration dictionary
        model_save_dir: Directory containing saved models
        result_save_dir: Directory to save test results
    """
    # Configuration
    print(f"Problem Type: {args['probType']}, Projection Type: {args['projType']}")
    args['proj_para']['useTestCorr'] = True

    # Extract configuration parameters
    nn_structure = args['predType']
    training_approach = args['nn_para']['approach']
    algo_type = args['algoType']

    # Setup device and data
    data.to_device(DEVICE)
    Xtest = data.testX
    Ytest = data.testY.squeeze()

    # Load trained model
    nn_save_dir = os.path.join('models', args['probType'], str(data))
    model_path = os.path.join(nn_save_dir, f'{training_approach}_{nn_structure}.pth')
    try:
        model = torch.load(model_path, map_location=DEVICE)
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Evaluate model performance
    epoch_stats = {}
    with torch.no_grad():
        eval_nn_solution(data, Xtest, Ytest, model, args, model_save_dir, 'test', epoch_stats)

    # Display results
    _display_test_results(epoch_stats, data)

    # Save results
    output_filename = f'{training_approach}_{algo_type}_test_stats.dict'
    output_path = os.path.join(result_save_dir, output_filename)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(epoch_stats, f)
        print(f"\nResults saved to: {output_path}")
    except IOError as e:
        print(f"Error saving results: {e}")

    return epoch_stats

def _display_test_results(stats, data):
    """
    Display formatted test results.

    Args:
        stats: Dictionary containing test statistics
        data: Data handler object for constraint information
    """
    # Extract metrics for cleaner formatting
    metrics = _extract_display_metrics(stats, data)

    # Format and print results
    results_template = """
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                           Neural Network Solver Test Results                   ║
    ╠════════════════════════════════════════════════════════════════════════════════╣
    ║ Feasibility & Timing:                                                          
    ║   Raw - Feasibility Rate: {raw_feas_rate:.4f}, Time: {raw_time:.4f}s         
    ║   Cor - Feasibility Rate: {cor_feas_rate:.4f}, Time: {cor_time:.4f}s  
    ║ Solution Accuracy:                                                            
    ║   Raw - MSE: {raw_mse:.4f}, MAE: {raw_mae:.4f}, MAPE: {raw_mape:.4f}             
    ║   Cor - MSE: {cor_mse:.4f}, MAE: {cor_mae:.4f}, MAPE: {cor_mape:.4f}                                                                                     
    ║ Objective Values:                                                                 
    ║   Raw - MSE: {raw_obj_mse:.4f}, MAE: {raw_obj_mae:.4f}, MAPE: {raw_obj_mape:.4f}  
    ║   Cor - MSE: {cor_obj_mse:.4f}, MAE: {cor_obj_mae:.4f}, MAPE: {cor_obj_mape:.4f}                                                                                  
    ║ Constraint Violations:                                                            
    ║   Raw Ineq - Max: {raw_ineq_max:.4f}, Sum: {raw_ineq_sum:.4f}  ║  Raw Eq   - Max: {raw_eq_max:.4f}, Sum: {raw_eq_sum:.4f}
    ║   Cor Ineq - Max: {cor_ineq_max:.4f}, Sum: {cor_ineq_sum:.4f}  ║  Cor Eq   - Max: {cor_eq_max:.4f}, Sum: {cor_eq_sum:.4f}                                                                                        
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """

    print(results_template.format(**metrics))

def _extract_display_metrics(stats, data):
    """
    Extract and compute metrics for display.

    Args:
        stats: Dictionary containing test statistics
        data: Data handler object

    Returns:
        dict: Formatted metrics for display
    """

    # Helper function to safely compute mean
    def safe_mean(key, default=0.0):
        return np.mean(stats.get(key, default)) if key in stats else default

    # Compute violation rates
    raw_ineq_rate = safe_mean('test_raw_ineq_num_viol_0') / max(data.nineq, 1)
    raw_eq_rate = safe_mean('test_raw_eq_num_viol_0') / max(data.neq, 1)
    cor_ineq_rate = safe_mean('test_cor_ineq_num_viol_0') / max(data.nineq, 1)
    cor_eq_rate = safe_mean('test_cor_eq_num_viol_0') / max(data.neq, 1)

    # Compute feasibility rates
    raw_feas_rate = 1 - safe_mean('test_raw_vio_instance')
    cor_feas_rate = 1 - safe_mean('test_cor_vio_instance')

    return {
        # Feasibility and timing
        'raw_feas_rate': raw_feas_rate,
        'raw_time': stats.get('test_raw_time', 0.0),
        'cor_feas_rate': cor_feas_rate,
        'cor_time': stats.get('test_raw_time', 0.0) + stats.get('test_proj_time', 0.0),

        # Solution accuracy metrics
        'raw_mse': safe_mean('test_raw_mse_loss'),
        'raw_mae': safe_mean('test_raw_mae_loss'),
        'raw_mape': safe_mean('test_raw_mape_loss'),
        'cor_mse': safe_mean('test_cor_mse_loss'),
        'cor_mae': safe_mean('test_cor_mae_loss'),
        'cor_mape': safe_mean('test_cor_mape_loss'),

        # Objective value metrics
        'raw_obj_mse': safe_mean('test_raw_obj_mse'),
        'raw_obj_mae': safe_mean('test_raw_obj_mae'),
        'raw_obj_mape': safe_mean('test_raw_obj_mape'),
        'cor_obj_mse': safe_mean('test_cor_obj_mse'),
        'cor_obj_mae': safe_mean('test_cor_obj_mae'),
        'cor_obj_mape': safe_mean('test_cor_obj_mape'),

        # Constraint violation metrics
        'raw_ineq_max': safe_mean('test_raw_ineq_max'),
        'raw_ineq_sum': safe_mean('test_raw_ineq_sum'),
        # 'raw_ineq_rate': raw_ineq_rate,
        'raw_eq_max': safe_mean('test_raw_eq_max'),
        'raw_eq_sum': safe_mean('test_raw_eq_sum'),
        # 'raw_eq_rate': raw_eq_rate,
        'cor_ineq_max': safe_mean('test_cor_ineq_max'),
        'cor_ineq_sum': safe_mean('test_cor_ineq_sum'),
        # 'cor_ineq_rate': cor_ineq_rate,
        'cor_eq_max': safe_mean('test_cor_eq_max'),
        'cor_eq_sum': safe_mean('test_cor_eq_sum'),
        # 'cor_eq_rate': cor_eq_rate,


    }



def eval_nn_solution(data, X, Ytarget, model, args, save_dir, prefix, stats):
    """
    Evaluate neural network solution predictions with optional post-processing for infeasible solutions.
    
    Args:
        data: Data handler object with problem-specific methods
        X: Input tensor
        Ytarget: Target solution tensor
        model: Neural network model
        args: Configuration dictionary
        save_dir: Directory for saved models
        prefix: Prefix for metric names
        stats: Dictionary to accumulate statistics
        
    Returns:
        stats: Updated statistics dictionary
    """
    # Generate NN predictions
    raw_start_time = time.time()
    model.eval()
    
    with torch.no_grad():
        Z_pred = model(X)
        
        # Add prediction noise if specified
        if args['ipnn_para']['sa_pred_noise'] > 0:
            torch.manual_seed(args['seed'])
            noise = torch.randn_like(Z_pred).to(Z_pred.device)
            Z_pred += noise * args['ipnn_para']['sa_pred_noise']
            Z_pred = torch.clamp(Z_pred, 0, 1)
        
        # Scale predictions
        Z_pred_scale = data.scale(X, Z_pred)
        
        # Complete partial solutions if needed
        if 'Eq' in args['predType']:
            Y = data.complete_partial(X, Z_pred_scale)
        else:
            Y = Z_pred_scale
            Z_pred = Z_pred[:, data.partial_vars_idx]
    
    raw_end_time = time.time()
    nn_pred_time = raw_end_time - raw_start_time
    
    # Check feasibility of predictions
    eps_converge = args['proj_para']['corrEps']
    penalty = data.check_feasibility(X, Y).abs()
    infeasible_index = (penalty.max(-1)[0] > eps_converge).view(-1)
    Y_pred_infeasible = Y[infeasible_index]
    num_infeasible = Y_pred_infeasible.shape[0]
    
    # Load post-processing models if needed
    if args['proj_para']['useTestCorr'] and num_infeasible > 0:
        if 'H_Proj' in args['algoType']:
            homeo_mapping = torch.load(
                os.path.join(save_dir, 'mdh_mapping.pth'), 
                map_location=DEVICE
            )
            homeo_mapping.eval()
        elif 'B_Proj' in args['algoType']:
            fixed_margin = args['ipnn_para']['fixed_margin']
            gamma = args['ipnn_para']['gamma']
            training_sample = args['ipnn_para']['training_sample']
            ipnn = torch.load(
                os.path.join(save_dir, f'ipnn_{training_sample}_{fixed_margin}_{gamma}.pth'), 
                map_location=DEVICE
            )
            ipnn.eval()
            with torch.no_grad():
                feasible_ip = ipnn(X[infeasible_index])
                feasible_ip = feasible_ip.view(feasible_ip.shape[0], 1, -1)
    
    # Post-process infeasible solutions
    cor_start_time = time.time()
    Ycorr = Y.detach().clone()
    
    if num_infeasible > 0 and args['proj_para']['useTestCorr']:
        X_infeasible = X[infeasible_index]
        Y_infeasible = Y[infeasible_index]
        Z_pred_infeasible = Z_pred[infeasible_index]
        
        # Apply appropriate projection method
        if 'H_Proj' in args['algoType']:
            Yproj, _ = homeo_bisection(homeo_mapping, data, args, Z_pred_infeasible, X_infeasible, eps_converge)
        elif 'B_Proj' in args['algoType']:
            Yproj, _ = ip_bisection(feasible_ip, data, args, Z_pred_infeasible, X_infeasible, eps_converge)
        elif 'D_Proj' in args['algoType']:
            Yproj, _ = diff_projection(data, X_infeasible, Y_infeasible, args, eps_converge)
        elif 'Proj' in args['algoType']:
            Yproj = data.opt_proj(X_infeasible, Y_infeasible).to(Y.device)
        elif 'WS' in args['algoType']:
            Yproj = data.opt_warmstart(X_infeasible, Y_infeasible).to(Y.device)
        else:
            Yproj = Y_infeasible
            
        Ycorr[infeasible_index] = Yproj
    
    cor_end_time = time.time()
    proj_time = cor_end_time - cor_start_time
    
    # Calculate metrics
    _compute_and_store_metrics(
        data, X, Y, Ycorr, Ytarget, 
        nn_pred_time, proj_time, num_infeasible, infeasible_index,
        eps_converge, prefix, stats
    )
    
    return stats

def _compute_and_store_metrics(data, X, Y, Ycorr, Ytarget, 
                               nn_pred_time, proj_time, num_infeasible, infeasible_index,
                               eps_converge, prefix, stats):
    """Helper function to compute and store evaluation metrics."""
    
    def make_prefix(metric_name):
        return f"{prefix}_{metric_name}"
    
    # Store timing metrics
    dict_agg(stats, make_prefix('time'), proj_time + nn_pred_time, op='sum')
    dict_agg(stats, make_prefix('proj_time'), proj_time, op='sum')
    dict_agg(stats, make_prefix('raw_time'), nn_pred_time, op='sum')
    
    # Store infeasibility metrics
    dict_agg(stats, make_prefix('num_infeasible'), num_infeasible)
    dict_agg(stats, make_prefix('index_infeasible'), infeasible_index.detach().cpu().numpy())
    
    # Calculate objective values
    Y_obj = data.obj_fn(Y).detach().cpu()
    Ycor_obj = data.obj_fn(Ycorr).detach().cpu()
    Ytarget_obj = data.obj_fn(Ytarget).detach().cpu()
    
    # Calculate constraint violations
    raw_ineq_vio = torch.abs(data.ineq_resid(X, Y)).detach().cpu()
    raw_eq_vio = torch.abs(data.eq_resid(X, Y)).detach().cpu()
    raw_vio = torch.abs(data.check_feasibility(X, Y)).detach().cpu()
    cor_ineq_vio = torch.abs(data.ineq_resid(X, Ycorr)).detach().cpu()
    cor_eq_vio = torch.abs(data.eq_resid(X, Ycorr)).detach().cpu()
    cor_vio = torch.abs(data.check_feasibility(X, Ycorr)).detach().cpu()
    
    # Move tensors to CPU for metric computation
    Y = Y.detach().cpu()
    X = X.detach().cpu()
    Ycorr = Ycorr.detach().cpu()
    Ytarget = Ytarget.detach().cpu()
    
    # Calculate solution errors
    solution_res = Y - Ytarget
    proj_solution_res = Ycorr - Ytarget
    target_solution_norm = torch.norm(Ytarget, dim=1, p=1)
    proj_dist_res = Ycorr - Y
    
    # Store solution accuracy metrics
    _store_solution_metrics(stats, make_prefix, solution_res, proj_solution_res, 
                           proj_dist_res, target_solution_norm)
    
    # Store objective value metrics
    _store_objective_metrics(stats, make_prefix, Y_obj, Ycor_obj, Ytarget_obj)
    
    # Store constraint violation metrics
    _store_violation_metrics(stats, make_prefix, 
                           raw_ineq_vio, raw_eq_vio, raw_vio,
                           cor_ineq_vio, cor_eq_vio, cor_vio,
                           eps_converge)

def _store_solution_metrics(stats, make_prefix, solution_res, proj_solution_res, 
                           proj_dist_res, target_solution_norm):
    """Store solution accuracy metrics."""
    # Raw prediction metrics
    raw_mae_loss = torch.norm(solution_res, dim=1, p=1)
    raw_mse_loss = torch.norm(solution_res, dim=1, p=2) ** 2
    raw_mape_loss = raw_mae_loss / target_solution_norm
    
    dict_agg(stats, make_prefix('raw_mae_loss'), raw_mae_loss.numpy())
    dict_agg(stats, make_prefix('raw_mse_loss'), raw_mse_loss.numpy())
    dict_agg(stats, make_prefix('raw_mape_loss'), raw_mape_loss.numpy())
    
    # Corrected prediction metrics
    cor_mae_loss = torch.norm(proj_solution_res, dim=1, p=1)
    cor_mse_loss = torch.norm(proj_solution_res, dim=1, p=2) ** 2
    cor_mape_loss = cor_mae_loss / target_solution_norm
    
    dict_agg(stats, make_prefix('cor_mae_loss'), cor_mae_loss.numpy())
    dict_agg(stats, make_prefix('cor_mse_loss'), cor_mse_loss.numpy())
    dict_agg(stats, make_prefix('cor_mape_loss'), cor_mape_loss.numpy())
    
    # Projection distance metrics
    proj_mae_dist = torch.norm(proj_dist_res, dim=1, p=1)
    proj_mse_dist = torch.norm(proj_dist_res, dim=1, p=2) ** 2
    proj_mape_dist = proj_mae_dist / target_solution_norm
    
    dict_agg(stats, make_prefix('proj_mae_dist'), proj_mae_dist.numpy())
    dict_agg(stats, make_prefix('proj_mse_dist'), proj_mse_dist.numpy())
    dict_agg(stats, make_prefix('proj_mape_dist'), proj_mape_dist.numpy())

def _store_objective_metrics(stats, make_prefix, Y_obj, Ycor_obj, Ytarget_obj):
    """Store objective value metrics."""
    # Raw prediction objective metrics
    dict_agg(stats, make_prefix('raw_obj_mae'), torch.abs(Y_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('raw_obj_mse'), torch.square(Y_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('raw_obj_mape'), 
             (torch.abs(Y_obj - Ytarget_obj) / torch.abs(Ytarget_obj)).numpy())
    
    # Corrected prediction objective metrics
    dict_agg(stats, make_prefix('cor_obj_mae'), torch.abs(Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mse'), torch.square(Ycor_obj - Ytarget_obj).numpy())
    dict_agg(stats, make_prefix('cor_obj_mape'), 
             (torch.abs(Ycor_obj - Ytarget_obj) / torch.abs(Ytarget_obj)).numpy())

def _store_violation_metrics(stats, make_prefix, 
                            raw_ineq_vio, raw_eq_vio, raw_vio,
                            cor_ineq_vio, cor_eq_vio, cor_vio,
                            eps_converge):
    """Store constraint violation metrics."""
    # Raw prediction violation metrics
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(raw_ineq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(raw_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_ineq_sum'), torch.sum(raw_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_0'), 
             torch.sum(raw_ineq_vio > eps_converge, dim=1).numpy())
    
    dict_agg(stats, make_prefix('raw_eq_max'), torch.max(raw_eq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'), torch.mean(raw_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_eq_sum'), torch.sum(raw_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_0'), 
             torch.sum(raw_eq_vio > eps_converge, dim=1).numpy())
    
    dict_agg(stats, make_prefix('raw_vio_instance'), 
             (torch.max(raw_vio, dim=1)[0] > eps_converge).numpy())
    
    # Corrected prediction violation metrics
    dict_agg(stats, make_prefix('cor_ineq_max'), torch.max(cor_ineq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('cor_ineq_mean'), torch.mean(cor_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_ineq_sum'), torch.sum(cor_ineq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_ineq_num_viol_0'), 
             torch.sum(cor_ineq_vio > eps_converge, dim=1).numpy())
    
    dict_agg(stats, make_prefix('cor_eq_max'), torch.max(cor_eq_vio, dim=1)[0].numpy())
    dict_agg(stats, make_prefix('cor_eq_mean'), torch.mean(cor_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_eq_sum'), torch.sum(cor_eq_vio, dim=1).numpy())
    dict_agg(stats, make_prefix('cor_eq_num_viol_0'), 
             torch.sum(cor_eq_vio > eps_converge, dim=1).numpy())
    
    dict_agg(stats, make_prefix('cor_vio_instance'), 
             (torch.max(cor_vio, dim=1)[0] > eps_converge).numpy())




def test_inf_time(data, args, result_save_dir):
    """
    Test inference time and calculate speed-up metrics for neural network solutions
    compared to traditional optimization solvers.
    
    Args:
        data: Data handler object containing test data and optimization methods
        args: Configuration dictionary with test parameters
        model_save_dir: Directory containing saved models
        result_save_dir: Directory to save results
    
    Returns:
        None (saves results to file)
    """
    # Configuration
    N_PROCESSES = 1
    SEED = args['seed']
    TEST_SIZE = args['testSize']
    CONVERGENCE_EPS = args['proj_para']['corrEps']

    solver_time_path  = os.path.join('results' , args['probType'], 'solver_time.txt')
    if os.path.exists(solver_time_path):
        with open(solver_time_path, 'r') as f:
            solver_time = float(f.read())
    else:
        # Set random seed for reproducibility
        np.random.seed(SEED)
        # Sample test indices
        total_test_samples = data.testX.shape[0]
        test_indices = np.random.choice( range(total_test_samples), TEST_SIZE, replace=False )
        # Get test data
        Xtest = data.testX.to(DEVICE)[test_indices]
        # Benchmark optimization solver time
        start_time = time.time()
        # Run optimization on batch
        _ = data.opt_solve(Xtest[:N_PROCESSES], tol=CONVERGENCE_EPS)
        end_time = time.time()
        # Calculate average time per sample
        solver_time = end_time - start_time
        with open(solver_time_path, 'w') as f:
            f.write(str(solver_time))
    
    # Load test statistics
    algo_type = args['algoType']
    training_approach = args['nn_para']['approach']
    stats_filename = f'{training_approach}_{algo_type}_test_stats.dict'
    stats_path = os.path.join(result_save_dir, stats_filename)
    try:
        epoch_stats = np.load(stats_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    
    # Calculate timing metrics
    timing_metrics = _calculate_timing_metrics( epoch_stats, solver_time  )
    
    # Update statistics with timing metrics
    epoch_stats.update(timing_metrics)

    # Save to CSV
    record_file = './results/BP_final.csv'
    csv_record(epoch_stats, data, args, record_file)

    # Save as pickle file
    output_path = os.path.join(result_save_dir, 'test_stats.dict')
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(epoch_stats, f)
        print(f"Results saved to: {output_path}")
    except IOError as e:
        print(f"Error saving results: {e}")

def _calculate_timing_metrics(epoch_stats, solver_time):
    """
    Calculate various timing metrics and speed-up ratios.
    
    Args:
        epoch_stats: Dictionary containing test statistics
        n_test_samples: Number of test samples
        opt_time_per_sample: Optimization time per sample
    
    Returns:
        dict: Timing metrics and speed-up ratios
    """
    # Extract base statistics
    total_raw_time = epoch_stats.get('test_raw_time', 0)
    total_proj_time = epoch_stats.get('test_proj_time', 0)
    total_time = epoch_stats.get('test_time', 0)
    # num_infeasible = epoch_stats.get('test_num_infeasible', 1)  # Avoid division by zero
    
    # Calculate average times
    # avg_raw_time = total_raw_time
    # avg_proj_time = total_proj_time
    # avg_total_time = total_time
    # avg_cor_time = total_time
    
    # Calculate speed-up ratios
    timing_metrics = {
        'batch_solver_time': solver_time,
        'batch_nn_raw_time': total_raw_time,
        'batch_nn_proj_time': total_proj_time,
        'batch_nn_total_time': total_time,
        # 'batch_raw_speed_up': opt_time_per_sample / avg_raw_time if avg_raw_time > 0 else float('inf'),
        # 'batch_solve_proj_time': avg_proj_time,
        # 'batch_proj_speed_up': opt_time_per_sample / avg_proj_time if avg_proj_time > 0 else float('inf'),
        # 'batch_solve_time': avg_total_time,
        # 'batch_speed_up': opt_time_per_sample / avg_total_time if avg_total_time > 0 else float('inf'),
        # 'batch_solve_cor_time': avg_cor_time,
        # 'batch_cor_speed_up': opt_time_per_sample / avg_cor_time if avg_cor_time > 0 else float('inf'),
        # 'opt_time_per_sample': opt_time_per_sample
    }
    
    return timing_metrics












###################################################################
# Unsupervised Training for Minimum-Distortion-Homeomoprhic Mapping
###################################################################
def unsupervised_training_mdh(MDH, data, x_tensor, input_tensor, save_dir, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    w_penalty = args['w_penalty']
    w_distortion = args['w_distortion']
    volume_list = []
    penalty_list = []
    dist_list = []
    trans_list = []
    n_dim = x_tensor.shape[1]
    bias_tensor = torch.ones(batch_size, n_dim).to(x_tensor.device) * args['center']
    # x_input = torch.cat([x_input, bias_tensor], dim=0)
    optimizer = optim.AdamW(MDH.parameters(),
                            lr=args['lr'],
                            weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=args['lr_decay_step'],
                            gamma=args['lr_decay'])

    for n in range(total_iteration):
        MDH.train()
        optimizer.zero_grad()
        batch_index = np.random.choice(range(x_tensor.shape[0]),  batch_size, replace=True)
        x_input = x_tensor[batch_index]
        batch_index = np.random.choice(range(input_tensor.shape[0]), batch_size, replace=True)
        input_batch = input_tensor[batch_index]
        xt, logdet, logdis = MDH(x_input, input_batch)
        volume = logdet
        xt_scale = data.scale(input_batch, xt)
        xt_full = data.complete_partial(input_batch, xt_scale)
        # eq_vio = data.eq_resid(input_batch, xt_full).abs()
        # obj = data.obj_fn(xt_full)
        ineq_vio = data.ineq_resid(input_batch, xt_full).abs()
        ineq_penalty = ineq_vio.sum(-1)
        penalty = ineq_penalty
        loss =   w_distortion * torch.mean(logdis) \
                 + w_penalty * torch.mean(penalty) \
                 - torch.mean(volume) / n_dim
        loss.backward()
        optimizer.step()
        scheduler.step()
        volume_list.append(torch.mean(logdet).detach().cpu().numpy()/n_dim)
        penalty_list.append(torch.mean(penalty).detach().cpu().numpy())
        dist_list.append(torch.mean(logdis).detach().cpu().numpy()/n_dim)

        if n % args['resultsSaveFreq']==0:
            torch.save(MDH, os.path.join(save_dir, 'mdh_mapping.pth'))
            MDH.eval()
            with torch.no_grad():
                batch_index = np.random.choice(range(input_tensor.shape[0]), batch_size, replace=True)
                input_batch = input_tensor[batch_index]
                x0,_,_ = MDH(bias_tensor, input_batch)
                x0_scale = data.scale(input_batch, x0)
                x0_full = data.complete_partial(input_batch, x0_scale)
                violation_0 = data.check_feasibility(input_batch, x0_full)
                penalty_0 = torch.abs(violation_0).detach().cpu()
            print(f'Iteration: {n}/{total_iteration}, '
                  f'Volume: {volume_list[-1]:.3f}, '
                  f'Penalty: {penalty_list[-1]:.3f}, '
                  f'Distortion: {dist_list[-1]:.3f}, '
                  f'Center Penalty: {penalty_0.sum(-1).max().numpy():.4f}, '
                  f'Valid rate: {(penalty_0.max(-1)[0] < 1e-5).numpy().mean():.2f}',
                  end='\n')
    training_record = {'volume_list': volume_list,
                       'penalty_list': penalty_list,
                       'dist_list': dist_list,
                       'trans_list': trans_list,}
    return MDH, training_record

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
    # bound = paras['bound']  # initial shape to [0,1] bounded constraint set
    # ball_sample = torch.rand(size=[paras['n_samples'], n_dim]).to(device=DEVICE) * (bound[1] - bound[0]) + bound[0]
    ball_sample = torch.randn(size=[paras['n_samples'], n_dim]).to(device=DEVICE) + args['inn_para']['center']
    #### Unsupervised Training for Homeo Mapping
    model, training_record = unsupervised_training_mdh(model, data, ball_sample, data.trainX, save_dir, paras)
    torch.save(model, os.path.join(save_dir, 'mdh_mapping.pth'))



###################################################################
# Unsupervised Training for Minimum-Eccentricity-IP Mapping
###################################################################
def general_boundary_sampling(
    data,
    input_tensor,
    feasible_ip,
    num_boundary_point,
    bisect_step=10,
    feasibility_tolerance=1e-5
):
    """
    Sample boundary points using bisection method along random directions.
    
    Args:
        data: Data object with scale, complete_partial, and check_feasibility methods
        input_tensor: Context tensor of shape (batch_size, c_dim)
        feasible_ip: Initial feasible interior points of shape (batch_size, n_dim)
        num_boundary_point: Number of boundary points to sample per batch
        bisect_step: Number of bisection iterations (default: 10)
        feasibility_tolerance: Tolerance for feasibility check (default: 1e-5)
    
    Returns:
        Tensor of boundary samples with shape (batch_size, num_boundary_point, n_dim)
    """
    batch_size, n_dim = feasible_ip.shape
    c_dim = input_tensor.shape[-1]
    device = feasible_ip.device
    
    # Sample random directions (not normalized to allow varied step sizes)
    unit_vec = torch.randn(
        size=(batch_size * num_boundary_point, n_dim),
        device=device
    )
    
    # Expand tensors for vectorized operations
    # Shape: (batch_size * num_boundary_point, n_dim)
    feasible_ip_expanded = feasible_ip.unsqueeze(1).repeat(1, num_boundary_point, 1).view(-1, n_dim)
    # Shape: (batch_size * num_boundary_point, c_dim)
    c_tensor_expanded = input_tensor.unsqueeze(1).repeat(1, num_boundary_point, 1).view(-1, c_dim)
    
    # Initialize bisection bounds
    lower_bound = torch.zeros(batch_size * num_boundary_point, 1, device=device)
    upper_bound = torch.ones(batch_size * num_boundary_point, 1, device=device)
    
    # Perform bisection to find boundary
    for _ in range(bisect_step):
        # Test upper bound
        alpha = upper_bound.clone()
        test_points = feasible_ip_expanded + unit_vec * alpha
        
        # Check feasibility at test points
        test_points_scaled = data.scale(c_tensor_expanded, test_points)
        test_points_full = data.complete_partial(c_tensor_expanded, test_points_scaled)
        penalties = data.check_feasibility(c_tensor_expanded, test_points_full).max(dim=-1)[0]
        
        # Update bounds based on feasibility
        is_infeasible = penalties > feasibility_tolerance
        is_feasible = ~is_infeasible
        
        # If infeasible: move upper bound down (bisect between lower and upper)
        upper_bound[is_infeasible] = (upper_bound[is_infeasible] + lower_bound[is_infeasible]) / 2
        
        # If feasible: move lower bound up and double upper bound to explore further
        lower_bound[is_feasible] = alpha[is_feasible]
        upper_bound[is_feasible] *= 2
    
    # Use lower bound as final boundary points (last known feasible points)
    boundary_samples = feasible_ip_expanded + unit_vec * lower_bound
    
    # Reshape to original batch structure
    return boundary_samples.view(batch_size, num_boundary_point, n_dim)


def _pretrain(IPNN, data, input_tensor, output_tensor,args, batch_size):
    """Pre-training phase with MSE loss"""
    for module in IPNN.modules():
        if isinstance(module, NoiseModule):
            if hasattr(module, 'log_gamma'):
                module.log_gamma.requires_grad = False
    pre_opt = optim.AdamW(IPNN.parameters(), lr=5e-4, weight_decay=1e-5)
    MSE = nn.MSELoss()
    # Ytrain = data.trainY.squeeze().to(DEVICE)[:args['training_sample']]

    for n in range(args['pre_training']):
        batch_idx = np.random.choice(len(input_tensor), batch_size)
        input_batch, y_batch = input_tensor[batch_idx], output_tensor[batch_idx]

        ip_batch = IPNN(input_batch)
        zt_scale = data.scale(input_batch, ip_batch)
        loss = MSE(zt_scale, y_batch[:, data.partial_vars_idx]).mean()

        loss.backward()
        pre_opt.step()
        pre_opt.zero_grad()

        if n % 1000 == 0:
            print(f'Pre-training at {n} iter with loss {loss.item():.6f}', end='\r')

def unsupervised_training_meip(IPNN, data, args, save_dir,seed=2023, input_tensor=None):
    total_iteration = args['total_iteration']
    fixed_margin = args['fixed_margin']
    gamma = args['gamma']
    training_sample = args['training_sample']
    batch_size = min(args['batch_size'], training_sample)
    cal_boudanry_dist = args['cal_boudanry_dist']
    ipnn_opt = optim.AdamW(IPNN.parameters(), lr=args['lr'], weight_decay=1e-5)
    ipnn_sche = optim.lr_scheduler.StepLR(ipnn_opt,
                            step_size=args['lr_decay_step'],
                            gamma=args['lr_decay'])
    # Initialize tracking lists
    metrics = {
        'penalty': [], 'gamma': [], 'eccentricity': [],
        'max_boundary_dist': [], 'min_boundary_dist': [],
        'valid_rate': [], 'training_time': []
    }
    if input_tensor is None:
        train_index = np.random.choice(np.arange(len(data.trainX)), training_sample)
        input_tensor = data.trainX.to(DEVICE)[train_index]
        output_tensor = data.trainY.to(DEVICE)[train_index]
    # Pre-training phase
    if args['pre_training'] > 0:
        _pretrain(IPNN, data, input_tensor, output_tensor, args, batch_size)
    if not fixed_margin:
        for module in IPNN.modules():
            if isinstance(module, NoiseModule):
                if hasattr(module, 'log_gamma'):
                    module.log_gamma.requires_grad = True

    for n in range(total_iteration):
        start_time = time.time()

        IPNN.train()
        batch_index = np.random.choice(np.arange(input_tensor.shape[0]), batch_size, replace=False)
        input_batch = input_tensor[batch_index]
        # penalty loss to find interior points
        ip_batch = IPNN(input_batch)
        zt_scale = data.scale(input_batch, ip_batch)
        yt_full = data.complete_partial(input_batch, zt_scale)
        ineq_vio = data.ineq_resid(input_batch, yt_full).abs()
        penalty_loss = (ineq_vio).mean()

        # Initialize regularization term
        reg_loss = None
        for module in IPNN.modules():
            if isinstance(module, NoiseModule):
                reg_loss = module.log_gamma / np.log(10)
        if fixed_margin: loss = penalty_loss
        else: loss = penalty_loss - 1e-2 * reg_loss

        loss.backward()
        ipnn_opt.step()
        ipnn_sche.step()
        ipnn_opt.zero_grad()

        # Record metrics
        metrics['penalty'].append(penalty_loss.detach().cpu())
        metrics['gamma'].append(reg_loss.detach().cpu() if gamma > 0 else 0.0)
        metrics['training_time'].append(time.time() - start_time)

        if (n) % args['resultsSaveFreq'] == 0:
            IPNN.eval()
            with torch.no_grad():
                batch_index = np.random.choice(np.arange(input_tensor.shape[0]), batch_size)
                input_batch = input_tensor[batch_index]
                ip_batch = IPNN(input_batch)
                zt_scale = data.scale(input_batch, ip_batch)
                yt_full = data.complete_partial(input_batch, zt_scale)
                ineq_vio = data.ineq_resid(input_batch, yt_full)
                penalty_max = torch.max(torch.abs(ineq_vio), dim=-1)[0]
                penalty_mean = torch.sum(torch.abs(ineq_vio), dim=-1)
                valid_rate = (penalty_max <= 1e-5).sum() / penalty_max.shape[0]
                metrics['valid_rate'].append(valid_rate.cpu().numpy())
                feasible_index = penalty_max <= 1e-5

                if cal_boudanry_dist and ip_batch[feasible_index].shape[0]>0:
                    num_boundary_point = args['n_boundary_point']
                    num_bisect_step = args['n_bisect_sampling']
                    feasible_ip = ip_batch[feasible_index]
                    feasible_input = input_batch[feasible_index]    
                    st = time.time()
                    boundary_point = general_boundary_sampling(data,
                                                                feasible_input,
                                                                feasible_ip, 
                                                                num_boundary_point,
                                                                num_bisect_step)
                    et = time.time()
                    point_to_boundary_dist = torch.norm(feasible_ip.view(feasible_ip.shape[0],1,-1) - boundary_point, dim=-1, p=2)
                    dist_max = point_to_boundary_dist.max(-1)[0]
                    dist_min = point_to_boundary_dist.min(-1)[0]
                    metrics['eccentricity'].append((dist_max - dist_min).mean().cpu().numpy())
                    metrics['max_boundary_dist'].append(dist_max.mean().cpu().numpy())
                    metrics['min_boundary_dist'].append(dist_min.mean().cpu().numpy())
                else:
                    metrics['eccentricity'].append(np.nan)
                    metrics['max_boundary_dist'].append(np.nan)
                    metrics['min_boundary_dist'].append(np.nan)


                print(f'iteration: {n}, penalty: {penalty_mean.cpu().numpy().mean():.4f}, ',
                      f'feasibility: {(penalty_max <  1e-5).cpu().numpy().mean():.4f},',
                      f'log-gamma: {metrics["gamma"][-1]:.4f}, ')
                torch.save(IPNN, os.path.join(save_dir, f'ipnn_{training_sample}_{fixed_margin}_{gamma}.pth'))

    # IPNN.eval()
    # with torch.no_grad():
    #     # torch.manual_seed(seed)
    #     # input_batch = torch.rand([4096, input_batch.shape[1]]).to(DEVICE)
    #     input_batch = data.testX
    #     input_batch = input_batch * (data.input_U - data.input_L) + data.input_L
    #     ip_batch = IPNN(input_batch)
    #     zt_scale = data.scale(input_batch, ip_batch)
    #     yt_full = data.complete_partial(input_batch, zt_scale)
    #     ineq_vio = data.ineq_resid(input_batch, yt_full)
    #     penalty_max = torch.max(torch.abs(ineq_vio), dim=-1)[0]
    #     ip_valid_rate = (penalty_max <= 1e-5).sum() / penalty_max.shape[0]
    #     print(f'ip_valid_rate: {ip_valid_rate:.4f}')

    training_record = {
        'perturbed_penalty_loss': np.stack([metrics['penalty'], metrics['gamma']]),
        'robust_margin': metrics['gamma'],
        'point_to_boudanry_dist': np.stack([
            metrics['eccentricity'],
            metrics['max_boundary_dist'],
            metrics['min_boundary_dist'] ]),
        'valid_rate_list': metrics['valid_rate'],
        'training_time_list': metrics['training_time']}
        # 'ip_valid_rate': ip_valid_rate.detach().cpu().numpy()}
    return IPNN, training_record

def train_meip_mapping(data, args, save_dir):
    seed = args['seed']
    paras = args['ipnn_para']
    fixed_margin = paras['fixed_margin']
    gamma = paras['gamma']
    training_sample = paras['training_sample']
    if 'Eq' in args['algoType']:
        out_dim = len(data.partial_vars_idx)
    else:
        out_dim = data.testY.shape[1]
    in_dim = data.xdim
    n_layer = args['nn_para']['num_layer']
    torch.manual_seed(seed)
    IPNN = NoiseResNet(in_dim, out_dim, out_dim+1, n_layer,
                       act='sigmoid',
                       fixed_margin=fixed_margin,
                       gamma=gamma,
                       noise_type='add').to(DEVICE)
    # input_tensor = torch.rand([paras['training_sample'], in_dim]).to(DEVICE)
    # input_tensor = input_tensor * (data.input_U - data.input_L) + data.input_L
    model, training_record = unsupervised_training_meip(IPNN, data, paras, save_dir, seed)
    np.save(os.path.join(save_dir, f'ipnn_{fixed_margin}_{gamma}.npy'), training_record)
    torch.save(model, os.path.join(save_dir, f'ipnn_{training_sample}_{fixed_margin}_{gamma}.pth'))
    return training_record





def eval_ipnn_solution(data, args, save_dir):
    fixed_margin = args['ipnn_para']['fixed_margin']
    gamma = args['ipnn_para']['gamma']
    training_sample = args['ipnn_para']['training_sample']
    eps_converge = args['proj_para']['corrEps']
    ipnn = torch.load(
        os.path.join(save_dir, f'ipnn_{training_sample}_{fixed_margin}_{gamma}.pth'),
        map_location=DEVICE
    )
    # nn_save_dir = os.path.join('models', args['probType'], str(data))
    # model_path = os.path.join(nn_save_dir, f'supervise_NN_Eq.pth')
    # NN_model = torch.load(model_path, map_location=DEVICE)
    with torch.no_grad():
        X = data.testX
        ipnn.eval()
        ip_batch = ipnn(X)
        # Scale predictions
        ip_scale = data.scale(X, ip_batch)
        # Complete partial solutions if needed
        Y_ip = data.complete_partial(X, ip_scale)

        # Add prediction noise if specified
        if args['ipnn_para']['sa_pred_noise'] > 0:
            torch.manual_seed(args['seed'])
            # Z_pred = NN_model(X)
            # noise = torch.randn_like(Z_pred).to(Z_pred.device)
            # Z_pred = Z_pred.clone() +  noise * args['ipnn_para']['sa_pred_noise']
            # Z_pred = torch.clamp(Z_pred, 0, 1)
            Z_pred = torch.rand_like(ip_batch).to(Y_ip.device) * args['ipnn_para']['sa_pred_noise']
            Z_scale = data.scale(X, Z_pred)
            Y_scale = data.complete_partial(X, Z_scale)
        Yproj, _ = ip_bisection(ip_batch.unsqueeze(1), data, args, Z_pred, X, eps_converge)

        IP_dist = (Y_scale - Y_ip).abs().mean(1)
        Proj_dist = (Yproj - Y_scale).abs().mean(1)
    return IP_dist.cpu().numpy(), Proj_dist.cpu().numpy()






###################################################################
# (Un)supervised Training for NN Solver for Power Flow
###################################################################
def train_pfnn_solver(data, args, save_dir):
    lr = args['pfnn_para']['lr']
    nepochs = args['pfnn_para']['total_iteration']
    batch_size = args['pfnn_para']['batch_size']
    lr_decay = args['pfnn_para']['lr_decay']
    lr_decay_step = args['pfnn_para']['lr_decay_step']
    NN_structure = args['predType']
    pre_training = args['pfnn_para']['pre_training']
    ### Equality completion
    in_dim = data.xdim + len(data.partial_vars_idx)
    out_dim = len(data.newton_vars_idx)
    n_layer = args['pfnn_para']['num_layer']
    PFNN = ResNet(in_dim, out_dim, (in_dim+out_dim)//2, n_layer)
    PFNN.to(DEVICE)
    solver_opt = optim.AdamW(PFNN.parameters(), lr=lr, weight_decay=1e-5)
    solver_shce = optim.lr_scheduler.StepLR(solver_opt,
                                            step_size=lr_decay_step,
                                            gamma=lr_decay)
    loss = nn.MSELoss()
    stats = {}
    PFNN.train()
    Xtrain = data.trainX.to(DEVICE)
    Ytrain = data.trainY.squeeze().to(DEVICE)
    Xtest = data.testX.to(DEVICE)
    Ytest = data.testY.squeeze().to(DEVICE)
    training_time_list = []
    total_time = 0
    """
    pre-training
    """
    for pn in range(pre_training):
        batch_index = np.random.choice(np.arange(Xtrain.shape[0]), 32, replace=False)
        # Get train loss
        Xtrain_batch = Xtrain[batch_index]
        Ytrain_batch = Ytrain[batch_index]
        # Z_pred_batch = model(Xtrain_batch)
        Z_pred_scale_batch = Ytrain_batch[:, data.partial_vars_idx]
        Z_pred = data.inverse_scale(Xtrain_batch, Z_pred_scale_batch)
        Y_full = data.neural_complete_partial(Xtrain_batch, Z_pred, PFNN)
        eq_vio = data.eq_resid(Xtrain_batch, Y_full).abs()
        mse_loss = loss(Ytrain_batch, Y_full).mean() + eq_vio.mean()
        mse_loss.backward()
        if pn % 1000 == 0:
            print('Pre-training loss', pn, mse_loss.mean(), end='\r')
        solver_opt.step()
        solver_shce.step()
        solver_opt.zero_grad()

    for i in range(nepochs + 1):
        epoch_stats = {}
        iter_st = time.time()
        """
        formal training
        """
        batch_index = np.random.choice(np.arange(Xtrain.shape[0]), batch_size, replace=False)
        Xtrain_batch = (Xtrain[batch_index]).view(batch_size,-1)
        Ytrain_batch = (Ytrain[batch_index]).view(batch_size,-1)
        Z_pred_scale_batch = Ytrain_batch[:, data.partial_vars_idx]
        Z_pred = data.inverse_scale(Xtrain_batch, Z_pred_scale_batch)
        Z_pred *= (1 + torch.randn_like(Z_pred).to(DEVICE) * 0.05)
        Y_full = data.neural_complete_partial(Xtrain_batch, Z_pred, PFNN)
        eq_vio = data.eq_resid(Xtrain_batch, Y_full).abs()
        train_loss = eq_vio.mean()

        train_loss.backward()
        solver_opt.step()
        solver_shce.step()
        solver_opt.zero_grad()

        iter_et = time.time()
        train_time = iter_et - iter_st
        training_time_list.append(train_time)
        total_time+= train_time
        # Print results
        if i % 1000 == 0:
            print(i, train_loss.detach(), end='\r')

    with open(os.path.join(save_dir, f'PFNN.pth'), 'wb') as f:
        torch.save(PFNN, f)
    return PFNN, stats







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
    training_approach = args['nn_para']['approach']
    ### Equality completion
    if args['probType'] in ['graph_acopf', 'power_control']:
        n_out = data.xdim
    elif args['probType'] in ['graph_qp']:
        n_out = len(data.partial_vars_idx)
    n_in = data.cdim # node context variables
    e_in = data.edim # edge context variables
    n_layer = args['nn_para']['num_layer']
    n_hid = max(n_in+1, 64)
    model = GraphNet(n_in, n_out, n_hid, e_in, n_layer, act='sigmoid')
    model.to(DEVICE)
    solver_opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
        model.train()
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

        Z_pred_batch = model(Ctrain_batch, Etrain_batch, Adjtrain_batch)
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
        if training_approach == 'supervise':
            mse_loss = ((Y_pred_scale_batch - Xtrain[batch_index]) ** 2).mean()
            train_loss = mse_loss \
                        + args['nn_para']['w_ineq'] * penalty.mean() \
                        + args['nn_para']['w_obj'] * training_obj.mean()
        else:
            train_loss = args['nn_para']['w_ineq'] * penalty.mean() \
                         + args['nn_para']['w_obj'] * training_obj.mean()
        train_loss.backward()
        solver_opt.step()
        solver_shce.step()
        solver_opt.zero_grad()

        # Print results
        if i % args['resultsSaveFreq'] == 0 and i > 0:
            model.eval()
            with torch.no_grad():
                graph_eval_nn_solution(data, Ctest, Etest, Xtest, Adjtest, model, args, save_dir, 'test', epoch_stats)
            print('\nEpoch:{}, Fea_rate:{:.2f}, '
                  '\nRaw_loss: MSE({:.4f}), MAP({:.4f}), '
                  '\nRaw_obj: MSE({:.4f}), MAP({:.4f}), '
                  '\nRaw_vio: Max({:.4f})'.format(
                i, epoch_stats['test_raw_fea_rate'],
                np.mean(epoch_stats['test_raw_mse_loss']), np.mean(epoch_stats['test_raw_mape_loss']),
                np.mean(epoch_stats['test_raw_obj_mse']),  np.mean(epoch_stats['test_raw_obj_mape']),
                np.mean(epoch_stats['test_raw_vio'])))
            with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
                torch.save(model, f)
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
    return model, stats

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

    model = torch.load(os.path.join(model_save_dir, 'model.pth'), map_location=DEVICE)
    epoch_stats = {}
    with torch.no_grad():
        graph_eval_nn_solution(data, Ctest, Etest, Xtest, Adjtest, model, args, model_save_dir, 'test', epoch_stats)
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

def graph_eval_nn_solution(data, C, E, Ytarget, Adjtest, model, args, save_dir, prefix, stats):
    ### NN solution prediction
    raw_start_time = time.time()
    model.eval()
    with torch.no_grad():
        Y_pred = model(C, E, Adjtest)
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

def unsupervised_training_graph_mdh(model, data, probType, x_tensor, input_tensor, e_tensor, adj_tensor, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    volume_list = []
    penalty_list = []
    distortion_list = []
    bias_tensor = x_tensor[:batch_size]
    bias_tensor[bias_tensor!=0] = np.mean(args['bound'])
    # torch.ones_like(x_tensor[:batch_size]).to(x_tensor.device) * np.mean(args['bound'])
    optimizer = optim.AdamW(model.parameters(),
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

        input_batch = adj_input = e_input = None
        if input_tensor is not None:
            batch_index = np.random.choice([i for i in range(input_tensor.shape[0])], batch_size, replace=True)
            input_batch = input_tensor[batch_index]
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
        xt, logdet, logdis = model(x_input, input_batch, e_input, adj_input)
        if 'acopf' in probType:
            input_batch, xt = data.extend_prediction(input_batch, xt)
        volume = logdet
        xt_scale = data.scale(input_batch, xt)
        xt_full = data.complete_partial(input_batch, xt_scale)
        if 'power_control' in probType:
            penalty = data.cal_penalty(e_input, xt_full, adj_input).sum(-1)
        else:
            penalty = data.cal_penalty(input_batch, xt_full, adj_input).sum(-1)
        loss =  ( torch.mean(logdis) - torch.mean(volume) ) / n_dim \
                + args['w_penalty'] * penalty.mean()
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
                #     input_batch, x0 = data.graph_prediction(input_batch, bias_tensor)
                batch_index = np.random.choice([i for i in range(input_tensor.shape[0])], batch_size, replace=True)
                input_batch = input_tensor[batch_index]
                x0, _, _ = model(bias_tensor, input_batch, e_input, adj_input)
                if 'acopf' in probType:
                    input_batch, x0 = data.extend_prediction(input_batch, x0)
                xt_scale = data.scale(input_batch, x0)
                xt_full = data.complete_partial(input_batch, xt_scale)
                if 'power_control' in probType:
                    penalty_0 = data.check_feasibility(e_input, xt_full, adj_input)
                else:
                    penalty_0 = data.check_feasibility(input_batch, xt_full, adj_input)
            print(f'Iteration: {n}/{total_iteration}, '
                  f'Volume: {volume_list[-1]:.4f}, '
                  f'Disotrtion: {distortion_list[-1]:.4f}, '
                  f'Penalty: {penalty_list[-1]:.4f}, '
                  f'Center Penalty: {penalty_0.sum(-1).cpu().numpy().mean():.4f}, '
                  f'Valid rate: {(penalty_0.max(-1)[0] < 1e-5).cpu().numpy().mean():.2f}',
                  end='\n')
    return model, volume_list, penalty_list








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



def csv_record(epoch_stats, data, args, record_file):
    labels = ['Prob', 'Train', 'NN', 'Proj',
              'Fea_rate',  'Ineq_vio',  'Eq_vio',
              'Sol_MAPE', 'Infea_Sol_MAPE', 'Obj_MAPE', 'Infea_Obj_MAPE',
              'Ave_speedup', 'Ave_proj_sppedup',]
    if not os.path.exists(record_file):
        data_record = pd.DataFrame(columns=labels)
        data_record.loc[0] = [str(0)]*len(labels)
    else:
        data_record = pd.read_csv(record_file, index_col=False)
    ### Record pure NN prediction & x-Proj post-processing
    infeasible_index = epoch_stats['test_index_infeasible']
    if args['projType'] == 'None':
        row_index = (data_record['Prob'] == str(data)) & \
                    (data_record['Train'] == args['nn_para']['approach']) & \
                    (data_record['NN'] == args['predType']) & \
                    (data_record['Proj'].isna())
    else:
        row_index = (data_record['Prob'] == str(data)) & \
                    (data_record['Train'] == args['nn_para']['approach']) & \
                    (data_record['NN'] == args['predType']) & \
                    (data_record['Proj'] == args['projType'])
    if not row_index.any():
        row_index = data_record.shape[0]
        data_record.loc[data_record.shape[0]] = {'Prob': str(data), \
                                                 'Train': args['nn_para']['approach'], \
                                                 'NN': args['predType'], \
                                                 'Proj': args['projType']}
    data_record.loc[row_index, 'Fea_rate'] = round((1-np.mean(epoch_stats['test_cor_vio_instance']))*100, 4)
    data_record.loc[row_index, 'Ineq_vio'] = round(np.mean(epoch_stats['test_cor_ineq_sum'][infeasible_index]), 4)
    data_record.loc[row_index, 'Eq_vio'] = round(np.mean(epoch_stats['test_cor_eq_sum'][infeasible_index]), 4)

    # data_record.loc[row_index, 'Sol_MAE'] = round(np.mean(epoch_stats['test_cor_mae_loss']), 2)
    data_record.loc[row_index, 'Sol_MAPE'] = round(np.mean(epoch_stats['test_cor_mape_loss'])*100, 6)
    # data_record.loc[row_index, 'Infea_Sol_MAE'] = round(np.mean(epoch_stats['test_cor_mae_loss'][infeasible_index]), 2)
    # data_record.loc[row_index, 'Infea_Sol_MAPE'] = round(np.mean(epoch_stats['test_cor_mape_loss'][infeasible_index])*100, 2)

    # data_record.loc[row_index, 'Obj_MAE'] = round(np.mean(epoch_stats['test_cor_obj_mae']), 2)
    data_record.loc[row_index, 'Obj_MAPE'] = round(np.mean(epoch_stats['test_cor_obj_mape'])*100, 6)
    # data_record.loc[row_index, 'Infea_Obj_MAE'] = round(np.mean(epoch_stats['test_cor_obj_mae'][infeasible_index]), 2)
    # data_record.loc[row_index, 'Infea_Obj_MAPE'] = round(np.mean(epoch_stats['test_cor_obj_mape'][infeasible_index])*100, 2)

    data_record.loc[row_index, 'solver_time'] = round(epoch_stats['batch_solver_time'], 4)
    data_record.loc[row_index, 'total_time'] = round(epoch_stats['batch_nn_total_time'], 4)
    data_record.loc[row_index, 'raw_time'] = round(epoch_stats['batch_nn_raw_time'], 4)
    data_record.loc[row_index, 'porj_time'] = round(epoch_stats['batch_nn_proj_time'], 4)

    data_record.to_csv(record_file, index=False)


