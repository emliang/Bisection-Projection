import numpy as np
import torch
import torch.optim as optim
import time
# torch.set_default_dtype(torch.float64)

###################################################################
# Unsupervised Training for Minimum-Distortion-Homeomoprhic Mapping
###################################################################
def unsupervised_training_mdh(model, constraints, x_tensor, c_tensor, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    penalty_coefficient = args['penalty_coefficient']
    distortion_coefficient = args['distortion_coefficient']
    transport_coefficient = args['transport_coefficient']
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
    model.train()
    for n in range(total_iteration):
        optimizer.zero_grad()
        batch_index = np.random.choice([i for i in range(x_tensor.shape[0])],  batch_size, replace=True)
        x_input = x_tensor[batch_index]
        batch_index = np.random.choice([i for i in range(c_tensor.shape[0])], batch_size, replace=True)
        c_input = c_tensor[batch_index]
        # x_input.requires_grad = True
        n_dim = x_input.shape[1]
        if args['scale_ratio']>1:
            xt, logdet, _ = model(x_input, c_input)
            _, _, logdis = model((x_input-bias_tensor)*args['scale_ratio']+bias_tensor, c_input)
        else:
            xt, logdet, logdis = model(x_input, c_input)
        trans = torch.mean((x_input - xt) ** 2, dim=1, keepdim=True)
        volume = logdet
        xt_scale = constraints.scale(c_input, xt)
        xt_full = constraints.complete_partial(c_input, xt_scale)
        violation = constraints.cal_penalty(c_input, xt_full)
        penalty = torch.sum(torch.abs(violation), dim=-1, keepdim=True)
        loss = -  torch.mean(volume) /n_dim  \
                +  penalty_coefficient * torch.mean(penalty) \
                +  distortion_coefficient * torch.mean(logdis) \
                +  transport_coefficient * torch.mean(trans) 
        loss.backward()
        optimizer.step()
        scheduler.step()
        volume_list.append(torch.mean(logdet).detach().cpu().numpy()/n_dim)
        penalty_list.append(torch.mean(penalty).detach().cpu().numpy())
        dist_list.append(torch.mean(logdis).detach().cpu().numpy()/args['num_layer'])
        trans_list.append(torch.mean(trans).detach().cpu().numpy())
        if n%1000==0 and n>0:
            model.eval()
            with torch.no_grad():
            # bias_tensor.requires_grad = True
                x0,_,_ = model(bias_tensor, c_input)
                x0_scale = constraints.scale(c_input, x0)
                x0_full = constraints.complete_partial(c_input, x0_scale)
                violation_0 = constraints.check_feasibility(c_input, x0_full)
                penalty_0 = torch.sum(torch.abs(violation_0), dim=-1, keepdim=True)
            print(f'Iteration: {n}/{total_iteration}, '
                  f'Volume: {volume_list[-1]:.4f}, '
                  f'Penalty: {penalty_list[-1]:.4f}, '
                  f'Distortion: {dist_list[-1]:.4f}, '
                  f'Transport: {trans_list[-1]:.4f}, '
                  f'Test valid: {torch.mean(penalty_0).detach().cpu().numpy():.8f}',
                  end='\n')
    return model, volume_list, penalty_list, dist_list, trans_list




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
    # unit_vec = unit_vec / torch.norm(unit_vec, dim=-1, p=2, keepdim=True)
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
        infeasible_index = penalty > 0
        feasible_index = penalty <= 0
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
    c_dim = c_tensor.shape[1]
    optimizer = optim.Adam(model.parameters(),
                           lr=args['lr'],
                           weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=args['lr_decay_step'],
                            gamma=args['lr_decay'])
    penalty_list = []
    eccentricity_list = []
    model.train()
    for n in range(total_iteration):
        optimizer.zero_grad()
        batch_index = np.random.choice(np.arange(c_tensor.shape[0]), batch_size)
        c_batch = c_tensor[batch_index]
        xt = model(c_batch)
        c_batch = c_batch.view(-1, 1, c_dim).repeat(1, n_ip, 1)
        ip_batch = xt.view(-1, n_ip, n_dim)
        c_batch_extend = c_batch.view(-1, c_dim)
        ip_batch_extend = ip_batch.view(-1, n_dim)
        xt_scale = constraints.scale(c_batch_extend, ip_batch_extend)
        xt_full = constraints.complete_partial(c_batch_extend, xt_scale)
        violation = constraints.cal_penalty(c_batch_extend, xt_full)
        penalty = torch.sum(torch.abs(violation), dim=-1)
        penalty = penalty.view(-1, n_ip).sum(dim=1)
        ### infeasible loss function
        infeasible_index = penalty > 0
        infeasible_loss = (penalty[infeasible_index]).mean()
        ### feasible loss function
        feasible_index = penalty <= 1e-5
        eccentric_dist = 0
        feasible_loss = 0
        sample_time =  0
        if ip_batch[feasible_index].shape[0]>0:
            feasible_ip = ip_batch[feasible_index]
            feasible_input = c_batch[feasible_index]
            ### boudary sampling
            st = time.time()
            ip_index = np.random.randint(n_ip)
            boundary_point = general_boundary_sampling(constraints,
                                                       feasible_input,
                                                       feasible_ip.detach(),
                                                       num_boundary_point,
                                                       num_bisect_step)
            et = time.time()
            point_to_boundary_dist = torch.norm(feasible_ip.view(-1, n_ip, 1, n_dim) - boundary_point, dim=-1, p=2)
            ### cal eccentric dist
            if args['softmin']:
                point_to_boundary_dist_min = logsumexp(point_to_boundary_dist, omega=-(n + 1)/10)
            else:
                point_to_boundary_dist_min = point_to_boundary_dist.min(1)[0]
            # softmin_time = et-st
            if args['softrange']:
                dist_max = logsumexp(point_to_boundary_dist_min, omega=(n + 1)/10)
                dist_min = logsumexp(point_to_boundary_dist_min, omega=-(n + 1)/10)
            else:
                dist_max = point_to_boundary_dist_min.max(-1)[0]
                dist_min = point_to_boundary_dist_min.min(-1)[0]
            # print(torch.abs(point_to_boundary_dist_min.max(-1)[0] - logsumexp(point_to_boundary_dist_min, omega=(n + 1))).max())
            # print(torch.abs(point_to_boundary_dist_min.min(-1)[0] - logsumexp(point_to_boundary_dist_min, omega=-(n + 1))).max())
            ### cal loss function
            feasible_loss = ((dist_max - dist_min)).mean()
            eccentric_dist = ((point_to_boundary_dist.min(1)[0]).max(1)[0]
                              - (point_to_boundary_dist.min(1)[0]).min(1)[0]).mean()
            eccentricity_list.append(eccentric_dist.detach().cpu())
            sample_time = et-st
        else:
            eccentricity_list.append(np.nan)
        penalty_list.append(infeasible_loss.detach().cpu())
        loss = args['w_penalty'] * infeasible_loss + feasible_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
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
                print(f'n_ip: {n_ip}, iteration: {n}, valid: {penalty.max():.4f}, sampling time: {sample_time:.4f}')
    return model, penalty_list, eccentricity_list




###################################################################
# Binary Search for Homeomorphic Projection
###################################################################
def homeo_bisection(model, constraints, args, x_infeasible, c_infeasible):
    model.eval()
    steps = args['proj_para']['corrTestMaxSteps']
    eps = args['proj_para']['corrEps']
    bis_step = args['proj_para']['corrBis']
    with torch.no_grad():
        bias = torch.tensor(np.mean(args['mapping_para']['bound']), device=x_infeasible.device).view(1, -1)
        x_latent_infeasible, _, _ = model.backward(x_infeasible, c_infeasible)
        alpha_upper = torch.ones([c_infeasible.shape[0], 1], device=x_infeasible.device)
        alpha_lower = torch.zeros([c_infeasible.shape[0], 1], device=x_infeasible.device)
        for k in range(steps):
            alpha = (1-bis_step)*alpha_lower + bis_step*alpha_upper
            xt, _, _ = model(alpha * (x_latent_infeasible - bias) + bias, c_infeasible)
            xt_scale = constraints.scale(c_infeasible, xt)
            xt_full = constraints.complete_partial(c_infeasible, xt_scale)
            violation = constraints.check_feasibility(c_infeasible, xt_full)
            penalty = torch.max(torch.abs(violation), dim=1)[0]
            sub_feasible_index = (penalty < eps)
            sub_infeasible_index = (penalty >= eps)
            alpha_lower[sub_feasible_index] = alpha[sub_feasible_index]
            alpha_upper[sub_infeasible_index] = alpha[sub_infeasible_index]
            if (alpha_upper-alpha_lower).max()<1e-2:
                break
        xt, _, _ = model(alpha_lower * (x_latent_infeasible - bias) + bias, c_infeasible)
        xt_scale = constraints.scale(c_infeasible, xt)
        xt_full = constraints.complete_partial(c_infeasible, xt_scale)
    return xt_full, k


###################################################################
# Binary Search in the Constraint Space
###################################################################
def gauge_bisection(model, constraints, args, x_infeasible, c_infeasible):
    model.eval()
    steps = args['proj_para']['corrTestMaxSteps']
    eps = args['proj_para']['corrEps']
    bis_step = args['proj_para']['corrBis']
    with torch.no_grad():
        bias_tensor = torch.ones_like(x_infeasible, device=x_infeasible.device) * np.mean(args['mapping_para']['bound'])
        try:
            x_interior_feasible, _, _ = model(bias_tensor, c_infeasible)
        except:
            x_interior_feasible = model.interior_forward(c_infeasible)
        # x_latent_infeasible, _, _ = model.backward(x_tensor_infeasible, c_tensor_infeasible)
        alpha_upper = torch.ones([c_infeasible.shape[0], 1], device=x_infeasible.device)
        alpha_lower = torch.zeros([c_infeasible.shape[0], 1], device=x_infeasible.device)
        for k in range(steps):
            alpha = (1-bis_step)*alpha_lower + bis_step*alpha_upper
            # xt, _,_ = model(alpha * (x_latent_infeasible - bias) + bias, c_tensor_infeasible)
            xt = alpha * (x_infeasible - x_interior_feasible) + x_interior_feasible
            xt_scale = constraints.scale(c_infeasible, xt)
            xt_full = constraints.complete_partial(c_infeasible, xt_scale)
            violation = constraints.check_feasibility(c_infeasible, xt_full)
            penalty = torch.max(torch.abs(violation), dim=1)[0]
            sub_feasible_index = (penalty < eps)
            sub_infeasible_index = (penalty >= eps)
            alpha_lower[sub_feasible_index] = alpha[sub_feasible_index]
            alpha_upper[sub_infeasible_index] = alpha[sub_infeasible_index]
            if (alpha_upper-alpha_lower).max()<1e-2:
                break
        xt = alpha_lower * (x_infeasible - x_interior_feasible) + x_interior_feasible
        # xt, _, _ = model(alpha_lower * (x_latent_infeasible - bias) + bias, c_tensor_infeasible)
        xt_scale = constraints.scale(c_infeasible, xt)
        xt_full = constraints.complete_partial(c_infeasible, xt_scale)
    return xt_full, k    


def ip_bisection(feasible_ip, constraints, args, x_infeasible, c_infeasible):
    steps = args['proj_para']['corrTestMaxSteps']
    eps = args['proj_para']['corrEps']
    bis_step = args['proj_para']['corrBis']
    batch_size = feasible_ip.shape[0]
    n_ip = feasible_ip.shape[1]
    n_dim = x_infeasible.shape[-1]
    c_dim = c_infeasible.shape[-1]
    # direction sampling
    # repeat for multiple IPs and extend to 2-dim vectors
    feasible_ip = feasible_ip.view(-1, n_dim)
    c_infeasible_extend = c_infeasible.view(-1, 1, c_dim).repeat(1,n_ip,1).view(-1, c_dim)
    x_infeasible_extend = x_infeasible.view(-1, 1, n_dim).repeat(1,n_ip,1).view(-1, n_dim)
    # lower bound and upper bound for bisection
    alpha_lower = torch.zeros(size=[batch_size * n_ip, 1]).to(feasible_ip.device)
    alpha_upper = torch.ones(size=[batch_size * n_ip, 1]).to(feasible_ip.device)
    for k in range(steps):
        alpha = (1-bis_step)*alpha_lower + bis_step*alpha_upper
        xt = alpha * (x_infeasible_extend - feasible_ip) + feasible_ip
        xt_scale = constraints.scale(c_infeasible_extend, xt)
        xt_full = constraints.complete_partial(c_infeasible_extend, xt_scale)
        violation = constraints.check_feasibility(c_infeasible_extend, xt_full)
        penalty = torch.max(torch.abs(violation), dim=1)[0]
        sub_feasible_index = (penalty < eps)
        sub_infeasible_index = (penalty >= eps)
        alpha_lower[sub_feasible_index] = alpha[sub_feasible_index]
        alpha_upper[sub_infeasible_index] = alpha[sub_infeasible_index]
        if (alpha_upper-alpha_lower).max()<1e-2:
            break
    x_feasible = alpha_lower * (x_infeasible_extend - feasible_ip) + feasible_ip
    x_feasible = x_feasible.view(batch_size, n_ip, n_dim)
    feasible_ip = feasible_ip.view(batch_size, n_ip, n_dim)
    x_infeasible = x_infeasible.view(-1, 1, n_dim)
    bis_dist = torch.norm(x_feasible - x_infeasible, dim=-1, p=2)
    min_idx = torch.argmin(bis_dist, dim=1).view(batch_size, 1, 1).repeat(1,1,n_dim)
    x_feasible_near = torch.gather(x_feasible, 1, min_idx).view(-1,n_dim)
    feasible_ip_near = torch.gather(feasible_ip, 1, min_idx).view(-1,n_dim)
    xt_scale = constraints.scale(c_infeasible, x_feasible_near)
    xt_full = constraints.complete_partial(c_infeasible, xt_scale)
    # x_full_infeasible = x_full_infeasible.view(batch_size, 1, -1)
    # bis_dist = torch.norm(xt_full - x_full_infeasible, dim=-1, ord=2)
    return xt_full, feasible_ip_near


###################################################################
# Gradient descent 
# Used only at test time, so let PyTorch avoid building the computational graph
###################################################################
def diff_projection(data, X, Y, args):
    take_grad_steps = args['proj_para']['useTestCorr']
    if take_grad_steps:
        lr = args['proj_para']['corrLr']
        eps_converge = args['proj_para']['corrEps']
        max_steps = args['proj_para']['corrTestMaxSteps']
        momentum = args['proj_para']['corrMomentum']
        partial_corr = True if args['proj_para']['corrMode'] == 'partial' else False
        Y_new = Y
        i = 0
        old_step = 0
        while i < max_steps:
            with torch.enable_grad():
                violation = data.cal_penalty(X, Y_new)
                if (torch.max(torch.abs(violation), dim=1)[0].max() < eps_converge):
                    break
                if partial_corr:
                    Y_step = data.ineq_partial_grad(X, Y_new)
                else:
                    ineq_step = data.ineq_grad(X, Y_new)
                    eq_step = data.eq_grad(X, Y_new)
                    Y_step = 0.5 * ineq_step + 0.5 * eq_step
                new_step = lr * Y_step + momentum * old_step
                Y_new = Y_new - new_step
                Y_new = data.complete_partial(X, Y_new[:,data.partial_vars_idx])
                old_step = new_step
                i += 1
        return Y_new, i
    else:
        return Y, 0






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
    record_file = 'results/all_bp_record.csv'
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


