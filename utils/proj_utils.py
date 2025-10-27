import numpy as np
import torch


###################################################################
# Binary Search for Homeomorphic Projection
###################################################################
def homeo_bisection(model, constraints, args, x_infeasible, c_infeasible,eps_converge=1e-3):
    model.eval()
    steps = args['proj_para']['corrTestMaxSteps']
    eps = args['proj_para']['corrEps']
    bis_step = args['proj_para']['corrBis']
    k = 0
    bias = args['inn_para']['center']
    with torch.no_grad():
        x_latent_infeasible, _, _ = model.inverse(x_infeasible, c_infeasible)
        if len(x_infeasible.shape)==2:
            alpha_upper = torch.ones([c_infeasible.shape[0], 1], device=x_infeasible.device)
            alpha_lower = torch.zeros([c_infeasible.shape[0], 1], device=x_infeasible.device)
        else:
            alpha_upper = torch.ones([c_infeasible.shape[0], 1, 1], device=x_infeasible.device)
            alpha_lower = torch.zeros([c_infeasible.shape[0], 1, 1], device=x_infeasible.device)
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
            if (alpha_upper-alpha_lower).max()<eps_converge:
                break
        xt, _, _ = model(alpha_lower * (x_latent_infeasible - bias) + bias, c_infeasible)
        xt_scale = constraints.scale(c_infeasible, xt)
        xt_full = constraints.complete_partial(c_infeasible, xt_scale)
    return xt_full, k


###################################################################
# Binary Search in the Constraint Space
###################################################################
def ip_bisection(feasible_ip, constraints, args, x_infeasible, c_infeasible, epsilon=1e-3):
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
        if (alpha_upper-alpha_lower).max()<epsilon:
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
def diff_projection(data, X, Y, args, eps_converge):
    take_grad_steps = args['proj_para']['useTestCorr']
    if take_grad_steps:
        lr = args['proj_para']['corrLr']
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
                new_step = (1-momentum) * Y_step + momentum * old_step
                Y_new = Y_new - lr * new_step
                Y_new = data.complete_partial(X, Y_new[:,data.partial_vars_idx])
                old_step = new_step
                i += 1
        return Y_new, i
    else:
        return Y, 0



