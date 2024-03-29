import numpy as np
import torch
from training_utils import *
from sampling_utils import *
from toy_utils import *

###################################################################
# Plot figures for homeomorphic projection
###################################################################
def scatter_constraint_approximation(model, constraints, x_tensor, instance_file, paras):
    simple_set = paras['shape']
    t_test = constraints.t_test
    t_num = len(t_test)
    model.eval()
    fig = plt.figure(figsize=[(t_num+1)*(4+0.2), 4])
    fig.tight_layout()
    n_samples = x_tensor.shape[0]
    grid = plt.GridSpec(1, t_num+1)
    x = x_tensor.detach().cpu().numpy()
    char_size = 26
    if simple_set=='sphere':
        x_norm = np.linalg.norm(x,ord=2, axis=1).T
        norm_tile = r'$\mathcal{B}$: $2$-norm ball'
    else:
        x_norm = np.linalg.norm(x,ord=np.inf, axis=1).T
        norm_tile = r'$\mathcal{B}$: $\infty$-norm ball'
    plt.subplot(grid[0,0])
    plt.scatter(x[:, 0], x[:, 1], s=0.1, alpha=0.7, c=x_norm, label='Sphere sampling')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xticks([-1,0,1], fontsize=18)
    plt.yticks([-1,0,1], fontsize=18)
    plt.title(norm_tile, fontsize=char_size)
    titles = [r'$\Phi(\mathcal{B},\theta_1)$',
                r'$\Phi(\mathcal{B},\theta_2)$',
                r'$\Phi(\mathcal{B},\theta_3)$',
                r'$\Phi(\mathcal{B},\theta_4)$',
                r'$\Phi(\mathcal{B},\theta_5)$',
                r'$\Phi(\mathcal{B},\theta_6)$',
                r'$\Phi(\mathcal{B},\theta_7)$',
                r'$\Phi(\mathcal{B},\theta_8)$',
                r'$\Phi(\mathcal{B},\theta_9)$',]
    for i, t in enumerate(t_test):
        plt.subplot(grid[0,i+1])
        t_tensor = torch.tensor(t).to(device=x_tensor.device).view(1, -1).repeat(n_samples, 1)
        # x_tensor.requires_grad = True
        with torch.no_grad():
            xt, _, _ = model(x_tensor, t_tensor)
        # x = x_tensor.detach().cpu().numpy()
        xt = xt.detach().cpu().numpy()
        constraints.plot_boundary(t)
        plt.scatter(xt[:, 0], xt[:, 1], s=0.1, alpha=0.7, c=x_norm ,label='Constraint approximation')
        plt.title(titles[i], fontsize=char_size)
        plt.xlim([-2.2, 2.2])
        plt.ylim([-2.2, 2.2])
        plt.xticks([-2,0,2], fontsize=18)
        plt.yticks([-2,0,2], fontsize=18)
    plt.subplots_adjust(wspace=0.15)
    seed = paras['seed']
    shape = paras['shape']
    dis_coff = paras['distortion_coefficient']
    plt.savefig(instance_file+f'/{constraints.__class__.__name__}_{shape}_{dis_coff}_{seed}_{len(t_test)}_constraint_approximation_scatter.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_convergence(volume_list, penalty_list, dist_list, trans_list, instance_file, paras):
    index = [i for i in range(len(volume_list))]
    t_num = 3
    fig = plt.figure(figsize=[t_num*(4+1.25),4])
    fig.tight_layout()
    char_size = 26

    plt.subplot(1, t_num, 1)
    plt.plot(index, volume_list, alpha=0.7, c='royalblue', label='Log-det')
    plt.xlabel('Iteration', fontsize=char_size)
    # plt.ylabel('Log-volume', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Ave. log-volume', fontsize=char_size)

    plt.subplot(1, t_num, 2)
    plt.plot(index, penalty_list, alpha=0.7, c='darkorange', label='Penalty')
    plt.xlabel('Iteration', fontsize=char_size)
    # plt.ylabel('Penalty', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Constraint violation', fontsize=char_size)

    plt.subplot(1, t_num, 3)
    plt.plot(index, dist_list, alpha=0.7, c='darkred', label='Dist')
    plt.xlabel('Iteration', fontsize=char_size)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.ylabel('Log-distortion', fontsize=18)
    plt.title('Ave. log-distortion', fontsize=char_size)

    # plt.subplot(1, 4, 4)
    # plt.plot(index, trans_list, alpha=0.7, c='seagreen', label='Dist')
    # plt.xlabel('Iteration', fontsize=15)
    # plt.ylabel('trans_list', fontsize=15)
    # plt.legend(['Transport cost'], fontsize=15)

    seed = paras['seed']
    shape = paras['shape']
    # plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.subplots_adjust(wspace=0.15)
    plt.savefig(instance_file+f'/convergence_{shape}_{seed}.png', bbox_inches='tight',  dpi=300)
    # plt.show()
    # plt.close()

def scatter_constraint_evolution(model, constraints, x_tensor, instance_file, paras):
    simple_set = paras['shape']
    t_test = constraints.t_test
    t_num = paras['num_layer']+1
    model.eval()
    fig = plt.figure(figsize=[t_num*(4+0.2),4])
    fig.tight_layout()
    n_samples = x_tensor.shape[0]
    grid = plt.GridSpec(1, t_num)
    x = x_tensor.detach().cpu().numpy()
    char_size = 26
    if simple_set=='sphere':
        x_norm = np.linalg.norm(x,ord=2, axis=1).T
        norm_tile = r'$\mathcal{B}$: $2$-norm ball'
    else:
        x_norm = np.linalg.norm(x,ord=np.inf, axis=1).T
        norm_tile = r'$\mathcal{B}$: $\infty$-norm ball'
    plt.subplot(grid[0,0])
    plt.scatter(x[:, 0], x[:, 1], s=0.1, alpha=0.7, c=x_norm, label='Sphere sampling')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xticks([-1,0,1], fontsize=18)
    plt.yticks([-1,0,1], fontsize=18)
    plt.title(norm_tile, fontsize=char_size)
    t = t_test[2]
    t_tensor = torch.tensor(t).to(device=x_tensor.device).view(1, -1).repeat(n_samples, 1)
    xt_list = model.forward_traj(x_tensor, t_tensor)
    k = 1
    for i, xt in enumerate(xt_list):
        if (i+1)%3==0: #or i== len(xt_list) - 1 :
            plt.subplot(grid[0, k])
            constraints.plot_boundary(t)
            xt = xt.detach().cpu().numpy()
            plt.scatter(xt[:, 0], xt[:, 1], s=0.1, alpha=0.7, c=x_norm ,label='Constraint evolution')
            plt.xlim([-3.2, 3.2])
            plt.ylim([-3.2, 3.2])
            # if (i+1)%3==0:
            plt.title(f'INN block {k}', fontsize=char_size)
            # else:
            #     plt.title(f'Final output', fontsize=char_size)
            plt.xticks([-2,0,2], fontsize=18)
            plt.yticks([-2,0,2], fontsize=18)
            k += 1
    plt.subplots_adjust(wspace=0.15)
    seed = paras['seed']
    shape = paras['shape']
    plt.savefig(instance_file+f'/{constraints.__class__.__name__}_{shape}_{seed}_{len(t_test)}_constraint_evolution_scatter.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def scatter_projection_error(model, constraints, x_tensor, t_tensor, instance_file, args):
    model.eval()
    proj_error_list =  []
    proj_x_list = []
    h_proj_x_list = []
    h_proj_error_list = []
    parameter_list = []
    penalty_list = []
    char_size = 20

    x_scale = constraints.scale(t_tensor, x_tensor)
    x_full = constraints.complete_partial(t_tensor, x_scale, backward=False)
    violation = constraints.check_feasibility(t_tensor, x_full)
    penalty = torch.max(torch.abs(violation), dim=1)[0]
    infeasible_index = (penalty > args['proj_para']['corrEps']).view(-1)
    x_full = x_full[infeasible_index]
    x_partial = x_tensor[infeasible_index]
    t_tensor = t_tensor[infeasible_index]
    penalty = penalty[infeasible_index]
    
    proj_x = constraints.opt_proj(t_tensor, x_full).to(x_full.device).view(x_full.shape)
    h_proj_x, h_steps = homeo_bisection(model, constraints, args, x_partial, t_tensor)
    g_proj_x, h_steps = gauge_bisection(model, constraints, args, x_partial, t_tensor)

    proj_error_list = torch.abs(proj_x - x_full).sum(-1).cpu().numpy()
    h_proj_error_list = torch.abs(h_proj_x - x_full).sum(-1).cpu().numpy()
    g_proj_error_list = torch.abs(g_proj_x - x_full).sum(-1).cpu().numpy()
    
    # proj_error_list = torch.cat(proj_error_list, dim=0).view(-1).cpu().numpy()
    # proj_x_list = torch.cat(proj_x_list, dim=0).cpu().numpy()
    # h_proj_x_list = torch.cat(h_proj_x_list, dim=0).cpu().numpy()
    # h_proj_error_list = torch.cat(h_proj_error_list, dim=0).view(-1).cpu().numpy()
    # penalty_list = torch.cat(penalty_list, dim=0).cpu().view(-1).numpy()

    # fig = plt.figure(figsize=[3*(4+1),4])
    # fig.tight_layout()
    # grid = plt.GridSpec(1, 3)
    # plt.subplot(grid[0,0])
    # for i in range(t_tensor.shape[0]):
    #     index = (parameter_list == i) & (~np.isnan(h_proj_error_list))
    #     plt.scatter(penalty_list[index], proj_error_list[index], s=25, alpha=0.5)
    # plt.legend([r'$\theta_1$',r'$\theta_2$',r'$\theta_3$'], loc='upper left', fontsize=char_size-2)
    # plt.xlabel(r'Constraint violation', fontsize=char_size) #: $|\rm{ReLU}(g(x,\theta))|_1$
    # plt.ylabel(r'Proj distance', fontsize=char_size) # : $|x-\rm{Proj}(x)|_1$
    # plt.title(r'Projection', fontsize=char_size)
    # plt.xlim([0, max(penalty_list[~np.isnan(h_proj_error_list)])])
    # plt.ylim([0, max(h_proj_error_list[~np.isnan(h_proj_error_list)])])
    # plt.subplot(grid[0,1])
    # for i in range(t_tensor.shape[0]):
    #     index = (parameter_list == i)& (~np.isnan(h_proj_error_list))
    #     plt.scatter(penalty_list[index], h_proj_error_list[index], s=25, alpha=0.5)
    # plt.legend([r'$\theta_1$',r'$\theta_2$',r'$\theta_3$'], loc='upper left', fontsize=char_size-2)
    # plt.xlabel(r'Constraint violation', fontsize=char_size) # : $|\rm{ReLU}(g(x,\theta))|_1
    # plt.ylabel(r'H-proj distance', fontsize=char_size) # : $|x-\rm{HB}(x)|_1
    # plt.title(r'Homeomorphic projection', fontsize=char_size)
    # plt.xlim([0, max(penalty_list[~np.isnan(h_proj_error_list)])])
    # plt.ylim([0, max(h_proj_error_list[~np.isnan(h_proj_error_list)])])
    # plt.subplot(grid[0,2])

    fig = plt.figure(figsize=[5,4])
    fig.tight_layout()
    index = ~np.isnan(h_proj_error_list)
    plt.scatter(proj_error_list[index], h_proj_error_list[index], s=30, alpha=0.4)
    index = ~np.isnan(g_proj_error_list)
    plt.scatter(proj_error_list[index], g_proj_error_list[index], s=30, alpha=0.4)
    index = ~np.isnan(h_proj_error_list)
    plt.legend(['h_proj', 'bis'])
    plt.plot([0,max(h_proj_error_list[index])], [0,max(h_proj_error_list[index])], c='powderblue', alpha=0.9, linewidth=3)
    # slope = np.mean((h_proj_error_list[index]))/np.mean(proj_error_list[index])
    # plt.text(((np.max(h_proj_error_list[index])+np.min(h_proj_error_list[index])))/2, 0,
    #          r'$\frac{\rm{H-proj\;\;distance}}{\rm{Proj\;\;distance}}\approx$'+'{:.4}'.format(slope),
    #          horizontalalignment='center',
    #          fontsize=char_size-4,
    #          bbox=dict(facecolor='lavender', alpha=0.2))
    # plt.legend(f'slope: {np.mean((h_proj_error_list[index]))/np.mean(proj_error_list[index])}')
    plt.xlabel(r'Proj distance', fontsize=16) # : $|x-\rm{Proj}(x)|_1
    plt.ylabel(r'H-proj distance', fontsize=16) # : $|x-\rm{HfB}(x)|_1
    plt.title(r'Proj vs H-Proj', fontsize=char_size)
    # print('\nH-Proj gap compared with Proj:', np.mean((h_proj_error_list[index])/proj_error_list[index]))
    # print('\nSlope for curve:', np.mean((h_proj_error_list[index]))/np.mean(proj_error_list[index]))
    shape = args['mapping_para']['shape']
    plt.savefig(instance_file+f'/{constraints.__class__.__name__}_{shape}_proj_error_scatter_{x_tensor.shape[1]}.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    return

def plot_projection_traj(model, t, simple_set, num_points=20):
    angles = torch.linspace(0,2*np.pi, num_points).view(-1,1)#sampling_surface(num_points, 2, simple_set) * 1.1
    x_surface = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if simple_set == 'square':
        x_surface = x_surface/torch.norm(x_surface, dim=1, p=torch.inf, keepdim=True)
    elif simple_set == 'sphere':
        x_surface = x_surface/torch.norm(x_surface, dim=1, p=2, keepdim=True)
    x_surface = x_surface * 1.15
    x_surface_tensor = torch.tensor(x_surface).view(-1, 2)
    center = torch.tensor([0,0]).view(-1, 2)
    z_list = []
    x_list = []
    num_interpolation = 200
    for i in range(num_points):
        x_start = x_surface_tensor[i:i+1]
        x_end = center
        inter = torch.linspace(0,1,num_interpolation).view(-1,1)
        x_traj = x_start * (1-inter) + x_end * inter
        t_tensor = torch.tensor(t).view(1, -1).repeat(inter.shape[0], 1)
        with torch.no_grad():
            xt, _, _ = model(x_traj.to(device), t_tensor.to(device))
        z_list.append(x_traj.cpu().numpy())
        x_list.append(xt.detach().cpu().numpy())
    return z_list, x_list

def visualize_homeo_projection(model, constraints, x_tensor, instance_file, paras):
    simple_set = paras['shape']
    t_test = constraints.t_test
    t_num = len(t_test)
    model.eval()
    fig = plt.figure(figsize=[(t_num+1)*(4+0.2), 4])
    fig.tight_layout()
    n_samples = x_tensor.shape[0]
    grid = plt.GridSpec(1, t_num+1)
    x = x_tensor.detach().cpu().numpy()
    char_size = 26
    if simple_set=='sphere':
        x_norm = np.linalg.norm(x,ord=2, axis=1).T
        norm_tile = r'$\mathcal{B}$: $2$-norm ball'
    else:
        x_norm = np.linalg.norm(x,ord=np.inf, axis=1).T
        norm_tile = r'$\mathcal{B}$: $\infty$-norm ball'

    titles = [r'$\Phi(\mathcal{B},\theta_1)$',
                r'$\Phi(\mathcal{B},\theta_2)$',
                r'$\Phi(\mathcal{B},\theta_3)$',
                r'$\Phi(\mathcal{B},\theta_4)$',
                r'$\Phi(\mathcal{B},\theta_5)$',
                r'$\Phi(\mathcal{B},\theta_6)$',
                r'$\Phi(\mathcal{B},\theta_7)$',
                r'$\Phi(\mathcal{B},\theta_8)$',
                r'$\Phi(\mathcal{B},\theta_9)$',]
    for i, t in enumerate(t_test):
        plt.subplot(grid[0,i+1])
        t_tensor = torch.tensor(t).to(device=x_tensor.device).view(1, -1).repeat(n_samples, 1)
        xt, _, _ = model(x_tensor, t_tensor)
        xt = xt.detach().cpu().numpy()
        plt.scatter(xt[:, 0], xt[:, 1], s=0.1, alpha=0.7, c=x_norm ,label='Constraint approximation')
        z_list, x_list = plot_projection_traj(model, t, simple_set, num_points=10)
        constraints.plot_boundary(t)
        for traj in x_list:
            plt.scatter(traj[0, 0], traj[0, 1], marker='.')
            plt.plot(traj[:,0],traj[:,1], linewidth=1.8, alpha=0.8)
        plt.scatter(traj[-1,0], traj[-1,1], marker='o', c='r')
        plt.title(titles[i], fontsize=char_size)
        plt.xlim([-2.2, 2.2])
        plt.ylim([-2.2, 2.2])
        plt.xticks([-2,0,2], fontsize=18)
        plt.yticks([-2,0,2], fontsize=18)

    plt.subplot(grid[0,0])
    plt.scatter(x[:, 0], x[:, 1], s=0.1, alpha=0.7, c=x_norm, label='Sphere sampling')
    for traj in z_list:
        plt.scatter(traj[0, 0], traj[0, 1], marker='.')
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1.8, alpha=0.8)
    plt.scatter(traj[-1,0], traj[-1,1], marker='o', c='r')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xticks([-1,0,1], fontsize=18)
    plt.yticks([-1,0,1], fontsize=18)
    plt.title(norm_tile, fontsize=char_size)

    plt.subplots_adjust(wspace=0.15)
    seed = paras['seed']
    shape = paras['shape']
    dis_coff = paras['distortion_coefficient']
    plt.savefig(instance_file+f'/{constraints.__class__.__name__}_proj_traj_{shape}_{dis_coff}_{seed}_{len(t_test)}_constraint_approximation_scatter.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()



###################################################################
# Plot figures for bisection projection
###################################################################
def offset_sampling(data,c,err,density):
    c_offset=c+torch.tensor([[err,err,np.sqrt(2)*err,np.sqrt(2)*err,np.sqrt(2)*err,np.sqrt(2)*err,0,0]]).to(device)
    offset=data.sampling_boudanry(c_offset.cpu().numpy())
    quadrant_1 = offset[((offset[:,0] > 0) & (offset[:,1] > 0))]
    quadrant_2 = offset[((offset[:,0] < 0) & (offset[:,1] > 0))]
    quadrant_3 = offset[((offset[:,0] < 0) & (offset[:,1] < 0))]
    quadrant_4 = offset[((offset[:,0] > 0) & (offset[:,1] < 0))]
    quadrant_1=quadrant_1[np.argsort(quadrant_1[:, 1]/quadrant_1[:, 0])]
    quadrant_2=quadrant_2[np.argsort(quadrant_2[:, 1]/quadrant_2[:, 0])]
    quadrant_3=quadrant_3[np.argsort(quadrant_3[:, 1]/quadrant_3[:, 0])]
    quadrant_4=quadrant_4[np.argsort(quadrant_4[:, 1]/quadrant_4[:, 0])]
    offset=np.concatenate((quadrant_1,quadrant_2,quadrant_3,quadrant_4),axis=0)
    samples=offset[::density,:]
    samples=torch.tensor(samples).to(device)
    return samples

def plot_bp_loss(data, paras, model_path, instance, n_dim, c_dim):
    """
    traininig loss plotting
    """
    loss_list = np.load(model_path + 'nns/'+ 'loss_' + instance + '.npy', allow_pickle=True)
    total_iteration = paras['total_iteration']
    ### constraint violation
    plt.figure(figsize=[14, 6])
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(total_iteration), loss_list[0], label='constraint violation')
    plt.legend(fontsize=18)
    plt.xlabel('Iteration', fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel('Constraint violation', fontsize=16)
    plt.yticks(fontsize=14)
    plt.title('Constraint violation for IP mapping', fontsize=22)
    ### eccentric distance
    plt.subplot(1, 2, 2)
    plt.scatter(np.arange(total_iteration), loss_list[1], label='eccentricity distance', marker='.', alpha=0.7, c='royalblue')
    plt.legend(fontsize=18)
    plt.xlabel('Iteration', fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel('Eccentric distance', fontsize=16)
    plt.yticks(fontsize=14)
    plt.title('Average eccentric distance for IPs', fontsize=22)
    plt.tight_layout()
    plt.savefig(model_path + 'pics/loss_' + instance + '.png')
    plt.show()
    # plt.close()

def bp_interior_points(data, paras, model_path, instance, n_dim, c_dim):
    """
    IPs visualization
    """
    n_ip = paras['n_ip']
    model = torch.load(model_path + 'model_' + instance + '.pth', map_location=device)
    if paras['fix_input']:
        c_sample = np.array([[0.5, 0.5, 2, 2, 2, 2, 1, 1]])
    else:
        np.random.seed(2002)
        c_sample = np.random.uniform(low=data.sampling_range[0],
                              high=data.sampling_range[1],
                              size=[4, c_dim], )

    c_tensor = torch.tensor(c_sample).to(device)
    n_test = c_tensor.shape[0]
    plt.figure(figsize=[(n_test)*(4+0.2), 4])
    for i in range(n_test):
        ct = c_tensor[i:i + 1]
        model.eval()
        with torch.no_grad():
            ip = model(ct).view(-1, paras['n_ip'], n_dim)
        ct_extend = ct.view(-1, 1, c_dim).repeat(1, paras['n_ip'], 1)
        plt.subplot(1, n_test, i + 1)
        data.plot_boundary(ct.cpu().numpy())
        boundary_point = general_boundary_sampling(data, ct_extend, ip, 50, 100)
        boundary_point = boundary_point.view(-1, n_dim).cpu().numpy()
        plt.scatter(boundary_point[:, 0], boundary_point[:, 1], marker='.')
        ip = ip.view(-1, n_dim).cpu().numpy()
        plt.scatter(ip[:, 0], ip[:, 1], c='crimson', marker='*')
        # plt.title(instance, fontsize=22)
        plt.xlim([-1.7, 1.7])
        plt.ylim([-1.7, 1.7])
        plt.xticks([-1, 0, 1], fontsize=14)
        plt.yticks([-1, 0, 1], fontsize=14)
    plt.tight_layout()
    plt.savefig(model_path + 'pics/ips_' + instance + '.png')
    plt.close()

def plot_bp_traj(data, ip, ct, x_tensor, paras):
    n_sample = x_tensor.shape[0]
    penalty = data.cal_penalty(ct.repeat(n_sample, 1), x_tensor).sum(-1)
    x_infeasible = x_tensor[penalty > 0]
    c_infeasible = ct.repeat(x_infeasible.shape[0], 1)
    ip = ip.repeat(x_infeasible.shape[0], 1, 1)
    x_feasible, ip_near = ip_bisection(ip, data, paras, x_infeasible, c_infeasible)
    x_infeasible = x_infeasible.cpu().numpy()
    x_feasible = x_feasible.cpu().numpy()
    ip_near = ip_near.cpu().numpy()
    return x_infeasible, x_feasible, ip, ip_near

def plot_bp_traj_varying_ip(data, paras, model_path, instance, n_dim, c_dim, err, density):
    """
    Bisection projection traj under different number of ips
    """
    fix_input = paras['fix_input']
    softmin = paras['softmin']
    softrange = paras['softrange']
    c_sample = np.array([[0.03, 0.03, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]])
    c_tensor = torch.tensor(c_sample).to(device)
    # modified infeasible sampling method
    x_tensor=offset_sampling(data,c_tensor,err,density).to(device)

    ip_list = [1,2,4]
    n_test = len(ip_list)
    title_list = ['IPNN 1', 'IPNN 2', 'IPNN 4', 'IPNN 8']
    fig = plt.figure(figsize=[(n_test)*(4+0.2), 4.2])
    for i, n_ip in enumerate(ip_list):
        ct = c_tensor
        instance = f'fix_{fix_input}_softmin_{softmin}_softrange_{softrange}_ip_{n_ip}'
        model = torch.load(model_path  + 'nns/' + 'model_' + instance + '.pth', map_location=device)
        model.eval()
        with torch.no_grad():
            ip = model(ct).view(-1, n_ip, n_dim)
        x_infeasible, x_feasible, ip_near, ip = plot_bp_traj(data, ip, ct, x_tensor, paras)
        plt.subplot(1, n_test, i + 1)
        data.plot_boundary(ct.cpu().numpy())
        for j in range(x_infeasible.shape[0]):
            plt.plot([x_infeasible[j, 0], x_feasible[j, 0]], [x_infeasible[j, 1], x_feasible[j, 1]],
                     linewidth=0.5, alpha=0.5, c='lightcoral')
        # for j in range(x_infeasible.shape[0]):
        #     plt.plot([x_feasible[j, 0], ip_near[j, 0]],
        #              [x_feasible[j, 1], ip_near[j, 1]],  '--', linewidth=0.5, alpha=0.1, c='lightcoral')
        plt.scatter(x_feasible[:, 0], x_feasible[:, 1], alpha=0.5, s=15, zorder=3, label='Bis. Projected points')
        plt.scatter(x_infeasible[:, 0], x_infeasible[:, 1], alpha=0.5, s=15, zorder=3, label='Infeasible points')
        plt.scatter(ip[:, 0], ip[:, 1], marker='*', c='crimson', zorder=3, label='Interior points')
        plt.xlim([-1.7, 1.7])
        plt.ylim([-1.7, 1.7])
        # if i in [1,2]:
        #     plt.xticks([], fontsize=14)
        # plt.xticks([-1, 0, 1], fontsize=14)
        # plt.yticks([-1, 0, 1], fontsize=14)
        plt.title(title_list[i], fontsize=19)
    lines, lables = fig.axes[0].get_legend_handles_labels()
    # lines1, labels1 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, lables, fontsize=16, ncol=3, bbox_to_anchor=[0.84, 0.13], framealpha=1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(model_path + 'pics/bis_ip_' + instance + '.png', dpi=300)
    plt.show()
    plt.close()

def plot_bp_traj_varying_input(data, paras, model_path, instance, n_dim, c_dim, err, density):
    """
    Bisection projection traj under different number of inputs
    """
    n_ip = paras['n_ip']
    model = torch.load(model_path + 'nns/' + 'model_' + instance + '.pth', map_location=device)
    if paras['fix_input']:
        c_sample = np.array([[0.5, 0.5, 2, 2, 2, 2, 1, 1]])
    else:
        np.random.seed(2000)
        c_sample = np.random.uniform(low=data.sampling_range[0],
                              high=data.sampling_range[1],
                              size=[3, c_dim], )
        # c_sample=np.array([[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        #                    [1.3, 0.1, 0.9211, 1.2651, 1.0, 0.6, 0.2, 0.3],
        #                    [0.5,0.3,1.4,0.7,0.7,1.4,1.1,1.1]])
    # [0.7, 1.4, 0.7, 0.7, 1.4, 1.4, 0.3, 0.1]

    c_tensor = torch.tensor(c_sample).to(device)
    n_test = c_tensor.shape[0]

    fig = plt.figure(figsize=[(n_test)*(4+0.2), 4.2])
    title_list = ['test input 1', 'test input 2', 'test input 3', 'test input 4']
    for i in range(n_test):
        ct = c_tensor[[i]]
        # modified infeasible sampling method
        x_tensor=offset_sampling(data,ct,err,density).to(device)
        model.eval()
        with torch.no_grad():
            ip = model(ct).view(-1, n_ip, n_dim)
        x_infeasible, x_feasible, ip_near, ip = plot_bp_traj(data, ip, ct, x_tensor, paras)
        plt.subplot(1, n_test, i + 1)
        data.plot_boundary(ct.cpu().numpy())
        for j in range(x_infeasible.shape[0]):
            plt.plot([x_infeasible[j, 0], x_feasible[j, 0]], [x_infeasible[j, 1], x_feasible[j, 1]],
                     linewidth=0.5, alpha=0.5, c='lightcoral')
        # for j in range(x_infeasible.shape[0]):
        #     plt.plot([x_feasible[j, 0], ip_near[j, 0]],
        #              [x_feasible[j, 1], ip_near[j, 1]],  '--', linewidth=0.5, alpha=0.1, c='lightcoral')
        plt.scatter(x_feasible[:, 0], x_feasible[:, 1], alpha=0.5, s=15, zorder=3, label='Bis. Projected points')
        plt.scatter(x_infeasible[:, 0], x_infeasible[:, 1], alpha=0.5, s=15, zorder=3, label='Infeasible points')
        plt.scatter(ip[:, 0], ip[:, 1], marker='*', c='crimson', zorder=3, label='Interior points')
        # plt.xlim([-1.7, 1.7])
        # plt.ylim([-1.7, 1.7])
        # if i in [1,2]:
        #     plt.xticks([], fontsize=14)
        # plt.xticks([-1, 0, 1], fontsize=14)
        # plt.yticks([-1, 0, 1], fontsize=14)
        plt.title(title_list[i], fontsize=19)
    lines, lables = fig.axes[0].get_legend_handles_labels()
    # lines1, labels1 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, lables, fontsize=15, ncol=3, bbox_to_anchor=[0.84, 0.13], framealpha=1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(model_path + 'pics/bis_' + instance + '.png', dpi=300)
    plt.show()
    # plt.close()

def plot_illustration(paras, model_path):
    data = Box_Constraints()
    c_sample = np.array([[0.2, 1]])
    ip_list = [[0.15, 0.95],[0,0],[[0, -0.6], [0,0] ,[0, 0.6]]]
    title_list = ['bad IP', 'MEIP', 'MEIPs']
    c_tensor = torch.tensor(c_sample).to(device)
    y = np.linspace(-1.2,1.2, 15)
    x = np.linspace(-0.4,0.4, 5)
    x_sample = [[xt,1.2] for xt in x] + [[xt,-1.2] for xt in x] + \
               [[0.4,yt] for yt in y] +  [[-0.4,yt] for yt in y]
    x_sample = np.reshape(np.array(x_sample), [-1,2])
    x_tensor = torch.tensor(x_sample).to(device)
    n_test = len(ip_list)
    fig = plt.figure(figsize=[(n_test)*(2.6+0.2), 4])
    for i, ip in enumerate(ip_list):
        ct = c_tensor
        ip = torch.tensor(ip).to(device)
        x_infeasible, x_feasible, ip_near, ip = plot_bp_traj(data, ip, ct, x_tensor, paras)
        plt.subplot(1, n_test, i + 1)
        data.plot_boundary(ct.cpu().numpy())
        for j in range(x_infeasible.shape[0]):
            plt.plot([x_infeasible[j, 0], x_feasible[j, 0]],
                     [x_infeasible[j, 1], x_feasible[j, 1]],
                     linewidth=0.5, alpha=0.5, c='lightcoral')
        # for j in range(x_infeasible.shape[0]):
        #     plt.plot([x_feasible[j, 0], ip_near[j, 0]],
        #              [x_feasible[j, 1], ip_near[j, 1]],
        #              '--', linewidth=0.5, alpha=0.1, c='lightcoral')
        plt.scatter(x_feasible[:, 0], x_feasible[:, 1], alpha=0.5, s=15, zorder=3, label='Bis. Projected points')
        plt.scatter(x_infeasible[:, 0], x_infeasible[:, 1], alpha=0.5, s=15, zorder=3, label='Infeasible points')
        plt.scatter(ip[:, 0], ip[:, 1], marker='*', c='crimson', zorder=3, label='Interior points')
        plt.xlim([-1, 1])
        plt.ylim([-1.5, 1.5])
        plt.xticks([], fontsize=14)
        plt.yticks([], fontsize=14)
        plt.title(title_list[i], fontsize=18)
    lines, lables = fig.axes[0].get_legend_handles_labels()
    # lines1, labels1 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, lables, fontsize=15, ncol=3, bbox_to_anchor=[0.99, 0.13])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(model_path + 'pics/ip_illus' + '.png', dpi=300)
    plt.show()
    # plt.close()
    

