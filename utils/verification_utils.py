import cvxpy as cp
import torch.nn as nn
import numpy as np
from .nn_utils import ResBlock


def nn_specification(model, prob, data):
    cl,cu = data.input_L.numpy(), data.input_L.numpy()
    xl, xu = data.L_np, data.U_np
    # Variables
    c = cp.Variable(data.xdim)  # input
    x = cp.Variable(data.ydim)
    s = cp.Variable(1)
    # Constraints
    constraints = [cl <= c, c <= cu]  # box constraints on input
    relax_constraint = [cl <= c, c <= cu]  # box constraints on input
    # Iterate over the layers 
    constraints, x_partial = nn_linear_relaxation_formulation(model, c, constraints, relax_constraint)
    ### upper/lower bound constraint
    if prob == 'qp':
        A_partial = data.A_partial.numpy()
        A_other_inv = data.A_other_inv.numpy()
        constraints.append(x[data.partial_vars_idx] == x_partial * (xu-xl) + xl)
        constraints.append(x[data.other_vars] == (c - x[data.partial_vars_idx] @ A_partial.T) @ A_other_inv.T)
        G = data.G_np
        h = data.h_np
        constraints+= [x @ G.T <= h + s]
    elif prob == 'convex_qcqp':
        A_partial = data.A_partial.numpy()
        A_other_inv = data.A_other_inv.numpy()
        constraints.append(x[data.partial_vars_idx] == x_partial * (xu-xl) + xl)
        constraints.append(x[data.other_vars] == (c - x[data.partial_vars_idx] @ A_partial.T) @ A_other_inv.T)
        for i in range(data.nineq):
            H = data.H_np[i,:,:]
            G = data.G_np[i]
            h = data.h_np[i]
            constraints.append( 0.5 * cp.quad_form(x, H) + G.T @ x <= h + s)
    elif prob == 'jccim':
        constraints.append(x ==  x_partial * (xu-xl) + xl)
        constraints += [data.A_np @ x >= c + data.W_np[i] - s for i in range(len(data.W_np))]
        constraints += [data.G_np @ x <= data.h_np + s]


    constraints += [x <= data.U_np + s,
                    x >= data.L_np - s]
    # Formulate the problem
    problem = cp.Problem(cp.Minimize(s), constraints)
    # Solve the problem
    problem.solve(solver=cp.GUROBI)  # or any other solver that supports MILP
    print(problem.status, problem.objective.value)

def nn_mix_integer_formulation(model, c, constraints, relax_constraint): 
    prev = c
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear): ## first/last linear layer
            # Extract weights and biases
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            # Variables
            h = cp.Variable(layer.out_features)  # output of current linear layer
            constraints.append(h == W @ prev + b)  # linear transformation for this layer
            relax_constraint.append(h == W @ prev + b)
            prev = h
        elif isinstance(layer, ResBlock): # linear + relu + linear
            L1, _, L2 = layer.net
            ### First linear layer
            W = L1.weight.detach().numpy()
            b = L1.bias.detach().numpy()
            y1 = cp.Variable(L1.out_features)  # output of current activation layer
            constraints.append(y1 == W @ prev + b)  # linear transformation for this layer
            relax_constraint.append(y1 == W @ prev + b)

            hl = np.empty_like(b)
            hu = np.empty_like(b)
            for j in range(L1.out_features):
                hl[j] = cp.Problem(cp.Minimize(y1[j]), relax_constraint).solve(solver=cp.GUROBI)
                hu[j] = cp.Problem(cp.Maximize(y1[j]), relax_constraint).solve(solver=cp.GUROBI)

            ### Activation layer
            y2 = cp.Variable(L1.out_features)  # output of current activation layer
            z = cp.Variable(L1.out_features, boolean=True)  # binary variables for each hidden neuron
            z_relax = cp.Variable(L1.out_features)  # binary variables for each hidden neuron
            constraints += [0 <= y2, y2<= cp.multiply(z, hu)]  # output of neurons if z=0
            constraints += [y1 <= y2, y2 <= y1 - cp.multiply(1-z, hl)]  # output of neurons if z=0

            relax_constraint += [0 <= z_relax, z_relax <= 1]
            relax_constraint += [0 <= y2, y2 <= cp.multiply(z_relax, hu)]  # output of neurons if z=0
            relax_constraint += [y1 <= y2, y2 <= y1 - cp.multiply(1-z_relax, hl)]  # output of neurons if z=1
            
            ### Second linear layer
            W = L2.weight.detach().numpy()
            b = L2.bias.detach().numpy()
            y3 = cp.Variable(L2.out_features)  # output of current activation layer
            constraints.append(y3 == W @ y2 + b + prev)  # linear transformation for this layer
            relax_constraint.append(y3 == W @ y2 + b + prev)

            prev = y3 
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
    return constraints, prev

def nn_linear_relaxation_formulation(model, c, constraints, relax_constraint):
    prev = c
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear): ## first/last linear layer
            # Extract weights and biases
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            # Variables
            h = cp.Variable(layer.out_features)  # output of current linear layer
            # constraints.append(h == W @ prev + b)  # linear transformation for this layer
            relax_constraint.append(h == W @ prev + b)
            prev = h
        elif isinstance(layer, ResBlock): # linear + relu + linear
            L1, _, L2 = layer.net
            ### First linear layer
            W = L1.weight.detach().numpy()
            b = L1.bias.detach().numpy()
            y1 = cp.Variable(L1.out_features)  # output of current activation layer
            relax_constraint.append(y1 == W @ prev + b)

            hl = np.empty_like(b)
            hu = np.empty_like(b)
            for j in range(L1.out_features):
                hl[j] = cp.Problem(cp.Minimize(y1[j]), relax_constraint).solve(solver=cp.GUROBI)
                hu[j] = cp.Problem(cp.Maximize(y1[j]), relax_constraint).solve(solver=cp.GUROBI)

            ### Activation layer
            y2 = cp.Variable(L1.out_features)  # output of current activation layer
            z_relax = cp.Variable(L1.out_features)  # binary variables for each hidden neuron

            relax_constraint += [0 <= z_relax, z_relax <= 1]
            relax_constraint += [0 <= y2, y2 <= cp.multiply(z_relax, hu)]  # output of neurons if z=0
            relax_constraint += [y1 <= y2, y2 <= y1 - cp.multiply(1-z_relax, hl)]  # output of neurons if z=1
            
            ### Second linear layer
            W = L2.weight.detach().numpy()
            b = L2.bias.detach().numpy()
            y3 = cp.Variable(L2.out_features)  # output of current activation layer
            relax_constraint.append(y3 == W @ y2 + b + prev)
            prev = y3
        # elif isinstance(layer, nn.Sigmoid):
        #     y1 = cp.Variable(layer.out_features)  # output of current activation layer
        #     relax_constraint.append(y1 == W @ prev + b)
        #     hl = np.empty_like(b)
        #     hu = np.empty_like(b)
        #     for j in range(L1.out_features):
        #         hl[j] = cp.Problem(cp.Minimize(y1[j]), relax_constraint).solve(solver=cp.GUROBI)
        #         hu[j] = cp.Problem(cp.Maximize(y1[j]), relax_constraint).solve(solver=cp.GUROBI)
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
    return relax_constraint, prev

