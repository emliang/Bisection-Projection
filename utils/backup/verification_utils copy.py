import cvxpy as cp
import torch
import torch.nn as nn
import numpy as np
# import sys
# sys.path.append("..") 
from nn_utils import FCNet, ResNet, ResBlock

def ResNet_verification_without_relaxation(model, input_constraint, output_constraint):
    l,u = input_constraint
    A,b = output_constraint
    # Variables
    x = cp.Variable(len(l))  # input
    t = None  # output of last layer
    # Constraints
    constraints = [l <= x, x <= u]  # box constraints on input
    linear_relax_constraint = [l <= x, x <= u]  # box constraints on input
    # Iterate over the layers
    prev = x
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            # Extract weights and biases
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            # Variables
            h = cp.Variable(layer.out_features)  # output of current linear layer
            constraints.append(h == W @ prev + b)  # linear transformation for this layer
            linear_relax_constraint.append(h == W @ prev + b)
        elif isinstance(layer, ResBlock): # linear + relu + linear
            for j, sub_layer in enumerate(layer.net):
                if j==0: # linear + relu
                    # Extract weights and biases
                    W = sub_layer.weight.detach().numpy()
                    b = sub_layer.bias.detach().numpy()
                    y1 = cp.Variable(sub_layer.out_features)  # output of current activation layer
                    constraints.append(y1 == W @ h + b)  # linear transformation for this layer
                    linear_relax_constraint.append(y1 == W @ h + b)
                    hl = np.empty_like(b)
                    hu = np.empty_like(b)
                    for j in range(sub_layer.out_features):
                        hl[j] = cp.Problem(cp.Minimize(y1[j]), linear_relax_constraint).solve(solver=cp.GUROBI)
                        hu[j] = cp.Problem(cp.Maximize(y1[j]), linear_relax_constraint).solve(solver=cp.GUROBI)
                    y2 = cp.Variable(sub_layer.out_features)  # output of current activation layer
                    z = cp.Variable(sub_layer.out_features, boolean=True)  # binary variables for each hidden neuron
                    z_relax = cp.Variable(sub_layer.out_features)  # binary variables for each hidden neuron
                    constraints += [0 <= y2, y2<= cp.multiply(z, hu)]  # output of neurons if z=0
                    constraints += [y1 <= y2, y2 <= y1 - cp.multiply(1-z, hl)]  # output of neurons if z=0
                    linear_relax_constraint += [0 <= y2, y2 <= cp.multiply(z_relax, hu)]  # output of neurons if z=0
                    linear_relax_constraint += [y1 <= y2, y2 <= y1 - cp.multiply(1-z_relax, hl)]  # output of neurons if z=0
                elif j==2: # linear + id
                    # Extract weights and biases
                    W = sub_layer.weight.detach().numpy()
                    b = sub_layer.bias.detach().numpy()
                    y3 = cp.Variable(sub_layer.out_features)  # output of current activation layer
                    constraints.append(y3 == W @ y2 + b + h)  # linear transformation for this layer
                    linear_relax_constraint.append(y3 == W @ y2 + b + h)
            h = y3 
            if i == len(model)-2:
                prev = h
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
    t = h  # output of last layer
    constraints.append(t <= A @ x + b)  # output must be less than or equal to Ax + b
    # Formulate the problem
    problem = cp.Problem(cp.Minimize(0), constraints)
    # Solve the problem
    problem.solve(solver=cp.GUROBI)  # or any other solver that supports MILP
    print(problem.status)

def ResNet_verification_with_sdp_relaxation(model, input_constraint, output_constraint):
    l,u = input_constraint
    A,b = output_constraint
    # Variables
    x = cp.Variable(len(l))  # input
    t = None  # output of last layer
    # Constraints
    sdp_relax_constraint = [l <= x, x <= u]  # box constraints on input
    # Iterate over the layers
    prev = x
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            # Extract weights and biases
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            # Variables
            h = cp.Variable(layer.out_features)  # output of current linear layer
            sdp_relax_constraint.append(h == W @ prev + b)  # linear transformation for this layer
        elif isinstance(layer, ResBlock):
            for j, sub_layer in enumerate(layer.net):
                if j==0: # linear + relu
                    # Extract weights and biases
                    W = sub_layer.weight.detach().numpy()
                    b = sub_layer.bias.detach().numpy()
                    y1 = cp.Variable(sub_layer.out_features)  # output of current activation layer
                    sdp_relax_constraint.append(y1 == W @ h + b)  # linear transformation for this layer
                    # If this is not the last linear layer, add ReLU constraints
                    y2 = cp.Variable(sub_layer.out_features)  # output of current activation layer
                    sdp_relax_constraint += [0 <= y2, y1 <= y2]
                    M = [cp.Variable((2, 2), PSD=True) for _ in range(sub_layer.out_features)]  # PSD matrix for each neuron
                    for j in range(sub_layer.out_features):
                        sdp_relax_constraint += [M[j][0, 0] == y2[j], M[j][1, 1] == y1[j], M[j][0, 1] == y2[j], M[j][1, 0] == y2[j]]
                elif j==2: # linear + id
                    # Extract weights and biases
                    W = sub_layer.weight.detach().numpy()
                    b = sub_layer.bias.detach().numpy()
                    y3 = cp.Variable(sub_layer.out_features)  # output of current activation layer
                    sdp_relax_constraint.append(y3 == W @ y2 + b + h)  # linear transformation for this layer
            h = y3 
            if i == len(model)-2:
                prev = h
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
    t = h  # output of last layer
    sdp_relax_constraint.append(t <= A @ x + b)  # output must be less than or equal to Ax + b
    # Formulate the problem
    problem = cp.Problem(cp.Minimize(0), sdp_relax_constraint)
    # Solve the problem
    problem.solve(solver=cp.MOSEK)  # or any other solver that supports MILP
    print(problem.status)

def NN_verification_without_relaxation(model, input_constraint, output_constraint):
    l,u = input_constraint
    A,b = output_constraint
    # Variables
    x = cp.Variable(len(l))  # input
    t = None  # output of last layer
    # Constraints
    constraints = [l <= x, x <= u]  # box constraints on input
    linear_relax_constraint = [l <= x, x <= u]
    # Iterate over the layers
    prev = x
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            # Extract weights and biases
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            # Variables
            h = cp.Variable(layer.out_features)  # output of current linear layer
            y = cp.Variable(layer.out_features)  # output of current activation layer
            # Constraints
            constraints.append(h == W @ prev + b)  # linear transformation for this layer
            linear_relax_constraint.append(h == W @ prev + b)
            # If this is not the last linear layer, add ReLU constraints
            if i < len(model) - 2:
                z = cp.Variable(layer.out_features, boolean=True)  # binary variables for each hidden neuron
                z_relax = cp.Variable(layer.out_features)  # binary variables for each hidden neuron
                linear_relax_constraint += [0<= z_relax, z_relax<=1]
                # Calculate bounds for next layer
                hl = np.empty_like(b)
                hu = np.empty_like(b)
                for j in range(layer.out_features):
                    hl[j] = cp.Problem(cp.Minimize(h[j]), linear_relax_constraint).solve(solver=cp.GUROBI)
                    hu[j] = cp.Problem(cp.Maximize(h[j]), linear_relax_constraint).solve(solver=cp.GUROBI)
                constraints += [0 <= y, y <= cp.multiply(z, hu)]  # output of neurons if z=0
                constraints += [h <= y, y <= h - cp.multiply(1-z, hl)]  # output of neurons if z=0
                linear_relax_constraint += [0 <= y, y <= cp.multiply(z_relax, hu)]  # output of neurons if z=0
                linear_relax_constraint += [h <= y, y <= h - cp.multiply(1-z_relax, hl)]  # output of neurons if z=0
                # l, u = l_next, u_next  # update bounds for next layer
                prev = y
            else:
                prev = h            
        elif isinstance(layer, nn.ReLU):
            continue
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
    t = prev  # output of last layer
    constraints.append(t <= A @ x + b)  # output must be less than or equal to Ax + b
    # Formulate the problem
    problem = cp.Problem(cp.Minimize(0), constraints)
    # Solve the problem
    problem.solve(solver=cp.GUROBI)  # or any other solver that supports MILP
    print(problem.status)

def NN_verification_with_linear_relaxation(model, input_constraint, output_constraint):
    l,u = input_constraint
    A,b = output_constraint
    # Variables
    x = cp.Variable(len(l))  # input
    t = None  # output of last layer
    # Constraints
    linear_relax_constraint = [l <= x, x <= u]  # box constraints on input
    prev = x
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            # Extract weights and biases
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            # Variables
            h = cp.Variable(layer.out_features)  # output of current linear layer
            y = cp.Variable(layer.out_features)  # output of current activation layer
            # Constraints
            linear_relax_constraint.append(h == W @ prev + b)
            # If this is not the last linear layer, add ReLU constraints
            if i < len(model) - 2:
                z_relax = cp.Variable(layer.out_features)  # binary variables for each hidden neuron
                linear_relax_constraint += [0<= z_relax, z_relax<=1]
                # Calculate bounds for next layer
                hl = np.empty_like(b)
                hu = np.empty_like(b)
                for j in range(layer.out_features):
                    hl[j] = cp.Problem(cp.Minimize(h[j]), linear_relax_constraint).solve(solver=cp.GUROBI)
                    hu[j] = cp.Problem(cp.Maximize(h[j]), linear_relax_constraint).solve(solver=cp.GUROBI)
                linear_relax_constraint += [0 <= y, y <= cp.multiply(z_relax, hu)]  # output of neurons if z=0
                linear_relax_constraint += [h <= y, y <= h - cp.multiply(1-z_relax, hl)]  # output of neurons if z=0
                # l, u = l_next, u_next  # update bounds for next layer
                prev = y
            else:
                prev = h            
        elif isinstance(layer, nn.ReLU):
            continue
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
    t = prev  # output of last layer
    linear_relax_constraint.append(t <= A @ x + b) # output must be less than or equal to Ax + b
    # Formulate the problem
    problem = cp.Problem(cp.Minimize(0), linear_relax_constraint)
    # Solve the problem
    problem.solve(solver=cp.GUROBI)  # or any other solver that supports MILP
    print(problem.status)

def NN_verification_with_sdp_relaxation(model, input_constraint, output_constraint):
    l,u = input_constraint
    A,b = output_constraint
    # Variables
    x = cp.Variable(len(l))  # input
    t = None  # output of last layer
    # Constraints
    sdp_relax_constraint = [l<= x, x<=u]
    # Iterate over the layers
    prev = x
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            # Extract weights and biases
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            # Variables
            h = cp.Variable(layer.out_features)  # output of current linear layer
            y = cp.Variable(layer.out_features)  # output of current activation layer
            # Constraints
            sdp_relax_constraint.append(h == W @ prev + b)
            # If this is not the last linear layer, add ReLU constraints
            if i < len(model) - 2:
                M = [cp.Variable((2, 2), PSD=True) for _ in range(layer.out_features)]  # PSD matrix for each neuron
                for j in range(layer.out_features):
                    sdp_relax_constraint += [M[j][0, 0] == y[j], M[j][1, 1] == h[j], M[j][0, 1] == y[j], M[j][1, 0] == y[j]]
                sdp_relax_constraint += [0 <= y, h <= y]
                prev = y
            else:
                prev = h
        elif isinstance(layer, nn.ReLU):
            continue
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")
    t = prev  # output of last layer
    sdp_relax_constraint.append(t <= A @ x + b) 
    # Formulate the problem
    problem = cp.Problem(cp.Minimize(0), sdp_relax_constraint)
    # Solve the problem
    problem.solve(solver=cp.MOSEK)  # or any other solver that supports MILP
    print(problem.status)




def __main__():
    n_in = 10
    n_out = 100
    n_hid = 100
    n_layer = 3
    # Define a NN
    # model = FCNet(n_in, n_out, n_hid, n_layer, act=None)
    model = ResNet(n_in, n_out, n_hid, n_layer, act=None)
    # Initialize weights and biases
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
            nn.init.normal_(m.bias)

    # Define bounds for the input
    l = np.ones(n_in) * -1
    u = np.ones(n_in)
    # Define bounds for the output
    A = np.random.randn(n_out, n_in)
    b = np.random.rand(n_out) * 10
    input_constraint = (l,u)
    output_constraint = (A,b)
    ResNet_verification_with_sdp_relaxation(model.net, input_constraint, output_constraint)


if __name__ == '__main__':
    __main__()