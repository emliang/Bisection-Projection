import torch
import numpy as np
import cvxpy as cp
import copy
import scipy as sp
import multiprocessing as mp
from .run_pf1 import runpf
from pypower.api import opf
torch.set_default_dtype(torch.float64)
n_process = 10
from .acopf_problem import ACOPF_Problem, Grpah_ACOPF_Problem
from .graph_problem import GraphQP_Problem
from .wireless_problem import PowerControl_Problem

###################################################################
# Iterative solver for optimization problem
###################################################################
def solve_opt_problem(args):
    prob_type = args[0]
    if prob_type == 'qp':
        prob_type, Q, p, A, G, h, L, U, Xi = args
        y = cp.Variable(len(Q))
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                          [G @ y <= h, y <= U, y >= L,
                           A @ y == Xi])
        prob.solve()
        sol = y.value
    elif prob_type == 'qcqp':
        prob_type, Q, p, A, G, H, h, L, U, Xi, ydim, nineq = args
        y = cp.Variable(ydim)
        constraints = [A @ y == Xi, y <= U, y >= L]
        for i in range(nineq):
            Ht = H[i]
            Gt = G[i]
            ht = h[i]
            constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                          constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'socp':
        prob_type, Xi, Q, p, A, G, h, C, d, L, U, ydim, nineq = args
        y = cp.Variable(ydim)
        soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(nineq)]
        constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y), constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'sdp':
        prob_type, Xi, Q, A, L, U, ymdim, neq = args
        y = cp.Variable((ymdim, ymdim), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.trace(Q @ y)),
                          [y >> 0] + [y <= U] + [y >= L] +
                          [cp.trace(A[i] @ y) == Xi[i] for i in range(neq)])
        prob.solve()
        sol = y.value
    elif prob_type == 'jccim':
        prob_type, Q, p, A, W, G, h, L, U, Xi = args
        num_scenario  = len(W)
        y = cp.Variable(len(Q))
        constraints = [y <= U, y >= L]
        constraints += [A @ y >= Xi + W[i] for i in range(num_scenario)]
        constraints += [G @ y <= h]
        # constraints.append(cp.sum(z) / num_scenario >= 0.9)
        prob = cp.Problem(cp.Minimize(p.T @ y), constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'acpf':
        prob_type, i, Xi, pgi, vmi, ppc, ppopt, PD, QD, PG, QG, VM, VA, nb, spv, pv_, baseMVA, baseMVA = args
        ppc['bus'][:, PD] = Xi[:nb] * baseMVA
        ppc['bus'][:, QD] = Xi[nb:] * baseMVA
        ppc['bus'][spv, VM] = vmi[spv]
        ppc['gen'][pv_, PG] = pgi[pv_]
        my_result = runpf(ppc, ppopt)[0]
        pg = my_result['gen'][:, PG] / baseMVA
        qg = my_result['gen'][:, QG] / baseMVA
        vm = my_result['bus'][:, VM]
        va = np.deg2rad(my_result['bus'][:, VA])
        y = np.concatenate([pg, qg, vm, va])
        sol = y.value
    elif prob_type == 'acopf':
        prob_type, i, Xi, ppc, ppopt, PD, QD, PG, QG, VM, VA, nb, baseMVA, baseMVA = args
        ppc['bus'][:, PD] = Xi[:nb] * baseMVA
        ppc['bus'][:, QD] = Xi[nb:] * baseMVA
        my_result = opf(ppc, ppopt)
        pg = my_result['gen'][:, PG] / baseMVA
        qg = my_result['gen'][:, QG] / baseMVA
        vm = my_result['bus'][:, VM]
        va = np.deg2rad(my_result['bus'][:, VA])
        sol = np.concatenate([pg, qg, vm, va])
    return sol

def solve_proj_problem(args):
    prob_type = args[0]
    if prob_type == 'qp':
        prob_type, Q, p, A, G, h, L, U, Xi, y_pred = args
        y = cp.Variable(len(Q))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                          [G @ y <= h, y <= U, y >= L,
                           A @ y == Xi])
        prob.solve()
        sol = y.value
    elif prob_type == 'qcqp':
        prob_type, Q, p, A, G, H, h, L, U, Xi, ydim, nineq, y_pred = args
        y = cp.Variable(ydim)
        constraints = [A @ y == Xi, y <= U, y >= L]
        for i in range(nineq):
            Ht = H[i]
            Gt = G[i]
            ht = h[i]
            constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                          constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'socp':
        prob_type, Xi, y_pred, Q, p, A, G, h, C, d, L, U, ydim, nineq = args
        y = cp.Variable(ydim)
        soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(nineq)]
        constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)), constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'sdp':
        prob_type, Xi, y_pred, Q, A, L, U, ymdim, neq = args
        y = cp.Variable((ymdim, ymdim), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                          [y >> 0] + [y <= U] + [y >= L] +
                          [cp.trace(A[i] @ y) == Xi[i] for i in range(neq)])
        prob.solve()
        sol = y.value
    elif prob_type == 'jccim':
        prob_type, Q, p, A, W, G, h, L, U, Xi, y_pred = args
        num_scenario  = len(W)
        y = cp.Variable(len(Q))
        t = cp.Variable(len(Q))
        constraints = [y <= U, y >= L]
        constraints += [A @ y >= Xi + W[i] for i in range(50)]
        constraints += [G @ y <= h]
        # constraints.append(cp.sum(z) / num_scenario >= 0.9)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)), constraints)
        # constraints += [t>= (y-y_pred), t<= -(y-y_pred)]
        # prob = cp.Problem(cp.Minimize(cp.sum(t)), constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'acopf':
        prob_type, i, Xi, pgi, qgi, vmi, vai, ppc, ppopt, PD, QD, PG, QG, VM, VA, nb, baseMVA, baseMVA = args
        ppc['bus'][:, PD] = Xi[:nb] * baseMVA
        ppc['bus'][:, QD] = Xi[nb:] * baseMVA
        ppc['gencost'][:, COST] = 1
        ppc['gencost'][:, COST + 1] = -2 * pgi
        # Set reduced voltage bounds if applicable
        ppc['bus'][:, idx_bus.VM] = vmi
        ppc['bus'][:, idx_bus.VA] = vai
        ppc['gen'][:, idx_gen.PG] = pgi
        ppc['gen'][:, idx_gen.QG] = qgi
        my_result = opf(ppc, ppopt)
        pg = my_result['gen'][:, PG] / baseMVA
        qg = my_result['gen'][:, QG] / baseMVA
        vm = my_result['bus'][:, VM]
        va = np.deg2rad(my_result['bus'][:, VA])
        sol = np.concatenate([pg, qg, vm, va])
    return sol

def solve_warmstart_problem(args):
    prob_type = args[0]
    if prob_type == 'qp':
        prob_type, Q, p, A, G, h, L, U, Xi, y_pred = args
        y = cp.Variable(len(Q))
        y.value = y_pred
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                          [G @ y <= h, y <= U, y >= L,
                           A @ y == Xi])
        prob.solve(warm_start=True)
        sol = y.value
    elif prob_type == 'qcqp':
        prob_type, Q, p, A, G, H, h, L, U, Xi, ydim, nineq, y_pred = args
        y = cp.Variable(ydim)
        y.value = y_pred
        constraints = [A @ y == Xi, y <= U, y >= L]
        for i in range(nineq):
            Ht = H[i]
            Gt = G[i]
            ht = h[i]
            constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                          constraints)
        prob.solve(warm_start=True)
        sol = y.value
    elif prob_type == 'socp':
        prob_type, Xi, y_pred, Q, p, A, G, h, C, d, L, U, ydim, nineq = args
        y = cp.Variable(ydim)
        y.value = y_pred
        soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(nineq)]
        constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y), constraints)
        prob.solve(warm_start=True)
        sol = y.value
    elif prob_type == 'sdp':
        prob_type, Xi, y_pred, Q, A, L, U, ymdim, neq = args
        y = cp.Variable((ymdim, ymdim), symmetric=True)
        y.value = y_pred
        prob = cp.Problem(cp.Minimize(cp.trace(Q @ y)),
                          [y >> 0] + [y <= U] + [y >= L] +
                          [cp.trace(A[i] @ y) == Xi[i] for i in range(neq)])
        prob.solve(warm_start=True)
        sol = y.value
    elif prob_type == 'jccim':
        prob_type, Q, p, A, W, G, h, L, U, Xi, y_pred = args
        num_scenario  = len(W)
        y = cp.Variable(len(Q))
        y.value = y_pred
        constraints = [y <= U, y >= L]
        constraints += [A @ y >= Xi + W[i] for i in range(num_scenario)]
        constraints += [G @ y <= h]
        # constraints.append(cp.sum(z) / num_scenario >= 0.9)
        prob = cp.Problem(cp.Minimize(p.T @ y), constraints)
        prob.solve(warm_start=True)
        sol = y.value
    elif prob_type == 'acopf':
        prob_type, i, Xi, pgi, qgi, vmi, vai, ppc, ppopt, PD, QD, PG, QG, VM, VA, nb, baseMVA, baseMVA = args
        ppc['bus'][:, PD] = Xi[:nb] * baseMVA
        ppc['bus'][:, QD] = Xi[nb:] * baseMVA
        # Set reduced voltage bounds if applicable
        ppc['bus'][:, idx_bus.VM] = vmi
        ppc['bus'][:, idx_bus.VA] = vai
        ppc['gen'][:, idx_gen.PG] = pgi
        ppc['gen'][:, idx_gen.QG] = qgi
        my_result = opf(ppc, ppopt)
        pg = my_result['gen'][:, PG] / baseMVA
        qg = my_result['gen'][:, QG] / baseMVA
        vm = my_result['bus'][:, VM]
        va = np.deg2rad(my_result['bus'][:, VA])
        sol = np.concatenate([pg, qg, vm, va])
    return sol

def solve_feasibility_problem(args):
    prob_type = args[0]
    if prob_type == 'qp':
        prob_type, Q, p, A, G, h, L, U, Xi = args
        y = cp.Variable(len(Q))
        prob = cp.Problem([G @ y <= h, y <= U, y >= L,
                           A @ y == Xi])
        prob.solve()
        sol = y.value
    elif prob_type == 'qcqp':
        prob_type, Q, p, A, G, H, h, L, U, Xi, ydim, nineq = args
        y = cp.Variable(ydim)
        constraints = [A @ y == Xi, y <= U, y >= L]
        for i in range(nineq):
            Ht = H[i]
            Gt = G[i]
            ht = h[i]
            constraints.append(0.5 * cp.quad_form(y, Ht) + Gt.T @ y <= ht)
        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'socp':
        prob_type, Xi, Q, p, A, G, h, C, d, L, U, ydim, nineq = args
        y = cp.Variable(ydim)
        soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(nineq)]
        constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
        prob = cp.Problem(constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'sdp':
        prob_type, Xi, Q, A, L, U, ymdim, neq = args
        y = cp.Variable((ymdim, ymdim), symmetric=True)
        prob = cp.Problem(
                          [y >> 0] + [y <= U] + [y >= L] +
                          [cp.trace(A[i] @ y) == Xi[i] for i in range(neq)])
        prob.solve()
        sol = y.value
    elif prob_type == 'jccim':
        prob_type, Q, p, A, W, G, h, L, U, Xi = args
        num_scenario = len(W)
        y = cp.Variable(len(Q))
        constraints = [y <= U, y >= L]
        constraints += [A @ y >= Xi + W[i] for i in range(num_scenario)]
        constraints += [G @ y <= h]
        # constraints.append(cp.sum(z) / num_scenario >= 0.9)
        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'acpf':
        prob_type, i, Xi, pgi, vmi, ppc, ppopt, PD, QD, PG, QG, VM, VA, nb, spv, pv_, baseMVA, baseMVA = args
        ppc['bus'][:, PD] = Xi[:nb] * baseMVA
        ppc['bus'][:, QD] = Xi[nb:] * baseMVA
        ppc['bus'][spv, VM] = vmi[spv]
        ppc['gen'][pv_, PG] = pgi[pv_]
        my_result = runpf(ppc, ppopt)[0]
        pg = my_result['gen'][:, PG] / baseMVA
        qg = my_result['gen'][:, QG] / baseMVA
        vm = my_result['bus'][:, VM]
        va = np.deg2rad(my_result['bus'][:, VA])
        y = np.concatenate([pg, qg, vm, va])
        sol = y.value
    elif prob_type == 'acopf':
        prob_type, i, Xi, ppc, ppopt, PD, QD, PG, QG, VM, VA, nb, baseMVA, baseMVA = args
        ppc['bus'][:, PD] = Xi[:nb] * baseMVA
        ppc['bus'][:, QD] = Xi[nb:] * baseMVA
        ppc['gencost'][:, COST] = 1
        ppc['gencost'][:, COST + 1] = 1
        my_result = opf(ppc, ppopt)
        pg = my_result['gen'][:, PG] / baseMVA
        qg = my_result['gen'][:, QG] / baseMVA
        vm = my_result['bus'][:, VM]
        va = np.deg2rad(my_result['bus'][:, VA])
        sol = np.concatenate([pg, qg, vm, va])
    return sol


###################################################################
# Base PROBLEM
###################################################################
class Base_Problem:
    def __init__(self, dataset, test_size):
        self.input_L = torch.tensor(dataset['XL'] )
        self.input_U = torch.tensor(dataset['XU'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.num = dataset['X'].shape[0]
        self.device = None#DEVICE
        # self.valid_frac = valid_frac
        # self.test_frac = test_frac

    def eq_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            x = X[n].view(1, -1)
            y = Y[n].view(1, -1)
            y = torch.autograd.Variable(y, requires_grad=True)
            eq_penalty = self.eq_resid(x, y) ** 2
            eq_penalty = torch.sum(eq_penalty, dim=-1, keepdim=True)
            grad = torch.autograd.grad(eq_penalty, y)[0]
            grad_list.append(grad.view(1, -1))
        grad = torch.cat(grad_list, dim=0)
        return grad

    def ineq_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            x = X[n].view(1, -1)
            y = Y[n].view(1, -1)
            y = torch.autograd.Variable(y, requires_grad=True)
            ineq_penalty = self.ineq_resid(x, y) ** 2
            ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
            grad = torch.autograd.grad(ineq_penalty, y)[0]
            grad_list.append(grad.view(1, -1))
        grad = torch.cat(grad_list, dim=0)
        return grad

    def ineq_partial_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            Y_pred = Y[n, self.partial_vars_idx].view(1, -1)
            x = X[n].view(1, -1)
            Y_pred = torch.autograd.Variable(Y_pred, requires_grad=True)
            y = self.complete_partial(x, Y_pred)
            ineq_penalty = self.ineq_resid(x, y) ** 2
            ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
            grad_pred = torch.autograd.grad(ineq_penalty, Y_pred)[0]
            grad = torch.zeros(1, self.ydim, device=X.device)
            grad[0, self.partial_vars_idx] = grad_pred
            grad[0, self.other_vars] = - (grad_pred @ self.A_partial.T) @ self.A_other_inv.T
            grad_list.append(grad)
        return torch.cat(grad_list, dim=0)

    def scale_full(self, X, Y):
        # lower_bound = self.L.view(1, -1)
        # upper_bound = self.U.view(1, -1)
        # The last layer of NN is sigmoid, scale to Opt bound
        scale_Y = Y * (self.U - self.L) + self.L
        return scale_Y

    def scale_partial(self, X, Y):
        # lower_bound = (self.L[self.partial_vars_idx]).view(1, -1)
        # upper_bound = (self.U[self.partial_vars_idx]).view(1, -1)
        scale_Y = Y * (self.U - self.L) + self.L
        return scale_Y

    def scale(self, X, Y):
        if Y.shape[1] < self.ydim:
            Y_scale = self.scale_partial(X, Y)
        else:
            Y_scale = self.scale_full(X, Y)
        return Y_scale

    def cal_penalty(self, X, Y):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y):
        return self.cal_penalty(X, Y)


###################################################################
# QP PROBLEM
###################################################################
class QP_Problem(Base_Problem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
                   L<= x <=U
    """
    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.Q_np = dataset['Q']
        self.p_np = dataset['p']
        self.A_np = dataset['A']
        self.G_np = dataset['G']
        self.h_np = dataset['h']
        self.L_np = dataset['YL']
        self.U_np = dataset['YU']
        self.Q = torch.tensor(dataset['Q'] )
        self.p = torch.tensor(dataset['p'] )
        self.A = torch.tensor(dataset['A'] )
        self.G = torch.tensor(dataset['G'] )
        self.h = torch.tensor(dataset['h'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.xdim = dataset['X'].shape[1]
        self.ydim = dataset['Q'].shape[0]
        self.neq = dataset['X'].shape[0]
        self.nineq = dataset['G'].shape[0]
        self.nknowns = 0

        best_partial = dataset['best_partial']
        self.partial_vars_idx = best_partial
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars_idx)
        self.A_partial = self.A[:, self.partial_vars_idx]
        self.A_other_inv = torch.inverse(self.A[:, self.other_vars])
        self.intrin_dim = max(len(self.partial_vars_idx), self.xdim)

        self.trainX = self.X[:-test_size]
        self.testX = self.X[-test_size:]
        self.trainY = self.Y[:-test_size]
        self.testY = self.Y[-test_size:]

    def __str__(self):
        return 'QP_Problem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * Y).sum(dim=1)/self.ydim

    def eq_resid(self, X, Y):
        return Y @ self.A.T - X

    def ineq_resid(self, X, Y):
        res = Y @ self.G.T - self.h.view(1, -1)
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def complete_partial(self, X, Y):
        Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y_full[:, self.partial_vars_idx] = Y
        Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        return Y_full

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            with mp.Pool(processes=n_process) as pool:
                params = [('qp', Q, p, A, G, h, L, U, Xi) for Xi in X_np]
                Y = pool.map(solve_opt_problem, params)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return sols

    def opt_ip(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            with mp.Pool(processes=n_process) as pool:
                params = [('qp', Q, p, A, G, h, L, U, Xi) for Xi in X_np]
                Y = pool.map(solve_feasibility_problem, params)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return sols

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred_np = Y_pred.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('qp', Q, p, A, G, h, L, U, Xi, y_pred) for Xi, y_pred in zip(X_np, Y_pred_np)]
                Y = pool.map(solve_proj_problem, params)

            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols)

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred_np = Y_pred.detach().cpu().numpy()
            with mp.Pool(processes=n_process) as pool:
                params = [('qp', Q, p, A, G, h, L, U, Xi, y_pred) for Xi, y_pred in zip(X_np, Y_pred_np)]
                Y = pool.map(solve_warmstart_problem, params)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols)


###################################################################
# QCQP Problem
###################################################################
class QCQP_Probem(QP_Problem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   1/2 * y^T H y + G^T y <= h
                   L<= x <=U
    """
    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.H_np = dataset['H']
        self.H = torch.tensor(dataset['H'] )

    def __str__(self):
        return 'QCQP_Problem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def ineq_resid(self, X, Y):
        res = []
        """
         1/2 * y^T H y + G^T y <= h
         H: m * n * n
         G: m * n
         y: 1 * n
         h: 1 * m
        """
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(Y, self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('qcqp',Q, p, A, G, H, h, L, U, Xi, self.ydim, self.nineq) for Xi in X_np]
                Y = pool.map(solve_opt_problem, params)

            sols = np.array(Y)
        else:
            raise NotImplementedError
        return sols

    def opt_ip(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('qcqp',Q, p, A, G, H, h, L, U, Xi, self.ydim, self.nineq) for Xi in X_np]
                Y = pool.map(solve_feasibility_problem, params)

            sols = np.array(Y)
        else:
            raise NotImplementedError
        return sols

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred_np = Y_pred.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('qcqp', Q, p, A, G, H, h, L, U, Xi, self.ydim, self.nineq, y_pred) for Xi, y_pred in
                          zip(X_np, Y_pred_np)]
                Y = pool.map(solve_proj_problem, params)

            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols)

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            Q, p, A, G, H, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.H_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred_np = Y_pred.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('qcqp', Q, p, A, G, H, h, L, U, Xi, self.ydim, self.nineq, y_pred) for Xi, y_pred in
                          zip(X_np, Y_pred_np)]
                Y = pool.map(solve_warmstart_problem, params)

            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols)


###################################################################
# SOCP Problem
###################################################################
class SOCP_Probem(QP_Problem):
    """
        minimize_y p^Ty
        s.t.       Ay =  x
                   ||G^T y + h||_2 <= c^Ty+d
                   L<= x <=U
    """

    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.c_np = dataset['C']
        self.d_np = dataset['d']
        self.C = torch.tensor(dataset['C'] )
        self.d = torch.tensor(dataset['d'] )

    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def ineq_resid(self, X, Y):
        res = []
        """
         ||G^T y + h||_2 <= C^Ty+d
         G: m * k * n
         h: m * k
         y: m * n
         C: m * n
         d: m * 1
        """
        q = torch.norm(torch.matmul(self.G, Y.T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.c_np, self.d_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('socp', Xi, Q, p, A, G, h, C, d, L, U, self.ydim, self.nineq) for Xi in X_np]
                results = pool.map(solve_opt_problem, params)
            sols = np.array(results)
        else:
            raise NotImplementedError
        return sols

    def opt_ip(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.c_np, self.d_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('socp', Xi, Q, p, A, G, h, C, d, L, U, self.ydim, self.nineq) for Xi in X_np]
                results = pool.map(solve_feasibility_problem, params)
            sols = np.array(results)
        else:
            raise NotImplementedError
        return sols

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.c_np, self.d_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('socp', Xi, y_pred, Q, p, A, G, h, C, d, L, U, self.ydim, self.nineq) for Xi, y_pred in
                        zip(X_np, Y_pred)]
                results = pool.map(solve_proj_problem, params)
            sols = np.array(results)
        else:
            raise NotImplementedError
        return torch.tensor(sols)

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, G, h, C, d, L, U = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.c_np, self.d_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()
            with mp.Pool(processes=n_process) as pool:
                params = [('socp', Xi, y_pred, Q, p, A, G, h, C, d, L, U, self.ydim, self.nineq) for Xi, y_pred in
                        zip(X_np, Y_pred)]
                results = pool.map(solve_warmstart_problem, params)
            sols = np.array(results)
        else:
            raise NotImplementedError
        return torch.tensor(sols)


###################################################################
# SDP Problem
###################################################################
class SDP_Probem(Base_Problem):
    """
        minimize_y tr(Qy)
        s.t.       tr(Ay) =  x
                   y >>0
                   L<= y <=U
    """
    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.Q_np = dataset['Q']
        self.A_np = dataset['A']
        self.G_np = dataset['G']
        self.h_np = dataset['h']
        self.L_np = dataset['YL']
        self.U_np = dataset['YU']
        self.Q = torch.tensor(dataset['Q'] )
        self.A = torch.tensor(dataset['A'] )
        self.xdim = dataset['X'].shape[1]
        self.ymdim = dataset['Q'].shape[0]
        # self.ydim = self.ymdim**2
        self.ydim = int(self.ymdim * (self.ymdim + 1) /2)
        self.num = dataset['X'].shape[0]
        self.neq = dataset['X'].shape[1]
        self.nineq = self.ymdim
        self.nknowns = 0

        self.A = torch.tensor(dataset['Ae'])
        self.G = torch.tensor(dataset['Ge'])
        self.h = torch.tensor(dataset['h'])
        self.Ye = torch.tensor(dataset['Ye'])
        self.tril_idx = torch.tril_indices(self.ymdim, self.ymdim)
        self.tril_idx_e = self.tril_idx[0] * self.ymdim + self.tril_idx[1]  # Convert to linear index
        # self.A = self.A.view(-1, self.ydim)
        # self.Y = self.Y.permute(0, 2, 1).contiguous().view(-1, self.ydim)
        # print(self.A.shape)
        # self.A = torch.tril(self.A) + torch.triu(self.A, 1).permute(0,2,1)
        # self.A = self.A[torch.tril_indices(self.ymdim)]
        # self.Y = self.Y[torch.tril_indices(self.ymdim)]
        # print(1/0)


        best_partial = dataset['best_partial']
        self.partial_vars_idx = best_partial
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars_idx)
        self.A_partial = self.A[:, self.partial_vars_idx]
        self.A_other_inv = torch.inverse(self.A[:, self.other_vars])
        self.intrin_dim = max(len(self.partial_vars_idx), self.xdim)
        self.trainX = self.X[:-test_size]
        self.testX = self.X[-test_size:]
        self.Y = self.get_lower_triangle_from_matrix_batch(self.Y, self.ymdim)
        self.trainY = self.Y[:-test_size]
        self.testY = self.Y[-test_size:]
        
    def __str__(self):
        return 'SDPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        # Ym = Y.view(Y.shape[0], self.ymdim, self.ymdim).permute(0, 2, 1)  # batch * n * n
        Y = self.recover_matrix_from_lower_triangle_batch(Y, self.ymdim)
        obj = torch.diagonal(torch.matmul(self.Q, Y), dim1=-2, dim2=-1)
        return torch.sum(obj, dim=1, keepdim=True)/self.ydim

    def eq_resid(self, X, Y):
        # Ye  = self.get_lower_triangle_from_matrix_batch(Y, self.ymdim)
        # Ye =  Y.permute(0, 2, 1).contiguous().view(-1, self.ydim)
        return Y @ self.A.T - X

    def ineq_resid(self, X, Y):
        Y = self.recover_matrix_from_lower_triangle_batch(Y, self.ymdim)
        """
        Y>>0 -> xYx > 0
        Definition of positive matrix
        """
        ## sample-based methods
        # num_sample = 4096
        # est = torch.randn(size=(1, self.ymdim, num_sample), device=X.device)  # batch * n * k
        # est = est# / torch.norm(est, dim=1, p=2, keepdim=True)
        # pel = torch.matmul(Y, est)  # batch * n * k
        # pel = torch.multiply(pel, est).sum(1)  # batch * n * k
        # r1 = -1*pel
        """
        Needell, D., Swartworth, W., & Woodruff, D. P. (2022, October). 
        Testing positive semidefiniteness using linear measurements. 
        In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS) (pp. 87-97). IEEE.
        """
        num_step = 100
        est = torch.randn(size=(X.shape[0], self.ymdim, 1), device=X.device)  # batch * n * 1
        est = est / torch.norm(est, dim=1, p=2, keepdim=True)
        est_list = [est]
        for _ in range(num_step-1):
            noise = torch.randn(size=(X.shape[0], self.ymdim, 1), device=X.device)
            noise = noise * noise.permute(0,2,1)
            noise = torch.matmul(noise, Y.detach())
            grad = torch.matmul(noise, est)
            # grad = torch.matmul(Y.detach(), est)
            est = est -  0.01 * grad
            est = est / torch.norm(est, dim=1, p=2, keepdim=True)
            est_list.append(est)
        est = torch.cat(est_list, dim=-1).detach()
        pel = torch.matmul(Y, est)  # batch * n * b
        pel = torch.multiply(pel, est).sum(1)  # batch * n * b
        r1 = -1 * pel

        # pel = torch.clamp(-1*pel, 0)
        # r1 = pel.view(-1, num_step).sum(1, keepdim=True)

        r2 = (self.L - Y).view(-1,self.ymdim**2)
        r3 = (Y - self.U).view(-1,self.ymdim**2)
        # Ye = self.get_lower_triangle_from_matrix_batch(Y, self.ymdim)
        # r4 = Ye @ self.G.T - self.h
        resids = torch.cat([r1, r2, r3], dim=1)
        return torch.clamp(resids, 0)

    def check_feasibility(self, X, Y):
        r1 = torch.abs(self.eq_resid(X, Y))
        Y = self.recover_matrix_from_lower_triangle_batch(Y, self.ymdim)
        l = (self.L - Y).view(-1,self.ymdim**2)
        u = (Y - self.U).view(-1,self.ymdim**2)
        r2 = torch.cat([l,u], dim=1)
        """
        Check the minimum eigenvalues
        """
        L, info = torch.linalg.cholesky_ex(Y)
        r3 = info.view(-1,1)
        # eigvals = torch.linalg.eigvalsh(Y)
        # r3 = -1 * torch.min(eigvals, dim=1, keepdim=True)[0]
        # r3,_ = torch.lobpcg(Y, largest=False)
        # r3=r3*-1
        # Ye = self.get_lower_triangle_from_matrix_batch(Y, self.ymdim)
        # r4 = Ye @ self.G.T - self.h
        resids = torch.cat([r1, r2, r3], dim=1)
        return  torch.clamp(resids, 0)

    def complete_partial(self, X, Y):
        Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y_full[:, self.partial_vars_idx] = Y
        Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        # Y_full = Y_full.view(Y.shape[0], self.ymdim, self.ymdim).permute(0, 2, 1).contiguous()  # batch * n * n
        # Y_full = self.recover_matrix_from_lower_triangle_batch(Y_full, self.ymdim)
        return Y_full

    # def recover_matrix_from_lower_triangle(self, x, n):
    #     idx = torch.tril_indices(n, n)
    #     X = torch.zeros((n, n), device=x.device)
    #     X[idx[0], idx[1]] = x
    #     X = X + X.T
    #     X = X - torch.diag(X.diag().div(2))
    #     return X

    # def recover_matrix_from_lower_triangle_batch(self, X, n):
        return torch.stack([self.recover_matrix_from_lower_triangle(x, n) for x in X])

    def get_lower_triangle_from_matrix_batch(self, X, n):
        X_extend = X.view(-1, n*n)
        tril_idx_e = self.tril_idx_e.to(device=X_extend.device, dtype=torch.long)
        return X_extend[:, tril_idx_e]

    def recover_matrix_from_lower_triangle_batch(self, X, n):
        X_recovered = torch.zeros((X.size(0), n * n), device=X.device)
        tril_idx_e = self.tril_idx_e.to(device=X.device, dtype=torch.long)
        X_recovered[:, tril_idx_e] = X
        X_recovered = X_recovered.view(-1, n, n)
        X_recovered = X_recovered + X_recovered.transpose(-2, -1)
        X_recovered = X_recovered - torch.diag_embed(X_recovered.diagonal(dim1=-2, dim2=-1) / 2)
        return X_recovered

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, A, L, U = self.Q_np, self.A_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                args = [('sdp', Xi, Q, A, L, U, self.ymdim, self.neq) for Xi in X_np]
                results = pool.map(solve_opt_problem, args)

            sols = np.array(results)
        else:
            raise NotImplementedError
        return sols

    def opt_ip(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, A, L, U = self.Q_np, self.A_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                args = [('sdp', Xi, Q, A, L, U, self.ymdim, self.neq) for Xi in X_np]
                results = pool.map(solve_feasibility_problem, args)

            sols = np.array(results)
        else:
            raise NotImplementedError
        return sols

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        Y_pred = self.recover_matrix_from_lower_triangle_batch(Y_pred, self.ymdim)
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, A, L, U = self.Q_np, self.A_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                args = [('sdp', Xi, y_pred, Q, A, L, U, self.ymdim, self.neq) for Xi, y_pred in zip(X_np, Y_pred)]
                results = pool.map(solve_proj_problem, args)

            sols = np.array(results)
        else:
            raise NotImplementedError
        sols = torch.tensor(sols)
        sols = self.get_lower_triangle_from_matrix_batch(sols, self.ymdim)
        return sols

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        Y_pred = self.recover_matrix_from_lower_triangle_batch(Y_pred, self.ymdim)
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, A, L, U = self.Q_np, self.A_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred = Y_pred.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                args = [('sdp', Xi, y_pred, Q, A, L, U, self.ymdim, self.neq) for Xi, y_pred in zip(X_np, Y_pred)]
                results = pool.map(solve_warmstart_problem, args)

            sols = np.array(results)
        else:
            raise NotImplementedError
        sols = torch.tensor(sols)
        sols = self.get_lower_triangle_from_matrix_batch(sols, self.ymdim)
        return sols


###################################################################
# JCC-IM (Joint-Chance-Constrained Inventory Management)
###################################################################
class JCCIM_Problem(Base_Problem):
    """
        minimize_y  p^Ty
        s.t.       Ay >=  x + w_i
                   Gy <= h
                   L<= x <=U
    """
    def __init__(self, dataset, test_size):
        super().__init__(dataset, test_size)
        self.Q_np = dataset['Q']
        self.p_np = dataset['p']
        self.A_np = dataset['A']
        self.W_np = dataset['W']
        self.G_np = dataset['G']
        self.h_np = dataset['h']
        self.L_np = dataset['YL']
        self.U_np = dataset['YU']
        self.Q = torch.tensor(dataset['Q'] )
        self.p = torch.tensor(dataset['p'] )
        self.A = torch.tensor(dataset['A'] )
        self.W = torch.tensor(dataset['W'] )
        self.G = torch.tensor(dataset['G'] )
        self.h = torch.tensor(dataset['h'] )
        self.L = torch.tensor(dataset['YL'] )
        self.U = torch.tensor(dataset['YU'] )
        self.X = torch.tensor(dataset['X'] )
        self.Y = torch.tensor(dataset['Y'] )
        self.xdim = dataset['X'].shape[1]
        self.ydim = dataset['Q'].shape[0]
        self.neq = dataset['X'].shape[0]
        self.nineq = dataset['G'].shape[0]
        self.nknowns = 0

        self.partial_vars_idx = np.arange(self.ydim)
        self.intrin_dim = max(len(self.partial_vars_idx), self.xdim)
        self.feasibility_level = 0.9
        # best_partial = dataset['best_partial']
        # self.partial_vars_idx = best_partial
        # self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars_idx)
        # self.A_partial = self.A[:, self.partial_vars_idx]
        # self.A_other_inv = torch.inverse(self.A[:, self.other_vars])
        # self.intrin_dim = max(len(self.partial_vars_idx), self.xdim)

        self.trainX = self.X[:-test_size]
        self.testX = self.X[-test_size:]
        self.trainY = self.Y[:-test_size]
        self.testY = self.Y[-test_size:]

    def __str__(self):
        return 'JCCIM_Problem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        # 0.5 * (Y @ self.Q) * Y +
        return (self.p * Y).mean(dim=1)

    def eq_resid(self, X, Y):
         # Y @ self.A.T - X
        return torch.zeros([X.shape[0],1]).to(X.device)

    def ineq_resid(self, X, Y, q=0.9):
        res_1 = Y @ self.G.T - self.h.view(1, -1)
        res_2 = [X + self.W[[i]] - Y @ self.A.T for i in range(self.W.shape[0])]
        res_2 = torch.stack(res_2, dim=1)
        res_2 = torch.clamp(res_2, 0)
        res_2 = res_2.mean(-1)
        quantile_penalty = torch.quantile(res_2, q=q, dim=1, keepdim=True)
        mask = res_2 > quantile_penalty
        mask_1 = res_2 <= quantile_penalty
        quantile_res = (res_2 * mask).sum(dim=1) * (q) + \
                       (res_2 * mask_1).sum(dim=1) * (1-q)
        quantile_res = quantile_res.view(-1,1)
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res_1, quantile_res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def check_feasibility(self, X, Y, q=0.9):
        res_1 = Y @ self.G.T - self.h.view(1, -1)
        res_2 = [X + self.W[[i]] - Y @ self.A.T for i in range(self.W.shape[0])]
        res_2 = torch.stack(res_2, dim=1)
        res_2 = torch.clamp(res_2, 0)
        res_2 = res_2.mean(-1)
        fea_mask = res_2<=1e-5
        fea_rate = (torch.ones_like(fea_mask, dtype=res_2.dtype)*fea_mask).mean(-1, keepdim=True)
        quantile_fea = q - fea_rate
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res_1, quantile_fea, l, u], dim=1)
        return torch.clamp(resids, 0)

    def complete_partial(self, X, Y):
        # Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        # Y_full[:, self.partial_vars_idx] = Y
        # Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        return Y

    def ineq_grad(self, X, Y):
        grad_list = []
        for n in range(Y.shape[0]):
            x = X[n].view(1, -1)
            y = Y[n].view(1, -1)
            y = torch.autograd.Variable(y, requires_grad=True)
            ineq_penalty = self.ineq_resid(x, y) ** 2
            ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
            grad = torch.autograd.grad(ineq_penalty, y)[0]
            grad_list.append(grad.view(1, -1))
        grad = torch.cat(grad_list, dim=0)
        return grad

    def ineq_partial_grad(self, X, Y):
        return self.ineq_grad(X, Y)

    def opt_solve(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, W, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.W_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            with mp.Pool(processes=n_process) as pool:
                params = [('jccim', Q, p, A, W, G, h, L, U, Xi) for Xi in X_np]
                Y = pool.map(solve_opt_problem, params)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return sols


    def opt_ip(self, X, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, W, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.W_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            with mp.Pool(processes=n_process) as pool:
                params = [('jccim', Q, p, A, W, G, h, L, U, Xi) for Xi in X_np]
                Y = pool.map(solve_feasibility_problem, params)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return sols

    def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, W, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.W_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred_np = Y_pred.detach().cpu().numpy()

            with mp.Pool(processes=n_process) as pool:
                params = [('jccim', Q, p, A, W, G, h, L, U, Xi, y_pred) for Xi, y_pred in zip(X_np, Y_pred_np)]
                Y = pool.map(solve_proj_problem, params)

            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols)

    def opt_warmstart(self, X, Y_pred, solver_type='cvxpy', tol=1e-5):
        if solver_type == 'cvxpy':
            print('running cvxpy', end='\r')
            Q, p, A, W, G, h, L, U = \
                self.Q_np, self.p_np, self.A_np, self.W_np, self.G_np, self.h_np, self.L_np, self.U_np
            X_np = X.detach().cpu().numpy()
            Y_pred_np = Y_pred.detach().cpu().numpy()
            with mp.Pool(processes=n_process) as pool:
                params = [('jccim', Q, p, A, W, G, h, L, U, Xi, y_pred) for Xi, y_pred in zip(X_np, Y_pred_np)]
                Y = pool.map(solve_warmstart_problem, params)
            sols = np.array(Y)
        else:
            raise NotImplementedError
        return torch.tensor(sols)

