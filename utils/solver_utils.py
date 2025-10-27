###################################################################
# AC-OPF (Alternating-Current Optimal Power FLow)
###################################################################
from pypower.api import makeYbus, ext2int
from pypower import idx_bus, idx_gen, idx_brch
from pypower.idx_cost import COST
from pypower.ppoption import ppoption

import torch
import numpy as np
import cvxpy as cp
from .run_pf1 import runpf
from pypower.api import opf
torch.set_default_dtype(torch.float64)
n_process = 100

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
        y.value = y_pred
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                          [G @ y <= h, y <= U, y >= L,
                           A @ y == Xi])
        prob.solve()
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
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                          constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'socp':
        prob_type, Xi, y_pred, Q, p, A, G, h, C, d, L, U, ydim, nineq = args
        y = cp.Variable(ydim)
        y.value = y_pred
        soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(nineq)]
        constraints = soc_constraints + [A @ y == Xi] + [y <= U] + [y >= L]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)), constraints)
        prob.solve()
        sol = y.value
    elif prob_type == 'sdp':
        prob_type, Xi, y_pred, Q, A, L, U, ymdim, neq = args
        y = cp.Variable((ymdim, ymdim), symmetric=True)
        y.value = y_pred
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
                          [y >> 0] + [y <= U] + [y >= L] +
                          [cp.trace(A[i] @ y) == Xi[i] for i in range(neq)])
        prob.solve()
        sol = y.value
    elif prob_type == 'jccim':
        prob_type, Q, p, A, W, G, h, L, U, Xi, y_pred = args
        num_scenario  = len(W)
        y = cp.Variable(len(Q))
        y.value = y_pred
        # t = cp.Variable(len(Q))
        constraints = [y <= U, y >= L]
        constraints += [A @ y >= Xi + W[i] for i in range(num_scenario)]
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
