from torch.autograd import Function
import torch
import numpy as np
import cvxpy as cp
import copy
import scipy as sp


###################################################################
# AC-OPF (Alternating-Current Optimal Power FLow)
###################################################################
from pypower.api import makeYbus, ext2int
from pypower import idx_bus, idx_gen, idx_brch
from pypower.idx_cost import COST
from pypower.ppoption import ppoption

import torch
import numpy as np
import copy
import scipy as sp
import multiprocessing as mp
torch.set_default_dtype(torch.float64)
n_process = 10
from .solver_utils import *


class ACOPF_Problem:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = 0   # voltage ang
                              vmag_slack = 1   # voltage mag
                              va_ij min <= va_ij <= va_ij max
                              (p_g - p_d) + (q_g - q_d)i = diag(V * conj(Ybus * V))
                              | diag(V * conj(Yf * V)) | <= s_max
                              | diag(V * conj(Yt * V)) | <= s_max
    """
    def __init__(self, data, test_size=1024, training=True):
        ## Define optimization problem input and output variables
        ppc = data['ppc']
        self.kron_reduction = True
        self.branch_constraint = False
        self.MaxChangeLoad = 0.1
        self.load_ppc(ppc)
        if self.kron_reduction:
            self.load_reduced_ppc(ppc)

        ## Keep parameters indicating how data was generated
        if training:
            ## Load data
            ## Define train/valid/test split
            # self.valid_frac = valid_frac
            # self.test_frac = test_frac
            X = np.concatenate([data['Pd'] / self.baseMVA,
                                data['Qd'] / self.baseMVA], axis=1)
            Y = np.concatenate([data['Pg'] / self.baseMVA,
                                data['Qg'] / self.baseMVA,
                                data['Vm'], data['Va']], axis=1)
            feas_mask = ~np.isnan(Y).any(axis=1)
            X = torch.tensor(X[feas_mask])
            Y = torch.tensor(Y[feas_mask])
            self.trainX = X[:-test_size]
            self.testX = X[-test_size:]
            self.trainY = Y[:-test_size]
            self.testY = Y[-test_size:]
            # self.num = X.shape[0]

        ### For Pytorch
        self.intrin_dim = len(self.partial_vars_idx) + 1
        self.device = None  # DEVICE
        print(f'neq:{self.neq}, nineq:{self.nineq}, '
              f'indim:{self.xdim}, outdim:{self.ydim}, '
              f'par_outdim:{len(self.partial_vars_idx)}, '
              f'pq_load:{len(self.pq_load)}, pq_mid:{len(self.pq_mid)} '
              f'datasize:{X.shape[0]}')

    def __str__(self):
        return 'ACOPF-{}-{}'.format(self.nb, self.MaxChangeLoad)

    def load_ppc(self, ppc):
        # self.baseMVA = ppc['gen'][:, idx_gen.MBASE]
        self.ppc = ppc
        self.baseMVA = ppc['baseMVA']
        self.ng = ppc['gen'].shape[0]
        self.nb = ppc['bus'].shape[0]
        self.nl = ppc['branch'].shape[0]
        self.xdim = self.nb * 2
        self.ydim = self.nb * 2 + self.ng * 2
        self.neq = 2 * self.nb
        self.nineq = 4 * self.ng + 2 * self.nb + 2 * self.nl
        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2 * self.ng
        self.va_start_yidx = 2 * self.ng + self.nb
        ## useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nb
        ## Define the index of different buses
        # pv: generators wihtout slack
        # spv: generators with slack bus (slack bus with known vol angle)
        # pq: load bus (zero Pg Qg generation)
        # ng = len(spv), npv = len(pv), nslack = len(slack), nb = ng + len(pq)
        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.pq = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 1)[0]
        self.pm = np.where(ppc['bus'][:, idx_bus.PD] == 0)[0]
        self.pq_load = np.setdiff1d(self.pq, self.pm)
        self.pq_mid = np.setdiff1d(self.pq, self.pq_load)
        self.non_pq_mid = np.setdiff1d(range(self.nb), self.pq_mid)
        self.spv = np.sort(np.concatenate([self.slack, self.pv]))
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))
        # indices within generators
        self.slack_g = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_g = np.array([np.where(x == self.spv)[0][0] for x in self.pv])
        self.branch_idxes = np.concatenate([[ppc['branch'][:, idx_brch.F_BUS]],
                                            [ppc['branch'][:, idx_brch.T_BUS]]], axis=0).T - 1
        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        self.quad_costs = torch.tensor(ppc['gencost'][:, 4])
        self.lin_costs = torch.tensor(ppc['gencost'][:, 5])
        self.const_cost = ppc['gencost'][:, 6].sum()

        ## Topology info
        ppc_copy = copy.deepcopy(ppc)
        ppc_copy['bus'][:, 0] -= 1
        ppc_copy['branch'][:, [0, 1]] -= 1
        Ybus, Yf, Yt = makeYbus(self.baseMVA, ppc_copy['bus'], ppc_copy['branch'])
        Ybus = Ybus.todense()
        Yf = Yf.todense()
        Yt = Yt.todense()
        self.Ybusr = torch.tensor(np.real(Ybus))
        self.Ybusi = torch.tensor(np.imag(Ybus))
        self.Yfr = torch.tensor(np.real(Yf))
        self.Yfi = torch.tensor(np.imag(Yf))
        self.Ytr = torch.tensor(np.real(Yt))
        self.Yti = torch.tensor(np.imag(Yt))

        ## initial values for solver
        self.pg_init = torch.tensor(ppc['gen'][:, idx_gen.PG] / self.baseMVA)
        self.qg_init = torch.tensor(ppc['gen'][:, idx_gen.QG] / self.baseMVA)
        self.vm_init = torch.tensor(ppc['bus'][:, idx_bus.VM])
        self.va_init = torch.tensor(np.deg2rad(ppc['bus'][:, idx_bus.VA]))
        self.pd_init = torch.tensor(ppc['bus'][:, idx_bus.PD] / self.baseMVA)
        self.qd_init = torch.tensor(ppc['bus'][:, idx_bus.QD] / self.baseMVA)
        self.output_init = torch.cat([self.pg_init, self.qg_init, self.vm_init, self.va_init], dim=0).view(1, -1)
        self.input_init = torch.cat([self.pd_init, self.qd_init], dim=0).view(1, -1)
        ## upper and lower bound
        self.input_L = self.input_init * (1 - self.MaxChangeLoad)
        self.input_U = self.input_init * (1 + self.MaxChangeLoad)
        self.pmax = torch.tensor(ppc['gen'][:, idx_gen.PMAX] / self.baseMVA)
        self.pmin = torch.tensor(ppc['gen'][:, idx_gen.PMIN] / self.baseMVA)
        self.qmax = torch.tensor(ppc['gen'][:, idx_gen.QMAX] / self.baseMVA)
        self.qmin = torch.tensor(ppc['gen'][:, idx_gen.QMIN] / self.baseMVA)
        self.vmax = torch.tensor(ppc['bus'][:, idx_bus.VMAX])
        self.vmin = torch.tensor(ppc['bus'][:, idx_bus.VMIN])
        self.output_L = torch.cat([self.pmin, self.qmin, self.vmin, -torch.ones_like(self.vmin) * torch.pi],
                                  dim=0).view(1, -1)
        self.output_U = torch.cat([self.pmax, self.qmax, self.vmax, torch.ones_like(self.vmax) * torch.pi],
                                  dim=0).view(1, -1)
        self.smax = torch.tensor(ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA)
        self.amax = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMAX]))
        self.amin = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMIN]))
        self.slackva = self.va_init[self.slack]

        ## Define variables and indices for "partial completion" neural network
        # pg (non-slack) and |v|_g (including slack) to be predict
        self.partial_vars_idx = np.concatenate([self.pg_start_yidx + self.pv_g,
                                                self.vm_start_yidx + self.spv])
        # exclude va at slack bus
        self.known_vars = np.concatenate([self.partial_vars_idx,
                                          self.va_start_yidx + self.slack])
        self.other_vars = np.setdiff1d(np.arange(self.nb*2+self.ng*2), self.known_vars)
        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2 * self.npv + self.nslack)
        # index for newton methods
        self.newton_eqs_idx = np.concatenate([
            self.pflow_start_eqidx + self.nonslack_idxes,  # real power flow at non-slack bus
            self.qflow_start_eqidx + self.pq])  # reactive power flow at load buses
        self.newton_vars_idx = np.concatenate([
            self.vm_start_yidx + self.pq,  # vm at load buses
            self.va_start_yidx + self.nonslack_idxes])  # va  at non-slack bus
        self.last_eqs_idx = np.concatenate([self.pflow_start_eqidx + self.slack,  # slack-bus pg
                                            self.qflow_start_eqidx + self.spv])  # pv-bus qg
        self.last_vars_idx = np.concatenate([self.pg_start_yidx + self.slack_g,  # slack-bus pg
                                             self.qg_start_yidx + np.arange(self.ng)])  # pv-bus qg

    def load_reduced_ppc(self, ppc):
        """Reduced Adimittance Matrix"""
        ppc_copy = copy.deepcopy(ppc)
        ppc_copy['bus'][:, 0] -= 1
        ppc_copy['branch'][:, [0, 1]] -= 1
        Ybus, Yf, Yt = makeYbus(self.baseMVA, ppc_copy['bus'], ppc_copy['branch'])
        Ybus = Ybus.todense()
        Y_ee = Ybus[self.non_pq_mid,:][:,self.non_pq_mid]
        Y_ei = Ybus[self.non_pq_mid,:][:,self.pq_mid]
        Y_ii = Ybus[self.pq_mid,:][:,self.pq_mid]
        Y_ie = Ybus[self.pq_mid,:][:,self.non_pq_mid]
        Yred = Y_ee - Y_ei @ np.linalg.inv(Y_ii) @ Y_ie
        self.Yredr = torch.tensor(np.real(Yred))
        self.Yredi = torch.tensor(np.imag(Yred))
        """Recover Matrix"""
        self.pf_mid_index = [i for i in self.pq_mid] + [i + self.nb for i in self.pq_mid]
        self.pf_non_mid_index = np.setdiff1d(range(self.nb * 2), self.pf_mid_index)
        self.pf_load_index = [i for i in self.pq_load] + [i + self.nb for i in self.pq_load]
        Ybus_mid_1 = torch.cat([self.Ybusr, -self.Ybusi], dim=1)
        Ybus_mid_2 = torch.cat([self.Ybusi, self.Ybusr], dim=1)
        Ybus_mid = torch.cat([Ybus_mid_1, Ybus_mid_2], dim=0)
        Ybus_mid_sub = Ybus_mid[self.pf_mid_index, :]  # 2N * M
        Ybus_mid_sub_inv = torch.inverse(Ybus_mid_sub[:, self.pf_mid_index])
        Ybus_non_mid_sub = Ybus_mid_sub[:, self.pf_non_mid_index]
        self.mid_complete = torch.matmul(Ybus_mid_sub_inv, Ybus_non_mid_sub)

        ## initial values for solver
        self.vm_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.VM])
        self.va_r_init = torch.tensor(np.deg2rad(ppc['bus'][self.non_pq_mid, idx_bus.VA]))
        self.pd_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.PD] / self.baseMVA)
        self.qd_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.QD] / self.baseMVA)
        self.output_r_init = torch.cat([self.pg_init, self.qg_init, self.vm_r_init, self.va_r_init], dim=0).view(1, -1)
        self.input_r_init = torch.cat([self.pd_r_init, self.qd_r_init], dim=0).view(1, -1)

        self.nb_r = len(self.non_pq_mid)
        self.yrdim = self.nb_r*2 + self.ng*2
        self.pg_r_start_yidx = 0
        self.qg_r_start_yidx = self.ng
        self.vm_r_start_yidx = 2 * self.ng
        self.va_r_start_yidx = 2 * self.ng + self.nb_r
        ## useful indices for equality constraints
        self.pflow_r_start_eqidx = 0
        self.qflow_r_start_eqidx = self.nb_r

        self.slack_r = np.where(ppc['bus'][self.non_pq_mid, idx_bus.BUS_TYPE] == 3)[0]
        self.pv_r = np.where(ppc['bus'][self.non_pq_mid, idx_bus.BUS_TYPE] == 2)[0]
        self.pq_r = np.where(ppc['bus'][self.non_pq_mid, idx_bus.BUS_TYPE] == 1)[0]
        self.spv_r = np.sort(np.concatenate([self.slack_r, self.pv_r]))
        self.nonslack_r = np.sort(np.concatenate([self.pq_r, self.pv_r]))


        ## Define variables and indices for "partial completion" neural network
        # pg (non-slack) and |v|_g (including slack) to be predict
        self.partial_r_vars_idx = np.concatenate([self.pg_r_start_yidx + self.pv_g,
                                                self.vm_r_start_yidx + self.spv_r])
        # exclude va at slack bus
        self.known_r_vars = np.concatenate([self.partial_r_vars_idx,
                                          self.va_r_start_yidx + self.slack_r])
        self.other_r_vars = np.setdiff1d(np.arange(self.yrdim), self.known_r_vars)
        # indices of useful quantities in partial solution
        # index for newton methods
        self.newton_r_eqs_idx = np.concatenate([
            self.pflow_r_start_eqidx + self.nonslack_r,  # real power flow at non-slack bus
            self.qflow_r_start_eqidx + self.pq_r])  # reactive power flow at load buses
        self.newton_r_vars_idx = np.concatenate([
            self.vm_r_start_yidx + self.pq_r,  # vm at load buses
            self.va_r_start_yidx + self.nonslack_r])  # va  at non-slack bus
        self.last_r_eqs_idx = np.concatenate([self.pflow_r_start_eqidx + self.slack_r,  # slack-bus pg
                                            self.qflow_r_start_eqidx + self.spv_r])  # pv-bus qg
        self.last_r_vars_idx = np.concatenate([self.pg_r_start_yidx + self.slack_g,  # slack-bus pg
                                             self.qg_r_start_yidx + np.arange(self.ng)])  # pv-bus qg

    def get_yvars(self, Y, kron_reduction=False):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2 * self.ng]
        if kron_reduction:
            vm = Y[:, -2 * self.nb_r:-self.nb_r]
            va = Y[:, -self.nb_r:]
        else:
            vm = Y[:, -2 * self.nb:-self.nb]
            va = Y[:, -self.nb:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.baseMVA).to(Y.device)
        cost = (self.quad_costs * pg_mw ** 2).sum(axis=1) + \
               (self.lin_costs * pg_mw).sum(axis=1) + \
               self.const_cost
        return cost / self.baseMVA / self.nb
        # return cost / (self.baseMVA.mean() ** 2)

    def solve_mid_bus(self, Y):
        if self.kron_reduction:
            vm = Y[:, self.vm_r_start_yidx:self.va_r_start_yidx]
            va = Y[:, self.va_r_start_yidx:]
        else:
            vm = Y[:, self.vm_start_yidx + self.non_pq_mid]
            va = Y[:, self.va_start_yidx + self.non_pq_mid]
        # solve mid-bus vm and va
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * cosva
        vi = vm * sinva
        vri_non_mid = torch.cat([vr, vi], dim=1)
        vri_mid = - vri_non_mid @ self.mid_complete.T
        # complete the close-form solving process
        vr_mid, vi_mid = torch.chunk(vri_mid, 2, dim=1)
        vm_mid_2 = vr_mid ** 2 + vi_mid ** 2
        vm_mid = torch.sqrt(vm_mid_2)
        va_mid = torch.arctan(vi_mid / vr_mid)
        if self.kron_reduction:
            Yfull = torch.zeros(size=[Y.shape[0], self.ydim]).to(Y.device)
            Yfull[:, :2*self.ng] = Y[:, :2*self.ng]
            Yfull[:, self.vm_start_yidx + self.non_pq_mid] = vm
            Yfull[:, self.va_start_yidx + self.non_pq_mid] = va
            Yfull[:, self.vm_start_yidx + self.pq_mid] = vm_mid
            Yfull[:, self.va_start_yidx + self.pq_mid] = va_mid
        else:
            Y[:, self.vm_start_yidx + self.pq_mid] = vm_mid
            Y[:, self.va_start_yidx + self.pq_mid] = va_mid
            Yfull = Y
        return Yfull

    def eq_resid(self, X, Y, kron_reduction=False):
        pg, qg, vm, va = self.get_yvars(Y, kron_reduction)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        if kron_reduction:
            Yr = self.Yredr
            Yi = self.Yredi
            nb = self.nb_r
            spv = self.spv_r
        else:
            Yr = self.Ybusr
            Yi = self.Ybusi
            nb = self.nb
            spv = self.spv
        ## power balance equations
        Ir = vr @ Yr.T - vi @ Yi.T
        Ii = vr @ Yi.T + vi @ Yr.T


        # real power
        pg_expand = torch.zeros(pg.shape[0], nb, device=X.device)
        pg_expand[:, spv] = pg
        real_resid = (pg_expand - X[:, :nb]) - (vr * Ir + vi * Ii)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], nb, device=X.device)
        qg_expand[:, spv] = qg
        react_resid = (qg_expand - X[:, nb:]) - (vi * Ir - vr * Ii)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)

        return resids

    def ineq_resid(self, X, Y):
        ## Bus & Branch limit
        # st = time.time()
        if self.branch_constraint:
            resids = torch.cat([Y - self.output_U, self.output_L - Y,
                                self.branch_ineq_resid(X, Y) ], dim=1)
        else:
            resids = torch.cat([Y - self.output_U, self.output_L - Y,], dim=1)
        # et = time.time()
        # print(et-st)
        return torch.clamp(resids, 0)

    def branch_ineq_resid(self, X, Y):
        _, _, vm, va = self.get_yvars(Y)
        ### Branch angele limit
        va_start = va[:, self.branch_idxes[:, 0]]
        va_end = va[:, self.branch_idxes[:, 1]]
        resids_brach_angle = (va_start - va_end).abs() - self.amax
        ### Branch flow limit
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        # Power at the "from" end
        If_real = torch.matmul(vr, self.Yfr.T) - torch.matmul(vi, self.Yfi.T)
        If_imag = torch.matmul(vi, self.Yfr.T) + torch.matmul(vr, self.Yfi.T)
        Sf_real = vr[:, self.branch_idxes[:, 0]] * If_real + vi[:, self.branch_idxes[:, 0]] * If_imag
        Sf_imag = vr[:, self.branch_idxes[:, 0]] * If_imag - vi[:, self.branch_idxes[:, 0]] * If_real
        # Power at the "to" end
        It_real = torch.matmul(vr, self.Ytr.T) - torch.matmul(vi, self.Yti.T)
        It_imag = torch.matmul(vi, self.Ytr.T) + torch.matmul(vr, self.Yti.T)
        St_real = vr[:, self.branch_idxes[:, 1]] * It_real + vi[:, self.branch_idxes[:, 1]] * It_imag
        St_imag = vr[:, self.branch_idxes[:, 1]] * It_imag - vi[:, self.branch_idxes[:, 1]] * It_real
        # Power magnitude
        sij = (Sf_real ** 2 + Sf_imag ** 2)
        sji = (St_real ** 2 + St_imag ** 2)
        resids_branch_flow = torch.maximum(sij, sji) - self.smax ** 2
        # print(resids_branch_flow.max())
        # print(1/0)
        resids_branch = torch.cat([resids_brach_angle, resids_branch_flow], dim=1)
        return torch.clamp(resids_branch, 0)

    def eq_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        eq_resid = self.eq_resid(X, Y)
        return 2 * eq_jac.transpose(1, 2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    def ineq_grad(self, X, Y, mode='autograd'):
        if mode == 'unfold':
            ineq_jac = self.ineq_jac(Y)
            ineq_resid = self.ineq_resid(X, Y)
            return 2 * ineq_jac.transpose(1, 2).bmm(ineq_resid.unsqueeze(-1)).squeeze(-1)
        elif mode == 'autograd':
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
        eq_jac = self.eq_jac(Y)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars_idx])

        direct_grad = self.ineq_grad(X, Y)
        indirect_partial_grad = dynz_dz.transpose(1, 2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars_idx]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=X.device)
        full_grad[:, self.partial_vars_idx] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)
        return full_grad

    def eq_jac_v(self, Y):
        # | dP / dVm dP / dVa |
        # | dQ / dVm dQ / dVa |
        _, _, vm, va = self.get_yvars(Y, self.kron_reduction)
        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(torch.multiply(v1, v2))
        Ydiagv = lambda Y, v: torch.multiply(Y.unsqueeze(0), v.unsqueeze(1))
        dtm = lambda v, M: torch.multiply(v.unsqueeze(2), M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * cosva
        vi = vm * sinva
        if self.kron_reduction:
            Yr = self.Yredr
            Yi = self.Yredi
        else:
            Yr = self.Ybusr
            Yi = self.Ybusi
        Ir = torch.matmul(vr, Yr) - torch.matmul(vi, Yi)
        Ii = torch.matmul(vr, Yi) + torch.matmul(vi, Yr)

        # Combined operations
        Ydiagv_Yi_cosva_Yr_sinva = Ydiagv(Yi, cosva) + Ydiagv(Yr, sinva)
        Ydiagv_Yr_cosva_Yi_sinva = Ydiagv(Yr, cosva) - Ydiagv(Yi, sinva)
        Ydiagv_Yi_vi_Yr_vr = Ydiagv(Yi, -vi) + Ydiagv(Yr, vr)
        Ydiagv_Yr_vi_Yi_vr = Ydiagv(Yr, -vi) - Ydiagv(Yi, vr)
        # real power equations
        dreal_dvm = -mdiag(cosva, Ir) - dtm(vr, Ydiagv_Yr_cosva_Yi_sinva) \
                    - mdiag(sinva, Ii) - dtm(vi, Ydiagv_Yi_cosva_Yr_sinva)
        dreal_dva = -mdiag(-vi, Ir) - dtm(vr, Ydiagv_Yr_vi_Yi_vr) \
                    - mdiag(vr, Ii) - dtm(vi, Ydiagv_Yi_vi_Yr_vr)

        # reactive power equations
        dreact_dvm = mdiag(cosva, Ii) + dtm(vr, Ydiagv_Yi_cosva_Yr_sinva) \
                     - mdiag(sinva, Ir) - dtm(vi, Ydiagv_Yr_cosva_Yi_sinva)
        dreact_dva = mdiag(-vi, Ii) + dtm(vr, Ydiagv_Yi_vi_Yr_vr) \
                     - mdiag(vr, Ir) - dtm(vi, Ydiagv_Yr_vi_Yi_vr)
        # dreal_dvm = dreact_dva = torch.zeros([Y.shape[0], self.nb, self.nb]).to(Y.device)
        jac = torch.cat([torch.cat([dreal_dvm, dreal_dva], dim=2),
                         torch.cat([dreact_dvm, dreact_dva], dim=2)], dim=1)
        return jac

    def eq_jac(self, Y):
        # | dP / dPg , dP / dQg , dP / dVm , dP / dVa |
        # | dQ / dPg , dQ / dQg , dQ / dVm , dQ / dVa |
        batch_size = Y.shape[0]
        # real power equations
        dreal_dpg = torch.zeros(self.nb, self.ng, device=Y.device)
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=Y.device)
        # reactive power equations
        dreact_dqg = torch.zeros(self.nb, self.ng, device=Y.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=Y.device)
        jac_p = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(batch_size, *dreal_dpg.shape),
                       torch.zeros(batch_size, self.nb, self.ng, device=Y.device)], dim=2),
            torch.cat([torch.zeros(batch_size, self.nb, self.ng, device=Y.device),
                       dreact_dqg.unsqueeze(0).expand(batch_size, *dreact_dqg.shape)], dim=2)], dim=1)

        jac_v = self.eq_jac_v(Y)
        return torch.cat([jac_p, jac_v], dim=2)

    def ineq_jac(self, Y):
        jac = torch.cat([
            torch.cat([torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device)], dim=1),
            torch.cat([-torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=Y.device),
                       torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.ng, self.ng, device=Y.device),
                       -torch.eye(self.ng, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device),
                       torch.zeros(self.ng, self.nb, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.nb, self.ng, device=Y.device),
                       torch.zeros(self.nb, self.ng, device=Y.device),
                       torch.eye(self.nb, device=Y.device),
                       torch.zeros(self.nb, self.nb, device=Y.device)], dim=1),
            torch.cat([torch.zeros(self.nb, self.ng, device=Y.device),
                       torch.zeros(self.nb, self.ng, device=Y.device),
                       -torch.eye(self.nb, device=Y.device),
                       torch.zeros(self.nb, self.nb, device=Y.device)], dim=1)
        ], dim=0)
        return jac.unsqueeze(0).expand(Y.shape[0], *jac.shape)

    def scale_full(self, X, Y):
        Y_scaled = Y * (self.output_U - self.output_L) + self.output_L
        # Y_scaled[:, self.va_start_yidx + self.slack] = self.slack_va#.unsqueeze(0).expand(X.shape[0], self.nslack)
        return Y_scaled

    def scale_partial(self, X, Y, scale_idx):
        Y_scaled = Y * (self.output_U[:, scale_idx] - self.output_L[:, scale_idx]) \
                   + self.output_L[:, scale_idx]
        return Y_scaled

    def scale(self, X, Y):
        if Y.shape[1] == len(self.partial_vars_idx):
            Y_scale = self.scale_partial(X, Y, self.partial_vars_idx)
        elif Y.shape[1] == len(self.newton_vars_idx):
            Y_scale = self.scale_partial(X, Y, self.newton_vars_idx)
        else:
            Y_scale = self.scale_full(X, Y)
        return Y_scale

    def complete_partial(self, X, Z, bsz=1024):
        X = X.detach()
        Yfull = []
        for b in range(0, X.shape[0], bsz):
            Xb = X[b:b + bsz]
            Zb = Z[b:b + bsz]
            if self.kron_reduction:
                Yb = PF_pgvm_Function(self)(Xb[:, self.pf_non_mid_index], Zb)
                Yb = self.solve_mid_bus(Yb)
            else:
                Yb = PF_pgvm_Function(self)(Xb, Zb)
            Yfull.append(Yb)
        return torch.cat(Yfull, dim=0)

    def cal_penalty(self, X, Y):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y):
        return self.cal_penalty(X, Y)

    def opt_solve(self, X, solver_type='pypower', tol=1e-5):
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc
        ppopt = ppoption(OPF_ALG=560, OUT_ALL=0, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
        if X.shape[0] > 1:
            with mp.Pool(processes=n_process) as pool:
                args = [('acopf', i, X_np[i], ppc, ppopt, \
                         idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
                         self.nb, self.baseMVA, self.baseMVA) for i in range(X_np.shape[0])]
                results = pool.map(solve_opt_problem, args)
        else:
            results = solve_opt_problem(('acopf', 0, X_np[0], ppc, ppopt, \
                                     idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
                                     self.nb, self.baseMVA, self.baseMVA))
        return torch.as_tensor(np.array(results)).to(X.device)

    def opt_ip(self, X, solver_type='pypower', tol=1e-5):
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc
        ppopt = ppoption(OPF_ALG=560, OUT_ALL=0, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
        if X.shape[0] > 1:
            with mp.Pool(processes=n_process) as pool:
                args = [('acopf', i, X_np[i], ppc, ppopt, \
                         idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
                         self.nb, self.baseMVA, self.baseMVA) for i in range(X_np.shape[0])]
                results = pool.map(solve_feasibility_problem, args)
        else:
            results = solve_feasibility_problem(('acopf', 0, X_np[0], ppc, ppopt, \
                                     idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
                                     self.nb, self.baseMVA, self.baseMVA))
        return torch.as_tensor(np.array(results)).to(X.device)

    def opt_proj(self, X, Y, solver_type='pypower', tol=1e-5):
        X_np = X.detach().cpu().numpy()
        pg, qg, vm, va = self.get_yvars(Y)
        pg_all = pg.detach().cpu().numpy() * self.baseMVA
        qg_all = qg.detach().cpu().numpy() * self.baseMVA
        vm_all = vm.detach().cpu().numpy()
        va_all = np.rad2deg(va.detach().cpu().numpy())
        ppc = self.ppc
        ppopt = ppoption(OPF_ALG=560, OUT_ALL=0, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
        with mp.Pool(processes=n_process) as pool:
            args = [('acopf', i, X_np[i], pg_all[i], qg_all[i], vm_all[i], va_all[i], ppc, ppopt, \
                     idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
                     self.nb, self.baseMVA, self.baseMVA) for i in range(X_np.shape[0])]
            results = pool.map(solve_proj_problem, args)
        return torch.tensor(np.array(results))

    def opt_warmstart(self, X, Y, solver_type='pypower', tol=1e-5):
        X_np = X.detach().cpu().numpy()
        pg, qg, vm, va = self.get_yvars(Y)
        pg_all = pg.detach().cpu().numpy() * self.baseMVA
        qg_all = qg.detach().cpu().numpy() * self.baseMVA
        vm_all = vm.detach().cpu().numpy()
        va_all = np.rad2deg(va.detach().cpu().numpy())
        ppc = self.ppc
        ppopt = ppoption(OPF_ALG=560, OUT_ALL=0, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
        with mp.Pool(processes=n_process) as pool:
            args = [('acopf', i, X_np[i], pg_all[i], qg_all[i], vm_all[i], va_all[i], ppc, ppopt, \
                     idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
                     self.nb, self.baseMVA, self.baseMVA) for i in range(X_np.shape[0])]
            results = pool.map(solve_warmstart_problem, args)
        # for i in range(X_np.shape[0]):
        #     args = ['acopf', i, X_np[i], pg_all[i], qg_all[i], vm_all[i], va_all[i], ppc, ppopt, \
        #              idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
        #              self.nb, self.baseMVA, self.baseMVA]
        #     results = solve_warmstart_problem(args)
        return torch.tensor(np.array(results))

    def opt_solve_pf(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        pg_all = pg.detach().cpu().numpy() * self.baseMVA
        # qg_all = qg.detach().cpu().numpy() * self.baseMVA
        vm_all = vm.detach().cpu().numpy()
        # va_all = np.rad2deg(va.detach().cpu().numpy())
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc
        ppopt = ppoption()
        ppopt = ppoption(ppopt, PF_ALG=1, OUT_ALL=0, VERBOSE=0, ENFORCE_Q_LIMS=False)
        with mp.Pool(processes=n_process) as pool:
            args = [('acpf', i, X_np[i], pg_all[i], vm_all[i], ppc, ppopt, \
                     idx_bus.PD, idx_bus.QD, idx_gen.PG, idx_gen.QG, idx_bus.VM, idx_bus.VA, \
                     self.nb, self.spv, self.pv_g, self.baseMVA, self.baseMVA) for i in range(X_np.shape[0])]
            results = pool.map(solve_opt_problem, args)
        return torch.as_tensor(np.array(results)).to(X.device)

def PF_pgvm_Function(data, tol=1e-5, max_iters=5):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            reduced = data.kron_reduction
            if not reduced:
                partial_vars_idx = data.partial_vars_idx
                newton_eqs_idx = data.newton_eqs_idx
                newton_vars_idx = data.newton_vars_idx
                last_eqs_idx = data.last_eqs_idx
                last_vars_idx = data.last_vars_idx
                Y = torch.zeros(X.shape[0], data.ydim, device=X.device) + data.output_init
            else:
                partial_vars_idx = data.partial_r_vars_idx
                newton_eqs_idx = data.newton_r_eqs_idx
                newton_vars_idx = data.newton_r_vars_idx
                last_eqs_idx = data.last_r_eqs_idx
                last_vars_idx = data.last_r_vars_idx
                Y = torch.zeros(X.shape[0], data.yrdim, device=X.device) + data.output_r_init
            # start_time = time.time()
            ## Step 1: known variables: pg at non-slack gens, vm at gens
            Y[:, partial_vars_idx] = Z
            ## Step 2: newton variables
            for n in range(max_iters):
                gy = data.eq_resid(X, Y, reduced)[:, newton_eqs_idx]
                jac_vmva = data.eq_jac_v(Y)
                jac_newton_eq_recon_var = jac_vmva[:, newton_eqs_idx, :][:, :, newton_vars_idx - 2 * data.ng]
                if torch.abs(gy).max() < tol:
                    break
                ### Newton update
                delta = torch.linalg.solve(jac_newton_eq_recon_var, gy.unsqueeze(-1)).squeeze(-1)
                Y[:, newton_vars_idx] -= delta
            if torch.abs(gy).max() > tol:
                print(f'Newton methods for Power Flow does not converge in {n} with error {torch.abs(gy).max()}',
                      end='\r')
            else:
                print(f'Newton methods for Power Flow converge in {n} iter with error {torch.abs(gy).max()}', end='\r')
            ## Step 3: last variables
            Y[:, last_vars_idx] -= data.eq_resid(X, Y, reduced)[:, last_eqs_idx]

            ## store information for backpropagation
            vm_start_yidx = data.vm_start_yidx
            partial_pg_yidx = partial_vars_idx[data.pg_pv_zidx]
            partial_vm_yidx = partial_vars_idx[data.vm_spv_zidx]
            # print('Newton methods error', n, torch.abs(data.eq_resid(X, Y)).max(), end='\r')

            ctx.save_for_backward(-jac_vmva[:, last_eqs_idx, :],  # jac_last_vars_vmva
                                  jac_newton_eq_recon_var,  # jac_newton_eq_recon_var
                                  jac_vmva[:, newton_eqs_idx, :][:, :,
                                  partial_vm_yidx - 2 * data.ng])  # jac_newton_eq_partial_vm_var
            ctx.data = [partial_vars_idx, partial_pg_yidx, partial_vm_yidx,
                        newton_vars_idx, last_vars_idx, vm_start_yidx]
            return Y

        @staticmethod
        def backward(ctx, dl_dy):
            jac_last_vmva, jac_newton_eq_recon_var, jac_newton_eq_partial_vm_var = ctx.saved_tensors
            partial_vars_idx, partial_pg_yidx, partial_vm_yidx, newton_vars_idx, last_vars_idx, vm_start_yidx = ctx.data
            dl_dx_total = dl_dz_total = None
            ### dl/dz
            if ctx.needs_input_grad[1]:
                # Step 1: gradient of all voltages through step-3 outputs
                dl_dy[:, vm_start_yidx:] += torch.matmul(dl_dy[:, last_vars_idx].unsqueeze(1),
                                                         jac_last_vmva).squeeze(1)
                ## Step 2:implicit gradient for newton solving step-2 outputs, (vector-jacobian trick)
                # jac_newton_eq_partial_pg_var = torch.zeros([dl_dy.shape[0],
                #                                             jac_newton_eq_partial_vm_var.shape[1],
                #                                             len(partial_pg_yidx)]).to(dl_dy.device)
                # jac_newton_eq_partial_pg_var[:, partial_pg_yidx, np.arange(len(partial_pg_yidx))] = 1
                # jac_newton_eq_partial_var = torch.cat([jac_newton_eq_partial_pg_var,
                #                                        jac_newton_eq_partial_vm_var], dim=2)
                d_int = torch.linalg.solve(jac_newton_eq_recon_var.transpose(1, 2),
                                           dl_dy[:, newton_vars_idx].unsqueeze(-1))
                # dl_dz_total = dl_dy[:, partial_vars_idx] + \
                #               torch.matmul(d_int.transpose(1, 2),
                #                            -jac_newton_eq_partial_var).squeeze(1)
                d_int = -d_int.transpose(1, 2)
                dl_dy[:, partial_pg_yidx] += (d_int[:, :, partial_pg_yidx]).squeeze(1)
                dl_dy[:, partial_vm_yidx] += torch.matmul(d_int, jac_newton_eq_partial_vm_var).squeeze(1)
                dl_dz_total = dl_dy[:, partial_vars_idx]

            return dl_dx_total, dl_dz_total

    return PFFunctionFn.apply

def scipy_solve(X, b):
    Xnp = X.to(torch.device("cpu"))
    bnp = b.to(torch.device("cpu"))
    Xb = []
    for i in range(X.shape[0]):
        Xb.append(sp.linalg.solve(Xnp[i], bnp[i]))
    Xb = np.stack(Xb, axis=0)
    return torch.as_tensor(Xb).to(X.device)




class Grpah_ACOPF_Problem(ACOPF_Problem):
    def __init__(self, data, test_size, MaxChangeLoad=0.1, training=True):
        super().__init__(data, test_size, MaxChangeLoad, training)
        """
        Represent ACOPF as Graph:
            Given loads for all nodes: X = [pd, qd]
            Predict deicsion for all nodes: Y = [pg, qg, vm, va]
            Pv node: predict pg, vm, solve qg, va
            Slack node: predict vm, solve pg, qg, known va
            Pq node: solve pg, qg, vm, va
        Operate over reduced graph
        """
        self.kron_reduction = True
        self.cdim = 2
        self.edim = 2
        self.xdim = 4
        self.num_node = self.nb_r
        if training:
            ## Load data
            ## Define train/valid/test split
            ## batch * buses * feature
            C = np.stack([data['Pd'][:, self.non_pq_mid] / self.baseMVA,
                          data['Qd'][:, self.non_pq_mid] / self.baseMVA], axis=2)
            # X = np.stack([np.zeros(shape=[C.shape[0], len(self.non_pq_mid)]),
            #               np.zeros(shape=[C.shape[0], len(self.non_pq_mid)]),
            #               data['Vm'][:, self.non_pq_mid],
            #               data['Va'][:, self.non_pq_mid]], axis=2)
            # X[:, self.spv_r, 0] = data['Pg'] / self.baseMVA
            # X[:, self.spv_r, 1] = data['Qg'] / self.baseMVA
            X = np.concatenate([data['Pg'] / self.baseMVA,
                                data['Qg'] / self.baseMVA,
                                data['Vm'], data['Va']], axis=1)

            feas_mask = ~np.isnan(X[:,:]).any(axis=1)
            C = torch.tensor(C[feas_mask])
            X = torch.tensor(X[feas_mask])
            self.trainC = C[:-test_size]
            self.testC = C[-test_size:]
            self.trainX = X[:-test_size]
            self.testX = X[-test_size:]
            self.trainE = self.testE = self.admittance_from_net()
            self.trainAdj = self.testAdj = None


    def admittance_from_net(self, scaling=1):
        N = self.nb_r
        # Y = (self.Yredr**2 + self.Yredi**2) ** 0.5
        # adj = torch.sign(Y)#torch.exp(-1/Y * scaling)
        # adj = adj - torch.eye(N)
        # adj = Y
        adj = torch.stack([self.Yredr, self.Yredi], dim=-1)
        return adj.view(1, N, N, -1)

    def extend_prediction(self, X, Y):
        Xe = torch.zeros(size=[X.shape[0], self.nb * 2]).to(X.device)
        Ye = torch.zeros(size=[X.shape[0], len(self.partial_r_vars_idx)]).to(X.device)
        Xe[:, self.non_pq_mid] = X[:, :, 0]
        Xe[:, self.nb + self.non_pq_mid] = X[:, :, 1]
        Ye[:, self.pg_pv_zidx] = Y[:, self.pv_r, 0]
        Ye[:, self.vm_spv_zidx] = Y[:, self.spv_r, 2]
        return Xe, Ye

    def graph_prediction(self, X, Y):
        Xe = torch.zeros(size=[X.shape[0], self.nb_r, 2]).to(X.device)
        Ye = torch.zeros(size=[X.shape[0], self.nb_r, 4]).to(X.device)
        Xe[:, :, 0] = X[:, self.non_pq_mid]
        Xe[:, :, 1] = X[:, self.nb + self.non_pq_mid]
        Ye[:, self.pv_r, 0] = Y[:, self.pg_pv_zidx]
        Ye[:, self.spv_r, 2] = Y[:, self.vm_spv_zidx]
        # Ye[:, :, 0] = -Xe[:, :, 0]
        # Ye[:, :, 1] = -Xe[:, :, 1]
        return Xe, Ye

    def node_penalty(self, X, Y):
        resids = torch.cat([Y - self.output_U, self.output_L - Y], dim=1)
        return torch.clamp(resids, 0)

    def edge_penalty(self, X, Y, Adj=None):
        resids = self.branch_ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def cal_penalty(self, X, Y, Adj=None):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y, Adj=None):
        return self.cal_penalty(X, Y)



# """Direct inverse"""
# newton_jac_inv = torch.inverse(jac)
# delta = torch.matmul(newton_jac_inv, gy.unsqueeze(-1)).squeeze(-1)
"""LU decomposition"""
# LU, pivots = torch.linalg.lu_factor(jac)
# delta = torch.linalg.lu_solve(LU, pivots, gy.unsqueeze(-1)).squeeze(-1)


