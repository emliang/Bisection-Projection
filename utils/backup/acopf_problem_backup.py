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
    def __init__(self, data, test_size=1024, MaxChangeLoad=0.1, training=True):
        ## Define optimization problem input and output variables
        ppc = data['ppc']
        self.MaxChangeLoad = MaxChangeLoad
        self.load_ppc(ppc)
        self.load_mid_bus_info()
        self.load_reduced_ppc(ppc)
        ## Keep parameters indicating how data was generated

        if training:
            ## Load data
            ## Define train/valid/test split
            # self.valid_frac = valid_frac
            # self.test_frac = test_frac
            demand = data['Dem'] / self.baseMVA
            gen = data['Gen'] / self.baseMVA
            voltage = data['Vol']
            X = np.concatenate([np.real(demand), np.imag(demand)], axis=1)
            Y = np.concatenate([np.real(gen), np.imag(gen),
                                np.abs(voltage), np.angle(voltage)], axis=1)
            # Yr = np.concatenate([np.real(gen), np.imag(gen),
            #                     np.abs(voltage)[:, self.non_pq_mid], np.angle(voltage)[:, self.non_pq_mid]], axis=1)
            feas_mask = ~np.isnan(Y).any(axis=1)
            self.X = torch.tensor(X[feas_mask])
            self.Y = torch.tensor(Y[feas_mask])
            # self.Yr = torch.tensor(Yr)
            # print(self.eq_resid(self.X, self.Y, False).max())
            # print(self.eq_resid(self.X[:, self.pf_non_mid_index], self.Yr, True).max())
            # print(1/0)
            # cons_vio = self.ineq_resid(self.X, self.Y).max(dim=1)[0] + self.eq_resid(self.X, self.Y).max(dim=1)[0]
            # feas_mask = cons_vio<=1e-5
            # self.X = self.X[feas_mask]
            # self.Y = self.Y[feas_mask]
            # print(self.X.shape)
            # print(1/0)

            self.xdim = X.shape[1]
            self.ydim = Y.shape[1]
            self.num = self.X.shape[0]
            self.neq = 2 * self.nb
            self.nineq = 4 * self.ng + 2 * self.nb + 2 * self.nl
            self.nknowns = self.nslack

            self.trainX = self.X[:-test_size]
            self.testX = self.X[-test_size:]
            self.trainY = self.Y[:-test_size]
            self.testY = self.Y[-test_size:]

        ## Define variables and indices for "partial completion" neural network
        self.PR = 'pgvm'  # 'vmva'
        self.use_mid_bus = True
        self.kron_reduction = True
        if self.PR == 'pgvm':
            # pg (non-slack) and |v|_g (including slack) to be predict
            self.partial_vars_idx = np.concatenate([self.pg_start_yidx + self.pv_,
                                                    self.vm_start_yidx + self.spv])
            # exclude va at slack bus
            self.known_vars = np.concatenate([self.partial_vars_idx,
                                              self.va_start_yidx + self.slack])
            self.other_vars = np.setdiff1d(np.arange(self.ydim), self.known_vars)
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
            self.last_vars_idx = np.concatenate([self.pg_start_yidx + self.slack_,  # slack-bus pg
                                                 self.qg_start_yidx + np.arange(self.ng)])  # pv-bus qg
        elif self.PR == 'vmva':
            # vm (spv) and va(spv)
            self.partial_vars_idx = np.concatenate([self.vm_start_yidx + self.spv,  # pv- & slack-bus vm
                                                    self.va_start_yidx + self.pv])  # pv-bus va
            self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars_idx)  # pg, qg, slack-va (known)
            self.vm_spv_zidx = np.arange(self.npv + self.nslack)
            self.va_pv_zidx = np.arange(self.npv + self.nslack, 2 * self.npv + self.nslack)
            # index for newton methods
            self.newton_eqs_idx = np.concatenate([self.pflow_start_eqidx + self.pq,  # pq bus active power balance
                                                  self.qflow_start_eqidx + self.pq])  # pq bus reactive power balance
            self.newton_vars_idx = np.concatenate([self.vm_start_yidx + self.pq,  # all vm at load buses
                                                   self.va_start_yidx + self.pq])  # all va at load buses
            self.last_eqs_idx = np.concatenate([self.pflow_start_eqidx + self.spv,  # all pg eq
                                                self.qflow_start_eqidx + self.spv])  # al qg eq
            self.last_vars_idx = np.concatenate([self.pg_start_yidx + np.arange(self.ng),  # all pg idx
                                                 self.pg_start_yidx + np.arange(self.ng), ])  # al qg idx
        ### For Pytorch
        self.intrin_dim = max(len(self.partial_vars_idx), 2 * len(self.pq_load))
        self.device = None  # DEVICE
        print(f'neq:{self.neq}, nineq:{self.nineq}, '
              f'indim:{self.xdim}, outdim:{self.ydim}, '
              f'par_outdim:{len(self.partial_vars_idx)}, '
              f'pq_load:{len(self.pq_load)}, pq_mid:{len(self.pq_mid)} '
              f'datasize:{self.X.shape[0]}')

    def __str__(self):
        return 'ACOPF-{}-{}'.format(self.nb, self.MaxChangeLoad)

    def load_ppc(self, ppc):
        # self.baseMVA = ppc['gen'][:, idx_gen.MBASE]
        self.ppc = ppc
        self.baseMVA = self.baseMVA = ppc['baseMVA']
        self.ng = ppc['gen'].shape[0]
        self.nb = ppc['bus'].shape[0]
        self.nl = ppc['branch'].shape[0]
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
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])
        self.branch_idxes = np.concatenate([[ppc['branch'][:, idx_brch.F_BUS]],
                                            [ppc['branch'][:, idx_brch.T_BUS]]], axis=0).T - 1
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2 * self.ng
        self.va_start_yidx = 2 * self.ng + self.nb
        ## useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nb

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
        self.output_L = torch.cat([self.pmin, self.qmin, self.vmin, -torch.ones_like(self.vmin) * torch.pi / 2],
                                  dim=0).view(1, -1)
        self.output_U = torch.cat([self.pmax, self.qmax, self.vmax, torch.ones_like(self.vmax) * torch.pi / 2],
                                  dim=0).view(1, -1)
        self.smax = torch.tensor(ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA)
        self.amax = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMAX]))
        self.amin = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMIN]))
        self.slackva = self.va_init[self.slack]

    def load_reduced_ppc(self, ppc):
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

        ## initial values for solver
        self.vm_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.VM])
        self.va_r_init = torch.tensor(np.deg2rad(ppc['bus'][self.non_pq_mid, idx_bus.VA]))
        self.pd_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.PD] / self.baseMVA)
        self.qd_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.QD] / self.baseMVA)
        self.output_r_init = torch.cat([self.pg_init, self.qg_init, self.vm_r_init, self.va_r_init], dim=0).view(1, -1)
        self.input_r_init = torch.cat([self.pd_r_init, self.qd_r_init], dim=0).view(1, -1)

        self.nb_r = len(self.non_pq_mid)
        self.yrdim = self.nb_r*2 + self.ng*2
        self.slack_r = np.where(ppc['bus'][self.non_pq_mid, idx_bus.BUS_TYPE] == 3)[0]
        self.pv_r = np.where(ppc['bus'][self.non_pq_mid, idx_bus.BUS_TYPE] == 2)[0]
        self.pq_r = np.where(ppc['bus'][self.non_pq_mid, idx_bus.BUS_TYPE] == 1)[0]
        self.spv_r = np.sort(np.concatenate([self.slack_r, self.pv_r]))
        self.nonslack_r = np.sort(np.concatenate([self.pq_r, self.pv_r]))
        self.pg_r_start_yidx = 0
        self.qg_r_start_yidx = self.ng
        self.vm_r_start_yidx = 2 * self.ng
        self.va_r_start_yidx = 2 * self.ng + self.nb_r
        ## useful indices for equality constraints
        self.pflow_r_start_eqidx = 0
        self.qflow_r_start_eqidx = self.nb_r

        # pg (non-slack) and |v|_g (including slack) to be predict
        self.partial_r_vars_idx = np.concatenate([self.pg_r_start_yidx + self.pv_,
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
        self.last_r_vars_idx = np.concatenate([self.pg_r_start_yidx + self.slack_,  # slack-bus pg
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

    def load_mid_bus_info(self):
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

        ### 2.1: non-mid bus used in newton methds (partial pv + pq load)
        self.newton_nomid_eqs_idx = np.concatenate(
            [self.pflow_start_eqidx + self.pv,  # real power flow at non-slack gens
             self.pflow_start_eqidx + self.pq_load,  # real power flow at load buses (=0)
             self.qflow_start_eqidx + self.pq_load])  # reactive power flow at load buses (=0)
        self.newton_nomid_vars_idx = np.concatenate([self.vm_start_yidx + self.pq_load,  # vm at load buses
                                                     self.va_start_yidx + self.pv,  # va at non-slack gens
                                                     self.va_start_yidx + self.pq_load])  # va at load buses
        ### 2.2: mid bus information
        self.mid_vars_idx = np.concatenate([self.vm_start_yidx + self.pq_mid,
                                            self.va_start_yidx + self.pq_mid])
        self.nomid_vars_idx = np.concatenate([self.vm_start_yidx + self.non_pq_mid,
                                              self.va_start_yidx + self.non_pq_mid])

    def solve_mid_bus(self, Y, return_jac=True):
        pq_mid_len = len(self.pq_mid)
        non_pq_mid_len = len(self.non_pq_mid)
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

        jac_full = None
        if return_jac:
            # solve mid-bus jacobian w.r.t non-mid bus
            ## jac_1 = d(vm_mid, va_mid)/d(vr_mid, vi_mid)
            jac_1 = torch.zeros([Y.shape[0], 2 * pq_mid_len, 2 * pq_mid_len], device=Y.device)
            d_vm_mid_d_vr_mid = vr_mid / vm_mid
            d_vm_mid_d_vi_mid = vi_mid / vm_mid
            d_va_mid_d_vr_mid = -  vi_mid / vm_mid_2
            d_va_mid_d_vi_mid = vr_mid / vm_mid_2
            jac_1[:, :pq_mid_len, :pq_mid_len] = torch.diag_embed(d_vm_mid_d_vr_mid)
            jac_1[:, :pq_mid_len, pq_mid_len:] = torch.diag_embed(d_vm_mid_d_vi_mid)
            jac_1[:, pq_mid_len:, :pq_mid_len] = torch.diag_embed(d_va_mid_d_vr_mid)
            jac_1[:, pq_mid_len:, pq_mid_len:] = torch.diag_embed(d_va_mid_d_vi_mid)
            ## jac_2 = d(vr_mid, vi_mid)/d(vr_non_mid, vi_non_mid)
            d_vri_mid_d_vri_non_mid = - self.mid_complete
            ## jac_3 = d(vr_non_mid, vi_non_mid)/d(vm_non_mid, va_non_mid)
            jac_3 = torch.zeros([Y.shape[0], 2 * non_pq_mid_len, 2 * non_pq_mid_len], device=Y.device)
            d_vr_non_mid_d_vm_non_mid = cosva
            d_vr_non_mid_d_va_non_mid = - vi
            d_vi_non_mid_d_vm_non_mid = sinva
            d_vi_non_mid_d_va_non_mid = vr
            jac_3[:, :non_pq_mid_len, :non_pq_mid_len] = torch.diag_embed(d_vr_non_mid_d_vm_non_mid)
            jac_3[:, :non_pq_mid_len, non_pq_mid_len:] = torch.diag_embed(d_vr_non_mid_d_va_non_mid)
            jac_3[:, non_pq_mid_len:, :non_pq_mid_len] = torch.diag_embed(d_vi_non_mid_d_vm_non_mid)
            jac_3[:, non_pq_mid_len:, non_pq_mid_len:] = torch.diag_embed(d_vi_non_mid_d_va_non_mid)
            jac = torch.matmul(torch.matmul(jac_1, d_vri_mid_d_vri_non_mid), jac_3)
            jac_full = torch.zeros([Y.shape[0], 2 * pq_mid_len, 2 * self.nb], device=Y.device)
            jac_full[:, :, self.pf_non_mid_index] = jac
        return Yfull, jac_full

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
        resids = torch.cat([Y - self.output_U, self.output_L - Y,
                            self.branch_ineq_resid(X, Y)
                            ], dim=1)
        # et = time.time()
        # print(et-st)
        return torch.clamp(resids, 0)

    def branch_ineq_resid(self, X, Y):
        _, _, vm, va = self.get_yvars(Y)
        ### Branch angele limit
        va_start = va[:, self.branch_idxes[:, 0]]
        va_end = va[:, self.branch_idxes[:, 1]]
        resids_brach_angle = torch.abs(va_start - va_end) - self.amax
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
        resids_branch_flow = torch.max(sij, sji) - self.smax ** 2
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
                Yb = PF_pgvm_Function_reduced(self)(Xb[:, self.pf_non_mid_index], Zb)
                Yb, _  = self.solve_mid_bus(Yb, return_jac=False)
            else:
                if self.use_mid_bus:
                    Yb = PF_pgvm_Function_mid(self)(Xb, Zb)
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
                     self.nb, self.spv, self.pv_, self.baseMVA, self.baseMVA) for i in range(X_np.shape[0])]
            results = pool.map(solve_opt_problem, args)
        return torch.as_tensor(np.array(results)).to(X.device)

def PF_pgvm_Function(data, tol=1e-5, max_iters=5):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # start_time = time.time()
            Y = torch.zeros(X.shape[0], data.ydim, device=X.device) + data.output_init
            ## Step 1: known variables: pg at non-slack gens, vm at gens
            partial_vars_idx = data.partial_vars_idx
            Y[:, data.partial_vars_idx] = Z
            ## Step 2: newton variables
            newton_eqs_idx = data.newton_eqs_idx
            newton_vars_idx = data.newton_vars_idx
            for n in range(max_iters):
                gy = data.eq_resid(X, Y)[:, newton_eqs_idx]
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
            last_eqs_idx = data.last_eqs_idx
            last_vars_idx = data.last_vars_idx
            Y[:, last_vars_idx] -= data.eq_resid(X, Y)[:, last_eqs_idx]

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

            ### dl/dx
            if ctx.needs_input_grad[0]:
                # gradient of pd at slack and qd at gens through step 3 outputs
                dl_dpdqd_3 = dl_dy[:, last_vars_idx]
                dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
                dl_dx_3[:, np.concatenate([data.slack, data.nb + data.spv])] = dl_dpdqd_3

                dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
                dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
                dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
                dl_dx_2[:, data.nb + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses
                # Final quantities
                dl_dx_total = dl_dx_3 + dl_dx_2
            return dl_dx_total, dl_dz_total

    return PFFunctionFn.apply

def PF_pgvm_Function_mid(data, tol=1e-5, max_iters=5):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # start_time = time.time()
            Y = torch.zeros(X.shape[0], data.ydim, device=X.device) + data.output_init
            ## Step 1: known variables: pg at non-slack gens, vm at gens
            partial_vars_idx = data.partial_vars_idx
            Y[:, data.partial_vars_idx] = Z
            ## Step 2: newton variables
            ### 2.1: non-mid bus information (load bus used in newton update)
            newton_eqs_idx = data.newton_nomid_eqs_idx
            newton_vars_idx = data.newton_nomid_vars_idx
            ### 2.2: mid bus information (mid bus used in newton update)
            newton_mid_vars_idx = data.mid_vars_idx
            for n in range(max_iters):
                Y, jac_mid_non_mid = data.solve_mid_bus(Y)
                jac_mid_newton_guess = jac_mid_non_mid[:, :, newton_vars_idx - 2 * data.ng]
                gy = data.eq_resid(X, Y)[:, newton_eqs_idx]
                jac_vmva = data.eq_jac_v(Y)
                jac_newton_eq_mid_var = jac_vmva[:, newton_eqs_idx, :][:, :, newton_mid_vars_idx - 2 * data.ng]
                jac_newton_eq_recon_var = jac_vmva[:, newton_eqs_idx, :][:, :, newton_vars_idx - 2 * data.ng] + \
                                          torch.matmul(jac_newton_eq_mid_var, jac_mid_newton_guess)
                if torch.abs(gy).max() < tol:
                    break
                ### Newton's update
                # delta = scipy_solve(jac_newton_eq_recon_var, gy.unsqueeze(-1)).squeeze(-1)
                delta = torch.linalg.solve(jac_newton_eq_recon_var, gy.unsqueeze(-1)).squeeze(-1)
                Y[:, newton_vars_idx] -= delta
            if torch.abs(gy).max() > tol:
                print(f'Newton methods for Power Flow does not converge in {n} with error {torch.abs(gy).max()}',
                      end='\r')
            else:
                print(f'Newton methods for Power Flow converge in {n} iter with error {torch.abs(gy).max()}', end='\r')

            ## Step 3: last variables
            last_eqs_idx = data.last_eqs_idx
            last_vars_idx = data.last_vars_idx
            Y[:, last_vars_idx] -= data.eq_resid(X, Y)[:, last_eqs_idx]

            ## store information for backpropagation
            vm_start_yidx = data.vm_start_yidx
            partial_pg_yidx = partial_vars_idx[data.pg_pv_zidx]
            partial_vm_yidx = partial_vars_idx[data.vm_spv_zidx]

            # jac_last_vmva = -jacs[:, last_eqs_idx, :][:, :, vm_start_yidx:]
            # # jac_newton_eq_mid_var = jacs[:, newton_eqs_idx, :][:,:,newton_mid_vars_idx]
            # # jac_newton_eq_recon_var = jacs[:, newton_eqs_idx, :][:,:,newton_vars_idx]\
            # #                           + torch.matmul(jac_newton_eq_mid_var, jacs_mid_non_mid)
            # jac_newton_eq_recon_var = jac_pq_load
            # jac_newton_eq_partial_var = jacs[:, newton_eqs_idx,:][:,:,partial_vars_idx]
            # print(jacs.shape,
            #     jac_newton_eq_recon_var.shape, jac_last_vmva.shape,
            #       jac_newton_eq_partial_var.shape, jacs_mid_non_mid.shape)
            # print(1/0)

            # jac_newton_eq_partial_var = jac_full[:, newton_eqs_idx,:][:,:,partial_vars_idx]
            jac_newton_eq_partial_vm_var = jac_vmva[:, newton_eqs_idx, :][:, :, partial_vm_yidx - 2 * data.ng]
            jac_newton_eq_partial_vm_var += torch.matmul(jac_newton_eq_mid_var, jac_mid_non_mid[:, :, data.spv])
            ctx.save_for_backward(-jac_vmva[:, last_eqs_idx, :],
                                  jac_newton_eq_recon_var,
                                  jac_newton_eq_partial_vm_var,
                                  jac_mid_non_mid)
            ctx.data = [partial_vars_idx, partial_pg_yidx, partial_vm_yidx, newton_vars_idx, newton_mid_vars_idx,
                        last_vars_idx, vm_start_yidx]
            return Y

        @staticmethod
        def backward(ctx, dl_dy):
            jac_last_vmva, jac_newton_eq_recon_var, jac_newton_eq_partial_vm_var, jac_mid_non_mid = ctx.saved_tensors
            partial_vars_idx, partial_pg_yidx, partial_vm_yidx, newton_vars_idx, newton_mid_vars_idx, last_vars_idx, vm_start_yidx = ctx.data
            dl_dx_total = dl_dz_total = None
            ### dl/dz
            if ctx.needs_input_grad[1]:
                # Step 1: gradient of all voltages through step-3 outputs
                dl_dy[:, vm_start_yidx:] += torch.matmul(dl_dy[:, last_vars_idx].unsqueeze(1),
                                                         jac_last_vmva).squeeze(1)
                # Step 2: gradient of all non-mid bus through step-2.2 outputs
                dl_dy[:, vm_start_yidx:] += torch.matmul((dl_dy[:, newton_mid_vars_idx]).unsqueeze(1),
                                                         jac_mid_non_mid).squeeze(1)
                ## Step 3:implicit gradient for newton solving step-2.1 outputs, (vector-jacobian trick)
                # jac_newton_eq_partial_pg_var = torch.zeros([dl_dy.shape[0],
                #                                             jac_newton_eq_partial_vm_var.shape[1],
                #                                             len(partial_pg_yidx)]).to(dl_dy.device)
                # jac_newton_eq_partial_pg_var[:, partial_pg_yidx, np.arange(len(partial_pg_yidx))] = 1
                # jac_newton_eq_partial_var = torch.cat([jac_newton_eq_partial_pg_var,
                #                                        jac_newton_eq_partial_vm_var], dim=2)
                d_int = torch.linalg.solve(jac_newton_eq_recon_var.transpose(1, 2),
                                           dl_dy[:, newton_vars_idx].unsqueeze(-1))
                d_int = -d_int.transpose(1, 2)
                dl_dy[:, partial_pg_yidx] += (d_int[:, :, partial_pg_yidx]).squeeze(1)
                dl_dy[:, partial_vm_yidx] += torch.matmul(d_int, jac_newton_eq_partial_vm_var).squeeze(1)
                dl_dz_total = dl_dy[:, partial_vars_idx]

            if ctx.needs_input_grad[0]:
                # gradient of pd at slack and qd at gens through step 3 outputs
                dl_dpdqd_3 = dl_dy[:, last_vars_idx]
                # insert into correct places in x and y loss vectors
                dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
                dl_dx_3[:, np.concatenate([data.slack, data.nb + data.spv])] = dl_dpdqd_3
                dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
                dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
                dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
                dl_dx_2[:, data.nb + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses
                # Final quantities
                dl_dx_total = dl_dx_3 + dl_dx_2
            return None, dl_dz_total

    return PFFunctionFn.apply

def PF_pgvm_Function_reduced(data, tol=1e-5, max_iters=5):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # start_time = time.time()
            reduced = data.kron_reduction
            Y = torch.zeros(X.shape[0], data.yrdim, device=X.device) + data.output_r_init
            ## Step 1: known variables: pg at non-slack gens, vm at gens
            partial_vars_idx = data.partial_r_vars_idx
            Y[:, partial_vars_idx] = Z
            ## Step 2: newton variables
            newton_eqs_idx = data.newton_r_eqs_idx
            newton_vars_idx = data.newton_r_vars_idx
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
            last_eqs_idx = data.last_r_eqs_idx
            last_vars_idx = data.last_r_vars_idx
            Y[:, last_vars_idx] -= data.eq_resid(X, Y, reduced)[:, last_eqs_idx]
            ## store information for backpropagation
            vm_start_yidx = data.vm_r_start_yidx
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

            ### dl/dx
            if ctx.needs_input_grad[0]:
                # gradient of pd at slack and qd at gens through step 3 outputs
                dl_dpdqd_3 = dl_dy[:, last_vars_idx]
                dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
                dl_dx_3[:, np.concatenate([data.slack, data.nb + data.spv])] = dl_dpdqd_3

                dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
                dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
                dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
                dl_dx_2[:, data.nb + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses
                # Final quantities
                dl_dx_total = dl_dx_3 + dl_dx_2
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


# def PF_vmva_Function(data, tol=1e-5, bsz=1024, max_iters=10):
#     class PFFunctionFn(Function):
#         @staticmethod
#         def forward(ctx, X, Z):
#             # start_time = time.time()
#             ## Step 1: Newton's method
#             Y = torch.zeros(X.shape[0], data.ydim, device=X.device) + data.output_init
#             ## Step 1: known variables: vm & va at pv bus
#             partial_vars_idx = data.partial_vars_idx
#             Y[:, data.partial_vars_idx] = Z
#             # init guesses for remaining values
#             Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = 0  # qg at gens (not used in Newton upd)
#             Y[:, data.pg_start_yidx:data.pg_start_yidx + data.ng] = 0  # pg at gens (not used in Newton upd)
#
#             newton_eqs_idx = data.newton_eqs_idx
#             newton_vars_idx = data.newton_vars_idx
#             last_eqs_idx = data.last_eqs_idx
#             last_vars_idx = data.last_vars_idx
#             jacs = []
#             for b in range(0, X.shape[0], bsz):
#                 X_b = X[b:b + bsz]
#                 Y_b = Y[b:b + bsz]
#                 for n in range(max_iters):
#                     gy = data.eq_resid(X_b, Y_b)[:, newton_eqs_idx]
#                     jac_full = data.eq_jac(Y_b)
#                     jac = jac_full[:, newton_eqs_idx, :]
#                     jac = jac[:, :, newton_vars_idx]
#                     if torch.abs(gy).max() < tol:
#                         break
#                     ### Newton update
#                     delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
#                     Y_b[:, newton_vars_idx] -= delta
#
#                 if torch.abs(gy).max() > tol:
#                     print('Newton methods for Power Flow does not converge:', torch.abs(gy).max())
#                 # else:
#                 #     print(n, torch.abs(gy).max())#, end='\r'
#                 jacs.append(jac_full)
#
#             ## Step 2: Solve for remaining variables
#             eq_resid = data.eq_resid(X, Y)
#             Y[:, last_vars_idx] = -eq_resid[:, last_eqs_idx]
#             # # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
#             # Y[:, data.qg_start_yidx + data.spv] =  -eq_resid[:, data.qflow_start_eqidx + data.spv]
#             # # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
#             # Y[:, data.pg_start_yidx + data.slack_] =  -eq_resid[:, data.pflow_start_eqidx + data.slack]
#
#             partial_vars_idx = data.partial_vars_idx
#             vm_start_yidx = data.vm_start_yidx
#             ctx.data = [partial_vars_idx, newton_vars_idx, newton_eqs_idx, last_vars_idx, last_eqs_idx, vm_start_yidx]
#             jacs = torch.cat(jacs)
#             # jac_last_vmva = -jacs[:, last_eqs_idx, :][:, :, vm_start_yidx:]
#             # jac_newton_eq_recon_var = jacs[:, newton_eqs_idx, :][:,:,newton_vars_idx]
#             # jac_newton_eq_partial_var = jacs[:, newton_eqs_idx,:][:,:,partial_vars_idx]
#             ctx.save_for_backward(-jacs[:, last_eqs_idx, :][:, :, vm_start_yidx:],
#                                   jacs[:, newton_eqs_idx, :][:,:,newton_vars_idx],
#                                   jacs[:, newton_eqs_idx,:][:,:,partial_vars_idx])
#             return Y
#
#         @staticmethod
#         def backward(ctx, dl_dy):
#             partial_vars_idx, newton_vars_idx, newton_eqs_idx, last_vars_idx, last_eqs_idx, vm_start_yidx = ctx.data
#             jac_last_vmva, jac_newton_eq_recon_var, jac_newton_eq_partial_var = ctx.saved_tensors
#             dl_dx_total = dl_dz_total = None
#             ### dl/dz
#             if ctx.needs_input_grad[1]:
#                 ## Step 2 (calc pg at slack and qg at gens)
#                 # gradient of all voltages through step 3 outputs
#                 # jac_last_vmva = -jac[:,last_eqs_idx,:][:,:,vm_start_yidx:]
#                 dl_dlast_dvmva = torch.matmul(dl_dy[:, last_vars_idx].unsqueeze(1),
#                                               jac_last_vmva).squeeze(1)
#                 dl_dy[:, vm_start_yidx:] += dl_dlast_dvmva # dl_dz1 + dl_dz2_dz2_dz1
#                 ## Step 1
#                 # Use precomputed inverse jacobian
#                 # jac2 = jac[:, newton_eqs_idx, :]
#                 d_int = torch.linalg.solve(jac_newton_eq_recon_var.transpose(1, 2),
#                                            dl_dy[:, newton_vars_idx].unsqueeze(-1))
#                 dl_dz_2 = torch.matmul(d_int.transpose(1, 2), -jac_newton_eq_partial_var).squeeze(1)
#                 dl_dz_total = dl_dz_2 + dl_dy[:, partial_vars_idx]
#             ### dl/dx
#             if ctx.needs_input_grad[0]:
#                 # gradient of pd at slack and qd at gens through step 3 outputs
#                 dl_dpdqd_3 = dl_dy[:, last_vars_idx]
#                 dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_3[:, np.concatenate([data.slack, data.nb + data.spv])] = dl_dpdqd_3
#
#                 dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
#                 dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
#                 dl_dx_2[:, data.nb + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses
#                 # Final quantities
#                 dl_dx_total = dl_dx_3 + dl_dx_2
#
#             return dl_dx_total, dl_dz_total
#
#     return PFFunctionFn.apply
#
# def PF_vmva_Function_plus(data, tol=1e-5, bsz=1024, max_iters=10):
#     class PFFunctionFn(Function):
#         @staticmethod
#         def forward(ctx, X, Z):
#             # start_time = time.time()
#             ## Step 1: Newton's method
#             Y = torch.zeros(X.shape[0], data.ydim, device=X.device)
#             # initialize vm va
#             Y[:, data.vm_start_yidx: data.vm_start_yidx+data.nb] = data.vm_init  # vm at all buses
#             Y[:, data.va_start_yidx: data.va_start_yidx+data.nb] = data.va_init   # va at all buses
#             # known/estimated values (vm at )
#             Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]  # vm at spv gnes
#             Y[:, data.va_start_yidx + data.pv] = Z[:, data.va_pv_zidx]  # va at pv gens
#             # init guesses for remaining values
#             Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = 0  # qg at gens (not used in Newton upd)
#             Y[:, data.pg_start_yidx:data.pg_start_yidx + data.ng] = 0  # pg at gens (not used in Newton upd)
#
#             newton_eqs_idx = np.concatenate([
#                 data.pflow_start_eqidx + data.pq_load,  # real power flow at non-slack gens
#                 data.qflow_start_eqidx + data.pq_load])  # reactive power flow at load buses (=0)
#             newton_vars_idx = np.concatenate([
#                 data.vm_start_yidx + data.pq_load,  # vm at load buses
#                 data.va_start_yidx + data.pq_load])  # va at load buses
#             newton_mid_vars_idx = np.concatenate([data.vm_start_yidx + data.pq_mid,
#                                              data.va_start_yidx + data.pq_mid])
#             last_eqs_idx = data.last_eqs_idx
#             last_vars_idx = data.last_vars_idx
#             jacs = []
#             jacs_mid_non_mid = []
#             for b in range(0, X.shape[0], bsz):
#                 X_b = X[b:b + bsz]
#                 Y_b = Y[b:b + bsz]
#                 for n in range(max_iters):
#                     Y_b, jac_mid_non_mid = data.solve_mid_bus(Y_b)
#                     jac_mid_non_mid = jac_mid_non_mid[:,:,newton_vars_idx-data.vm_start_yidx]
#                     gy = data.eq_resid(X_b, Y_b)[:, newton_eqs_idx]
#                     jac_full = data.eq_jac(Y_b)
#                     jac = jac_full[:, newton_eqs_idx, :]
#                     jac_pq_mid = jac[:, :, newton_mid_vars_idx]
#                     jac_pq_load = jac[:, :, newton_vars_idx] + torch.matmul(jac_pq_mid, jac_mid_non_mid)
#                     if torch.abs(gy).max() < tol:
#                         break
#                     ### Newton's update
#                     delta = torch.linalg.solve(jac_pq_load, gy.unsqueeze(-1)).squeeze(-1)
#                     Y_b[:, newton_vars_idx] -= delta
#                 if torch.abs(gy).max() > tol:
#                     print('Newton methods for Power Flow does not converge:', torch.abs(gy).max(), end='\r')
#                 # else:
#                 #     print(n, torch.abs(gy).max())#, end='\r'
#                 jacs.append(jac_full)
#                 jacs_mid_non_mid.append(jac_mid_non_mid)
#             ## Step 2: Solve for remaining variables
#             eq_resid = data.eq_resid(X, Y)
#             Y[:, last_vars_idx] = -eq_resid[:, last_eqs_idx]
#             # # solve for pg at all gens
#             # Y[:, data.pg_start_yidx:data.pg_start_yidx + data.ng] =  -eq_resid[:, data.pflow_start_eqidx + data.spv]
#             # # solve for qg at all gens
#             # Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] =  -eq_resid[:, data.qflow_start_eqidx + data.spv]
#
#             partial_vars_idx = data.partial_vars_idx
#             vm_start_yidx = data.vm_start_yidx
#             ng = data.ng
#             ctx.data = [partial_vars_idx, newton_vars_idx, newton_mid_vars_idx, newton_eqs_idx, last_vars_idx, last_eqs_idx, vm_start_yidx, ng]
#
#             jacs = torch.cat(jacs)
#             jacs_mid_non_mid = torch.cat(jacs_mid_non_mid)
#             # jac_last_vmva = -jacs[:, last_eqs_idx, :][:, :, vm_start_yidx:]
#             # jac_newton_eq_mid_var = jacs[:, newton_eqs_idx, :][:,:,newton_mid_vars_idx]
#             # jac_newton_eq_recon_var = jacs[:, newton_eqs_idx, :][:,:,newton_vars_idx]\
#             #                           + torch.matmul(jac_newton_eq_mid_var, jacs_mid_non_mid)
#             # jac_newton_eq_recon_var = jac_pq_load
#             # jac_newton_eq_partial_var = jacs[:, newton_eqs_idx,:][:,:,partial_vars_idx]
#             ctx.save_for_backward(-jacs[:, last_eqs_idx, :][:, :, vm_start_yidx:],
#                                   jac_pq_load,
#                                   jacs[:, newton_eqs_idx,:][:,:,partial_vars_idx],
#                                   jacs_mid_non_mid)
#             return Y
#
#         @staticmethod
#         def backward(ctx, dl_dy):
#             partial_vars_idx, newton_vars_idx, newton_mid_vars_idx, newton_eqs_idx, last_vars_idx, last_eqs_idx, vm_start_yidx, ng = ctx.data
#             jac_last_vmva, jac_newton_eq_recon_var, jac_newton_eq_partial_var, jac_mid_non_mid = ctx.saved_tensors
#             dl_dx_total = dl_dz_total = None
#             ### dl/dz
#             if ctx.needs_input_grad[1]:
#                 # jac_last_vmva = -jac[:,last_eqs_idx,:][:,:,vm_start_yidx:]
#                 dl_dlast_dvmva = torch.matmul(dl_dy[:, last_vars_idx].unsqueeze(1),
#                                               jac_last_vmva).squeeze(1)
#                 dl_dmid_dnonmid = torch.matmul((dl_dy[:, newton_mid_vars_idx]).unsqueeze(1),
#                                                jac_mid_non_mid).squeeze(1)
#                 dl_dlast_dnonmid_dmid = torch.matmul( (dl_dlast_dvmva[:, newton_mid_vars_idx - 2 * ng]).unsqueeze(1),
#                                                       jac_mid_non_mid).squeeze(1)
#                 dl_dy[:, vm_start_yidx:] += dl_dlast_dvmva
#                 dl_dy[:, newton_vars_idx] += dl_dmid_dnonmid + dl_dlast_dnonmid_dmid
#
#                 ## Step 1
#                 # Use precomputed inverse jacobian
#                 # jac2 = jac[:, newton_eqs_idx, :]
#                 # jac_mid = jac2[:, :, newton_mid_vars_idx]
#                 # jac_non_mid = jac2[:, :, newton_vars_idx] + torch.matmul(jac_mid, jac_mid_non_mid)
#                 d_int = torch.linalg.solve(jac_newton_eq_recon_var.transpose(1, 2),
#                                            dl_dy[:, newton_vars_idx].unsqueeze(-1))
#                 dl_dz_2 = torch.matmul(d_int.transpose(1, 2), -jac_newton_eq_partial_var).squeeze(1)
#                 dl_dz_total = dl_dz_2 + dl_dy[:, partial_vars_idx]
#
#             if ctx.needs_input_grad[0]:
#                 # gradient of pd at slack and qd at gens through step 3 outputs
#                 dl_dpdqd_3 = dl_dy[:, last_vars_idx]
#                 # insert into correct places in x and y loss vectors
#                 dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_3[:, np.concatenate([data.slack, data.nb + data.spv])] = dl_dpdqd_3
#                 dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
#                 dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
#                 dl_dx_2[:, data.nb + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses
#                 # Final quantities
#                 dl_dx_total = dl_dx_3 + dl_dx_2
#             return dl_dx_total, dl_dz_total
#
#     return PFFunctionFn.apply
#

# """Direct inverse"""
# newton_jac_inv = torch.inverse(jac)
# delta = torch.matmul(newton_jac_inv, gy.unsqueeze(-1)).squeeze(-1)
"""LU decomposition"""
# LU, pivots = torch.linalg.lu_factor(jac)
# delta = torch.linalg.lu_solve(LU, pivots, gy.unsqueeze(-1)).squeeze(-1)


# def PF_pgvm_Function_v0(data, tol=1e-5, max_iters=10):
#     class PFFunctionFn(Function):
#         @staticmethod
#         def forward(ctx, X, Z):
#             # start_time = time.time()
#             Y = torch.zeros(X.shape[0], data.ydim, device=X.device) + data.output_init
#             ## Step 1: known variables: pg at non-slack gens, vm at gens
#             partial_vars_idx = data.partial_vars_idx
#             Y[:, data.partial_vars_idx] = Z
#             ## Step 2: newton variables
#             newton_eqs_idx = data.newton_eqs_idx
#             newton_vars_idx = data.newton_vars_idx
#             for n in range(max_iters):
#                 gy = data.eq_resid(X, Y)[:, newton_eqs_idx]
#                 jac_full = data.eq_jac(Y)
#                 jac_newton_eq_recon_var = jac_full[:, newton_eqs_idx, :][:, :, newton_vars_idx]
#                 if torch.abs(gy).max() < tol:
#                     break
#                 ### Newton update
#                 delta = torch.linalg.solve(jac_newton_eq_recon_var, gy.unsqueeze(-1)).squeeze(-1)
#                 Y[:, newton_vars_idx] -= delta
#             if torch.abs(gy).max() > tol:
#                 print(f'Newton methods for Power Flow does not converge in {n} with error {torch.abs(gy).max()}', end='\r')
#             else:
#                 print(f'Newton methods for Power Flow converge in {n} iter with error {torch.abs(gy).max()}', end='\r')
#             ## Step 3: last variables
#             last_eqs_idx = data.last_eqs_idx
#             last_vars_idx = data.last_vars_idx
#             Y[:, last_vars_idx]  -= data.eq_resid(X, Y)[:, last_eqs_idx]
#
#             ## store information for backpropagation
#             vm_start_yidx = data.vm_start_yidx
#             # print('Newton methods error', n, torch.abs(data.eq_resid(X, Y)).max(), end='\r')
#             # jac_last_vmva = -jac_eq[:, last_eqs_idx, :][:, :, vm_start_yidx:]
#             # jac_newton_eq_recon_var = jac_eq[:, newton_eqs_idx, :][:,:,newton_vars_idx]
#             # jac_newton_eq_partial_var = jac_eq[:, newton_eqs_idx,:][:,:,partial_vars_idx]
#             # print(jac_full.shape, jac.shape, jac_last_vmva.shape, jac_newton_eq_recon_var.shape, jac_newton_eq_partial_var.shape)
#             # print(1/0)
#             ctx.save_for_backward(-jac_full[:, last_eqs_idx, :][:, :, vm_start_yidx:],
#                                   jac_newton_eq_recon_var,
#                                   jac_full[:, newton_eqs_idx,:][:,:,partial_vars_idx])
#             ctx.data = [partial_vars_idx, newton_vars_idx, last_vars_idx, vm_start_yidx]
#             return Y
#
#         @staticmethod
#         def backward(ctx, dl_dy):
#             jac_last_vmva, jac_newton_eq_recon_var, jac_newton_eq_partial_var = ctx.saved_tensors
#             partial_vars_idx, newton_vars_idx, last_vars_idx, vm_start_yidx = ctx.data
#             dl_dx_total = dl_dz_total = None
#             ### dl/dz
#             if ctx.needs_input_grad[1]:
#                 # Step 1: gradient of all voltages through step-3 outputs
#                 dl_dy[:, vm_start_yidx:] += torch.matmul(dl_dy[:, last_vars_idx].unsqueeze(1),
#                                               jac_last_vmva).squeeze(1)
#                 ## Step 2:implicit gradient for newton solving step-2 outputs, (vector-jacobian trick)
#                 d_int = torch.linalg.solve(jac_newton_eq_recon_var.transpose(1, 2),
#                                            dl_dy[:, newton_vars_idx].unsqueeze(-1))
#                 dl_dz_total = dl_dy[:, partial_vars_idx] + \
#                               torch.matmul(d_int.transpose(1, 2),
#                                            -jac_newton_eq_partial_var).squeeze(1)
#             ### dl/dx
#             if ctx.needs_input_grad[0]:
#                 # gradient of pd at slack and qd at gens through step 3 outputs
#                 dl_dpdqd_3 = dl_dy[:, last_vars_idx]
#                 dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_3[:, np.concatenate([data.slack, data.nb + data.spv])] = dl_dpdqd_3
#
#                 dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
#                 dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
#                 dl_dx_2[:, data.nb + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses
#                 # Final quantities
#                 dl_dx_total = dl_dx_3 + dl_dx_2
#             return dl_dx_total, dl_dz_total
#     return PFFunctionFn.apply

#
# def PF_pgvm_Function_plus(data, tol=1e-5, max_iters=10):
#     class PFFunctionFn(Function):
#         @staticmethod
#         def forward(ctx, X, Z):
#             # start_time = time.time()
#             Y = torch.zeros(X.shape[0], data.ydim, device=X.device) + data.output_init
#             ## Step 1: known variables: pg at non-slack gens, vm at gens
#             partial_vars_idx = data.partial_vars_idx
#             Y[:, data.partial_vars_idx] = Z
#             ## Step 2: newton variables
#             ### 2.1: non-mid bus information (load bus used in newton update)
#             newton_eqs_idx = data.newton_nomid_eqs_idx
#             newton_vars_idx = data.newton_nomid_vars_idx
#             ### 2.2: mid bus information (mid bus used in newton update)
#             newton_mid_vars_idx = data.mid_vars_idx
#             non_mid_idx = data.nomid_vars_idx
#             for n in range(max_iters):
#                 Y, jac_mid_non_mid = data.solve_mid_bus(Y)
#                 # jac_mid_pred_vm = jac_mid_non_mid[:,:,data.spv]
#                 # jac_mid_non_mid = jac_mid_non_mid[:,:,newton_vars_idx-data.vm_start_yidx]
#                 jac_mid_newton_guess = jac_mid_non_mid[:,:,newton_vars_idx-data.vm_start_yidx]
#                 gy = data.eq_resid(X, Y)[:, newton_eqs_idx]
#                 jac_full = data.eq_jac(Y)
#                 jac_newton_eq_mid_var = jac_full[:, newton_eqs_idx,:][:,:,newton_mid_vars_idx]
#                 jac_newton_eq_recon_var = jac_full[:, newton_eqs_idx,:][:,:,newton_vars_idx] + \
#                                           torch.matmul(jac_newton_eq_mid_var, jac_mid_newton_guess)
#                 if torch.abs(gy).max() < tol:
#                     break
#                 ### Newton's update
#                 # delta = scipy_solve(jac_newton_eq_recon_var, gy.unsqueeze(-1)).squeeze(-1)
#                 delta = torch.linalg.solve(jac_newton_eq_recon_var, gy.unsqueeze(-1)).squeeze(-1)
#                 Y[:, newton_vars_idx] -= delta
#             if torch.abs(gy).max() > tol:
#                 print(f'Newton methods for Power Flow does not converge in {n} with error {torch.abs(gy).max()}', end='\r')
#             else:
#                 print(f'Newton methods for Power Flow converge in {n} iter with error {torch.abs(gy).max()}', end='\r')
#
#             ## Step 3: last variables
#             last_eqs_idx = data.last_eqs_idx
#             last_vars_idx = data.last_vars_idx
#             Y[:, last_vars_idx] -= data.eq_resid(X, Y)[:, last_eqs_idx]
#
#             ## store information for backpropagation
#             vm_start_yidx = data.vm_start_yidx
#
#             ctx.data = [partial_vars_idx, newton_vars_idx, newton_mid_vars_idx, last_vars_idx, vm_start_yidx]
#             # jac_last_vmva = -jacs[:, last_eqs_idx, :][:, :, vm_start_yidx:]
#             # # jac_newton_eq_mid_var = jacs[:, newton_eqs_idx, :][:,:,newton_mid_vars_idx]
#             # # jac_newton_eq_recon_var = jacs[:, newton_eqs_idx, :][:,:,newton_vars_idx]\
#             # #                           + torch.matmul(jac_newton_eq_mid_var, jacs_mid_non_mid)
#             # jac_newton_eq_recon_var = jac_pq_load
#             # jac_newton_eq_partial_var = jacs[:, newton_eqs_idx,:][:,:,partial_vars_idx]
#             # print(jacs.shape,
#             #     jac_newton_eq_recon_var.shape, jac_last_vmva.shape,
#             #       jac_newton_eq_partial_var.shape, jacs_mid_non_mid.shape)
#             # print(1/0)
#
#             jac_newton_eq_partial_var = jac_full[:, newton_eqs_idx,:][:,:,partial_vars_idx]
#             jac_newton_eq_partial_var[:,:, data.vm_spv_zidx] += torch.matmul(jac_full[:, newton_eqs_idx,:][:,:,newton_mid_vars_idx],
#                                                                          jac_mid_non_mid[:,:,data.spv])
#             ctx.save_for_backward(-jac_full[:, last_eqs_idx, :][:, :, vm_start_yidx:],
#                                   jac_newton_eq_recon_var,
#                                   jac_newton_eq_partial_var,
#                                   jac_mid_non_mid)
#             return Y
#
#         @staticmethod
#         def backward(ctx, dl_dy):
#             jac_last_vmva, jac_newton_eq_recon_var, jac_newton_eq_partial_var, jac_mid_non_mid = ctx.saved_tensors
#             partial_vars_idx, newton_vars_idx, newton_mid_vars_idx, last_vars_idx, vm_start_yidx = ctx.data
#             dl_dx_total = dl_dz_total = None
#             ### dl/dz
#             if ctx.needs_input_grad[1]:
#                 # Step 1: gradient of all voltages through step-3 outputs
#                 dl_dy[:, vm_start_yidx:] += torch.matmul(dl_dy[:, last_vars_idx].unsqueeze(1),
#                                               jac_last_vmva).squeeze(1)
#                 # Step 1: gradient of all non-mid bus through step-2.2 outputs
#                 dl_dy[:, vm_start_yidx:] +=  torch.matmul((dl_dy[:, newton_mid_vars_idx]).unsqueeze(1),
#                                                jac_mid_non_mid).squeeze(1)
#                 ## Step 1
#                 ## Step 2:implicit gradient for newton solving step-2.1 outputs, (vector-jacobian trick)
#                 d_int = torch.linalg.solve(jac_newton_eq_recon_var.transpose(1, 2),
#                                            dl_dy[:, newton_vars_idx].unsqueeze(-1))
#                 dl_dz_total = dl_dy[:, partial_vars_idx] + \
#                               torch.matmul(d_int.transpose(1, 2),
#                                            -jac_newton_eq_partial_var).squeeze(1)
#
#             if ctx.needs_input_grad[0]:
#                 # gradient of pd at slack and qd at gens through step 3 outputs
#                 dl_dpdqd_3 = dl_dy[:, last_vars_idx]
#                 # insert into correct places in x and y loss vectors
#                 dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_3[:, np.concatenate([data.slack, data.nb + data.spv])] = dl_dpdqd_3
#                 dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=dl_dy.device)
#                 dl_dx_2[:, data.pv] = d_int[:, :data.npv]  # dl_dpd at pv buses
#                 dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
#                 dl_dx_2[:, data.nb + data.pq] = d_int[:, -len(data.pq):]  # dl_dqd at pq buses
#                 # Final quantities
#                 dl_dx_total = dl_dx_3 + dl_dx_2
#             return dl_dx_total, dl_dz_total
#     return PFFunctionFn.apply










# ###################################################################
# # NONCONVEX PROBLEM
# ###################################################################
# class NonconvexProblem(QP_Problem):
#     """
#         minimize_y 1/2 * y^T Q y + p^T sin(y)
#         s.t.       Ay =  x
#                    Gy <= h
#     """
#     def __str__(self):
#         return 'NonconvexProblem-{}-{}-{}-{}'.format(
#             str(self.ydim), str(self.nineq), str(self.neq), str(self.num))
#
#     def obj_fn(self, Y):
#         return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).mean(dim=1)
#
#     def opt_solve(self, X, solver_type='ipopt', tol=1e-6):
#         Q, p, A, G, h, L, U = \
#             self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
#         X_np = X.detach().cpu().numpy()
#         Y = []
#         total_time = 0
#         for Xi in X_np:
#             if solver_type == 'ipopt':
#                 y0 = np.linalg.pinv(A) @ Xi  # feasible initial point
#                 # upper and lower bounds on variables
#                 lb = L
#                 ub = U
#                 # upper and lower bounds on constraints
#                 cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
#                 cu = np.hstack([Xi, h])
#                 nlp = ipopt.problem(
#                     n=len(y0),
#                     m=len(cl),
#                     problem_obj=nonconvex_ipopt(Q, p, A, G),
#                     lb=lb,
#                     ub=ub,
#                     cl=cl,
#                     cu=cu)
#                 nlp.addOption('tol', tol)
#                 nlp.addOption('print_level', 0)  # 3)
#                 start_time = time.time()
#                 y, info = nlp.solve(y0)
#                 end_time = time.time()
#                 Y.append(y)
#                 total_time += (end_time - start_time)
#             else:
#                 raise NotImplementedError
#         sols = np.array(Y)
#         parallel_time = total_time / len(X_np)
#         return sols, parallel_time
#
#     def opt_proj(self, X, Y_pred, solver_type='cvxpy', tol=1e-6):
#
#         if solver_type == 'cvxpy':
#             print('running cvxpy', end='\r')
#             Q, p, A, G, h, L, U = \
#                 self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
#             X_np = X.detach().cpu().numpy()
#             Y_pred = Y_pred.detach().cpu().numpy()
#             Y = []
#             total_time = 0
#             n = 0
#             for Xi, y_pred in zip(X_np, Y_pred):
#                 y = cp.Variable(self.ydim)
#
#                 prob = cp.Problem(cp.Minimize(cp.sum_squares(y - y_pred)),
#                                   [G @ y <= h, y <= U, y >= L,
#                                    A @ y == Xi])
#                 start_time = time.time()
#                 prob.solve()
#                 end_time = time.time()
#                 print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
#                 n += 1
#                 Y.append(y.value)
#                 total_time += (end_time - start_time)
#             sols = np.array(Y)
#             parallel_time = total_time / len(X_np)
#         else:
#             raise NotImplementedError
#         return torch.tensor(sols )
#
#     def opt_warmstart(self, X, Y_pred, solver_type='ipopt', tol=1e-6):
#         if solver_type == 'ipopt':
#             Q, p, A, G, h, L, U = \
#                 self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np, self.L_np, self.U_np
#             X_np = X.detach().cpu().numpy()
#             Y_pred = Y_pred.detach().cpu().numpy()
#             Y = []
#             total_time = 0
#             for Xi, y_pred in zip(X_np, Y_pred):
#                 if solver_type == 'ipopt':
#                     y0 = y_pred
#                     # upper and lower bounds on variables
#                     lb = L
#                     ub = U
#                     # upper and lower bounds on constraints
#                     cl = np.hstack([Xi, -np.inf * np.ones(G.shape[0])])
#                     cu = np.hstack([Xi, h])
#                     nlp = ipopt.problem(
#                         n=len(y0),
#                         m=len(cl),
#                         problem_obj=nonconvex_ipopt(Q, p, A, G),
#                         lb=lb,
#                         ub=ub,
#                         cl=cl,
#                         cu=cu)
#                     nlp.addOption('tol', tol)
#                     nlp.addOption('print_level', 0)  # 3)
#                     start_time = time.time()
#                     y, info = nlp.solve(y0)
#                     end_time = time.time()
#                     Y.append(y)
#                     total_time += (end_time - start_time)
#                 else:
#                     raise NotImplementedError
#         sols = np.array(Y)
#         return sols
#
# class nonconvex_ipopt(object):
#     def __init__(self, Q, p, A, G):
#         self.Q = Q
#         self.p = p
#         self.A = A
#         self.G = G
#         self.tril_indices = np.tril_indices(Q.shape[0])
#
#     def objective(self, y):
#         return 0.5 * (y @ self.Q @ y) + self.p @ np.sin(y)
#
#     def gradient(self, y):
#         return self.Q @ y + (self.p * np.cos(y))
#
#     def constraints(self, y):
#         return np.hstack([self.A @ y, self.G @ y])
#
#     def jacobian(self, y):
#         return np.concatenate([self.A.flatten(), self.G.flatten()])
#
#     # # Don't use: In general, more efficient with numerical approx
#     # def hessian(self, y, lagrange, obj_factor):
#     #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
#     #     return H[self.tril_indices]
#
#     # def intermediate(self, alg_mod, iter_count, obj_value,
#     #         inf_pr, inf_du, mu, d_norm, regularization_size,
#     #         alpha_du, alpha_pr, ls_trials):
#     #     print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


