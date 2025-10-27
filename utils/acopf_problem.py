import torch

from .solver_utils import *
import os
from torch.autograd import Function
import copy
import scipy as sp
import multiprocessing as mp
import random

"""
ACOPF_Problem:
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
class ACOPF_Problem:
    def __init__(self, data,
                kron_reduction: bool = True,
                branch_constraint: bool = True,
                relax_factor: list =[0., 0., 0.],  # Vm, Pg, Sl
                training: bool = True,
                test_size: int = 1024,):
        ppc = data['ppc']
        self.kron_reduction = kron_reduction
        self.branch_constraint = branch_constraint
        self.relax_factor = relax_factor
        # if ppc['bus'].shape[0] >= 300:
        #     self.relax_factor = [0.1, 0.1, 0.1]
        self.MaxChangeLoad = data['MaxChangeLoad'][0][0]
        try:
            self.BaseLoad = data['BaseLoad'][0][0]
        except:
            self.BaseLoad = 1
        self.load_ppc(ppc)
        if self.kron_reduction:
            self.load_reduced_ppc(ppc)

        if training:
            ## Load data
            ## Define train/valid/test split
            X = np.concatenate([data['Pd'] / self.baseMVA,
                                data['Qd'] / self.baseMVA], axis=1)
            Y = np.concatenate([data['Pg'] / self.baseMVA,
                                data['Qg'] / self.baseMVA,
                                data['Vm'], data['Va']], axis=1)
            feas_mask = ~np.isnan(Y).any(axis=1)
            X = torch.tensor(X[feas_mask])
            Y = torch.tensor(Y[feas_mask])
            # self.output_init = Y.mean(0).view(1, -1)
            self.trainX = X[:-test_size]
            self.testX = X[-test_size:]
            self.trainY = Y[:-test_size]
            self.testY = Y[-test_size:]
            # self.num = X.shape[0]
            # Y_full = self.complete_partial(self.testX[:10], self.testY[:10, self.partial_vars_idx])
            # print((Y_full - self.testY[:10]).abs().max())
            # # print(self.ineq_resid(self.testX, self.testY).max())
            # print(1/0)

        print(f'neq:{self.neq}, nineq:{self.nineq}, '
              f'indim:{self.xdim}, outdim:{self.ydim}, '
              f'par_outdim:{len(self.partial_vars_idx)}, '
              f'pq_load:{len(self.pq_load)}, pq_mid:{len(self.pq_mid)} '
              f'datasize:{X.shape[0]}')
        self.device = None  # DEVICE

    def __str__(self):
        return 'ACOPF-{}-{}-{}'.format(self.nb, self.BaseLoad,self.MaxChangeLoad)

    def to_device(self, device):
        self.device = device
        for attr in dir(self):
            var = getattr(self, attr)
            if torch.is_tensor(var):
                try:
                    setattr(self, attr, var.to(device))
                except AttributeError:
                    pass

    def load_ppc(self, ppc):
        # Store the entire ppc
        self.ppc = ppc # Power system case data.
        self.baseMVA = ppc['baseMVA'] # Base power for the system (MVA).

        # System dimensions
        self.ng = ppc['gen'].shape[0] # Number of generators.
        self.nb = ppc['bus'].shape[0] # Number of buses.
        self.nl = ppc['branch'].shape[0] # Number of branches.
        self.xdim = self.nb * 2
        self.ydim = self.nb * 2 + self.ng * 2
        self.neq = 2 * self.nb
        self.nineq = 4 * self.ng + 2 * self.nb + 2 * self.nl

        # Define starting indices for output variables
        self.pg_start_yidx = 0 # Starting index for active power generation in output vector.
        self.qg_start_yidx = self.ng # Starting index for reactive power generation in output vector.
        self.vm_start_yidx = 2 * self.ng # Starting index for voltage magnitudes in output vector.
        self.va_start_yidx = 2 * self.ng + self.nb # Starting index for voltage angles in output vector.

        # Define starting indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nb

        # Define the index of different buses based on type
        # BUS_TYPE: 1 = PQ, 2 = PV, 3 = Slack
        self.pq = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 1)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.spv = np.sort(np.concatenate([self.slack, self.pv])) # indices of slack and PV buses (spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv])) # indices of non-slack buses
        self.nslack = len(self.slack)
        self.npv = len(self.pv)
        # Buses with zero load demand
        self.pm = np.where((np.abs(ppc['bus'][:, idx_bus.PD]) +
                            np.abs(ppc['bus'][:, idx_bus.QD])) == 0)[0]
        # Determine PQ bus with load and w.o. load
        self.pq_load = np.setdiff1d(self.pq, self.pm)
        self.pq_mid = np.setdiff1d(self.pq, self.pq_load)
        self.non_pq_mid = np.setdiff1d(range(self.nb), self.pq_mid) # Bus with net injection (load or generation)
        # Indices within generators for slack and PV
        self.slack_g = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_g = np.array([np.where(x == self.spv)[0][0] for x in self.pv])
        # Branch indices (from and to buses), converted to zero-based indexing
        self.branch_idxes = np.concatenate([[ppc['branch'][:, idx_brch.F_BUS]],
                                            [ppc['branch'][:, idx_brch.T_BUS]]], axis=0).T - 1
        self.quad_costs = torch.tensor(ppc['gencost'][:, 4])
        self.lin_costs = torch.tensor(ppc['gencost'][:, 5])
        self.const_cost = ppc['gencost'][:, 6].sum()

        # Topology information
        ppc_copy = copy.deepcopy(ppc)
        ppc_copy['bus'][:, 0] -= 1  # Adjust bus indices for zero-based indexing
        ppc_copy['branch'][:, [0, 1]] -= 1  # Adjust branch indices
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
        self.slackva = self.va_init[self.slack]
        self.output_init = torch.cat([self.pg_init, self.qg_init, self.vm_init, self.va_init], dim=0).view(1, -1)
        self.input_init = torch.cat([self.pd_init, self.qd_init], dim=0).view(1, -1)

        ## decision upper and lower bound
        self.vmax = torch.tensor(ppc['bus'][:, idx_bus.VMAX])
        self.vmin = torch.tensor(ppc['bus'][:, idx_bus.VMIN])
        self.pmax = torch.tensor(ppc['gen'][:, idx_gen.PMAX] / self.baseMVA)
        self.pmin = torch.tensor(ppc['gen'][:, idx_gen.PMIN] / self.baseMVA)
        self.qmax = torch.tensor(ppc['gen'][:, idx_gen.QMAX] / self.baseMVA)
        self.qmin = torch.tensor(ppc['gen'][:, idx_gen.QMIN] / self.baseMVA)
        self.smax = torch.tensor(ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA)
        self.amax = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMAX]))
        # self.amin = torch.tensor(np.deg2rad(ppc['branch'][:, idx_brch.ANGMIN]))
        self.constraint_relaxation()

        self.input_L = self.input_init * self.BaseLoad * (1 - self.MaxChangeLoad)
        self.input_U = self.input_init * self.BaseLoad * (1 + self.MaxChangeLoad)
        self.L = torch.cat([self.pmin, self.qmin, self.vmin, -torch.ones(self.nb) * torch.pi/2],dim=0).view(1, -1)
        self.U = torch.cat([self.pmax, self.qmax, self.vmax, torch.ones(self.nb) * torch.pi/2],dim=0).view(1, -1)

        # Define variables and indices for "partial completion" neural network
        # Predict pg (non-slack) and |v|_g (including slack)
        total_vars = self.nb * 2 + self.ng * 2
        self.partial_vars_idx = np.concatenate([
            self.pg_start_yidx + self.pv_g,
            self.vm_start_yidx + self.spv])
        # exclude va at slack bus
        self.known_vars = np.concatenate([
            self.partial_vars_idx,
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

    def constraint_relaxation(self):
        self.vmax *= (1 + self.relax_factor[0])
        self.vmin *= (1 - self.relax_factor[0])
        self.pmax *= (1 + self.relax_factor[1])
        self.pmin *= (1 - self.relax_factor[1])
        self.qmax += self.pmax * self.relax_factor[1]
        self.qmin -= self.pmax * self.relax_factor[1]
        self.smax *= (1 + self.relax_factor[2])
        self.amax *= (1 + self.relax_factor[2])

    def load_reduced_ppc(self, ppc):
        # Create a deep copy of ppc to avoid modifying the original data
        ppc_copy = copy.deepcopy(ppc)
        # Adjust bus and branch indices for zero-based indexing
        ppc_copy['bus'][:, 0] -= 1
        ppc_copy['branch'][:, [0, 1]] -= 1
        # Generate the Y-bus matrix and related matrices
        Ybus, Yf, Yt = makeYbus(self.baseMVA, ppc_copy['bus'], ppc_copy['branch'])
        Ybus = Ybus.todense()
        # Partition Ybus into submatrices based on bus types
        Y_ee = Ybus[np.ix_(self.non_pq_mid, self.non_pq_mid)]
        Y_ei = Ybus[np.ix_(self.non_pq_mid, self.pq_mid)]
        Y_ii = Ybus[np.ix_(self.pq_mid, self.pq_mid)]
        Y_ie = Ybus[np.ix_(self.pq_mid, self.non_pq_mid)]
        # Compute the reduced admittance matrix using Schur complement
        Yred = Y_ee - Y_ei @ np.linalg.inv(Y_ii) @ Y_ie
        # Store the real and imaginary parts as torch tensors
        self.Yredr = torch.tensor(np.real(Yred))
        self.Yredi = torch.tensor(np.imag(Yred))
        # Recover Matrix Preparation
        self.pf_mid_index = np.concatenate((self.pq_mid, self.pq_mid + self.nb))
        self.pf_non_mid_index = np.setdiff1d(np.arange(self.nb * 2), self.pf_mid_index)
        self.pf_load_index = np.concatenate((self.pq_load, self.pq_load + self.nb))
        # Subset Ybus_mid for pf_mid_index and compute its inverse
        Ybus_mid_1 = torch.cat([self.Ybusr, -self.Ybusi], dim=1)
        Ybus_mid_2 = torch.cat([self.Ybusi, self.Ybusr], dim=1)
        Ybus_mid = torch.cat([Ybus_mid_1, Ybus_mid_2], dim=0)
        Ybus_mid_sub = Ybus_mid[self.pf_mid_index, :]
        Ybus_mid_sub_inv = torch.inverse(Ybus_mid_sub[:, self.pf_mid_index])
        Ybus_non_mid_sub = Ybus_mid_sub[:, self.pf_non_mid_index]
        self.mid_complete = Ybus_mid_sub_inv @ Ybus_non_mid_sub

        # Initialize values for the solver from reduced ppc data
        self.vm_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.VM])
        self.va_r_init = torch.tensor(np.deg2rad(ppc['bus'][self.non_pq_mid, idx_bus.VA]))
        self.pd_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.PD] / self.baseMVA)
        self.qd_r_init = torch.tensor(ppc['bus'][self.non_pq_mid, idx_bus.QD] / self.baseMVA)
        self.output_r_init = torch.cat([self.pg_init, self.qg_init, self.vm_r_init, self.va_r_init], dim=0).view(1, -1)
        self.input_r_init = torch.cat([self.pd_r_init, self.qd_r_init], dim=0).view(1, -1)

        # Define dimensionality parameters
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

    def solve_mid_bus(self, Y):
        _, _, vm, va = self.get_yvars(Y, self.kron_reduction)
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
        # if self.kron_reduction:
        Yfull = torch.zeros(size=[Y.shape[0], self.ydim]).to(Y.device)
        Yfull[:, :2*self.ng] = Y[:, :2*self.ng]
        Yfull[:, self.vm_start_yidx + self.non_pq_mid] = vm
        Yfull[:, self.va_start_yidx + self.non_pq_mid] = va
        Yfull[:, self.vm_start_yidx + self.pq_mid] = vm_mid
        Yfull[:, self.va_start_yidx + self.pq_mid] = va_mid
        # else:
        #     Y[:, self.vm_start_yidx + self.pq_mid] = vm_mid
        #     Y[:, self.va_start_yidx + self.pq_mid] = va_mid
        #     Yfull = Y
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
        pg, qg, vm, va = self.get_yvars(Y)
        node_resid = torch.cat([pg - self.pmax, self.pmin - pg,
                                qg - self.qmax, self.qmin - qg,
                                vm - self.vmax, self.vmin - vm], dim=1)
        if self.branch_constraint:
            branch_resid = self.branch_ineq_resid(X, Y)
            resids = torch.cat([node_resid, branch_resid], dim=1)
        else:
            resids = node_resid
        # et = time.time()
        # print(et-st)
        return torch.relu(resids)

    def branch_ineq_resid(self, X, Y):
        _, _, vm, va = self.get_yvars(Y)
        f_index = self.branch_idxes[:, 0]
        t_index = self.branch_idxes[:, 1]
        ### Branch angele limit
        resids_brach_angle = (va[:, f_index] -
                              va[:, t_index]).abs() - self.amax
        ### Branch flow limit
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        # Power at the "from" end
        If_real = torch.matmul(vr, self.Yfr.T) - torch.matmul(vi, self.Yfi.T)
        If_imag = torch.matmul(vi, self.Yfr.T) + torch.matmul(vr, self.Yfi.T)
        Sf_real = vr[:, f_index] * If_real + vi[:, f_index] * If_imag
        Sf_imag = vr[:, f_index] * If_imag - vi[:, f_index] * If_real
        # Power at the "to" end
        It_real = torch.matmul(vr, self.Ytr.T) - torch.matmul(vi, self.Yti.T)
        It_imag = torch.matmul(vi, self.Ytr.T) + torch.matmul(vr, self.Yti.T)
        St_real = vr[:, t_index] * It_real + vi[:, t_index] * It_imag
        St_imag = vr[:, t_index] * It_imag - vi[:, t_index] * It_real
        # Power magnitude
        resids_branch_flow = torch.maximum((Sf_real ** 2 + Sf_imag ** 2),
                                           (St_real ** 2 + St_imag ** 2)) - self.smax ** 2
        resids_branch = torch.cat([resids_brach_angle, resids_branch_flow], dim=1)
        return resids_branch

    def eq_jac_v(self, Y, kron_reduction=False):
        """
        Calculate the Jacobian matrix for AC power flow equations with respect to
        voltage magnitudes (Vm) and voltage angles (Va).
        The Jacobian is structured as:
            | dP/dVm  dP/dVa |
            | dQ/dVm  dQ/dVa |
        Args:
            Y (torch.Tensor): Tensor containing voltage magnitudes and angles,
                              shape [batch_size, num_vars].
            kron_reduction (bool): Flag indicating whether Kron reduction is applied.
        Returns:
            torch.Tensor: The Jacobian matrix, shape [batch_size, 2*num_buses, 2*num_buses].
        """
        # ====== Step 1: Extract Voltage Magnitudes and Angles ======
        _, _, V_mag, V_ang = self.get_yvars(Y, kron_reduction)
        # ====== Step 2: Compute Trigonometric Functions ======
        cos_va = torch.cos(V_ang)  # [batch_size, num_buses]
        sin_va = torch.sin(V_ang)  # [batch_size, num_buses]
        # ====== Step 3: Convert to Rectangular Coordinates ======
        V_real = V_mag * cos_va  # [batch_size, num_buses]
        V_imag = V_mag * sin_va  # [batch_size, num_buses]
        # ====== Step 4: Select Admittance Matrices ======
        Y_real = self.Yredr if kron_reduction else self.Ybusr  # [num_buses, num_buses]
        Y_imag = self.Yredi if kron_reduction else self.Ybusi  # [num_buses, num_buses]
        # ====== Step 5: Compute Current Injections ======
        I_real = torch.matmul(V_real, Y_real) - torch.matmul(V_imag, Y_imag)  # [batch_size, num_buses]
        I_imag = torch.matmul(V_real, Y_imag) + torch.matmul(V_imag, Y_real)  # [batch_size, num_buses]
        # ====== Step 6: Compute Combined Ydiagv Terms ======
        # These terms are reused across multiple derivative calculations
        Ydiagv_Yi_cosva_Yr_sinva = compute_Ydiagv(Y_imag, cos_va) + compute_Ydiagv(Y_real, sin_va)
        Ydiagv_Yr_cosva_Yi_sinva = compute_Ydiagv(Y_real, cos_va) - compute_Ydiagv(Y_imag, sin_va)
        Ydiagv_Yi_vi_Yr_vr = compute_Ydiagv(Y_imag, -V_imag) + compute_Ydiagv(Y_real, V_real)
        Ydiagv_Yr_vi_Yi_vr = compute_Ydiagv(Y_real, -V_imag) - compute_Ydiagv(Y_imag, V_real)
        # ====== Step 7: Compute Partial Derivatives ======
        # Derivatives of Real Power (P) w.r.t Vm and Va
        dP_dVm = (-create_diagonal_batch(cos_va, I_real)
                  - compute_dtm(V_real, Ydiagv_Yr_cosva_Yi_sinva)
                  - create_diagonal_batch(sin_va, I_imag)
                  - compute_dtm(V_imag, Ydiagv_Yi_cosva_Yr_sinva))
        dP_dVa = (-create_diagonal_batch(-V_imag, I_real)
                  - compute_dtm(V_real, Ydiagv_Yr_vi_Yi_vr)
                  - create_diagonal_batch(V_real, I_imag)
                  - compute_dtm(V_imag, Ydiagv_Yi_vi_Yr_vr))
        # Derivatives of Reactive Power (Q) w.r.t Vm and Va
        dQ_dVm = (create_diagonal_batch(cos_va, I_imag)
                  + compute_dtm(V_real, Ydiagv_Yi_cosva_Yr_sinva)
                  - create_diagonal_batch(sin_va, I_real)
                  - compute_dtm(V_imag, Ydiagv_Yr_cosva_Yi_sinva))
        dQ_dVa = (create_diagonal_batch(-V_imag, I_imag)
                  + compute_dtm(V_real, Ydiagv_Yi_vi_Yr_vr)
                  - create_diagonal_batch(V_real, I_real)
                  - compute_dtm(V_imag, Ydiagv_Yr_vi_Yi_vr))
        # ====== Step 8: Assemble the Jacobian Matrix ======
        # Concatenate derivatives to form the full Jacobian
        # Shape: [batch_size, 2*num_buses, 2*num_buses]
        jacobian_p = torch.cat([dP_dVm, dP_dVa], dim=2)  # [batch_size, num_buses, 2*num_buses]
        jacobian_q = torch.cat([dQ_dVm, dQ_dVa], dim=2)  # [batch_size, num_buses, 2*num_buses]
        jacobian = torch.cat([jacobian_p, jacobian_q], dim=1)  # [batch_size, 2*num_buses, 2*num_buses]
        return jacobian

    def eq_jac(self, Y, kron_reduction=False):
        # | dP / dPg , dP / dQg , dP / dVm , dP / dVa |
        # | dQ / dPg , dQ / dQg , dQ / dVm , dQ / dVa |
        batch_size = Y.shape[0]
        device = Y.device
        # real power equations
        dP_dPg = torch.zeros(batch_size, self.nb, self.ng, device=device)
        dP_dQg = torch.zeros(batch_size, self.nb, self.ng, device=device)
        dP_dPg[:, self.spv, :] = torch.eye(self.ng, device=device).unsqueeze(0).expand(batch_size, self.ng, self.ng)
        # Initialize dQ/dPg and dQ/dQg
        dQ_dPg = torch.zeros(batch_size, self.nb, self.ng, device=device)
        dQ_dQg = torch.zeros(batch_size, self.nb, self.ng, device=device)
        dQ_dQg[:, self.spv, :] = torch.eye(self.ng, device=device).unsqueeze(0).expand(batch_size, self.ng, self.ng)
        # Stack dP/dPg and dP/dQg
        jacobian_pq_pg_qg = torch.cat([torch.cat([dP_dPg, dQ_dPg],dim=1),
                                      torch.cat([dP_dQg, dQ_dQg],dim=1)], dim=2)  # Shape: [batch_size, nb, 2*ng]
        # Calculate Jacobian with respect to Vm and Va
        jacobian_pq_vm_va = self.eq_jac_v(Y, kron_reduction)  # Shape: [batch_size, 2*nb, 2*nb]
        # Assemble the Full Jacobian
        full_jacobian = torch.cat([jacobian_pq_pg_qg, jacobian_pq_vm_va], dim=2)  # Shape: [batch_size, 2*nb, 2*ng + 2*nb]
        return full_jacobian

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
                ineq_penalty = self.ineq_resid(x, y)
                ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
                grad = torch.autograd.grad(ineq_penalty, y)[0]
                grad_list.append(grad.view(1, -1))
            grad = torch.cat(grad_list, dim=0)
            return grad

    def ineq_partial_grad(self, X, Y):
        eq_jac = self.eq_jac(Y)
        I = torch.eye(len(self.other_vars)).to(X.device).unsqueeze(0)
        dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]+1e-5*I).bmm(eq_jac[:, :, self.partial_vars_idx])
        direct_grad = self.ineq_grad(X, Y)

        indirect_partial_grad = dynz_dz.transpose(1, 2).bmm(
            direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

        full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars_idx]

        full_grad = torch.zeros(X.shape[0], self.ydim, device=X.device)
        full_grad[:, self.partial_vars_idx] = full_partial_grad
        full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)
        return full_grad

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

    def scale(self, X, Y):
        if Y.shape[1] == len(self.partial_vars_idx):
            Y_scale = Y * (self.U[:, self.partial_vars_idx]
                           - self.L[:, self.partial_vars_idx]) \
                        + self.L[:, self.partial_vars_idx]
        elif Y.shape[1] == len(self.newton_vars_idx):
            Y_scale = Y * (self.U[:, self.newton_vars_idx]
                           - self.L[:, self.newton_vars_idx]) \
                        + self.L[:, self.newton_vars_idx]
        else:
            Y_scale = Y * (self.U - self.L) + self.L
        return Y_scale

    def inverse_scale(self, X, Y):
        if Y.shape[1] == len(self.partial_vars_idx):
            Y_inv_scale = ((Y - self.L[:, self.partial_vars_idx]) /
                                     (self.U[:, self.partial_vars_idx] -
                                      self.L[:, self.partial_vars_idx] + 1e-8))
        elif Y.shape[1] == len(self.newton_vars_idx):
            Y_inv_scale = ((Y - self.L[:, self.newton_vars_idx]) /
                           (self.U[:, self.newton_vars_idx]
                            - self.L[:, self.newton_vars_idx] + 1e-8))
        else:
            Y_inv_scale = (Y - self.L) / (self.U - self.L + 1e-8)
        return Y_inv_scale

    def complete_partial(self, X, Z, bsz=1024):
        X = X.detach()
        Yfull = []
        for b in range(0, X.shape[0], bsz):
            Xb = X[b:b + bsz]
            Zb = Z[b:b + bsz]
            if self.kron_reduction:
                Yb_r = PF_pgvm_Function(self)(Xb[:, self.pf_non_mid_index], Zb)
                Yb = self.solve_mid_bus(Yb_r)
            else:
                Yb = PF_pgvm_Function(self)(Xb, Zb)
            Yfull.append(Yb)
        return torch.cat(Yfull, dim=0)

    def neural_complete_partial(self, X, Z, NN_PF):
        Zo = NN_PF(torch.cat([X.detach(),Z.detach()],dim=1))
        Zo_scale = self.scale(X, Zo)
        Z_scale = self.scale(X, Z)
        Yfull = torch.zeros(X.shape[0], self.ydim, device=X.device) + self.output_init
        Yfull[:, self.partial_vars_idx]  = Z_scale
        Yfull[:, self.newton_vars_idx] = Zo_scale
        return Yfull








    def penalty_zero_order_est(self, X, Z):
        return penalty_function(self)(X, Z)

    def cal_penalty(self, X, Y):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y):
        return self.cal_penalty(X, Y)

    def opt_solve(self, X, tol=1e-5):
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

    def opt_ip(self, X,  tol=1e-5):
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

    def opt_proj(self, X, Y,  tol=1e-5):
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

    def opt_warmstart(self, X, Y,  tol=1e-5):
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



def PF_pgvm_Function(data, tol=1e-5, max_iters=4):
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
            ## Step 1: known variables: pg at non-slack gens, vm at gens
            Y[:, partial_vars_idx] = Z
            I = torch.eye(len(newton_eqs_idx)).unsqueeze(0).to(X.device)
            ## Step 2: newton variables
            for n in range(max_iters):
                gy = data.eq_resid(X, Y, reduced)[:, newton_eqs_idx]
                jac_vmva = data.eq_jac_v(Y, reduced)
                jac_newton_eq_recon_var = jac_vmva[:, newton_eqs_idx, :][:, :, newton_vars_idx - 2 * data.ng]
                if torch.abs(gy).max() < tol:
                    break
                ### Newton update
                delta = torch.linalg.solve(jac_newton_eq_recon_var + 1e-5 * I, gy.unsqueeze(-1)).squeeze(-1)
                Y[:, newton_vars_idx] -=  delta
            if torch.abs(gy).max() > tol:
                print(f'PF non-converge in {n} with max {torch.abs(gy).max():.4f}', end='\r')
                # and mean # {torch.abs(gy).mean(): .4f}
            else:
                print(f'PF converge in {n} iter with error {torch.abs(gy).max():.4f}', end='\r')
            ## Step 3: last variables
            Y[:, last_vars_idx] -= data.eq_resid(X, Y, reduced)[:, last_eqs_idx]

            ## store information for backpropagation
            vm_start_yidx = data.vm_start_yidx
            partial_pg_yidx = partial_vars_idx[data.pg_pv_zidx]
            partial_vm_yidx = partial_vars_idx[data.vm_spv_zidx]
            # print('Newton methods error', n, torch.abs(data.eq_resid(X, Y)).max(), end='\r')

            ctx.save_for_backward(-jac_vmva[:, last_eqs_idx, :],  # jac_last_vars_vmva
                                  jac_newton_eq_recon_var + 1e-5 * I,  # jac_newton_eq_recon_var
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
                d_int = torch.linalg.solve(jac_newton_eq_recon_var.transpose(1, 2),
                                           dl_dy[:, newton_vars_idx].unsqueeze(-1))
                d_int = -d_int.transpose(1, 2)
                dl_dy[:, partial_pg_yidx] += (d_int[:, :, partial_pg_yidx]).squeeze(1)
                dl_dy[:, partial_vm_yidx] += torch.matmul(d_int, jac_newton_eq_partial_vm_var).squeeze(1)
                dl_dz_total = dl_dy[:, partial_vars_idx]

            return dl_dx_total, dl_dz_total

    return PFFunctionFn.apply

def penalty_function(data):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            with torch.no_grad():
               Y = data.complete_partial(X, Z)
            penalty = data.ineq_resid(X, Y).sum(-1).view(-1,1)
            ctx.data = [data, X, Z, Y, penalty]
            return penalty

        @staticmethod
        def backward(ctx, dl_dy):
            data, X, Z, Y, P = ctx.data
            dl_dx_total = dl_dz_total = None
            ### dl/dz
            if ctx.needs_input_grad[1]:
                epsilon = 1e-4
                v = torch.randn_like(Z).to(Z.device)
                # v = v/torch.norm(v, p=2, dim=-1, keepdim=True)
                with torch.no_grad():
                    batch_size = X.shape[0]
                    Xe = torch.cat([X,X], dim=0)
                    Ze = torch.cat([Z + epsilon * v,
                                    Z - epsilon * v], dim=0)
                    Ye = data.complete_partial(Xe, Ze)
                    Pe = data.ineq_resid(Xe, Ye).sum(-1).view(-1,1)
                dl_dz_total = Z.shape[1] * v / (2*epsilon) * \
                              (Pe[:batch_size] - Pe[batch_size:])
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


# ====== Helper Functions ======
def create_diagonal_batch(v1, v2):
    """
    Creates a batch of diagonal matrices with v1 * v2 on the diagonals.
    Args:
        v1 (torch.Tensor): Tensor of shape [batch_size, num_buses].
        v2 (torch.Tensor): Tensor of shape [batch_size, num_buses].
    Returns:
        torch.Tensor: Tensor of shape [batch_size, num_buses, num_buses].
    """
    return torch.diag_embed(v1 * v2)

def compute_Ydiagv(Y_matrix, v_vector):
    """
    Computes Y * v for each bus in a batch, resulting in a batch of matrices.
    Args:
        Y_matrix (torch.Tensor): Tensor of shape [num_buses, num_buses].
        v_vector (torch.Tensor): Tensor of shape [batch_size, num_buses].
    Returns:
        torch.Tensor: Tensor of shape [batch_size, num_buses, num_buses].
    """
    return Y_matrix.unsqueeze(0) * v_vector.unsqueeze(1)

def compute_dtm(v_vector, M_matrix):
    """
    Element-wise multiplies v_vector with M_matrix across the third dimension.
    Args:
        v_vector (torch.Tensor): Tensor of shape [batch_size, num_buses].
        M_matrix (torch.Tensor): Tensor of shape [batch_size, num_buses, num_buses].
    Returns:
        torch.Tensor: Tensor of shape [batch_size, num_buses, num_buses].
    """
    return v_vector.unsqueeze(2) * M_matrix




from contextlib import contextmanager

@contextmanager
def temp_seed(seed):
    # 保存当前状态
    cpu_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    np_state = np.random.get_state()
    py_state = random.getstate()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available():
            for i, state in enumerate(cuda_state):
                torch.cuda.set_rng_state(state)
        np.random.set_state(np_state)
        random.setstate(py_state)


def generate_positive_definite_cov(n, min_corr=0, max_corr=1):
    # n = len(stds)
    # Generate random matrix A
    A = np.random.randn(n, n)
    # Create positive definite matrix by A*A^T
    R = A @ A.T
    # Convert to correlation matrix by normalizing
    d = np.sqrt(np.diag(R))
    R = R / d[:, None] / d[None, :]
    # Scale correlations to desired range
    R = R * (max_corr - min_corr) + min_corr
    np.fill_diagonal(R, 1)
    # Convert correlation to covariance
    std_matrix = np.diag(np.ones(n))
    cov = std_matrix @ R @ std_matrix
    return cov

"""
CC_ACOPF_Problem
    minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
    s.t.   (p_g - p_d(\omega)) + (q_g - q_d(\omega))i = diag(V * conj(Ybus * V))
             Prob{        p_g min   <= p_g  <= p_g max
                          q_g min   <= q_g  <= q_g max
                          vmag min  <= vmag <= vmag max
                          vang_slack = 0   # voltage ang
                          vmag_slack = 1   # voltage mag
                          va_ij min <= va_ij <= va_ij max
                          | diag(V * conj(Yf * V)) | <= s_max
                          | diag(V * conj(Yt * V)) | <= s_max   } \ge 1-\delta
"""
class CC_ACOPF_Problem(ACOPF_Problem):
    def __init__(self, data,
                kron_reduction: bool = True,
                branch_constraint: bool = True,
                relax_factor: list =[0.0, 0.0, 0.0],  # Vm, Pg, Sl
                training: bool = True,
                test_size: int = 1024,
                test_scenario: int = 1000,
                eval_scenario: int = 10,
                uncertainty: str = 'ind_gaussian'):
        super().__init__(data, kron_reduction, branch_constraint, relax_factor, training, test_size)
        self.uncertainty = uncertainty
        self.num_test_scenario = test_scenario
        self.num_eval_scenario = eval_scenario
        self.joint_gaussian = 0.5
        self.skewed_gaussian = 0.25
        self.laplace_scale = 0.5
        if self.nb<300:
            self.sigma = 0.1
            self.uncertainty_bus_idx = np.array([i for i in range(self.nb)
                                                 if (self.pd_init[i] > 0) and
                                                    (self.pd_init[i] < 1)])
        else:
            self.sigma = 0.05
            self.uncertainty_bus_idx = np.array([i for i in range(self.nb)
                                                 if (self.pd_init[i] > 0) and
                                                    (self.pd_init[i] < 0.25)])
        self.nb_uncertain = len(self.uncertainty_bus_idx)
        print(f'{uncertainty}, num of uncertain bus: {self.nb_uncertain}, uncertain bus ratio {self.nb_uncertain/self.nb * 100:.2f}%')
        self.uncertainty_load_idx = np.concatenate([self.uncertainty_bus_idx,
                                                    self.uncertainty_bus_idx + self.nb])
        self.deter_load_idx = np.setdiff1d(np.arange(self.nb*2), self.uncertainty_load_idx)
        mat_file = {'mpc': self.ppc,
                    'BaseLoad': self.BaseLoad,
                    'MaxChangeLoad': self.MaxChangeLoad,
                    'UncertainBusIndex': self.uncertainty_bus_idx,
                    'TestLoadMean': self.testX * self.baseMVA,
                    'TestLoadStdMax': self.sigma,
                    'TestScenario': self.num_test_scenario,}
        sp.io.savemat(f"./datasets/ccacopf/casefile/{self.nb}_instance_{test_size}_scenario_{self.num_test_scenario}.mat", mat_file)
        # noise_file = f'./datasets/ccacopf/noise/{self.uncertainty}/'
        # if not os.path.exists(noise_file):
        #     os.makedirs(noise_file)
        # test_noise_path = noise_file + f'{self.nb}_noise_{test_size}_{self.num_test_scenario}_{self.uncertainty}_{self.sigma}.npy'
        # if not os.path.exists(test_noise_path):
        #     with temp_seed(2025):
        #         noise = self.sampling_scenario(test_size * self.num_test_scenario, self.nb* 2)
        #     noise = noise.view(test_size, self.num_test_scenario, self.nb * 2)
        #     self.test_noise = noise
        #     np.save(test_noise_path, noise.numpy())
        #     print("sampling testing scenarios")
        # else:
        #     noise = np.load(test_noise_path, allow_pickle=True)
        #     self.test_noise = torch.as_tensor(noise)
        #     print("load testing scenarios")

    def __str__(self):
        return 'CC-ACOPF-{}-{}-{}-{}-{}'.format(self.nb, self.BaseLoad,self.MaxChangeLoad,self.uncertainty,self.sigma)

    def sampling_scenario(self, N: int, m: int):
        """
        Generate uncertainty scenarios based on the specified distribution.
        Args:
            N (int): Number of samples.
            m (int): Number of variables.
            mean (float, optional): Mean of the distribution. Defaults to 1.0.
            sigma (float, optional): Standard deviation. Defaults to 0.1.
            cor (float, optional): Correlation coefficient for joint Gaussian. Defaults to 0.5.
            skewness (float, optional): Skewness parameter for skewed Gaussian. Defaults to 0.5.
        Returns:
            Tensor: Generated scenarios of shape (N, m).
        """
        mean = 1
        std = torch.rand(N, m) * self.sigma
        z = torch.randn(N, m)
        if self.uncertainty == 'ind_gaussian':
            samples = z * std + mean
        elif self.uncertainty == 'joi_gaussian':
            """ generate_correlation_matrix """
            cov = generate_positive_definite_cov(m, min_corr=0, max_corr=self.joint_gaussian)
            L = torch.as_tensor(np.linalg.cholesky(cov))
            samples = z @ L * std + mean
        elif self.uncertainty == 'ind_skew_gaussian':
            samples = z + self.skewed_gaussian * torch.nn.functional.softplus(z)
            samples = samples * std + mean
        elif self.uncertainty == 'ind_laplace':
            laplace_dist = torch.distributions.Laplace(loc=0, scale=self.laplace_scale)
            samples = laplace_dist.sample((N, m))
            samples = samples * std + mean
        else:
            raise ValueError(f"Unsupported uncertainty type: {self.uncertainty}")
        # samples = torch.clip(samples, min=mean - 3*self.sigma, max=mean + 3*self.sigma)
        samples[:, self.deter_load_idx] = 1
        return samples

    def expect_obj_fn(self, X, Z, eval_mode='train'):
        batch_size, load_dim = X.shape
        _, Ys = self.complete_partial_scenario(X, Z, eval_mode)
        cost = self.obj_fn(Ys).view(batch_size, -1).mean(1)
        return cost

    def chance_ineq_resid(self, X, Z, eval_mode='train'):
        batch_size, load_dim = X.shape
        Xs, Ys = self.complete_partial_scenario(X, Z, eval_mode)
        # Compute inequality residuals
        residuals = self.ineq_resid(Xs, Ys)  # Shape depends on ineq_resid implementation
        # Calculate penalties as sum of absolute residuals
        penalty = residuals.abs().sum(dim=-1)  # (batch_size * num_test_scenario)
        penalty = penalty.view(batch_size,  -1) # (batch_size, num_test_scenario)
        return penalty

    def complete_partial_scenario(self, X, Z, eval_mode='train'):
        ### sampling load uncertainty: (1 +/- p%) * base_load
        batch_size, load_dim = X.shape
        _, gen_dim = Z.shape
        if eval_mode == 'test':
            num_test_scenario = self.num_test_scenario
        elif eval_mode=='train':
            num_test_scenario = 1
        elif eval_mode == 'eval':
            num_test_scenario = self.num_eval_scenario
        # Expand X and Z for all scenarios
        X_expanded = X.unsqueeze(1).repeat(1, num_test_scenario, 1)  # (batch_size, num_test_scenario, load_dim)
        Z_expanded = Z.unsqueeze(1).repeat(1, num_test_scenario, 1)  # (batch_size, num_test_scenario, gen_dim)
        # Sample uncertainty scenarios
        if eval_mode == 'test':
            if X.shape[0] == 1024:
                rate = self.test_noise
            else:
                rate = self.test_noise[[self.eval_index]]
        else:
            rate = self.sampling_scenario(batch_size * num_test_scenario, self.nb * 2).to(X.device)
            rate = rate.view(batch_size, num_test_scenario, self.nb * 2)
        # Apply uncertainty to X: (batch_size, num_test_scenario, load_dim)
        X_expanded = X_expanded * rate
        # Reshape for processing
        X_flat = X_expanded.view(batch_size * num_test_scenario, load_dim)
        Z_flat = Z_expanded.view(batch_size * num_test_scenario, gen_dim)
        # Complete system variables
        Ys = self.complete_partial(X_flat, Z_flat, bsz=1000)  # Assumes batch processing
        return X_flat, Ys

    def cal_penalty(self, X, Z,):
        penalty = self.chance_ineq_resid(X, Z)
        average_penalty = penalty.mean(dim=1)
        return average_penalty

    def check_instance_feasibility(self, X, Z, risk_level=0.1, eval_mode='train', eps=1e-5):
        # feasibile w.r.t. chance constraint for each instance
        penalty = self.chance_ineq_resid(X, Z, eval_mode)
        fea_mask = penalty <= eps
        feasibility_rate = fea_mask.float().mean(dim=1, keepdim=True)
        fea_mask = torch.clamp(1 - feasibility_rate - risk_level , min=0)
        return fea_mask

    def check_scenario_feasibility(self, X, Z, eval_mode='train', eps=1e-5):
        # feasibile rate averaged over sampled scenarios for each instance
        penalty = self.chance_ineq_resid(X, Z, eval_mode)
        fea_mask = penalty <= eps
        feasibility_rate = fea_mask.float().mean(dim=1, keepdim=True)
        return feasibility_rate














"""
Graph formulation of AC-OPF problem:
    Represent ACOPF as a direct Graph:
        Given loads for all nodes: X = [pd, qd]
        Predict decisions for all nodes: Y = [pg, qg, vm, va]
        Pv node: predict pg, vm | solve qg, va
        Pq node:                | solve pg, qg, vm, va
        Slack node: predict vm  | solve pg, qg, known va
    kron reduction to reduce graph size
"""
class Grpah_ACOPF_Problem(ACOPF_Problem):
    def __init__(self, data,
                kron_reduction: bool = True,
                branch_constraint: bool = False,
                relax_factor: list =[0.0, 0.0, 0.],  # Vm, Pg, Sl
                training: bool = True,
                test_size: int = 1024,):
        super().__init__(data, kron_reduction, branch_constraint, relax_factor, training, test_size)

        self.ndim = 2 # node dim [pd, qd]
        self.edim = 2 # edge dim [G, B]
        self.ydim = 4 # deicsion dimension [pg, qg, vm, va]
        self.num_node = self.nb_r if self.kron_reduction else self.nb
        self.trainE = self.testE = self.admittance_from_net()
        self.trainAdj = self.testAdj = None

    def admittance_from_net(self):
        if self.kron_reduction:
            adm = torch.stack([self.Ybusr, self.Ybusi], dim=-1)
        else:
            adm = torch.stack([self.Yredr, self.Yredi], dim=-1)
        return adm.view(1, self.num_node, self.num_node, 2)

    def graph_to_compact_prediction(self, X, Z):
        Xe = torch.zeros(size=[X.shape[0], self.nb * 2]).to(X.device)
        Ze = torch.zeros(size=[X.shape[0], len(self.partial_r_vars_idx)]).to(X.device)
        if self.kron_reduction:
            Xe[:, self.non_pq_mid] = X[:, :, 0]
            Xe[:, self.nb + self.non_pq_mid] = X[:, :, 1]
            Ze[:, self.pg_pv_zidx] = Z[:, self.pv_r, 0]
            Ze[:, self.vm_spv_zidx] = Z[:, self.spv_r, 2]
        else:
            Xe[:, :self.nb] = X[:, :, 0]
            Xe[:, self.nb:] = X[:, :, 1]
            Ze[:, self.pg_pv_zidx] = Z[:, self.pv, 0]
            Ze[:, self.vm_spv_zidx] = Z[:, self.spv, 2]
        return Xe, Ze

    def compact_to_graph_prediction(self, X, Z):
        if self.kron_reduction:
            Xe = torch.zeros(size=[X.shape[0], self.nb_r, 2]).to(X.device)
            Ze = torch.zeros(size=[X.shape[0], self.nb_r, 4]).to(X.device)
            Xe[:, :, 0] = X[:, self.non_pq_mid]
            Xe[:, :, 1] = X[:, self.nb + self.non_pq_mid]
            Ze[:, self.pv_r, 0] = Z[:, self.pg_pv_zidx]
            Ze[:, self.spv_r, 2] = Z[:, self.vm_spv_zidx]
        else:
            Xe = torch.zeros(size=[X.shape[0], self.nb, 2]).to(X.device)
            Ze = torch.zeros(size=[X.shape[0], self.nb, 4]).to(X.device)
            Xe[:, :, 0] = X[:, :self.nb]
            Xe[:, :, 1] = X[:, self.nb:]
            Ze[:, self.pv, 0] = Z[:, self.pg_pv_zidx]
            Ze[:, self.spv, 2] = Z[:, self.vm_spv_zidx]
        return Xe, Ze

    def edge_penalty(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        resids = torch.cat([pg - self.pmax, self.pmin - pg,
                                qg - self.qmax, self.qmin - qg,
                                vm - self.vmax, self.vmin - vm], dim=1)
        return torch.clamp(resids, 0)

    def edge_penalty(self, X, Y, Adj=None):
        resids = self.branch_ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def cal_penalty(self, X, Y, Adj=None):
        penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
        return torch.abs(penalty)

    def check_feasibility(self, X, Y, Adj=None):
        return self.cal_penalty(X, Y)

