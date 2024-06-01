import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .activation import Swish

import torch.nn.init as init

###################################################################
# Lipschitz layers
###################################################################
class GraphLipNet(nn.Module):
    def __init__(self,
                 adj,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 lip = 0.99):
        super(GraphLipNet, self).__init__()
        num_node = adj.shape[1]
        self.num_node = num_node
        self.adj = (adj + torch.eye(self.num_node, device=adj.device))
        # self.w_adj = torch.nn.Parameter(torch.rand(self.adj.shape)).to(self.adj.device)
        self.deg = (self.adj.sum(1)).view(1,self.num_node,1)
        self.lip = lip
        if num_cond_inputs is not None:
            self.w = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden), nn.Tanh())
            self.b = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden), nn.ReLU())
        self.emb = nn.Sequential(LinearNormalized(num_inputs, num_hidden), Swish())
        self.cat = nn.Sequential(LinearNormalized(num_hidden, num_hidden), Swish(),
                                 LinearNormalized(num_hidden, num_inputs))

    def forward(self, inputs, cond_inputs=None):

        ndim = inputs.shape[-1]
        emb = inputs.view(-1, self.num_node, ndim)
        emb = torch.matmul(emb.permute(0, 2, 1), self.adj)
        emb = emb.permute(0, 2, 1).contiguous() / self.deg
        emb = emb.view(-1, ndim)

        emb = self.emb(emb)
        if cond_inputs is not None:
            w = self.w(cond_inputs)
            b = self.b(cond_inputs)
            emb = w * emb + b

        gx = self.cat(emb)
        return gx * self.lip



class LipNet(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 lip = 0.99):
        super(LipNet, self).__init__()
        self.lip = lip
        if num_cond_inputs is not None:
            # self.w = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden), nn.Tanh())
            self.b = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden))
        self.emb = nn.Sequential(SpectralNormLinear(num_inputs, num_hidden))
        # self.emb = nn.Sequential(PartialLinearNormalized(num_inputs, num_hidden, num_cond_inputs), Swish())
        self.cat = nn.Sequential(SpectralNormLinear(num_hidden, num_hidden), nn.ReLU(),
                                 SpectralNormLinear(num_hidden, num_inputs))

    def forward(self, inputs, cond_inputs=None):
        emb = self.emb(inputs)
        if cond_inputs is not None:
        #     w = self.w(cond_inputs)
            emb =  emb + self.b(cond_inputs)
        # emb = self.emb(torch.cat([inputs, cond_inputs], dim=1))
        gx = self.cat(emb)
        return gx * self.lip

class SpectralNormLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, coeff=0.99, n_iterations=1, atol=None, rtol=None, **unused_kwargs
    ):
        del unused_kwargs
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', F.normalize(self.weight.new_empty(h).normal_(0, 1), dim=0))
        self.register_buffer('v', F.normalize(self.weight.new_empty(w).normal_(0, 1), dim=0))
        self.compute_weight(True, 1)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        if update:
            with torch.no_grad():
                itrs_used = 0.
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()
            sigma = torch.dot(u, torch.mv(weight, v))
            with torch.no_grad():
                self.scale.copy_(sigma)
            # soft normalization: only when sigma larger than coeff
            self.factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / self.factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=self.training)
        # lip = torch.linalg.norm(weight, ord=2)
        # print(lip)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, coeff={}, n_iters={}, atol={}, rtol={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.coeff, self.n_iterations, self.atol,
            self.rtol
        )

class LinearNormalized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.linear = spectral_norm(nn.Linear(in_features, out_features))
        # nn.init.orthogonal_(self.linear.weight)

    def forward(self, x):
        # lip = torch.linalg.norm(self.linear.weight, ord=2)
        # print(lip)
        return self.linear(x)

class PartialLinearNormalized(nn.Module):
    def __init__(self, input_dim, output_dim, con_dim):
        super(PartialLinearNormalized, self).__init__()
        self.con_dim = con_dim
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim + con_dim, output_dim)
        self.linear_1 = LinearNormalized(input_dim, output_dim)

    def forward(self, x):
        with torch.no_grad():
            weight_copy = self.linear_1.weight.data.clone()
            self.linear.weight.data[:, :self.input_dim] = weight_copy
        return self.linear(x)


"""
Convex Potential Layer:
https://arxiv.org/pdf/2110.12690.pdf
"""
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.empty(1).fill_(
            self.weight.norm().item()), requires_grad=True)

        self.Q_cached = None

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

        # self.Q = None
        self.Q_cached = None

    def forward(self, X):
        if self.training:
            self.Q_cached = None
            # self.Q = cayley(self.alpha * self.weight / self.weight.norm())
            Q = cayley(self.alpha * self.weight / self.weight.norm())
        else:
            if self.Q_cached is None:
                with torch.no_grad():
                    self.Q_cached = cayley(
                        self.alpha * self.weight / self.weight.norm())
            Q = self.Q_cached
            # with torch.no_grad():
            #     self.Q = cayley(self.alpha * self.weight / self.weight.norm())

        # return F.linear(X, self.Q, self.bias)
        return F.linear(X, Q, self.bias)

class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(
            1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x

class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(
            1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(
            out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B*h
        if self.psi is not None:
            # sqrt(2) \Psi^{-1} B * h
            x = x * torch.exp(-self.psi) * (2 ** 0.5)
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x

"""
Convex Potential Layer:
https://arxiv.org/pdf/2110.12690.pdf
"""
class SpectralNormPowerMethod(nn.Module):
    def __init__(self, input_size, eps=1e-8):
        super(SpectralNormPowerMethod, self).__init__()
        self.input_size = input_size
        self.eps = eps
        self.u = torch.randn(input_size)
        self.u = self.u / self.u.norm(p=2)
        self.u = nn.Parameter(self.u, requires_grad=False)

    def normalize(self, arr):
        norm = torch.sqrt((arr ** 2).sum())
        return arr / (norm + 1e-12)

    def _compute_dense(self, M, max_iter):
        """Compute the largest singular value with a small number of
        iteration for training"""
        for _ in range(max_iter):
            v = self.normalize(F.linear(self.u, M))
            self.u.data = self.normalize(F.linear(v, M.T))
        z = F.linear(self.u, M)
        sigma = torch.mul(z, v).sum()
        return sigma

    def _compute_conv(self, kernel, max_iter):
        """Compute the largest singular value with a small number of
        iteration for training"""
        pad = (1, 1, 1, 1)
        pad_ = (-1, -1, -1, -1)
        for i in range(max_iter):
            v = self.normalize(F.conv2d(F.pad(self.u, pad), kernel))
            self.u.data = self.normalize(F.pad(F.conv_transpose2d(v, kernel), pad_))
        u_hat, v_hat = self.u, v

        z = F.conv2d(F.pad(u_hat, pad), kernel)
        sigma = torch.mul(z, v_hat).sum()
        return sigma

    def forward(self, M, max_iter):
        """ Return the highest singular value of a matrix
        """
        if len(M.shape) == 4:
            return self._compute_conv(M, max_iter)
        elif len(M.shape) == 2:
            return self._compute_dense(M, max_iter)

class ConvexPotentialLayerLinear(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-4):
        super(ConvexPotentialLayerLinear, self).__init__()
        self.activation = nn.ReLU(inplace=False)
        self.register_buffer('eval_sv_max', torch.Tensor([0]))

        self.weights = torch.zeros(cout, cin)
        self.bias = torch.zeros(cout)

        self.weights = nn.Parameter(self.weights)
        self.bias = nn.Parameter(self.bias)

        self.pm = SpectralNormPowerMethod((1, cin))
        self.train_max_iter = 1
        self.eval_max_iter = 100

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon
        self.alpha = torch.zeros(1)
        self.alpha = nn.Parameter(self.alpha)

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        res = F.linear(res, self.weights.t())
        if self.training == True:
            self.eval_sv_max -= self.eval_sv_max
            sv_max = self.pm(self.weights, self.train_max_iter)
            h = 2 / (sv_max ** 2 + self.epsilon)
        else:
            if self.eval_sv_max == 0:
                self.eval_sv_max += self.pm(self.weights, self.eval_max_iter)
            h = 2 / (self.eval_sv_max ** 2 + self.epsilon)

        out = x - h * res
        return out

"""
SDP-based Lipschitz Layers,
introduced in paper https://openreview.net/pdf?id=k71IGLC8cfc.
Code (adapted) from
https://github.com/araujoalexandre/Lipschitz-SLL-Networks/commit/faaa02e34ce4a81cfece26c411fcdf1f711b0579

Note that an updated version can be found in https://github.com/araujoalexandre/Lipschitz-SLL-Networks/blob/main/core/models/layers.py
where numerical issues have been resolved.

"""
import numpy as np
class SLLLinear(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6):
        super().__init__()

        self.activation = nn.ReLU(inplace=False)
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1 / (q_abs + self.epsilon))[:, None]
        T = 2 / (torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1)
                 + self.epsilon)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out