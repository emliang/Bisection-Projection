import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lipschitz import LipNet
from .coupling import LUInvertibleMM
import logging

logger = logging.getLogger()

__all__ = ['iResidualLayer', 'iGraphResidualLayer']
torch.set_default_dtype(torch.float64)

###################################################################
# Invertible Residual layers
# paper: https://arxiv.org/abs/1906.02735
# code: https://github.com/rtqichen/residual-flows/tree/master
###################################################################
class iResidualLayer(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                geom_p=0.5,
                lamb=2.,
                n_power_series=None,
                exact_trace=False,
                brute_force=False,
                n_samples=1,
                n_exact_terms=2,
                n_dist='geometric',
                neumann_grad=True,
                grad_in_forward=True,):
        super(iResidualLayer, self).__init__()
        self.lipnet = LipNet(num_inputs, num_hidden, num_cond_inputs)
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                gx, logdet = self.logdet_estimator(inputs, cond_inputs)
                logdet = logdet.view(inputs.shape[0],-1)
            else:
                gx = self.lipnet(inputs, cond_inputs)
                logdet = None
            return inputs + gx, logdet
        else:
            return self.inverse_fixed_point(inputs, cond_inputs), None

    def inverse_fixed_point(self, y, cond_inputs=None, atol=1e-5, rtol=1e-5):
        with torch.no_grad():
            x, x_prev = y - self.lipnet(y, cond_inputs), y
            i = 0
            tol = atol + y.abs() * rtol
            while not torch.all((x - x_prev)**2 / tol < 1):
                x, x_prev = y - self.lipnet(x, cond_inputs), x
                i += 1
                if i > 1000:
                    logger.info('Iterations exceeded 1000 for inverse.')
                    break
        return x

    def logdet_estimator(self, x, c):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""
        with torch.enable_grad():
            # x = x.requires_grad_(True)
            # g = self.lipnet(x, c)
            # jac = batch_jacobian(g, x)
            # distortion = jac.norm(2, dim=-1)
            # print(distortion.shape)
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * \
                    sum(n_samples >= k - 20) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(x)
                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator
                # Do backprop-in-forward to save memory.
                if self.training and self.grad_in_forward:
                    g, logdetgrad = mem_eff_wrapper(
                        estimator_fn, self.lipnet, x, c, n_power_series, vareps, coeff_fn, self.training)
                else:
                    x = x.requires_grad_(True)
                    g = self.lipnet(x, c)
                    logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, self.training)
            else:
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = self.lipnet(x,c)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + (-1) ** (k + 1) / k * coeff_fn(k) * batch_trace(jac_k)
            return g, logdetgrad.view(-1, 1)


###################################################################
# Invertible Graph Residual layers
# paper: https://arxiv.org/abs/1909.13521
###################################################################
class iGraphResidualLayer(iResidualLayer):
    def __init__(self, num_node, num_inputs, num_hidden, num_cond_inputs, num_edge_input):
        super(iGraphResidualLayer, self).__init__(num_inputs, num_hidden, num_cond_inputs)
        self.lipnet = LipNet(num_inputs, num_hidden, num_cond_inputs)
        self.w_1 = torch.nn.Parameter(torch.rand(num_node, int(np.log(num_node))+1)) #
        self.w_2 = torch.nn.Parameter(torch.rand(int(np.log(num_node))+1, num_node)) #
        # self.w_adj = torch.nn.Parameter(torch.rand(num_node, num_node))
        self.num_node = num_node
        self.edge_emb = nn.Sequential(nn.Linear(num_edge_input, 8), nn.ReLU(), nn.Linear(8,1))
        # self.adj_embedding = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8,1))
        # w_1 = self.w_1
        # w_2 = self.w_2
        # adj = torch.matmul(w_1, w_2).unsqueeze(0)
        # adj = self.adj_embedding(adj).squeeze(-1)
        # adj = torch.softmax(adj, dim=2)

    def mixture(self, x, c=None, e=None, adj=None):
        # num_node = self.num_node
        # diag_indices = torch.arange(num_node)
        ndim = x.shape[-1]
        x = x.view(-1, self.num_node, ndim)
        if e is not None:
            # w_1 = self.w_1
            # w_2 = self.w_2
            # adj = torch.matmul(w_1, w_2).unsqueeze(0)
            # adj = torch.softmax(adj, dim=2)
            edge_emb = self.edge_emb(e).squeeze(-1)
            adj = torch.softmax(edge_emb, dim=2)
        elif adj is not None:
            adj = (adj + torch.eye( self.num_node, device=x.device).unsqueeze(0))
            adj = adj / (adj.sum(2)).view(-1,  self.num_node,1)

        emb = torch.matmul(adj, x)
        return emb.view(-1, ndim)

    def forward(self, inputs, c=None, e=None, adj=None, mode='direct'):
        """
        Graph input: batch * [node * feature]
        """
        if mode == 'direct':
            emb = self.mixture(inputs, c, e, adj)
            if self.training:
                gx, logdet = self.logdet_estimator(emb, c)
                logdet = logdet.view(inputs.shape[0],-1)
            else:
                gx = self.lipnet(emb, c)
                logdet = None
            return inputs + gx, logdet
        else:
            return self.inverse_fixed_point(inputs, c, adj), None


    def inverse_fixed_point(self, y, c=None, adj=None, atol=1e-5, rtol=1e-5):
        num_node = adj.shape[0]
        adj = (adj + torch.eye(num_node, device=y.device).unsqueeze(0))
        deg = (adj.sum(2)).view(-1, num_node,1)
        with torch.no_grad():
            ndim = y.shape[-1]
            emb = y.view(-1, num_node, ndim)
            emb = torch.matmul(emb.permute(0, 2, 1), adj)
            emb = emb.permute(0, 2, 1).contiguous() / deg
            emb = emb.view(-1, ndim)
            gx = self.lipnet(emb, c)
            x, x_prev = y - gx, y
            i = 0
            tol = atol + y.abs() * rtol
            while not torch.all((x - x_prev)**2 / tol < 1):
                emb = x.view(-1, num_node, ndim)
                emb = torch.matmul(emb.permute(0, 2, 1), adj)
                emb = emb.permute(0, 2, 1).contiguous() / deg
                emb = emb.view(-1, ndim)
                gx = self.lipnet(emb, c)
                x, x_prev = y - gx, x
                i += 1
                if i > 1000:
                    logger.info('Iterations exceeded 1000 for inverse.')
                    break
        return x



#####################
# Logdet Estimators helper
#####################
def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)


class iResBlock(nn.Module):

    def __init__(
        self,
        nnet,
        geom_p=0.5,
        lamb=2.,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist='geometric',
        neumann_grad=True,
        grad_in_forward=False,
    ):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
            brute_force: Computes the exact logdet. Only available for 2D inputs.
        """
        nn.Module.__init__(self)
        self.nnet = nnet
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

        # store the samples of n.
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))

    def forward(self, x, logpx=None):
        if logpx is None:
            y = x + self.nnet(x)
            return y
        else:
            g, logdetgrad = self._logdetgrad(x)
            return x + g, logpx - logdetgrad

    def inverse(self, y, logpy=None):
        x = self._inverse_fixed_point(y)
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)[1]

    def _inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.nnet(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev)**2 / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                logger.info('Iterations exceeded 1000 for inverse.')
                break
        return x

    def _logdetgrad(self, x):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""

        with torch.enable_grad():
            if (self.brute_force or not self.training) and (x.ndimension() == 2 and x.shape[1] == 2):
                ###########################################
                # Brute-force compute Jacobian determinant.
                ###########################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                # Brute-force logdet only available for 2D.
                jac = batch_jacobian(g, x)
                batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[:, 0, 1] * jac[:, 1, 0]
                return g, torch.log(torch.abs(batch_dets)).view(-1, 1)

            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * \
                    sum(n_samples >= k - 20) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(x)

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                if self.training and self.grad_in_forward:
                    g, logdetgrad = mem_eff_wrapper(
                        estimator_fn, self.nnet, x, n_power_series, vareps, coeff_fn, self.training
                    )
                else:
                    x = x.requires_grad_(True)
                    g = self.nnet(x)
                    logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, self.training)
            else:
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + (-1)**(k + 1) / k * coeff_fn(k) * batch_trace(jac_k)

            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))


            return g, logdetgrad.view(-1, 1)

    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace, self.brute_force
        )


class MemoryEfficientLogDetEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, c, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x, c)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None, None) + grad_params


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = (-1)**(k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad


def mem_eff_wrapper(estimator_fn, gnet, x, c, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, gnet, x, c, n_power_series, vareps, coeff_fn, training, *list(gnet.parameters())
    )


# -------- Helper distribution functions --------
# These take python ints or floats, not PyTorch tensors.


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


# -------------- Helper functions --------------


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def _flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [p.reshape(-1) if p is not None else torch.zeros_like(q).view(-1) for p, q in zip(sequence, like_sequence)]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])