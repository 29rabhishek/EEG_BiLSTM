import torch
import numpy as np
import warnings

#FastICA Parallel Implementation in Torch


def _logcosh(x, device, fun_args=None):
    alpha = 1 if fun_args == None else fun_args  # comment it out?

    x *= alpha
    gx = torch.tanh(x)  # apply the tanh inplace
    g_x = torch.empty(x.shape[0], dtype=x.dtype, device=device)
    # XXX compute in chunks to avoid extra allocation
    for i, gx_i in enumerate(gx):  # please don't vectorize.
        g_x[i] = (alpha * (1 - gx_i**2)).mean()
    return gx, g_x


def _sym_decorrelation(W):
    # ((W,W.T)-1/2)*W
    K = torch.mm(W, W.t())
    s, u = torch.linalg.eigh(K)
    s = torch.sqrt(torch.clamp(s, min=torch.finfo(W.dtype).tiny)) 
    # scaling eigenvector in u by sqrt of inv of eigenvalue     
    return torch.linalg.multi_dot([u,torch.diag((1.0 / torch.sqrt(s))), u.T, W])

def _ica_par(X, tol, g, fun_args, max_iter, w_init, device):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = torch.tensor(X.shape[1], dtype = torch.float, device=device)
    for ii in range(max_iter):
        gwtx, g_wtx = g(torch.matmul(W, X), device, fun_args = fun_args)
        W1 = _sym_decorrelation(torch.matmul(gwtx, X.T) / p_ - g_wtx[:,None]* W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        # np.einsum allows having the lowest memory footprint.
        # It is faster than np.diag(np.dot(W1, W.T)).
        # lim = torch.max(torch.abs(torch.abs(torch.einsum("ij,ij->i", W1, W)) - 1))
        lim = torch.max(torch.abs(W1 -W)) 
        if lim < tol:
            break
        W = W1
    else:
        warnings.warn(
            (
                "FastICA did not converge. Consider increasing "
                "tolerance or the maximum number of iterations."
            ),
        )

    return W, ii + 1

class FastICA_Torch():
    def __init__(
        self, 
        n_components = None,
        max_iter = 200,
        fun_args = 1,
        tol=1e-4,
        non_linearity = 'tanh',
        w_init = None,
        device = 'cpu',
        random_state = None,
        whiten = 'unit_varience',
        whiten_solver = 'eigh'
        ):
        self.n_components = n_components
        self.non_linearity = non_linearity
        self.tol = tol
        self.max_iter = max_iter
        self.fun_args = fun_args
        self.w_init = w_init
        self.device = device
        self.random_state = random_state
        self.whiten = whiten
        self.whiten_solver = whiten_solver
        self.components_ = None
        self.mixing_ = None
        self._unmixing = None
        self.whitening_ = None
        self.mean_ = None
        self.n_iter = None

    def fit_transform(self, X):
        n_features, n_samples = X.shape

        # checking and selecting n_components
        if self.n_components is None:
            self.n_components = min(n_features, n_samples)
        elif self.n_components > min(n_features, n_samples):
            self.n_components = min(n_features, n_samples)
            warnings.warn(
                f"n_components set to {self.n_components}"
            )
        # checking and genrating w_init
        w_init = self.w_init
        if w_init is None:
            if self.random_state is not None:
                w_init = torch.randn(
                    self.n_components, self.n_components,
                    dtype=X.dtype,
                    generator=self.random_state, 
                    device=self.device
                    )
            else:
                w_init = torch.randn(
                    self.n_components, self.n_components,
                    dtype=X.dtype,
                    device= self.device
                )
        else:
            w_init = torch.tensor(w_init)
            if w_init.shape != (self.n_components, self.n_components):
                raise ValueError(
                    "w_init has invalid shape -- should be %(shape)s"
                    % {"shape": (self.n_components, self.n_components)}
                )

        # Whitening process
        if self.whiten:
            X_mean = torch.mean(X, keepdim = True, dim = -1)
            X -= X_mean
            if self.whiten_solver == 'eigh':
                d, u = torch.linalg.eigh(torch.matmul(X, X.T))
                sort_indices = torch.argsort(d).flip(dims=[0])
                eps = torch.finfo(d.dtype).eps
                degenrate_idx = d < eps
                
                if np.any(degenrate_idx.clone().cpu().numpy()):
                    warnings.warn(
                        "There are some small singular values, using "
                        "whiten_solver = 'svd' might lead to more "
                        "accurate results."
                    )
                d[degenrate_idx] = eps
                d = torch.sqrt(d)
                d, u = d[sort_indices], u[sort_indices]
            elif self.whiten_solver == 'svd':
                u, d = torch.linalg.svd(X, full_matrices = False)[:2]
            # u *= torch.sign(u[0])
            K = torch.matmul(torch.diag(1.0/d),u.T)[:self.n_components]
            # K = (u/d).T[:self.n_components]
            del u, d
            X1 = torch.matmul(K, X)
            X1 *= torch.sqrt(torch.tensor(n_samples))

        # at that time only availabel function implementaiton is
        # 'logcosh'
        self.non_linearity = 'logcosh'
        if self.non_linearity == 'logcosh':
            g = _logcosh
        
        kwargs = {
            "tol": self.tol,
            "g": g,
            "fun_args": self.fun_args,
            "max_iter": self.max_iter,
            "w_init": w_init,
            "device": self.device  
        }
        W, n_iter = _ica_par(X1, **kwargs)
        self.n_iter = n_iter
        del X1
        if self.whiten:
            S = torch.linalg.multi_dot([W,K, X])
        else:
            S = torch.matmul(W, X)

        if self.whiten == 'unit_varience':
            S_std = torch.std(S, dim = -1, keepdim=True)
            S /= S_std
            W /= S_std
            self.components_ = torch.matmul(W,K)
            self.mean_ = X_mean
            self.whitening_ = K
        else:
            self.components_ = W

        self.mixing_ = torch.linalg.pinv(self.components_)
        self._unmixing = W
        
        return S
              