import torch
from torch.autograd import Function

class FastICAFunction(Function):
    @staticmethod
    def _symmetric_decorrelation(W):
        K = torch.mm(W, W.t())
        s, u = torch.linalg.eigh(K)
        return torch.mm(torch.mm(torch.mm(u, torch.diag(1.0 / torch.sqrt(s))), u.t()), K)

    @staticmethod
    def _g(x):
        # Implement the nonlinearity function (e.g., tanh)
        return torch.tanh(x)

    @staticmethod
    def forward(ctx, x):
        # Centering the data
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_centered = x - x_mean

        # Whitening the data
        U, S, V = torch.linalg.svd(x_centered / torch.sqrt(torch.tensor(x.shape[1])))
        U = torch.mm(torch.diag(1.0 / S), U.t())
        x_white = torch.mm(U, x_centered)

        # Initialize unmixing matrix randomly
        W = torch.randn(x.shape[0], x.shape[0], dtype=x.dtype, device=x.device)
        W = FastICAFunction._symmetric_decorrelation(W)

        # FastICA algorithm iterations
        for _ in range(100):
            wx = torch.mm(W, x_white)
            g_wx = FastICAFunction._g(wx)
            g_wx_mean = torch.mean(g_wx, dim=1, keepdim=True)
            W_delta = (x_white * g_wx).mean(dim=1) - g_wx_mean * W
            W_delta = FastICAFunction._symmetric_decorrelation(W_delta)
            W += W_delta

        # Save variables for backward pass
        ctx.save_for_backward(W)

        # Unmix the signals
        S = torch.mm(W, x_white)

        return S

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved variables
        W, = ctx.saved_tensors

        # Compute gradient
        dW = torch.mm(grad_output, W.t())

        return dW

