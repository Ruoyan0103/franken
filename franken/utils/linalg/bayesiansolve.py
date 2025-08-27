import warnings

import torch


try:
    import cupy.cuda
    from cupy_backends.cuda.libs import cublas, cusolver
except ImportError:
    cupy = None
    cusolver = None
    cublas = None


def bayesian_linear(
    cov: torch.Tensor, rhs: torch.Tensor, alpha: float, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Bayesian linear regression to get posterior mean and covariance of weights.

    Multiple right-hand sides are supported. Instead of providing the data
    matrix (commonly :math:`X` in ridge-regression notation), and labels (commonly :math:`y`),
    we are given directly :math:`\text{cov} = X^T X` and :math:`\text{rhs} = X^T y`.
    Since :attr:`cov` is symmetric only its **upper triangle** will be accessed.

    To limit memory usage, the :attr:`cov` matrix **may be overwritten**, and :math:`rhs`
    may also be overwritten (depending on its memory layout).
    
    Args:
        cov (Tensor): covariance of the linear system
        rhs (Tensor): right hand side (one or more) of the linear system
        alpha (float) : prior precision (1 / prior variance)
        beta (float)  : likelihood precision (1 / noise variance)

    Returns:
        mu (Tensor): posterior mean
        Sigma (Tensor): posterior covariance
    """
    n = cov.shape[0]
    
    # Posterior precision matrix: A = alpha * I + beta * X^T X
    A = alpha * torch.eye(n, device=cov.device, dtype=cov.dtype) + beta * cov
    
    # Cholesky decomposition for numerical stability
    L = torch.linalg.cholesky(A, upper=True)
    
    # Posterior covariance: Sigma = A^{-1}
    Sigma = torch.cholesky_inverse(L, upper=True)
    
    # Posterior mean: mu = beta * Sigma * rhs
    rhs_shape = rhs.shape
    mu = beta * (Sigma @ rhs.view(n, -1))
    mu = mu.view(rhs_shape)

    return mu, Sigma