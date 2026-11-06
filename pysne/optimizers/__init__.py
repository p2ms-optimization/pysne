from .base import BaseOptimizer
from .spo.engine import spiral_optimization
from .spo.optimizer import SPOOptimizer
from .scipy_root import ScipyRootOptimizer

OPTIMIZERS = {
    'spo': SPOOptimizer,
    'newton': ScipyRootOptimizer,
    'scipy_root': ScipyRootOptimizer,
    'hybr': lambda: ScipyRootOptimizer(method='hybr'),
    'lm': lambda: ScipyRootOptimizer(method='lm'),
}

def get_optimizer(optimizer_input):
    """
    Resolves an optimizer from a string identifier, BaseOptimizer class,
    BaseOptimizer instance, or a callable.

    Parameters
    ----------
    optimizer_input : str, BaseOptimizer, or callable
        The optimizer configuration.

    Returns
    -------
    BaseOptimizer or callable
        The resolved optimizer instance or function.
    """
    if isinstance(optimizer_input, str):
        key = optimizer_input.lower()
        if key not in OPTIMIZERS:
            available = list(OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer '{optimizer_input}'. Available: {available}")
        opt_factory = OPTIMIZERS[key]
        return opt_factory()
    elif isinstance(optimizer_input, BaseOptimizer):
        return optimizer_input
    elif isinstance(optimizer_input, type) and issubclass(optimizer_input, BaseOptimizer):
        return optimizer_input()
    elif callable(optimizer_input):
        return optimizer_input
    else:
        raise TypeError(
            f"Invalid optimizer type: {type(optimizer_input)}. "
            "Must be a string, BaseOptimizer subclass/instance, or callable."
        )

__all__ = [
    'BaseOptimizer',
    'SPOOptimizer',
    'ScipyRootOptimizer',
    'spiral_optimization',
    'get_optimizer',
    'OPTIMIZERS'
]