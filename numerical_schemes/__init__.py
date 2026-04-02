from .base import *
from .explicit import *

SCHEME_REGISTRY = {
    "Euler": Euler,
    "EulerForPrices": Euler,
    "Milstein": Milstein,
    "ModifiedMilsteinCIR": ModifiedMilsteinCIR
}

def retrieve_scheme(scheme_name:str, *args, **kwargs):
    if scheme_name not in SCHEME_REGISTRY:
        raise ValueError(f"Scheme '{scheme_name}' not found. Options: {list(SCHEME_REGISTRY.keys())}")
    return SCHEME_REGISTRY[scheme_name](*args, **kwargs)