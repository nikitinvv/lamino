from pkg_resources import get_distribution, DistributionNotFound

from lamcg.lamusfft import *
from lamcg.solver_lam import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass