import numpy as np
import numpy.linalg as lin
def whitening(x):
    n,_ = x.shape
    SIG = (1.0 / n) * x * x.T
    ei = lin.eig(SIG)


