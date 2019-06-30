
import numpy as np

from time import clock

class Timer():
    def __init__(self,msg):
        self.msg = msg
    
    def __enter__(self):
#        print("")
#        print("{:*^80}".format(""))
#        print("{:*^80}".format("  {}  ".format(self.msg)))
        self.t1 = clock()
        
        def timerfun():
            return clock() - self.t1
        
        return timerfun
    
    def __exit__(self,*args):
        # args not needed here
        t2 = clock()
        print("{}, took {:.2f} s".format(self.msg, t2-self.t1))

#        print("{:*^80}".format("  took {:.2f} s  ".format(t2-self.t1)))
#        print("{:*^80}".format(""))
#        print("")
        

def compute_area(domain):
    """ Computes a cross sectional area of given domain """
    jac = domain.jacobians(1)
    detg = (jac @ jac.T).det()
    sqrt_detg = detg.unary_map(lambda x: np.sqrt(x)[...,None,None])

    return np.sum((sqrt_detg.integrate().assemble()).toarray())


def compute_c(domain):
    """ Computes a source divider matrix """

    #Adom = compute_area(domain)

    detJ = domain.det_jacobians(1, [0,1])
    N = domain.basis(1)

    return (N.T * detJ).integrate().assemble()

def compute_stiffness_matrix(domain, reluctivity):
    """ Computes a stiffness matrix in given domain """
    detJ = domain.det_jacobians(1, [0,1])
    dN = domain.d_basis(1)
    jac = domain.jacobians(1, [0,1])
    dNxyz = jac.backslash(dN)
    
    return (reluctivity * dNxyz.T @ dNxyz * detJ).integrate().assemble()

    #assemble(integrate(integrand, detJ), shape=(ndofs, ndofs))

def compute_damping_matrix(domain, conductivity):
    """ Computes a damping matrix in given domain """
    detJ = domain.det_jacobians(2, [0,1])
    N = domain.basis(2)
    return (conductivity * N.T @ N * detJ).integrate().assemble()
    

def info(msg):
    print("{: ^80}".format(msg))
    
def stop():
    raise RuntimeError("Not an error. Stopping the execution here")    