import numpy as np
from .basis import scalarbasis as sbasis
import scipy.sparse as sps
from itertools import product
from .npstruct import NPStruct

# helper functions to iterate dictionaries
def _dictzip(d1,d2):
    return ((k, d1[k], d2[k]) for k in d1.keys() & d2.keys())

def _dictmatch(d1, d2):
    """ Write a dictzip which throws an error if either one of the
    dicts contains some entry which isn't in the other """
    # if some keys in either one but not both
    if(d1.keys() ^ d2.keys()):
        raise RuntimeError("Fields do not match")
    return _dictzip(d1,d2)

def _refill_ndim(arr, ndim):
    times = ndim - arr.ndim
    for _ in range(times):
        arr = np.expand_dims(arr, axis=-1)
    return arr

def _domain_lazy_map(domain, fun):
    """ Generator for iterating over a domain and yielding a pair
    (elementgroup, fun(elementgroup)) for constructing NPStructs """
    for _, egs in domain:
        for eg in egs:
            yield (eg, fun(eg))


def pullback_form1(domain, jacobian, intdeg):
    out = {}
    for physnum, egs in domain:
        for eg in egs:
            intp, intw = eg.get_integration_points(intdeg)
            N = eg.eval_basis(sbasis, intp)
            points = (N @ eg.lnop_coords).squeeze(2)
            Jac = jacobian(points)
            out[eg] = Jac.swapaxes(-1,-2)[None,...]
    return NPStruct(out, intdeg)    
    
def pullback_form2(domain, jacobian, intdeg):
    """
        WARNING: Not tested if rows and columns are 
        right way around
        
    """
    
    out = {}
    for physnum, egs in domain:
        for eg in egs:
            intp, intw = eg.get_integration_points(intdeg)
            intps = intp.shape[0]
            
            N = eg.eval_basis(sbasis, intp)
            points = (N @ eg.lnop_coords).squeeze(2)
            JT = jacobian(points).swapaxes(-1,-2)
            
            # Transpose changes rows and columns of jacobian
            # -> -1 is rows, -2 -> columns
            target_dim = JT.shape[-1]
            domain_dim = JT.shape[-2]
            
            # 2 forms transform using the determinants of 2x2
            # submatrices of the transpose of the jacobian
            if target_dim == 3:
                row_iter = [(0,1), (1,2), (2,0)]
                det_matrix_rows = 3
            elif target_dim == 2:
                row_iter = [(0,1)]
                det_matrix_rows = 1
            else:
                raise ValueError('Two forms exist only on 2D and 3D but '
                                 'Jacobian has {} rows'.format(target_dim))
            
            if domain_dim == 3:
                col_iter = [(0,1), (1,2), (2,0)]
                det_matrix_cols = 3
            elif domain_dim == 2:
                col_iter = [(0,1)]
                det_matrix_cols = 1
            else:
                raise ValueError('Two forms exist only on 2D and 3D but '
                                 'Jacobian has {} columns'.format(domain_dim))

            # a bit unorthodox waý to select submatrices but who cares
            sdets = list(np.linalg.det(JT[:,:,rows,:][:,:,:,cols])   
                         for rows in row_iter                 
                         for cols in col_iter)

           
            out[eg] = np.reshape(np.stack(sdets, -1), 
                                 (intps, -1, det_matrix_rows, det_matrix_cols))
            
    return NPStruct(out, intdeg)    
    
#class Form1(NPStruct):
    
#    def pullback(self, jacobian):
        
        