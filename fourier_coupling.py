import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from collections import defaultdict, namedtuple


def wavenumbers(K, no_zero=False): 
    """ Return integers [0,1,-1,1,-2... int(K/2), -int(K/2)]
    
    If K even, sequence ends at int(K/2), if K odd, -int(K/2)
    """
#    def w(k):
#        mod = k%2
#        return (1-mod)*int(k/2) - mod*int((k-1)/2)
#
#    return [w(k) for k in range(1,K+1)]
    
    P = int(K/2)+1   
#    N = P-1*K%2

    tail = np.stack([np.arange(1, P), -np.arange(1,P)], 
                     axis=1).reshape(-1)[0:K-1]
    if no_zero:
        return tail
    else:
        return np.concatenate([np.array([0]), tail])
    #return list(range(0,K))


def nudft_matrix(points, K=None):
    """ Construct a non-uniform discret Fourier-transform matrix
    
    Inputs:
        points - np.array of sample points (inside [0,1])
        K - number of frequencies to sample, defaults to number of points
        
    Outputs:
        np.array of shape (K, len(points))
    """
    if(K is None):
        K = len(points)
        N = K
    else:
        N = len(points)
    
    if(points.ndim > 1):
        points = points.reshape(-1)
        

    ks = wavenumbers(K)
    
    D = np.stack([np.exp(-1j*2*np.pi*k*points) for k in ks])
    
    return 1/N*D

def center_interpolation_1D(N):
    """ Construct a matrix which adds interpolated center point values.
    
    Inputs :    
        N - number of nodes
    
    Outputs: 
        I - matrix which turns a nodal value array into array containing
        nodal values and center point values 
        
        I @ np.array(a1, a2, ..., aN) 
                        = np.array(a1, cp12, a2, cp23, ... ,cp(N-1)N, aN)     
    """
    spread = np.eye(2*N,2*N-1)[0::2].T    
    # next node value
    spread_u = np.eye(2*N,2*N-1)[1::2].T*0.5     
    # previous node value
    spread_l = np.eye(2*N,2*N-1,-1)[0::2].T*0.5        
    I = spread + spread_u + spread_l
    return I


def build_parametric_domain(bdom, pcoordinates):
    
    def build_parametric_egs(dt, et, eg):
        # renumber the nodes to domain-local numbering
        l_nodeids = bdom.global2local[eg.global_nodeids]
        l_nop = bdom.global2local[eg.gnop]
        # get parametric coordinates for nodes
        new_lnode_coords = pcoordinates[l_nodeids, None]
        new_lnop_coords = pcoordinates[l_nop, None]
        
        # shallow copy/functional update
        neweg = eg._shallow_partial_copy(lnode_coords=new_lnode_coords,
                                         nop_coords=new_lnop_coords) 
        return dt, et, neweg
    
    # replace the coordinates of the nodes in elementgroups
    # with the parametric coordinates     
    eg_list = [build_parametric_egs(*x) for x in bdom]
    
    # new domain containing the parametric egs    
    return bdom.__class__(bdom.mesh, eg_list)
    
    
def compute_nudft_fem(bdom, undo_cut, pcoordinates, **kwargs):
    """ Computes nudft matrix with FEM approach.
    
    Computes the Fourier transform integral 
        \hat{f}_k = int_0^1 f(x) exp(-2j pi x k) dx
    by writing 
        f(x) = sum_i f_i * psi_i(x)
    i.e. in FE-basis and computes the integral over the mesh to mitigate the
    effect of the non-uniformity of the mesh.
    
    Inputs: 
        bdom:
            A domain object (the cutted version)
        undo_cut:
            The cut information
        pcoordinates:
            parametric coordinates of the nodes in the domain
    
    Outputs:
        nudft_matrix: 
            A (ndofs-1, ndofs-1) nudft matrix (because the cut/artificial
            nodes will be undoed 
    """
        
    intdeg = kwargs.pop('intdeg', 2)
    
    pbdom = build_parametric_domain(bdom, pcoordinates)

    detj = pbdom.det_jacobians(intdeg)
    N = pbdom.basis(intdeg)
    
    def make_exp(dimtag, etag, up):

        # shape should be (intpoints, elements=1, dof, K)
        exp = np.stack([np.exp(-2j*np.pi*up*k) 
                        for k in wavenumbers(bdom.ndofs-1)], axis=-2)
        return exp
    
    # build the exp( -2 pi j k_ p_i) part as npstruct tensor field
    exp = pbdom.tensor_field(intdeg, make_exp)
    
    integrated = ((exp@N)*detj).integrate()
    
    # restore the uncut dof numbering
    renum = np.arange(0,pbdom.ndofs)
    artif_lnode = pbdom.global2local[undo_cut[0].artificial]
    renum[artif_lnode] = pbdom.global2local[undo_cut[0].real]

    
    def col_numbering(arr_unass, nopt):
        # column index is the node index, with the undoed cut
        nop_table = renum[nopt]
        index_shape = arr_unass.shape[-2:]
        index_j = nop_table.transpose((1, 0, 2))
        index_j = index_j*np.ones(index_shape)
        return index_j
    
    def row_numbering(arr_unass, nop_table):
        # row index is the wave number index (Fourier series term index)
        # produce shape (elem, dofs, dofs) shape but like this
        # [[[1,1],[2,2],[3,3], ... [dofs,dofs,]]]
        ax0,ax3,ax2 = arr_unass.shape
        return (np.ones((ax0, 1, ax2))*np.arange(0,ax3)[None,:,None])

    nudft_mat = integrated.assemble(shape=(pbdom.ndofs-1, pbdom.ndofs-1), 
                                    local=True,
                                    row_numbering=row_numbering,
                                    col_numbering=col_numbering).toarray()
    return nudft_mat


def fourier_integrals_1D(x1s, x2s, ks):
    """ Compute fourier integrals for batch of elements for frequencies ks
    
    Compute Fourier integrals for elements up to K:th order
    where local node1 are x1s, local node2 are x2s. Uses numpy broadcasting
    for ultimate win
    
    Inputs:
        x1s: np.array of shape (elements,) containing parametric coordinates
                for local nodes 1 
        x2s: np.array of shape (elements,) containing parametric coordinates
                for local nodes 2
        ks: np.array of shape (frequencies,) containing frequencies and 
              ks[0] == 0
              
    Output:
        np.array of shape (ldofs, elements, frequencies)
    """
    assert ks[0] == 0
    
    Ls = x2s-x1s
    
    
    x1 = x1s[..., None]
    L = Ls[..., None]
    # drop zero out
    k = ks[None, 1:]

    D0 = L/2
    
    # N1
    c11 = np.divide(1, L*(np.pi*k)**2)
    t11 = (-0.5j*np.pi*L*k + 0.25)*np.exp(2j*np.pi*k*(L+x1))
    t21 = -0.25*np.exp(2j*np.pi*k*x1)
    c31 = np.exp(-2j*np.pi*k*(L+2*x1))
    
    D11 = c11*(t11+t21)*c31
    
    # N2    
    c12 = np.divide(1, L*(np.pi*k)**2)
    t12 = (0.5j*np.pi*L*k + 0.25)*np.exp(2j*np.pi*k*x1)
    t22 = -0.25*np.exp(2j*np.pi*k*(L+x1))
    c32 = np.exp(-2j*np.pi*k*(L+2*x1))
        
    D22 = c12*(t12+t22)*c32
    
    D1 = np.concatenate([D0, D11], axis=1)
    D2 = np.concatenate([D0, D22], axis=1)    

    return np.stack([D1 , D2], axis=0)

    
def compute_nudft_analyt(bdom, undo_cut, pcoordinates,  **kwargs):
    """ Computes the dft matrix using analytically derived formula 
    
    """

    pbdom = build_parametric_domain(bdom, pcoordinates)
    
    C = bdom.ndofs-1 
    K = kwargs.pop('K', C)
    
    ws = wavenumbers(K)
    
    index_i = []
    index_j = []
    arr = []
    
    for eg in pbdom.elementgroups():
        
        # domain-local gnop matrix
        dlgnop = pbdom.global2local[eg.gnop]
        x1s, x2s = eg.lnop_coords.squeeze(-1).T
        
        nel = len(eg.gnop)
        
        # Is, shape (2, elems, freqs)
        Is = fourier_integrals_1D(x1s, x2s, ws)
        arr.append(Is.reshape(-1))

        # frequency indices need to be [0,1,2, ... ,0,1,2,... ...] nel*2 times
        freqinds = (np.arange(0, K)*np.ones((nel*2,1))).reshape(-1)
        index_i.append(freqinds)

        # dof indices need to be [dof1, dof1 ..., dof2, dof2 ...] each dof
        # freq times
        dofi = (dlgnop[None,...]*np.ones((K,1,1))).swapaxes(0,-1).reshape(-1)
        index_j.append(dofi)

    index_i = np.concatenate(index_i)
    index_j = np.concatenate(index_j)
    
    # remove the artificial node
    real, artificial = undo_cut[0]
    index_j[index_j == pbdom.global2local[artificial]] = pbdom.global2local[real]

    arr = np.concatenate(arr)
    
    return sps.csr_matrix((arr, (index_i, index_j)), shape=(K, C))


                
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                             UNIT TESTS                            # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#if __name__ == '__main__':
#    
#    import unittest
    
#    class TestNufft(unittest.TestCase):
#    
#        def setUp(self):            
#
#    
#        def test_find_simplices_1d(self):
#
#    unittest.main()