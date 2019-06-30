import numpy as np
import scipy.sparse as sps

def csr_set_rows(csr, rows, value=0):
    """ Set the rows of a csr matrix to a value. Value can be an iterator.
    Then sets Mat[rows[i],:] = value[i], i.e. whole row to same value.

    Useful in zeroing out rows, or setting rows of sparse vectors"""

    if not isinstance(csr, sps.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')

    try:
        for row, val in zip(rows, value):
            csr.data[csr.indptr[row]:csr.indptr[row+1]] = val

    except TypeError:
        # value (or row) not an iterator, so try this, throws an error
        # again if rows not an iterator
        for row in rows:
            csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

        if value == 0:
            csr.eliminate_zeros()

def zero_rowcols_mapping(dofs, dim):
    """ Make a csr_matrix which zeroes out rows given by dofs from (dim, dim)
    matrices when multiplied from both sides.

    These can be composed with multiplication!

    Example:

        K = csr_matrix(..., shape=(10,10))
        Omap = rowcols_zero_mapping([1,2,3], 10)
        K_0 = Omap*K*Omap
        # K_0[[1,2,3],:] == all zero
        # K_0[:, [1,2,3]] == all zero
    """
    chi_interior = np.ones(dim)
    chi_interior[dofs] = 0.0
    return sps.spdiags(chi_interior, [0], dim, dim).tocsr()

def ones_to_diag_matrix(dofs, dim):
    """ Make a csr_matrix which has ones in diagonal elements [dim, dim] ,
    otherwise full of zeros

    These can be composed with addition, iff there are no overlapping
    dofs between diag matrices. Doublecheck with e.g. np.max after composing.

    """
    chi_boundary = np.zeros(dim)
    chi_boundary[dofs] = 1.0
    return sps.spdiags(chi_boundary, [0], dim, dim).tocsr()

def robust_set_rows(F, inds, values):
    """ Set rows given by inds of F to values given by values robustly. F can
    be either sparse or dense array. For sparse arrays, csr_set_rows may be
    better.

    If inds is a list or an array (dim,) this assignment needs to be
    F[inds] = (something of shape (dim,1)) for some reason.
    User can sometimes give values as (dim,) or (dim,1) and this shouldn't
    make a difference.
    """
    try:
        if(values.ndim < 2):
            F[inds] = values[:,None]
        else:
            F[inds] = values
    except AttributeError:
        # values was not an np.array
        # assuming its a number then
        F[inds] = values
        
def robust_subtract(F,Q):
    """ Robustly calculate F - Q.

    There are scenarios where user gives  F as (dim,) or (dim,1) and
    this shouldn't make a difference. Broadcasting rules cause these two
    to have different behaviour...
    """
    if(F.ndim < 2):
        return F[:,None] - Q
    else:
        return F - Q


def set_dofs(K, F, dofs, val):
    """ Sets dirichlet rows to zero, ones to diagonal. Results in nonsymmetric
    system.

    A bit dirty but usually the fastest way to get a solvable full system.

    """
    csr_set_rows(K, dofs)
    K[dofs,dofs] = 1 # SLOW
    robust_set_rows(F,dofs,val)

def set_dofs_dense(K, F, dofs, val):
    K[dofs, :] = 0
    K[dofs,dofs] = 1
    F[dofs] = val

def set_dofs_zero_symm(K, dirichlet_zero_dofs):
    """ Set dirichlet rows and columns to zero and adds ones to the
    diagonal (system stays symmetric).
    """
    dim = K.shape[0]
    I_interior = zero_rowcols_mapping(dirichlet_zero_dofs, dim)
    I_boundary = ones_to_diag_matrix(dirichlet_zero_dofs, dim)

    return I_interior * K * I_interior + I_boundary

def set_dofs_rows_and_columns_zero_symm(K, dofs):
    """ Set dirichlet rows and columns to zero (system stays symmetric).
    Does not put ones to the diagonal.
    """
    dim = K.shape[0]
    I_interior = zero_rowcols_mapping(dofs, dim)

    return I_interior * K * I_interior

def set_dofs_symm(K, F, dofs, values):
    """ Set Dirichlet boundary conditions to the system K*x = F.
    This is really fast way to get a symmetric stiffness matrix.
    Does not mutate K and F, returns a copy Kmod and Fmod.
    """

    #B = sps.csr_matrix(F.shape)
    dim = K.shape[0]
    bdim = len(dofs)
    B = sps.csr_matrix((values, (dofs, np.zeros(bdim,))), shape=(dim,1))
    Q = (K @ B).toarray()

    Fmod = robust_subtract(F, Q)

    robust_set_rows(Fmod, dofs, values)

    Kmod = set_dofs_zero_symm(K, dofs)

    return (Kmod, Fmod)

def set_dofs_zero_naive(K, dofs):
    """ Naive way to set dirichlet rows and columns to zero and diagonal to
    one. Results in symmetric stiffness matrix

    This is REALLY REALLY SLOW so do not really use this...
    """
    Kout = K.tolil()
    Kout[dofs,:] = 0 # SLOW
    Kout[:, dofs] = 0 # "fast"
    Kout[dofs,dofs] = 1 # SLOWW
    return Kout.tocsr()

def coupling(K, dofs, vals):
    """ Computes the vector "Q" which needs to be substracted from
    the RHS of equation system """
    dim = K.shape[0]
    bdim = len(dofs)

    B = sps.csr_matrix((vals, (dofs, np.zeros(bdim,))), shape=(dim,1))
    return K@B

def set_dofs_dynamic_symm(K, M, F, dofs, vals=None, d_vals=None):
    """ Set dirichlet boundary conditions to dynamic system with possibly
    time varying boundary values. If vals and d_vals == None, sets homogeneous
    conditions.

    Except that this can't really be used at all in the intended context,
    so, yeah. Another two-hours wasted but hey, that's life :)
    """
    dim = K.shape[0]
    I_interior = zero_rowcols_mapping(dofs, dim)
    I_boundary = ones_to_diag_matrix(dofs, dim)

    bdim = len(dofs)

    if(vals is not None):
        B = sps.csr_matrix((vals, (dofs, np.zeros(bdim,))), shape=(dim,1))
        Q = K@B
    else:
        Q = sps.csr_matrix(shape=(dim,1))

    if(d_vals is not None):
        dBdt = sps.csr_matrix((d_vals, (dofs, np.zeros(bdim,))), shape=(dim,1))
        W = (M@dBdt).toarray()
        Q = Q + W

    Kmod = I_interior * K * I_interior + I_boundary
    Mmod = I_interior * M * I_interior


    Fmod = robust_subtract(F, Q)
    robust_set_rows(Fmod, dofs, vals)


    return (Kmod, Mmod, Fmod)
