from .npstruct import NPStruct

""" Generic constructor functions for different needs """

def jacobians(domain, intdeg, columns=None):
    """ Construct an NPStruct of isoparametric jacobian matrices
    """
    return domain.jacobians(intdeg, columns)

def det_jacobians(domain, intdeg, columns=None):
    """ Construct an NPStruct of isoparametric jacobian matrice determinants
    """
    return domain.det_jacobians(intdeg, columns)

def tensor_field(domain, intdeg, field_fun):
    return domain.tensor_field(intdeg, field_fun)

def basis(domain, intdeg, basis=None):
    """ Evaluate basis functions to npstruct form
    Inputs:
        intdeg: integration degree
        basis: basis type
    Output:
        npstruct containing basis function array representations with
        following axes:
            (elements=1, intpoints, components=1, dofs)
    """
    return domain.basis(intdeg, basis)

def d_basis(domain, intdeg, basis=None):
    """ Evaluate derivatives of basis functions to npstruct form
    Inputs:
        intdeg: integration degree
        basis: basis type
    Output:
        npstruct containing basis function array representations with
        following axes:
            (elements=1, intpoints, components=3/2/1, dofs)
    """
    return domain.d_basis(intdeg, basis)

def field_in_elements(domain, f, intdeg=1, offset=0):
    """ Distribute the nodal field f locally to degrees of freedom
    in the elements.

    Parameter f must have the shape (timesteps, nodes, ijk...)
    """
    return domain.field_in_elements(f, intdeg, offset)

def initializer(domain, initfun, intdeg):
    """ Initialize an NPStruct with a initializer function.
    Initfun must be of type
        (dim_tag, enttag, elementgroup, intdeg) -> nparray of correct shape
    """
    return domain.initializer(initfun, intdeg)


def matmul(f1, f2):
    """ This is basically the @ operator but can be used to
    also multiply NPStruct from the left with a bare numpy array.
    Numpy arrays have a bug that they don't dispatch nparr @ npstruct to
    npstructs __rmatmul__ method.

    https://github.com/numpy/numpy/issues/9028
    """
    if(isinstance(f1, NPStruct)):
        return f1.__matmul__(f2)
    else:
        return f2.__rmatmul__(f1)

def integrate(field):
    """ Integrate an npstruct object. Field must be representing a integrable
    field (must be defined in integration points of elements).

    Returns an npstruct with one less dimensions.
    """
    return field.integrate()

def assemble(field, **kwargs):
    """ Assembles the npstruct object. The object must represent either
    local stiffness matrices (array of (..., dofs, dofs)) or
    local residual vectors (array of (..., 1, dofs) or (..., dofs, 1))

    Input:
        field: npstruct object to be assembled
        shape: the shape of the global system
               defaults to the amount of nodes on the mesh object
        kwargs: local - True -> only consider dofs in the domain,
                        renumbers the global dofs locally
    Output:
        a scipy csr_matrix
    """
    return field.assemble(**kwargs)

def nodes_in(domain_field_mesh):
    """ Return a np.array containing all unique node labels in entity """
    return domain_field_mesh.nodes_in()
