""" Contains Element Group implementation
which implements a set of element operations to a group of elements. """

import numpy as np
from collections import namedtuple
from .basis_functions import scalarbasis as sbasis


"""
Shorthand for datatypes used in numpy arrays
"""
t_float = np.float64
t_int = np.int64
t_uint = np.uint64
# if your mesh has more than 4294967295 nodes or elements, change this to
# np.uint64
t_tags = np.uint32

IntegrationPoints = namedtuple('IntegrationPoints', 'points weights')

""" ElementGroup is a base class.
It is used to derive classes for a set of elements of the same type.
All operations should(if possible) be implemented as vectorized
operations using numpy arrays """

class ElementGroup():

    # store node coordinates inside the element
    def __init__(self, entity_tag, global_nodeids, lnode_coords,
                 gnop, nop_coords, element_tags):
        self.entity_tag = entity_tag
        # coordinates in nop matrix form
        self.lnop_coords = nop_coords
        # global nodeids in each element (elements x nodel)
        self.gnop = np.array(gnop, dtype=t_tags)
        # global nodeids present in this group
        self.global_nodeids = np.array(global_nodeids, dtype=t_tags)
        # node coordinates in array (glob nodes x xyz)
        self.lnode_coords = lnode_coords

        # global element tags from gmsh, needed in plotting etc.
        self.element_tags = element_tags

        # for each global index gid replace it with a local index
        # replaces global indexes with complete local index set
        # starting from 0,1,2...

        # this may seem bad (nested loops) but it is actually stupidly fast.
        d = dict(map(reversed, enumerate(global_nodeids)))
        self.lnop = np.array([[d[i] for i in row] for row in gnop],
                             dtype=t_tags)


    def _shallow_partial_copy(self, **kwargs):
        """ Create a shallow copy of the object so that
        kwargs can be given to overwrite the attributes. Otherwise they
        are taken from self. So a new ElementGroup will be returned but
        same objects are still used in it. """
        entity_tag = kwargs.pop('entity_tag', self.entity_tag)
        global_nodeids = kwargs.pop('global_nodeids', self.global_nodeids)
        lnode_coords = kwargs.pop('lnode_coords', self.lnode_coords)
        gnop = kwargs.pop('gnop', self.gnop)
        nop_coords = kwargs.pop('nop_coords', self.lnop_coords)
        element_tags = kwargs.pop('element_tags', self.element_tags)
        # kwargs has to be empty at this point
        assert(not kwargs)
        # create instance of the same class (possibly polymorphic)
        return self.__class__(entity_tag, global_nodeids, lnode_coords, gnop,
                              nop_coords, element_tags)

    def coord_map_isop(self, points):
        """ Maps from ref element to global elements """
        coefficients = self.eval_basis(sbasis, points)
        coordinates = np.matmul(coefficients, self.lnop_coords)
        return coordinates

    def get_jacobians_isop(self, points, columns=None):
        """ Calculate coordinate mapping jacobian matrix
            for each element in the group """
        dN = self.eval_d_basis(sbasis, points)
        J = np.matmul(dN, self.lnop_coords)

        if columns is None:
            return J

        return J[:,:,:,columns]

    def get_detj_isop(self, points, columns=None):
        J = self.get_jacobians_isop(points, columns)
        detJ = np.linalg.det(J)[np.newaxis,np.newaxis,:,:].transpose((2,3,0,1))
        return detJ

    def evaluate_nodal(self, values_glob, arr_loc):
        #warning: not tested yet!
        nop_table = self.lnop[np.newaxis,:]
        nop_table = nop_table.transpose((1, 2, 0))
        coeff = values_glob[nop_table]
        return np.matmul(arr_loc, coeff)


    def interp_ncoeff_fromfunction(self, sc_func):
        coeffs = self.evaluate_function(sc_func,
                            self._nodepoints).transpose((2, 1, 0, 3))
        return coeffs

    def evaluate_function(self, f, coords):
        #returns a (points,elements,dimension,1)-array, where
        #f is evaluated at reference points "points" in all "elements".
        #"dimension" is dependent on the dimension of the codomain of f
        coords_glob = self.coord_map_isop(coords)
        interpolated = f(coords_glob)
        if interpolated.ndim == 3:
            interpolated = interpolated[np.newaxis,:]
        interpolated = interpolated.transpose((1,2,0,3))
        return interpolated

class ElementGroupGmshProxy:
    """ Represents an element group and uses Gmsh as a backend to
    compute basis functions, jacobians etc.
    TOTHINK: Should we get the stuff "lazily"? Only when needed but cache
    the results to avoid retreiving them multiple times?
        -> probably not worth the complexity
    TODO: Add data types to numpy arrays (int/uint/float)?
    """

    def __init__(self, gmsh, gdim, gtag, gname, enttag, etype):

        self.gmsh = gmsh
        self.group_dimension = gdim
        self.group_tag = gtag
        self.group_name = gname
        self.entity_tag = enttag
        self.element_type = etype

        etags, node_tags = gmsh.model.mesh.getElementsByType(etype, enttag)

        num_elements = len(etags)
        #num_nodes = len(node_tags)

        # Gmsh starts numbering of nodes and elements from 1, hence -1
        self.gnop = np.array(node_tags,
                             dtype=t_tags).reshape((num_elements,-1))-1
        self.global_nodeids = np.unique(np.array(node_tags, dtype=t_tags))-1
        self.element_tags = np.array(etags, dtype=t_tags)-1

    def integration_string(self, degree):
        """Returns a gmsh Gauss integration method string for order degree """
        return "Gauss{}".format(degree)

    def integration_points_xyz(self, integ):
        res = self.gmsh.model.mesh.getJacobians(self.element_type, integ,
                                                self.entity_tag)

        num_elements = self.gnop.shape[0]

        # reshape to (intpoints, elements, dim, dim) and
        # (intpoints, elements, 1, 1)
        xyz = np.reshape(res[2], (num_elements, -1, 3)).swapaxes(0,1)

        return xyz

    def jacobians(self, integ, columns=None):
        """ Calculate coordinate mapping jacobian matrix
            for each element in the group """
        res = self.gmsh.model.mesh.getJacobians(self.element_type, integ,
                                                self.entity_tag)

        num_elements = self.gnop.shape[0]

        # reshape to (intpoints, elements, dim, dim) and
        # (intpoints, elements, 1, 1)
        jac = np.reshape(res[0], (num_elements, -1, 3 , 3)).swapaxes(0,1)
        detj = np.reshape(res[1], (num_elements,-1)).swapaxes(0,1)

        if columns is None:
            return jac, detj

        return jac[:,:,:,columns], detj

    def basis_functions(self, basis, integ):
        """ Get basis integration points, weights and basis functions
        in elements of this element group.
        Inputs:
            basis: type of basis ('Lagrange' or 'GradLagrange' for now)
            integ: integration type ('Gauss<n>' where n is degree)
        """
        mesh = self.gmsh.model.mesh

        uvwo, ncomp, baso = mesh.getBasisFunctions(self.element_type, integ,
                                                   basis)
        uvwq = np.reshape(uvwo, (-1,4))
        uvw = uvwq[:,0:3]
        weights = uvwq[:,3]

        nint_points = len(weights)

        # intp x ncomponents x dof
        bas = np.reshape(baso, (nint_points, -1, ncomp)).swapaxes(1,2)

        return uvw,weights,bas


    def eval_basis(self, intdeg):
        """ Get integration points and scalar (Lagrange) basis functions in
        integration points.
        Inputs:
            intdeg - degree of Gauss integration
        """
        return self.basis_functions("Lagrange",
                                    self.integration_string(intdeg))

    def eval_d_basis(self, intdeg):
        """ Get gradient (GradLagrange) of scalar basis functions in
        integration points.
        Inputs:
            intdeg - degree of Gauss integration
        """
        return self.basis_functions("GradLagrange",
                                    self.integration_string(intdeg))
