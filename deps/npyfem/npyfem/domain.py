import numpy as np
import scipy.sparse as sps
from .basis_functions import scalarbasis as sbasis
from .npstruct import NPStruct

from itertools import repeat


class DomainContext():
    """ Attach a context on top of an elementgroup, e.g. awareness
    of some local dof numbering in case of a mesh partition """

    def __init__(self, elementgroup, domain_global_nodeids):
        self._elementgroup = elementgroup
        self.global_nodeids = domain_global_nodeids



        # Construct a connectivity array where global dof id:s are replaced
        # with according domain specific dof id:s
        d = dict(map(reversed, enumerate(domain_global_nodeids)))
        self.connectivity = np.array([[d[i] for i in row]
                                     for row in elementgroup.gnop], dtype=int)

class Domain():
    """ Defines a domain consisting of a set of element groups
    and a local set of degrees of freedom (all those which belong to
    the domain) and a map from/to global degrees of freedom

    Let's assume you seriously never want to get an element group
    using the tag, so a list of tuples will be sufficient (instead of dict)
    """

    def __init__(self, mesh, eg_list):
        # eg_list : [ (phys_dimTag, entity_tag, element_group)  ]
        self.mesh = mesh
        self._elgroups = [(dimtag, entity_tag, eg)
                         for dimtag, entity_tag, eg in eg_list]

        self.global_nodeids = np.unique(np.concatenate([eg.global_nodeids
                                        for eg in self.elementgroups()]))

        d = dict(map(reversed, enumerate(self.global_nodeids)))

        inds = np.array(list(d.keys()), dtype=int)
        vals = np.array(list(d.values()), dtype=int)
        C = sps.coo_matrix((vals, (inds, np.zeros(inds.shape, dtype=int))))
        self.global2local = C.toarray().squeeze()

        self.ndofs = len(self.global_nodeids)


    def __iter__(self):
        """ Returns an iterator which iterates through the domain
        yielding (DimTag, entity tag, element goup) """
        return iter(self._elgroups)

    def connectivity(self, eg):
        return self.global2local[eg.gnop]

#    def local_ids(self, eg, gnodes=None):
#        if(gnodes is None):
#            return self._local2global[eg][eg.global_nodids]
#        return self._local2global[eg][gnodes]

    def nodes_in(self):
        """ Return a np.array containing all unique node labels in domain """
        ldofs = np.unique(np.concatenate([eg.global_nodeids
                                      for eg in self.elementgroups()]))
        return ldofs

    def elementgroups(self):
        """ Returns an iterator which iterates through the elementgroups
        in the domain """
        return (eg for _,_,eg in self._elgroups)

    def __repr__(self):
        over = [(dt, et) for dt, et, _ in self._elgroups]
        return "Domain over {} ".format(over)

    def plot_mesh(self, ax, **kwargs):
        """ Plot the mesh of the definition domain of the field or the mesh
        of a definition domain directly.

        Inputs:
            field_or_domain: the npstruct object or a domain object
            ax: an axis where to plot

            kwargs: pmap - a function to map node points
                    unit_scale - applied to the values before plotting
                                 e.g. 1000 to plot in millimeters
                    colors - a dict of colors for each physical group
            
            Rest of kwargs are passed directly to matplotlib triplot function.
            """

        pmap = kwargs.pop('pmap', None)
        uscale = kwargs.pop('unit_scale', 1)
        colors = kwargs.pop('colors', None)
                
        out = []

        for dimtag, _, eg in self:
            if pmap is None:
                nodes = eg.lnode_coords
            else:
                nodes = pmap(eg.lnode_coords)

            if colors is not None:
                kwargs['color'] = colors[dimtag]

            out.append(ax.triplot(uscale*nodes[:,0], uscale*nodes[:,1],
                                  eg.lnop, **kwargs))

        return out


    def plot_contour(self, ax, F, **kwargs):
        """ A contour plot of a scalar field in no points """
        kwargs['width'] = kwargs.get('width', 0.0015)
        pmap = kwargs.pop('pmap', None)
        uscale = kwargs.pop('unit_scale', 1)

        offset = kwargs.pop('offset', 0)
        for eg in self.elementgroups():

            if pmap is None:
                node_coords = eg.lnode_coords
            else:
                node_coords = pmap(eg.lnode_coords)

            x = uscale*node_coords[:,0]
            y = uscale*node_coords[:,1]
            tri = eg.lnop
            z = F[eg.global_nodeids+offset,...]

            ax.tricontour(x, y, tri, z,**kwargs)


    def plot_contourf(self, ax, F, **kwargs):
        """ A contour plot of a scalar field in no points """
        kwargs['width'] = kwargs.get('width', 0.0015)
        pmap = kwargs.pop('pmap', None)
        uscale = kwargs.pop('unit_scale', 1)

        out = []

        offset = kwargs.pop('offset', 0)
        for eg in self.elementgroups():

            if pmap is None:
                node_coords = eg.lnode_coords
            else:
                node_coords = pmap(eg.lnode_coords)

            x = uscale*node_coords[:,0]
            y = uscale*node_coords[:,1]
            tri = eg.lnop
            z = F[eg.global_nodeids+offset,...]

            out.append(ax.tricontourf(x, y, tri, z,**kwargs))

        return out

    def jacobians(self, intdeg, columns=None):
        """ Construct an NPStruct of isoparametric jacobian matrices
        """
        out = {}
        for eg in self.elementgroups():
            try:
                # old way
                # will be removed soon
                intp, intw = eg.get_integration_points(intdeg)
                J = eg.get_jacobians_isop(intp, columns)
                out[eg] = J
            except AttributeError:
                #gmsh backend
                integ = eg.integration_string(intdeg)
                jac, _ = eg.jacobians(integ, columns)
                out[eg] = jac

        return NPStruct(out, self, intdeg)


    def det_jacobians(self, intdeg, columns=None):
        """ Construct an NPStruct of isoparametric jacobian matrice determinants
        """
        out = {}
        for eg in self.elementgroups():
            try:
                # old way
                # will be removed soon
                intp, intw = eg.get_integration_points(intdeg)
                detJ = eg.get_detj_isop(intp, columns)
                out[eg] = detJ
            except AttributeError:
                #gmsh backend
                integ = eg.integration_string(intdeg)
                _, detj = eg.jacobians(integ)
                # expand dimensions
                out[eg] = detj[..., None, None]

        return NPStruct(out, self, intdeg)


    def tensor_field(self, intdeg, field_fun):
        out = {}
        for dim_tag, enttag, eg in self:
            try:
                intp, intw = eg.get_integration_points(intdeg)
                N = eg.eval_basis(sbasis, intp)
                points = (N @ eg.lnop_coords).squeeze(2)
                out[eg] = field_fun(dim_tag, enttag, points)

            except AttributeError:
                #gmsh backend
                integ = eg.integration_string(intdeg)
                points = eg.integration_points_xyz(integ)
                out[eg] = field_fun(dim_tag, enttag, points)

        return NPStruct(out, self, intdeg)

    def basis(self, intdeg, basis=None):
        """ Evaluate basis functions to npstruct form
        Inputs:
            intdeg: integration degree
            basis: basis type
        Output:
            npstruct containing basis function array representations with
            following axes:
                (elements=1, intpoints, components=1, dofs)
        """
        data = {}
        for eg in self.elementgroups():
            try:
                # old way
                # will be removed soon
                if(basis is None):
                    basis = sbasis
                intp, _ = eg.get_integration_points(intdeg)
                basf = eg.eval_basis(sbasis, intp)
                data[eg] = basf
            except AttributeError:
                #gmsh backend
                _, _, basf = eg.eval_basis(intdeg)
                # shape (intp, elements, dim, dof)
                data[eg] = basf[:, None, ...]

        return NPStruct(data, self, intdeg)

    def d_basis(self, intdeg, basis=None):
        """ Evaluate derivatives of basis functions to npstruct form
        Inputs:
            intdeg: integration degree
            basis: basis type
        Output:
            npstruct containing basis function array representations with
            following axes:
                (elements=1, intpoints, components=3/2/1, dofs)
        """
        data = {}
        for eg in self.elementgroups():
            try:
                # old way
                # will be removed soon
                if(basis is None):
                    basis = sbasis
                intp, _ = eg.get_integration_points(intdeg)
                basf = eg.eval_d_basis(sbasis, intp)
                data[eg] = basf
            except:
                # gmsh backend
                _, _, basf = eg.eval_d_basis(intdeg)
                # shape (intp, elements, dim, dof)
                data[eg] = basf[:, None, ...]
        return NPStruct(data, self, intdeg)

    def field_in_elements(self, f, intdeg=1, offset=0):
        """ Distribute the nodal field f locally to degrees of freedom
        in the elements.

        Parameter f must have the shape (nodes, ijk...)
        """
        data = {}
        for eg in self.elementgroups():
            gnop = eg.gnop+offset
            # this should result in (intp=1, elements, dofs, ijk...)
            f_vals = f[None,gnop,...]

            data[eg] = f_vals

        return NPStruct(data, self, intdeg)

    def initializer(self, initfun, intdeg):
        """ Initialize an NPStruct with a initializer function.
        Initfun must be of type
            (dim_tag, enttag, elementgroup, intdeg) -> nparray of correct shape
        """
        data = {}
        for dim_tag, enttag, eg in self:
            data[eg] = initfun(dim_tag, enttag, eg, intdeg)

        return NPStruct(data, self, intdeg)

#    def nodes_in(self):
#        """ Return a np.array containing all unique global node
#        labels in domain """
#        ldofs = np.unique(np.concatenate([eg.global_nodeids
#                                      for eg in self.elementgroups()]))
#        return ldofs
