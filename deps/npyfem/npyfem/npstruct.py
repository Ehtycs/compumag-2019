import numpy as np
import scipy.sparse as sps
from .basis_functions import scalarbasis as sbasis
from .element_group import t_tags

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
            
            
def _row_numbering(arr_unass, nop_table):
    """ Standard row numbering function used in NPStruct.assemble.
    
    Returns the ROW index vector fed to CSR_matrix constructor """

    index_shape = arr_unass.shape[-2:]
    row_shape = index_shape[0]

    if row_shape == 1:
        # assembling wide or tall "residual vector"
        index_i = np.zeros(arr_unass.shape, dtype=t_tags)
    else:
        # assembling "stiffness matrix"
        index_i = nop_table.transpose((1, 2, 0))
        index_i = index_i*np.ones(index_shape, dtype=t_tags)

    return index_i

def _column_numbering(arr_unass, nop_table):
    """ Standard column numbering function used in NPStruct.assemble.
    
    Returns the COLUMN index vector fed to CSR_matrix constructor """

    index_shape = arr_unass.shape[-2:]
    column_shape = index_shape[1]

    if column_shape == 1:
        # assembling tall "residual vector"
        index_j = np.zeros(arr_unass.shape, dtype=t_tags)
    else:
        # assembling "stiffness matrix"
        index_j = nop_table.transpose((1, 0, 2))
        index_j = index_j*np.ones(index_shape, dtype=t_tags)
        
    return index_j


class NPStruct(object):
    """ NPStruct class implements a dictionary of numpy tensors which
    can be multiplied together. This enables the user to focus on
    computing global (on mesh level) quantities with single expressions.

    The keys of the dictionary are element groups and data is a numpy
    multidomensional array representing a field in that
    elementgroup (e.g. basis functions).

    IT IS ASSUMED THAT THE INTERNALS ARE KEPT UNMUTATED!!

    AVOID MUTATING THINGS UNLESS YOU KNOW WHAT YOU ARE DOING!!

    NPStruct recycles naively what it can so the mutations will spread
    uncontrollably.
    """

    def __init__(self, data, domain, intdeg=1):
        self.data = data
        self.domain = domain
        self.intdeg = intdeg

    @property
    def shape(self):
        return [f.shape for f in self.data.values()]

    def elementgroups(self):
        return (k for k in self.data.keys())

    def __repr__(self):
        return "".join(["NPStruct object: \n",
                "Shape: {}\n".format(self.shape),
                "Defined in: {}\n".format(self.domain.__str__())])

    def __iter__(self):
        """ Iterate over the data attribute as if it would be a dict
        Shortens the for expressions from field.data.items() to just field """
        return ((k,v) for k,v in self.data.items())

    def _shallow_partial_copy(self, **kwargs):
        """ Create a shallow copy of the object so that
        kwargs can be given to overwrite the attributes. Otherwise they
        are taken from self. So a new NPStruct will be returned but
        same objects are still used in it. """
        data = kwargs.pop('data', self.data)
        intdeg = kwargs.pop('intdeg', self.intdeg)
        domain = kwargs.pop('domain', self.domain)
        # kwargs has to be empty at this point
        assert(not kwargs)
        return NPStruct(data, domain, intdeg)


    def __matmul__(self, obj):
        """ Left matrix multiplication """
        if(isinstance(obj, NPStruct)):
            data = {k: f1 @ f2
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: f1 @ obj
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __rmatmul__(self, obj):
        """ Right matrix multiplication """
        if(isinstance(obj, NPStruct)):
            data = {k: f2 @ f1
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: obj @ f1
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __mul__(self, obj):
        """ Left multiplication """
        if(isinstance(obj, NPStruct)):
            data = {k: f1 * f2
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: f1 * obj
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __rmul__(self, obj):
        """ Right multiplication """
        if(isinstance(obj, NPStruct)):
            data = {k: f2 * f1
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: obj * f1
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __add__(self, obj):
        if(isinstance(obj, NPStruct)):
            data = {k: f1 + f2
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: obj + f1
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __radd__(self, obj):
        if(isinstance(obj, NPStruct)):
            data = {k: f2 + f1
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: f1 + obj
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __sub__(self, obj):
        if(isinstance(obj, NPStruct)):
            data = {k: f1 - f2
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: obj - f1
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __rsub__(self, obj):
        if(isinstance(obj, NPStruct)):
            data = {k: f2 - f1
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: f1 - obj
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __truediv__(self, obj):
        if(isinstance(obj, NPStruct)):
            data = {k: np.divide(f1,f2)
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: np.divide(f1, obj)
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def __rtruediv__(self, obj):
        if(isinstance(obj, NPStruct)):
            data = {k: np.divide(f2,f1)
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: np.divide(obj, f1)
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def get_T(self):
        """ Flip two rightmost axes """
        data = {k: ent.swapaxes(-1,-2)
                for k, ent in self.data.items()}

        return self._shallow_partial_copy(data=data)

    T = property(get_T)

    def transpose(self, axes):
        #axes_full = [0,1]+(np.array(axes)+2).tolist()
        axes_full = [0,1,2]+(np.array(axes)+3).tolist()
        data = {k: ent.transpose(axes_full)
                for k, ent in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def unary_map(self, oper):
        """ Applies a mapping oper for each np array of the struct """
        newdata = {eg: oper(field) for eg, field in self.data.items()}
        return self._shallow_partial_copy(data=newdata)

    def binary_map(self, obj, oper):
        """ Applies a mapping oper for each matched pair of this and
        obj's arrays.

        Copies all attributes from "self" to the resulting NPStruct"""
        newdata = {eg: oper(f1, f2)
            for eg, f1, f2 in _dictmatch(self.data, obj.data)}
        return self._shallow_partial_copy(data=newdata)


    def get_nth(self, num):
        """for debugging purposes"""
        return list(self.data.values())[num]

    def det(self):
        data = {eg: np.linalg.det(f)[...,None,None]
                for eg, f in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def max(self):
        return max([np.max(F) for F in self.data.values()])

    def min(self):
        return min([np.min(F) for F in self.data.values()])

    def norm(self, axis=-1):
        return self.unary_map(lambda x: np.linalg.norm(x, axis=axis))

    def mean(self, axis=None):
        return np.mean([np.mean(F) for F in self.data.values()])

    def backslash(self, obj):
        """ A "backslash" operator, aka self \ obj which solves
        an equation system np.linalg.solve(obj, self) and outputs the
        solution as npstruct
        """
        if(isinstance(obj, NPStruct)):
            data = {k: np.linalg.solve(f1, f2)
                    for k, f1, f2 in _dictmatch(self.data, obj.data)}
        else:
            data = {k: np.linalg.solve(f1, obj)
                    for k, f1 in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def squeeze(self, axis=None):
        data = {eg: np.squeeze(f, axis)
                for eg, f in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def real(self):
        return self.unary_map(np.real)

    def imag(self):
        return self.unary_map(np.imag)

    def expand_dims(self, axis):
        data = {eg: np.expand_dims(f, axis)
                for eg, f in self.data.items()}
        return self._shallow_partial_copy(data=data)

    def nodes_in(self):
        """ Return a np.array containing all unique node labels in domain """
        ldofs = np.unique(np.concatenate([eg.global_nodeids
                                      for eg in self.elementgroups()]))
        return ldofs


    def plot_mesh(self, ax, **kwargs):
        """ Plot the mesh of the definition domain of the field or the mesh
        of a definition domain directly.

        Inputs:
            field_or_domain: the npstruct object or a domain object
            ax: an axis where to plot

            kwargs: pmap - a function to map node points
                    unit_scale - applied to the values before plotting
                                 e.g. 1000 to plot in millimeters
            """

        pmap = kwargs.pop('pmap', None)
        uscale = kwargs.pop('unit_scale', 1)
        out = []

        for eg in self.elementgroups():
            if pmap is None:
                nodes = eg.lnode_coords
            else:
                nodes = pmap(eg.lnode_coords)

            out.append(ax.triplot(uscale*nodes[:,0], uscale*nodes[:,1],
                                  eg.lnop, **kwargs))

        return out

    def plot_colormap(self, ax, pmap=None, **kwargs):
        """ A tricontour plot of a scalar field in an elementgroup.

        Outputs all colormap objects created from calls to tripcolor """
        out = []
        uscale = kwargs.pop('unit_scale', 1)


        if('vmin' not in kwargs):
            kwargs['vmin'] = self.min()

        if('vmax' not in kwargs):
            kwargs['vmax'] = self.max()

        for eg, F in self.data.items():

            if pmap is None:
                node_coords = eg.lnode_coords
            else:
                node_coords = pmap(eg.lnode_coords)

            xs = node_coords[:,0]
            ys = node_coords[:,1]

            out.append(ax.tripcolor(uscale*xs, uscale*ys,
                                    eg.lnop, F[0,:,0,0], **kwargs))

        return out

    def plot_quiver(self, ax, **kwargs):
        """ A quiver plot of a vector field in center points """

        kwargs['width'] = kwargs.get('width', 0.0015)
        pmap = kwargs.pop('pmap', None)
        axis = kwargs.pop('axis', (0,1))
        uscale = kwargs.pop('unit_scale', 1)

        for eg, F in self.data.items():

            if pmap is None:
                node_coords = eg.lnode_coords
            else:
                node_coords = pmap(eg.lnode_coords)

            nop_coords = node_coords[eg.lnop,:]

            intp,_ = eg.get_integration_points(self.intdeg)
            N = eg.eval_basis(sbasis, intp)

            pnts = (N @ nop_coords).squeeze()

            ax.quiver(uscale*pnts[:,0], uscale*pnts[:,1],
                         F[0,:,axis[0],0], F[0,:,axis[1],0], **kwargs)

    def integrate(self):
        """ Integrate an npstruct object. Field must be representing a integrable
        field (must be defined in integration points of elements).

        Returns an npstruct with one less dimensions.
        """
        out = {}
        for eg, f in self.data.items():
            try:
                _, intw = eg.get_integration_points(self.intdeg)
                out[eg] = np.tensordot(f, intw, axes=((0),(0)))
            except AttributeError:
                _, intw, _ = eg.eval_basis(self.intdeg)
                out[eg] = np.tensordot(f, intw, axes=((0),(0)))
        return self._shallow_partial_copy(data=out)

    def assemble(self, **kwargs):
        """ Assembles the npstruct object to a sparse csr matrix. 
        
        The object must represent either
        local stiffness matrices (array of (..., dofs, dofs)) or
        local residual vectors (array of (..., 1, dofs) or (..., dofs, 1))
        
        If you provide kwargs row_numbering, col_numbering you probably already
        know what you are doing...

        Inputs:
            None
        Kwargs: 
            shape :: (Int, Int): 
                The shape of the resulting global system. Defaults to number 
                of nodes in the mesh if local is False, 
                if local is True defaults to the number of nodes in the domain.
            local :: Bool:  
                True -> only consider dofs in the domain and use the 
                domainwise local numbering for the dofs
                False (default) -> use meshwise global numbering for 
                dofs
            row_numbering :: f(arr_unass, nop_table) -> (index_i, rows)
                Function which should take an unassembled array of 
                local stiffness matrices and the connectivity array and 
                return 
        Output:
            a scipy csr_matrix
        """

        local = kwargs.pop('local', False)
        shape = kwargs.pop('shape', None)
        
        row_numbering = kwargs.pop('row_numbering', _row_numbering)
        column_numbering = kwargs.pop('col_numbering', _column_numbering)
        
        if(shape is None):
            if local:
                ndofs = self.domain.ndofs
            else:
                ndofs = self.domain.mesh.ndofs
            rows = ndofs
            cols = ndofs
        else:
            rows, cols = shape

        gindex_i = []
        gindex_j = []
        garr = []

        for eg, arr_unass in self:

            if local:
                nop_table = self.domain.connectivity(eg)[None,...]
            else :
                nop_table = eg.gnop[None,...]

            index_i = row_numbering(arr_unass, nop_table)
            index_j = column_numbering(arr_unass, nop_table)
            
            index_shape = arr_unass.shape[-2:]
            
            if(index_shape[0] == 1):
                rows = 1
            if(index_shape[1] == 1):
                cols = 1

            gindex_i.append(index_i.reshape((-1,)))
            gindex_j.append(index_j.reshape((-1,)))
            garr.append(arr_unass.reshape((-1,)))

        gindex_i = np.concatenate(gindex_i)
        gindex_j = np.concatenate(gindex_j)
        garr = np.concatenate(garr)

        return sps.coo_matrix((garr,
                              (gindex_i, gindex_j)), shape=(rows, cols)).tocsr()
