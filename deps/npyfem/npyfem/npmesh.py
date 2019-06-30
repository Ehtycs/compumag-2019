import numpy as np
from functools import partial
from .element_group_factory import element_factory_gmsh, element_factory_mio
from .element_group import ElementGroupGmshProxy, t_tags

from .domain import Domain

from collections import namedtuple

# type for physical group identifier, gmsh uses a tuple of dimension
# and physical group tag
DimTag = namedtuple('DimTag', 'dim tag')

# Helper functions

def _iterate_physical_groups(model):
    """ Generator which yields a tuple
    (group_dim, group_tag, group_name)
    """
    return ((gdim, gtag, model.getPhysicalName(gdim, gtag))
            for gdim, gtag in model.getPhysicalGroups())

def _iterate_ents_etypes(model, gdim, gtag):
    return ((enttag, etype)
            for enttag in model.getEntitiesForPhysicalGroup(gdim,gtag)
            for etype in model.mesh.getElementTypes(gdim, enttag))


def from_gmsh(*args, **kwargs):

    backend = kwargs.pop('backend', 'npyfem')

    if(backend == 'npyfem'):
        return NPMeshGmsh(*args)
    elif(backend == 'gmsh'):
        return NPMeshGmshProxy(*args)
    else:
        ValueError("Unknown backend: {}".format(backend))


def from_mio(*args):
    return NPMeshNpyfem(*args)


class NPMesh():

    """ Common parts for all NPMesh instances """

    def get_global_nodeids(self, phystags=None):
        """ For backwards compatibility... """
        return self.global_nodes_in(phystags)


    def global_nodes_in(self, dim_tags=None):
        """ Returns a 1D np.array of node id:s included in physical
        domains given in 'phystags'. This concatenates the global_nodeids
        properties of elementgroups in the order of phystags
        """
        iterator = self.iterate_physical_domains(dim_tags)

        return np.unique(np.concatenate([eg.global_nodeids
                               for _,_,eg in iterator], axis=0))


    def define_domain(self, physical_tags):
        """ Returns an object which defines a domain. This can be given to
        functions in npstruct module. """

        return Domain(self, self.iterate_physical_domains(physical_tags))


#    def get_phys(self, phystag):
#        """ Get element group based on physical tag. If one group,
#        it will be returned as is. Otherwise a dict with element types
#        is returned. """
#        physg = self.physical_groups[phystag]
#        if len(physg) > 1:
#            return physg
#        else:
#            return next(iter(physg.values()))



class NPMeshNpyfem(NPMesh):
    """ A Mesh class which uses numpy magic inside.
        - Contains all information related to the mesh."""

    ndofs = None

    def __init__(self, mio_mesh):
        points = mio_mesh.points
        cells = mio_mesh.cells
        point_data = mio_mesh.point_data
        cell_data = mio_mesh.cell_data
        field_data = mio_mesh.field_data

        self.ndofs = points.shape[0]
        self.nodes = points

        builder = partial(element_factory_mio, points)

        # potential way to make the things as they used to be:
        # better way to let elementgroups deal with physical domains
        # get all physical region numbers present in the mesh accompanied with
        # the element type
        tagsntypes = ((tag,typ)
                      for typ in cells.keys()
                      for tag in np.unique(cell_data[typ]['gmsh:physical']))

        self.physical_groups = {}
        self.elementgroups = []

        for egid, (tag, typ) in enumerate(tagsntypes):
            if tag not in self.physical_groups.keys():
                self.physical_groups[tag] = {}

            # If I understand this correctly this should never happen
            assert typ not in self.physical_groups[tag].keys()

            # feed correct type and slice of elements array
            # based on the physical domain number
            elements = cells[typ][cell_data[typ]['gmsh:physical'] == tag]
            eg = builder(typ, egid, elements)
            self.physical_groups[tag][typ] = eg
            self.elementgroups.append(eg)

    def iterate_physical_domains(self, phystags=None):
        """ Iterates over elementgroups which belong to given physical
        domains.
        Input: phystags: iterable of tag numbers
        Output: generator object of tuples (physical tag, elementgroup) """

        # if none given, iterate all
        if(phystags == None):
            phystags = self.physical_groups.keys()

        for p in phystags:
            for typ in self.physical_groups[p].keys():
                # Because the builtin version throws entity tags away
                # insert None to enttag to remain consistent with the
                # amount of items to unpack
                yield  (p, None, self.physical_groups[p][typ])
        return



class NPMeshGmshProxy(NPMesh):
    """ Contains all mesh related stuff, e.g. element groups and their
    structure.

    Builds elementgroups which keep a dependency to GMSH and all basis
    functions etc. are queried from GMSH when needed.

    Due to gmsh being a global lib with it own global state this is
    absolutely definitely not thread safe at all.

    """

    def __init__(self, gmsh):
        self.gmsh = gmsh

        model = self.gmsh.model

        # First output is nodeTags, in this case it's redundant information
        _, coord, _ = gmsh.model.mesh.getNodes(-1,-1)
        # coord is a flat array 3*number of nodes

        # TODO: We should not query this information before we need it?
        # -> implement accessors!
        self.nodes = np.reshape(coord, (-1,3))
        self.ndofs = self.nodes.shape[0]

        # phys[(dim,grouptag)][entitytag][elementtype] = elementgroup
        phys = {}

        self.physicals_name2tag = {}
        self.physicals_tag2name = {}

        for gdim, gtag, gname in _iterate_physical_groups(model):
            # save the physical groups and their names in two way mapping
            self.physicals_name2tag[gname] = gtag
            self.physicals_tag2name[gtag] = gname
            # get the geometrical entities belonging to this group
            for enttag, etype in _iterate_ents_etypes(model, gdim, gtag):
                eg = ElementGroupGmshProxy(gmsh, gdim, gtag, gname, enttag,
                                           etype)
                # insert eg to this deeeep dict structure
                dimtag = DimTag(gdim, gtag)
                if(dimtag not in phys):
                    phys[dimtag] = {}
                if(enttag not in phys[dimtag]):
                    phys[dimtag][enttag] = {}

                assert etype not in phys[dimtag][enttag]

                phys[dimtag][enttag][etype] = eg

        self.physical_groups = phys

    def iterate_physical_domains(self, dim_tags=None):
        """ Iterates over elementgroups which belong to given physical groups.
        Input: phystags: iterable of tag numbers, group names
        Output: generator object of tuples
                (dim, group tag, entity tag, elementgroup) """

        if(dim_tags == None):
            dim_tags = self.physical_groups.keys()

        for dt in dim_tags:
            entities = self.physical_groups[dt]
            for enttag, entity in entities.items():
                for _, eg in entity.items():
                    # no need for element type, it's in eg
                    yield (dt, enttag, eg)


class NPMeshGmsh(NPMeshGmshProxy):
    """ Contains all mesh related stuff, e.g. element groups and their
    structure.

    Builds old school elementgroups which don't depend on GMSH after creation.
    """
    def __init__(self, gmsh):
        model = gmsh.model
        # First output is nodeTags, in this case it's redundant information
        _, coord, _ = gmsh.model.mesh.getNodes(-1,-1)
        # coord is a flat array 3*number of nodes

        self.nodes = np.reshape(coord, (-1,3))
        self.ndofs = self.nodes.shape[0]

        # phys[(dim,grouptag)][entitytag][elementtype] = elementgroup
        phys = {}

        self.physicals_name2tag = {}
        self.physicals_tag2name = {}

        for gdim, gtag, gname in _iterate_physical_groups(model):
            # save the physical groups and their names in two way mapping
            self.physicals_name2tag[gname] = gtag
            self.physicals_tag2name[gtag] = gname
            # get the geometrical entities belonging to this group
            for enttag, etype in _iterate_ents_etypes(model, gdim, gtag):

                eg = element_factory_gmsh(gmsh, self.nodes, etype, gdim,
                                      gtag, enttag)
                dimtag = DimTag(gdim, gtag)
                # insert eg to this deeeep dict structure
                if(dimtag not in phys):
                    phys[dimtag] = {}
                if(enttag not in phys[dimtag]):
                    phys[dimtag][enttag] = {}

                assert etype not in phys[dimtag][enttag]

                phys[dimtag][enttag][etype] = eg

        self.physical_groups = phys
