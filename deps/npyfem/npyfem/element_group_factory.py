""" Element registry, add the number corresponding to element and the
constructor of element here later. The numbers are from gmsh ascii file
format specification """

import numpy as np
from .element_group import t_uint, t_tags
from .mesh_exceptions import ElementNotImplementedExeption

from .points import Points
from .tetrahedras import Tetrahedras4
from .lines import Lines2, Lines3
from .triangles import Triangles3, Triangles6
from .quadrangles import Quadrangles4, Quadrangles9

"""
Element registry, add the number corresponding to element and the constructor
of element here later. The strings are the identifiers meshio uses for elements
"""
element_registry_mio = {
    'vertex': Points,
    'line': Lines2,
    'line3': Lines3,
    'triangle': Triangles3,
    'triangle6': Triangles6,
    'quad': Quadrangles4,
    'quad9': Quadrangles9,
    'tetra': Tetrahedras4,
}

def element_factory_mio(nodes, etype, egid, gnop):
    """ Factory function which constructs and returns an instance of correct
    ElementGroup subclass for an element type and list of elements """
    try:
        # list of global node id:s belonging to the elements of this group
        global_nodeids = np.unique(gnop)
        # node coordinates in traditional format (nodes x coordinates)
        node_coords = nodes[global_nodeids,:]
        # node coordinates in gnop format (elements x nodel x coordinates)
        nop_coords = nodes[gnop,:]

        return element_registry_mio[etype](egid, global_nodeids, node_coords,
                                       gnop, nop_coords, None)
    except ValueError:
        raise ElementNotImplementedExeption("Element of type "+str(etype)+
                                            " not implemented")

element_registry_gmsh = {
    15: Points,
    1: Lines2,
    8: Lines3,
    2: Triangles3,
    9: Triangles6,
    3: Quadrangles4,
    10: Quadrangles9,
    4: Tetrahedras4,
}

def element_factory_gmsh(gmsh, nodes, element_type, group_dim, group_tag,
                         entity_tag):
    """ Factory function which constructs and returns an instance of correct
    ElementGroup subclass for an element type and list of elements """
    try:
        etags, node_tags = gmsh.model.mesh.getElementsByType(element_type,
                                                             entity_tag)
        num_elements = len(etags)

        # Gmsh starts numbering of nodes and elements from 1, hence -1
        gnop = np.array(node_tags,
                             dtype=t_tags).reshape((num_elements,-1))-1
        global_nodeids = np.unique(np.array(node_tags, dtype=t_tags))-1
        element_tags = np.array(etags, dtype=t_tags)-1

        # node coordinates in traditional format (nodes x coordinates)
        lnode_coords = nodes[global_nodeids,:]
        # node coordinates in gnop format (elements x nodel x coordinates)
        nop_coords = nodes[gnop,:]

        builder = element_registry_gmsh[element_type]

        return builder(entity_tag, global_nodeids, lnode_coords, gnop, 
                       nop_coords, element_tags)
    except ValueError:
        raise ElementNotImplementedExeption("Element type {} not "
                                            "implemented".format(element_type))
