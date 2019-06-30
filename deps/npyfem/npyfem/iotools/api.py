
"""
    INTERFACE DEFINITION (at least a try...):

    Each module must implement at least these functions:

    read(filename) -> nodes, cells, celldata, fielddata
    write(filename, nodes, cells, celldata, fielddata)

    nodes, cells and celldata according to meshio specifications:

    The coordinates of the nodes of the mesh
        nodes: 2d nparray (nodes, dim)

    The connectivity of elements distributed to different element types
        cells: {element type: gnop: 2d nparray (elements, nodel), ...}

    To which physical or geometrical domain each of the elements belong
        celldata: {'physical': {type: np.1darray, ...},
                   'geometrical': {element type: np.1darray, ...}}

    Fields associated with this mesh
    fielddata: {
        'time': nparray (timestep values), None or removed if static
        'node': {
            scalar_fieldn: nparray (time, nodes), ...
            vector_fieldn: nparray (time, nodes, n), ...
            tensor_fieldn: nparray (time, nodes, i, j, k,...) ...
        },
        'element': {
            scalar_field: nparray (time, elements), ...
            vector_field: nparray (time, elements, n), ...
            tensor_field: nparray (time, elements, i, j, k,...), ...
            ...
        }
    }

    "scalar_fieldn" etc. are field identifiers. For example "A", "B" or
    "current".

    In import/export interface, element data is associated with center points
    of elements. (At least for now. Huge thing to deal with different kinds
    of elements and different amounts of integration points to export them in
    combatible way...)

    If static solution, time axis is 1
"""
