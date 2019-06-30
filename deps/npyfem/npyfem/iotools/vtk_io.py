from pyevtk.hl import unstructuredGridToVTK
import numpy as np
from itertools import chain
from pyevtk.vtk import VtkGroup, VtkTriangle, VtkLine, VtkQuad
from npyfem.mesh.lines import Lines
from npyfem.mesh.triangles import Triangles
from npyfem.mesh.quadrangles import Quadrangles

eg_cases = [(Lines, VtkLine), 
            (Triangles, VtkTriangle), 
            (Quadrangles, VtkQuad)]

gnop_cases = { 2: VtkLine, 3: VtkTriangle}

def eg_to_vtk_tid(eg):
    """ Maps element group instance to Vtk type id """
    # this would be a huge elseif
    for egtyp, vtktyp in eg_cases:
        if(isinstance(eg, egtyp)):
            return vtktyp.tid
    else:
        raise TypeError("Given parameter doesn't have a Vtk type ID")

def gnop_to_vtk_tid(gnop):
    """ Maps a nop table to vtk type id just by checking how many nodes 
    the elements consist of.
    """
    # no way of knowing as of now about fancier element types so
    # assuming linear elements
    nodes_in_element = gnop.shape[1]
    typ = gnop_cases.get(nodes_in_element, None)
    if typ is not None:
        return typ.tid
    else:
        raise TypeError("Given parameter with {} nodes "
                        "doesn't have a Vtk type ID".format(nodes_in_element))

def connectivity(gnops):
    """ Create a connectivity vector which vtk requires from a nop table 
    Connectivity vector is kind of a flattened version of nop table.
    Outputs np.array of (elements * nodes_in_elements, )"""
    # calculate the size of the conn array and create the conn array
    size = np.sum([gnop.shape[1]*gnop.shape[0]
                   for gnop in gnops])
    conn = np.fromiter(chain.from_iterable((gnop.flatten()
                                            for gnop in gnops)), int, size)
    return conn

def offsets(gnops):
    """ Calculate the offset vector required by vtk which tells where each
    element starts and how many nodes it has in the connectivity vector 
    (the offset).
    outputs np.array of dimension (elements,)"""
    # Define offsets of last vertex of each element
    gen = (np.full(gnop.shape[0], gnop.shape[1]) for gnop in gnops)
    offset = np.cumsum(np.concatenate(list(gen)))

    return offset

def celltypes_from_elementgroups(elemgroups):
    """ Create cell type vector from elementgroups 
    outputs np.array of (elements,) """
    # Define cell types
    genct = (np.full(eg.gnop.shape[0], eg_to_vtk_tid(eg)) for eg in elemgroups)
    ctype = np.concatenate(list(genct))

    return ctype

def celltypes_from_gnop(gnop):
    """ Create cell type vector from a nop table """
    # Define cell types
    ctype = np.full(gnop.shape[0], gnop_to_vtk_tid(gnop))

    return ctype

def extract_nodes(nodearray):
    """ Return a tuple of contiguous arrays where nodal coordinates are """
    x = np.asfortranarray(nodearray[:,0])
    y = np.asfortranarray(nodearray[:,1])
    z = np.asfortranarray(nodearray[:,2])

    return (x,y,z)

def pack_field(f):
    """ Return a contiguous array of a field. 
    If input is scalar field, returns the field array as contiguous array.
    If input is 3d vector field, returns a tuple (fx,fy,fz) where each 
    component is a contiguous array.
    """
    if(f.ndim == 1):
        return np.ascontiguousarray(f[:])
    elif(f.shape[1]== 1):
        return np.ascontiguousarray(f[:,0])
    elif(f.shape[1] == 3):
        return (np.ascontiguousarray(f[:,0]),
                np.ascontiguousarray(f[:,1]),
                np.ascontiguousarray(f[:,2]))
    else:
        raise RuntimeError("Field needs to be either "
                           "scalar or 3 dim vector")


def time_fields_write_data(fname, x, y, z, time, conn, offs, ctypes, **kwargs):
    """ Write data to vtk files """
    pfields = kwargs.get('pointdata', None)
    cfields = kwargs.get('celldata', None)

    if(pfields is None and cfields is None):
        raise RuntimeError("No field data given, give either "
                           "a point field or cell field or both.")

    group = VtkGroup(fname+"/"+fname)

    for i, t in enumerate(time):
        if pfields is None:
            pdata = None
        else:
            pdata = {k: pack_field(f[i,...]) for k,f in pfields.items()}

        if cfields is None:
            cdata = None
        else:
            cdata = {k: pack_field(f[i,...]) for k,f in cfields.items()}

        unstructuredGridToVTK(fname+"/step_"+str(i), x, y, z,
                              connectivity = conn, offsets = offs,
                              cell_types = ctypes, pointData = pdata,
                              cellData = cdata)
        group.addFile(filepath = fname+"/step_"+str(i)+".vtu", sim_time = t)

    group.save()


def export_time_fields(msh, fname, time, **kwargs):
    """ Export time dependent fields by extracting the information from 
    NPMesh object """
    x, y, z = extract_nodes(msh.nodes)

    phys = kwargs.get('phys', None)
    egs = [eg for _, eg in msh.iterate_physical_domains(phys)]
    gnops = [eg.gnop for eg in egs]
    conn = connectivity(gnops)
    offs = offsets(gnops)
    ctypes = celltypes_from_elementgroups(egs)

    time_fields_write_data(fname, x, y, z, time, conn, offs, ctypes, **kwargs)

    
    
def export_time_fields_nomsh(nodes, gnop, fname, time, **kwargs):
    """ Export time dependent fields by using provided nodes and gnop arrays
    """
    x, y, z = extract_nodes(nodes)

    conn = connectivity([gnop])
    offs = offsets([gnop])
    ctypes = celltypes_from_gnop(gnop)
    
    time_fields_write_data(fname, x, y, z, time, conn, offs, ctypes, **kwargs)

