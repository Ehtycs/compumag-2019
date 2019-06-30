import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

import npyfem
import npyfem.dirichletbc as dbc
from npyfem.basis_functions import scalarbasis as sbasis
from npyfem.domain import Domain

from collections import defaultdict, namedtuple

""" 
Make coupling boundary mapping using parametrization of curves/surfaces:
Let M be a curve or surface shared between two domain
Assumptions:
1. Two discretizations for M s.t. the corners are equal in both meshes
   e.g. the endpoints of a curve or the corners of a plane

   This makes the parametrized coordinates u=0/1, v=0/1 match on both 
   parametrized domains.

"""

# Cut information for a domain
UndoCut = namedtuple('UndoCut', 'real artificial')

def find_domain_simplices(nodes, domain, points):
    """ Searches for simplices where points belong inside a domain
    Calculates an extent for the elementgroups to eliminate candidates
    where the points can not lie.
    """
    
    out = defaultdict(list)
    
    rowmasktpl = np.ones((len(points),1), dtype=bool)

    #print("Total: {}".format(len(points)))
    
    eps = np.finfo(float).eps

    for eg in domain.elementgroups():
        dnop = domain.connectivity(eg)
        # This gumgum avoids giving hopeless points to 
        # find_simplices function 
        # Compute rough boundaries of eg, and drop points which are 
        # outside those
        nodesin = nodes[domain.global2local[eg.global_nodeids]]
        nmin = np.min(nodesin,axis=0)
        nmax = np.max(nodesin,axis=0)
        rowmask = rowmasktpl.copy()
        for minv, maxv in zip(nmin, nmax):
            rowmask *= np.logical_and(points > minv-eps, points < maxv+eps)
        
        # Keep up with the indices of the points
        indices = np.arange(0,len(points))[rowmask[:,0]]
        #print("   Narrowed to {}".format(len(indices)))
            
        pelmap = find_simplices(nodes, dnop, points[rowmask[:,0], :])
        for pind, elinds in pelmap:
            out[indices[pind]].append((eg, elinds))
        
    return dict(out)

def find_simplices(nodes, connectivity, points):
    """ Returns a list of lists which indicates
    to which elements a point belongs.
    
    Inputs:
        nodes - np.array of node coordinates (nodes, xyz) 
        connectivity - gnop array, (elements, nodel)
        points - np.array of points to find (points, xyz)
        
        xyz should be the parametric coordinates
        
    Outputs: 
        a list of lists = [[n1 element_indices], ...] 
        
    """
    
    ## TODO: create larger system to be fed to det once
    
    
    # If 1D domain, append extra dimension implicitely for ease of use
    if nodes.ndim == 1:
        nodes = nodes[...,None]   
    if points.ndim == 1:
        points = points[...,None]
    # Node coordinates for each simplex in (elements, nodel, xyz)
    ngnops = nodes[connectivity]

    # shape for building the concatenation array
    concshape = np.ones((*ngnops.shape[0:2], 1))
    # A numpy array of shape (elems, nodel, xyz1)
    # contains the nodes augemented by a column of 1:s
    #print(ngnops.shape, concshape.shape)
    elems_template = np.concatenate([ngnops, concshape], axis=-1)
    #print(np.linalg.det(elems_template))
    # num of nodes in a simplex
    nelems, nodel = connectivity.shape
    # Compute the orientations of the elements
    det_signs = np.sign(np.linalg.det(elems_template))
    # machine epsilon
    eps = np.finfo(float).eps
    point_el_map = []
    for indp, p in enumerate(points):
        # an array of shape (elems, 1, xyz1). Will be substituted to 
        # elems_template to form the determinant arrays 
        points_aug1 = np.concatenate([p,np.array([1])])
        points_template = np.ones((nelems, 1)) * points_aug1

        # Start with all elements in
        # each loop, drop all elements which are not possible out
        handle = np.arange(0, nelems)
        
        for i in range(0,nodel):
            # Exlude one node at a time and replace the node with 
            # the candidate point
            elems = elems_template[handle].copy()
            elems[:,i,:] = points_template[handle]
            # compute the determinants and apply "sign correction"
            dets = det_signs[handle]*np.linalg.det(elems)
            # If on the boudary of elements, determinant will be exactly
            # zero, so allow for negative values up to -machine epsilon
            handle = handle[dets > -eps]
    
        # now we should have only the elements associated with the point
        elind = handle.tolist()
        if elind:
        #print("Point({}): {} belongs to element(s) {}".format(indp, p, elind))
            point_el_map.append((indp, elind))
        
    return point_el_map



def cut_domain(domain, ldof):
    """ Duplicate a global node so that the domain is "cut" 
    from the place of that node. 
    Return a new element group in place of the modified one.
    
    # This is a bit ugly solution but ultimately this should not be needed
    # Parametrization should be already available.
    
    """
    
    #ldofs = npyfem.nodes_in(domain)
    nextgdof = max(domain.global_nodeids)+1
    #nextldof = len(domain.global_nodeids)
    gdof = domain.global_nodeids[ldof]
    first = [True]
    undo_cut = [UndoCut(gdof, nextgdof)]
    
    def replace_if(eg):
        # Replace the elementgroup where the global node gid is 
        # with a new one where the cutting node is inserted
        # UGLY: This just takes the last node of the last element
        # if it matches the cut node and and throws the new node in
        if(first[0] and gdof == eg.gnop[-1, -1]):
            global_nodeids = eg.global_nodeids.copy()
            global_nodeids[global_nodeids == gdof] = nextgdof
            gnop = eg.gnop.copy()
            gnop[gnop == gdof] = nextgdof
            # Construct the same object which we are dealing with
            # PROBABLY Lines2 object
            neweg = eg.__class__(eg.entity_tag, global_nodeids, 
                                 eg.lnode_coords, gnop, eg.lnop_coords,
                                 eg.element_tags)
            first[0] = False
            return neweg
        return eg
    
    # kind of like map over the structure
    newdomain = [(dimtag, entitytag, replace_if(eg)) 
                 for dimtag, entitytag, eg in domain]

    assert not first[0], "Cutting went south"

    return Domain(domain.mesh, newdomain), undo_cut


def make_mortar_assembler(nodepoints_uvw):
    """ Returns a function which can be used to compute the mortar map """
    
    def fun(bdry_domain):
        ## solve for parametric coordinates
        u, bdom, undo_cut = solve_parametric_coordinates(bdry_domain)
        M, ipgel = assemble_node_projection(u[:, None], bdom, 
                                            nodepoints_uvw, undo_cut)

        return M, u, ipgel

    return fun

def make_mortar_assembler_intpoint(intpoints_uvw, mortar_part, mortar_dampmat):
    """ Returns a function which can be used to compute the mortar map """
    
    def fun(bdry_domain):
        ## solve for parametric coordinates
        u, bdom, undo_cut = solve_parametric_coordinates(bdry_domain)
        
    #    info(("Integration points in the "
    #          "boundary: {}").format(len(integration_points_uvw)))
    
        nonmortar_part, ipgel = assemble_nonmortar_part(u[:,None], bdom, 
                                                        intpoints_uvw,
                                                        undo_cut)
        D = (nonmortar_part @ mortar_part).T
        M = spsolve(mortar_dampmat, D)
        
        return M, u, ipgel

    return fun

def solve_parametric_coordinates(bdry_domain, cut_node=0):
    """ Solves for parametric coordinates """ 
    
    bdom, undo_cut = cut_domain(bdry_domain, cut_node)
    bnodeids = npyfem.nodes_in(bdom)
    nbnodes = bnodeids.shape[0]
    #info("Degrees of freedom in the boundary {}".format(nbnodes))
    zero = cut_node
    one = len(bnodeids)-1
    
    dN = bdom.d_basis( 1)
    jac = bdom.jacobians(1)
    g = jac @ jac.T

    invg = g.unary_map(np.linalg.inv)

    detJ = g.det().unary_map(np.sqrt)
    K = ((dN.T @ invg @ dN) * detJ).integrate().assemble(local=True)
    
    O = dbc.zero_rowcols_mapping([zero, one], nbnodes)
    I = dbc.ones_to_diag_matrix([zero, one], nbnodes)
    
    Q = dbc.coupling(K, [zero,one], np.array([0,1]))
    
    K0 = O*K*O + I
    F0 = -Q
    F0[zero] = 0
    F0[one] = 1
    
    u = spsolve(K0, F0)
    
    return u, bdom, undo_cut

def assemble_mortar_part(domain, nodes_uvw, undo_cut=None):
    """ Build a matrix of shape (ip*elems , dofs) which contains 
    pre computed everything from the mortar (COIL) side.

    This is half what is needed to compute  
        int( (_) psi_i(p_k) ) = sum_{i=1}^N (_) psi_i(p_k) * det(J(p_k)) * w_k
    where psi is the master side basis function and (_) is the hole for 
    the slave side basis functions.     
    
    Input: 
        domain: domain object of the boundary
        nodes_uvw: parametric coordinates of nodes
        undo_cut: information about duplicate nodes
    Output: (C, itps)
        C: csr_matrix containing "master half" of the linear mapping
        intps: np.array containing the xyz coordinates of integration points
               on the boundary
        
    """
    
    if(nodes_uvw.ndim == 1):
        nodes_uvw = nodes_uvw[:,None]
    
    # We must use 2 order integration scheme since 
    # N1 * N2 is second order function
    N = domain.basis(2)
    # calculate detJ by sqrt(jac @ jac.T) because our isop. mapping is 
    # from R1 -> R3
    jac = domain.jacobians(2)
    detJ = (jac@jac.T).det().unary_map(np.sqrt)
    field = (N * detJ)
    bdofs = np.unique(np.concatenate([eg.global_nodeids 
                                      for eg in field.data.keys()]))
    
    # construct index vector for global->local map
    gnode2lnode = np.zeros((np.max(bdofs)+1,), dtype=int)
    for lind, gind in enumerate(bdofs):
        gnode2lnode[gind] = lind
    
    gindex_i = []
    gindex_j = []
    garr = []
    
    # increment counters for the element count
    elinc = 0
    pelinc = 0
    
    # store the xyz coordinates of integration points here
    ipcords = []
    
    if undo_cut is not None:
        degenerate_dofs = np.arange(0, len(gnode2lnode))
        real, art = np.array(undo_cut).T
        degenerate_dofs[art] = real

    # assemble a matrix of shape (intp*elems x nodel)
    for eg, F in field.data.items():
        intp, intw = eg.get_integration_points(field.intdeg)
        A = F * intw[:,None,None,None]
        print(A.shape)
        nintps = intw.shape[0]
        nel = F.shape[1]
        ndofs = F.shape[-1]
        
        # Calculate the coordinates of integration points 
        # in parametric coordinates
        lnop_coords = nodes_uvw[gnode2lnode[eg.gnop]]
        lN = eg.eval_basis(sbasis,intp)
        ipcords.append((lN@lnop_coords).squeeze(-2).swapaxes(0,1).reshape(nel*nintps,-1))
        
        # row indices are related to integration points so we need a 
        # huge index vector
        # [e1_ip1_dof1, e1_ip2_dof2, e2_ip1_dof1, ... ipn_ek_dofp]    
        
        elinc += nintps*nel         
        
        # gindi is now [e1_ip1_dof1, e1_ip1_dof2, e1_ip2_dof1 ...]
        gindi = np.ravel(np.tile(np.arange(pelinc,elinc)[None,:], ndofs))
        gindex_i.append(gindi)
    
        # column index is related to global dof so this one we take         
        # from gnop, but we need to repeat it "intpoints" times
        # we also need to map global indices to indices only in the boundary
        if undo_cut is not None:
            lnop = gnode2lnode[degenerate_dofs[eg.gnop]]
        else:
            lnop = gnode2lnode[eg.gnop]
            
        gnopinds = np.tile(np.ravel(lnop), nintps)
        gindex_j.append(gnopinds)
        garr.append(np.ravel(A))
        pelinc = elinc
    
    garr = np.concatenate(garr)
    gindex_i = np.concatenate(gindex_i)
    gindex_j = np.concatenate(gindex_j)
    ipcords = np.concatenate(ipcords, axis=0)
    C = sps.coo_matrix((garr, (gindex_i, gindex_j))).tocsr()
    
    return C, ipcords

def assemble_nonmortar_part(nodes, domain, intpoints_uvw, undo_cut=None):

    ipgel = find_domain_simplices(nodes, domain, intpoints_uvw)
    
    rhs_aug = np.array([1])[None,:]
    
    Nlist = []
    rowindlist = []
    colindlist = []
    
    if undo_cut is not None:
        degenerate_dofs = np.arange(0, len(domain.global2local))
        real, art = np.array(undo_cut).T
        degenerate_dofs[art] = real
    
    for pind, egels in ipgel.items():
        #print("Point {}\n".format(pind))
        for eg, elems in egels:
            #print("   Eg: {} \n   handling elements {}\n".format(eg,elems))
            nodel = eg.gnop.shape[1]
            nop = domain.global2local[eg.gnop[elems,:]]
            # Create an array of 1:s which is the partition of unity condition 
            concat = np.ones((len(nop),nodel,1))
            # LHS needs to be of shape (elements containing points, xyz+1, nodes):
            #[ [ [x1, x2 ...], [y1, y2 ...], ..., [1,1,...] ], ... ]
            lhs = np.concatenate([nodes[nop], concat], axis=-1).transpose(0,2,1)
            # RHS contains the point coordinates and 1
            # the extra equation comes from the partition of unity
            rhs = np.concatenate([intpoints_uvw[pind:pind+1], rhs_aug], axis=-1)
            N = np.linalg.solve(lhs, rhs)
            Nravel = N.ravel()
            Nlist.append(Nravel)
            colindlist.append(np.ones(Nravel.shape, dtype=int)*pind)
            if undo_cut is not None:  
                uncut_nop = domain.global2local[degenerate_dofs[eg.gnop[elems,:]]]
                rowindlist.append(uncut_nop.ravel())
            else:
                rowindlist.append(nop.ravel())
    
    Ns = np.concatenate(Nlist)
    colinds = np.concatenate(colindlist)
    rowinds = np.concatenate(rowindlist)

    return sps.coo_matrix((Ns, (rowinds, colinds))).tocsr(), ipgel

def assemble_node_projection(nodes, domain, nodepoints_uvw, undo_cut=None):
    """ Assembles a linear map M which maps a field given in nodes of "nodes"
    to nodes of "nodepoints_uvw".
    
    nodes - parametric coordinates of nonmortar side nodes (main)
    domain - domain object of the nonmortar boundary (main)
    nodepoints_uvw - parametric coordinates of nodes in mortar side (coil)
    undo_cut - information on artificial nodes created in cutting a domain
    
    returns: M, ipgel
    M - the projection matrix (csr) of shape 
        (len(nodepoints_uvw), len(nodes - degenerated))
    ipgel - the information in which element the nodepoints_uvw belongs to
            (given by find_domain_simplices...)
    """
    
    ipgel = find_domain_simplices(nodes, domain, nodepoints_uvw)
    
    # partition of unity row
    rhs_aug = np.array([1])[None,:]
    
    Nlist = []
    rowindlist = []
    colindlist = []
        
    if undo_cut is not None:
        degenerate_dofs = np.arange(0, len(domain.global2local))
        real, art = np.array(undo_cut).T
        degenerate_dofs[art] = real
    
    for pind, egels in ipgel.items():
        
        nbelongs_to_elements = sum([len(els) for _, els in egels])

        #print("Point {}\n".format(pind))
        for eg, elems in egels:
            #print("   Eg: {} \n   handling elements {}\n".format(eg,elems))
            nodel = eg.gnop.shape[1]
            nop = domain.global2local[eg.gnop[elems,:]]
            # Create an array of 1:s which is the partition of unity condition 
            concat = np.ones((len(nop),nodel,1))
            # LHS needs to be of shape (elements containing points, xyz+1, nodes):
            #[ [ [x1, x2 ...], [y1, y2 ...], ..., [1,1,...] ], ... ]
            lhs = np.concatenate([nodes[nop], concat], axis=-1).transpose(0,2,1)
            # RHS contains the point coordinates and 1
            # the extra equation comes from the partition of unity
            rhs = np.concatenate([nodepoints_uvw[pind:pind+1], rhs_aug], axis=-1)
            N = np.linalg.solve(lhs, rhs)
            Nravel = 1/nbelongs_to_elements*N.ravel()
            #Nravel = N.ravel()
            Nlist.append(Nravel)
            
            # column index: global dof of nonmortar (main) side
            rowindlist.append(np.ones(Nravel.shape, dtype=int)*pind)
            if undo_cut is not None:  
                uncut_nop = domain.global2local[degenerate_dofs[eg.gnop[elems,:]]]
                colindlist.append(uncut_nop.ravel())
            else:
                colindlist.append(nop.ravel())
    
    Ns = np.concatenate(Nlist)
    colinds = np.concatenate(colindlist)
    rowinds = np.concatenate(rowindlist)
    
    # This has a 50-60 chance of working in higher dimensions also...
    shape = (len(nodepoints_uvw), len(nodes)-len(undo_cut))
    
    return sps.coo_matrix((Ns, (rowinds, colinds)), shape=shape).tocsr(), ipgel



def boundary_mapping(dom_in, bound_in):     
    """ Make a matrix which picks the boundary nodes from all nodes 
    in the domain
    dom_in - domain, either np.array of global nodes or an elementgroup
    bound_in - boundary  either np.array of global nodes or an elementgroup
    """
    try:
        bound = bound_in.global_nodeids
    except:
        bound = bound_in
        
    try:
        dom = dom_in.global_nodeids
    except:
        dom = dom_in

        
    cols = dom.shape[0]
    rows = bound.shape[0]

    ii = np.arange(0,bound.shape[0])
    I = np.ones((bound.shape[0],))

    return sps.csr_matrix((I, (ii, bound)), shape=(rows, cols))

def internal_mapping(dom_in, bound_in):
    """ Make a matrix which picks the internal nodes from all nodes 
    in the domain
    dom_in - domain, either np.array of global nodes or an elementgroup
    bound_in - boundary  either np.array of global nodes or an elementgroup
    """
    try:
        bound = bound_in.global_nodeids
    except:
        bound = bound_in
        
    try:
        dom = dom_in.global_nodeids
    except:
        dom = dom_in

#    bmask = np.ones(dom, dtype=bool)
    indses = np.concatenate([np.where(dom == x) for x in bound]).reshape(-1)
    intnodes = np.delete(dom, indses)
        
    cols = dom.shape[0]
    rows = cols - bound.shape[0]

    ii = np.arange(0,rows)
    I = np.ones((rows,))

    return sps.csr_matrix((I, (ii, intnodes)), shape=(rows, cols))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                             UNIT TESTS                            # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == '__main__':
    
    import unittest
    
    class TestFindSimplices(unittest.TestCase):
    
        def setUp(self):            
            self.nodes2d = np.array([[0,0], [1,0], [1,1], [0,1], [0.5,0.5]])
            self.gnop2d = np.array([[0,1,4], [2,1,4], [2,3,4], [3,0,4]])
            
            self.nodes1d = np.array([0,0.5,0.75,1])[:, None]
            self.gnop1d = np.array([[1,0], [1,2], [2,3]])
    
        def test_find_simplices_1d(self):
            points1d = np.array([0.1, 0.6, 0.75, 1])[:,None]
            out = find_simplices(self.nodes1d, self.gnop1d, points1d)
            should_be = [(i, l) 
                         for i ,l in enumerate([[0], [1], [1,2], [2]])]
            self.assertEqual(out, should_be)
            
        def test_find_simplices_2d(self):
            points2d = np.array([[0.5,0.25], # in e0
                                 [0.6,0.7], # in e2
                                 [0.6,0.6], # an edge point, in e1 e2
                                 [0.5,0.5]]) # a node point in all
            out = find_simplices(self.nodes2d, self.gnop2d, points2d)
            should_be = [(i, l) 
                         for i ,l in enumerate([[0], [2], [1,2], [0,1,2,3]])]
            
            self.assertEqual(out, should_be)
    unittest.main()