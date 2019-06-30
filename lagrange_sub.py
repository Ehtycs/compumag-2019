import deps

import gmsh

import numpy as np
import scipy.sparse as sps

from scipy.sparse.linalg import spsolve

import npyfem
from npyfem import dirichletbc as dbc

import utils 
from coupling import (assemble_mortar_part, cut_domain, 
                                boundary_mapping, internal_mapping,
                                solve_parametric_coordinates)
from fourier_coupling import (nudft_matrix, 
                                        wavenumbers, 
                                        compute_nudft_analyt)

from collections import namedtuple

import geometry as geo

from timeit import default_timer as clock

from datatypes import ReducedModelDataFourier, PreProcessingData


nu0 = 1/(np.pi*4e-7)

AIR = (2,1)
WINDL = [(2,x) for x in range(100, 106)]
WINDR = [(2,x) for x in range(200, 206)]
BDRY = (1,4)


def make_mesh(gmsh, densities):    
    gmsh.clear()    
    gmsh.model.add("SubProblem")
    geo.sub.create(gmsh, densities)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    return npyfem.from_gmsh(gmsh)

def build_reduced_model(gmsh, **kwargs):
    """ Take some configuration info in and build a reduced order model
    """
    
    t1 = clock()
    
    densities = kwargs.pop('densities', {})
    
    mesh = make_mesh(gmsh, densities)
    
    domain = mesh.define_domain([AIR] + WINDL + WINDR)
    windl = mesh.define_domain(WINDL)
    windr = mesh.define_domain(WINDR)
    
    Kcoil = utils.compute_stiffness_matrix(domain, nu0)
    
    Cc = 1/utils.compute_area(windl)*utils.compute_c(windl) \
          - 1/utils.compute_area(windr)*utils.compute_c(windr)
          
    boundary_domain = mesh.define_domain([BDRY])
    boundary_nodes = boundary_domain.nodes_in()
    
    Bm = boundary_mapping(domain, boundary_nodes)
    Int = internal_mapping(domain, boundary_nodes)
    
    Scc = Int @ Kcoil @ Int.T
    
    Sbc = Bm @ Kcoil @ Int.T
    Scb = Int @ Kcoil @ Bm.T
    Sbb = Bm @ Kcoil @ Bm.T
    
    # used in flux retrieval
    Lic = spsolve(Scc, Int@Cc)
    Lbc = spsolve(Scc, Scb)
   
    # used in total system
    Li = Sbc @ Lic
    Lb = (Sbb - Sbc @ Lbc).toarray()
    #%%
    # compute "the other" mortar matrix

    print("Computing parametric coordinates")
    u, bdom, undo_cut  = solve_parametric_coordinates(boundary_domain)

    nodes_uvw = u[:-1]
    
    Mc = compute_nudft_analyt(bdom, undo_cut, u).todense()
    
   
    #%% Save the results
    nodes_xyz = mesh.nodes[boundary_domain.nodes_in()]
    
    
    t2 = clock()
    runtime = t2-t1
    flux_Bbc = -(Int @ Cc).T @ Lbc
    flux_I = (Int @ Cc).T @ Lic
    
    permute = sps.bmat([[Int],[Bm]])
    
    # Mapping which takes an "input vector" (i, ab1, ab2, ... ) 
    # and returns the solution in all nodes for plotting
    to_full = sps.bmat([[Lic[:,None], -Lbc],
                        [None, sps.eye(Lbc.shape[1], Lbc.shape[1])]])
    
    reduced_model = {
                     'Li': Li,
                     'Lb': Lb,
                     'Lic': Lic,
                     'Lbc': Lbc,
                     'flux_Bbc': flux_Bbc,
                     'flux_I': flux_I,
                     'nodes_xyz': nodes_xyz,
                     'nodes_uvw': nodes_uvw,
                     'M': Mc,
                     'mesh': mesh,
                     'permute': permute,
                     'to_full': to_full,
                     }
    
        
    pp_data = {'full_dimension': Kcoil.shape[0],
               'cpl_dimension': Bm.shape[0],
               'runtime': runtime,
               }
    
    return reduced_model, pp_data
