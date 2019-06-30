import deps

from mordecomp import gmsh_tricks
gmsh = gmsh_tricks.gmsh

import numpy as np
import scipy.sparse as sps


import npyfem

from mordecomp import utils, storage
from mordecomp.utils import Timer, info, stop

from mordecomp.coupling import make_mortar_assembler, boundary_mapping
import geometry as geo

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from datatypes import CouplingData, Solver, DecomposedModel

nu0 = 1/(np.pi*4e-7)

#gmsh.fltk.run()

##%%
AIR = (2,1)
SLOTU = (1,2)
SLOTD = (1,3)
BDRY = (1,4)

def make_decomposed_model(gmsh, rmodel, **kwargs):
    
    MNAME = "MainProblem"
    
    model = gmsh.model
    factory = gmsh.model.occ
    model.add(MNAME)
    
    pos = kwargs.pop("coil_position", (0.0, 0.006))
    angle = kwargs.pop("coil_angle", 0)
    
    holes = [pos, (0.0,-0.01)]
    angles = [(*pos, angle), (0.0, 0.0, 0.0)]
    
    geo.main.create(gmsh, holes, angles)
    factory.synchronize()
    model.mesh.generate(2)    
    
    mesh = npyfem.from_gmsh(gmsh)
    
    domain = mesh.define_domain([AIR])
    Kmedium = utils.compute_stiffness_matrix(domain, nu0)
    
    boundary_nodes = mesh.global_nodes_in([BDRY])
    
#    bvals0 = np.zeros(boundary_nodes.shape)
    
    I1 = npyfem.dirichletbc.ones_to_diag_matrix(boundary_nodes, mesh.ndofs)
    O1 = npyfem.dirichletbc.zero_rowcols_mapping(boundary_nodes, mesh.ndofs)
    
    Ko = O1*Kmedium*O1 + I1
    Fo = np.zeros((Kmedium.shape[0],1))
    
    #%%
    #with Timer("Loading mortar side data"):
#    store = storage.loader("wpt_mortar_compumag")
    #mortar_part = store.mortar_part
    #mortar_dampmat = store.mortar_dampmat
    nodes_xyz = rmodel.nodes_xyz
    nodes_uvw = rmodel.nodes_uvw
    #integration_points_uvw = store.integration_points_uvw
    psi = rmodel.psi
    rKcoil = rmodel.redK
    rBcs = rmodel.redBsc
    rCc = rmodel.redCc
    
    assemble_mortar_map = make_mortar_assembler(nodes_uvw)
    
    #fCc = store.load("Cc_full")
    #fKcoil = store.load("K_full")
    #fBcs = store.load("Bsc_full")
    
    boundary_slotu = mesh.define_domain([SLOTU])
    boundary_slotd = mesh.define_domain([SLOTD])
    
    Mu, unodes_uvw, ipgelu = assemble_mortar_map(boundary_slotu)
    Md, dnodes_uvw, ipgeld = assemble_mortar_map(boundary_slotd)
    
    all_nodeids = mesh.get_global_nodeids()
    bslotu_nodeids = mesh.get_global_nodeids([SLOTU])
    bslotd_nodeids = mesh.get_global_nodeids([SLOTD])
    
    Bmu = boundary_mapping(all_nodeids, bslotu_nodeids)
    Bmd = boundary_mapping(all_nodeids, bslotd_nodeids)
    
    MBmu = Mu @ Bmu
    MBmd = Md @ Bmd
    
    ## NEXT ASSEMBLE THE TOTAL SYSTEM AND SOLVE
    
    # number of nodes in coupling upper and lower coupling boundaries
    ncplu = Mu.shape[0]
    ncpll = Md.shape[0]
    
    Ou = sps.csr_matrix((ncplu, 1))
    Ol = sps.csr_matrix((ncpll, 1))
    
    rSglob = sps.bmat([[Ko   , None   , None   , MBmu.T , MBmd.T ],
                       [None , rKcoil , None   , -rBcs.T , None   ],
                       [None , None   , rKcoil , None   , -rBcs.T ],
                       [MBmu , -rBcs   , None   , None   , None   ],
                       [MBmd , None   , -rBcs   , None   , None   ]], format='csr')
    
    #%%
    # Some mappings to restore the full order solution in couils and 
    # get rid of lagrange multipliers
    Mtofull = sps.block_diag([sps.eye(Ko.shape[0]), psi, psi],
                             format='csr')
    # Mask which removes all Lagrange multipliers from the result
    mask_red_dofs = np.zeros((rSglob.shape[0],), dtype=bool)
    rnodes = Ko.shape[0] + rKcoil.shape[0]*2
    mask_red_dofs[0:rnodes] = np.ones((rnodes,), dtype=bool)
    reduced_dimension = psi.shape[1]
        
    def solve_reduced(i1, i2):
        rFglob = sps.bmat([[Fo], [i1*rCc], [i2*rCc], [Ou], [Ol]]).tocsr()
        redaglob = sps.linalg.spsolve(rSglob, rFglob)
        return redaglob
    
    ndofs = Ko.shape[0]
    redudofs = rKcoil.shape[0]
    
    def coupling_coefficient():
        
        a1 = solve_reduced(6,0)
        a2 = solve_reduced(0,6)
        
        L1 = (rCc.T@a1[ndofs:ndofs+redudofs])[0]
        M12 = (rCc.T@a1[ndofs+redudofs:ndofs+2*redudofs])[0]
        M21 = (rCc.T@a2[ndofs:ndofs+redudofs])[0]
        L2 = (rCc.T@a2[ndofs+redudofs:ndofs+2*redudofs])[0]
        
        k = M12/np.sqrt(L1*L2)
        
        assert np.linalg.norm(M12-M21)/np.linalg.norm(M12) < 0.01

        return CouplingData(L1, M12, M21, L2, k)
            
    
    emb1 = geo.sub.embedding(pos, angle)
    emb2 = geo.sub.embedding((0, -0.01), 0)
        
    def plotting(solution):    
        
        aglob = (Mtofull@solution[mask_red_dofs])
        
        sub_mesh = rmodel.mesh
        
        main_domain = mesh.define_domain([AIR])
        
        
        COIL_AIR = (2,1)
        WIND_LEFT = [(2,x) for x in range(100, 106)]
        WIND_RIGHT = [(2,x) for x in range(200, 206)]
        BDRY = (1,4)

        coil_domain = sub_mesh.define_domain([COIL_AIR] + WIND_LEFT + WIND_RIGHT)
    
        subcolors1 = dict([(dt, 'red') for dt in WIND_LEFT] +\
                          [(dt, 'blue') for dt in WIND_RIGHT] +\
                          [((2,1), 'green')])

        subcolors2 = dict([(dt, 'red') for dt in WIND_LEFT] +\
                          [(dt, 'blue') for dt in WIND_RIGHT] +\
                          [((2,1), 'orange')]) 
       
        ### main domain plot
        fig = plt.figure(11, figsize=(12,9))
        fig.clear()
        ax = plt.gca()
        main_domain.plot_mesh(ax, unit_scale=1000, color='black', 
                              linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        fig.savefig('figs/maindomain.pdf')
        
        ### Combined plot
        fig = plt.figure(11, figsize=(12,9))
        fig.clear()
        ax = plt.gca()
        main_domain.plot_mesh(ax, unit_scale=1000, color='black', 
                              linewidth=0.5)
        coil_domain.plot_mesh(ax, unit_scale=1000, colors=subcolors1,
                              linewidth=0.5, pmap=emb1)
        coil_domain.plot_mesh(ax, unit_scale=1000, colors=subcolors2,
                              linewidth=0.5, pmap=emb2)
        ax.set_aspect('equal')
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        fig.savefig('figs/combined.pdf')
        
        ### Solution plot
        coilu = aglob[mesh.ndofs:mesh.ndofs+sub_mesh.ndofs]
        coill = aglob[mesh.ndofs+sub_mesh.ndofs:mesh.ndofs+2*sub_mesh.ndofs]
        
        glob_min = min(np.min(coill), np.min(coilu))
        glob_max = max(np.max(coill), np.max(coilu))
        
        fig = plt.figure(44, figsize=(12,6))
        fig.clear()
        ax = plt.gca()
        ax.set_aspect('equal')
        levls = np.linspace(glob_min,
                            glob_max,
                            20)
        
        domain.plot_contourf(ax, aglob, levels=levls, unit_scale=1000)
        c = coil_domain.plot_contourf(ax, coilu, levels=levls,
                                      pmap=emb1, unit_scale=1000)
        c = coil_domain.plot_contourf(ax, coill, levels=levls,
                                      pmap=emb2, unit_scale=1000)
        
        domain.plot_contour(ax, aglob, levels=levls, linewidths=0.5,
                            colors='k', unit_scale=1000)   
        coil_domain.plot_contour(ax, coilu, levels=levls, pmap=emb1,
                                 linewidths=0.5, colors='k', unit_scale=1000)
        coil_domain.plot_contour(ax, coill, levels=levls, pmap=emb2,
                                 linewidths=0.5, colors='k', unit_scale=1000)
        
        domain.plot_mesh(ax, linewidth=0.5, color='black', markersize=6,
                      alpha=0.3, unit_scale=1000)
        coil_domain.plot_mesh(ax, pmap=emb1, linewidth=0.5,
                      color='black', markersize=6, alpha=0.3, unit_scale=1000)
        coil_domain.plot_mesh(ax,pmap=emb2, linewidth=0.5,
                      color='black', markersize=6, alpha=0.3, unit_scale=1000)
        
        ax.set_title(r'$A_z$ in $\frac{Vs}{m}$ with $1\,$A in both coils')
        ax.set_ylabel('mm')
        ax.set_xlabel('mm')
        fig.tight_layout()
        fig.colorbar(c[0], ax=ax, format='%.3e')
        
        fig.savefig('figs/combined_solution.pdf')
        plt.show()

        
    return Solver(solve_reduced, coupling_coefficient), plotting
