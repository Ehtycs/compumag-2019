import deps

import gmsh

import numpy as np
import scipy.sparse as sps

from scipy.sparse.linalg import spsolve


import npyfem

from timeit import default_timer as timer

from coupling import (boundary_mapping, 
                                solve_parametric_coordinates,
                                internal_mapping)
from fourier_coupling import (nudft_matrix, 
                                        wavenumbers, 
                                        compute_nudft_analyt)
import geometry as geo

import utils

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from datatypes import CouplingData, Solver, DecomposedModel

nu0 = 1/(np.pi*4e-7)

def debug(msgs):
    with open("fourier_decomposed.log", 'a') as f:
        for msg in msgs:
            f.write(msg)
            f.write("\n")
    

##%%
AIR = (2,1)
SLOTU = (1,2)
SLOTD = (1,3)
BDRY = (1,4)

def make_decomposed_model(gmsh, rmodel, **kwargs):
    """ Builds a decomposed model, returns a set of functions 
    
    rmodel needs to be a reduced model, gmsh just a handle to gmsh API object
    
    Solver(solve_reduced, coupling_coefficient), plotting, model_data
    
    where 
        Solver(...) - is a tuple, it has "methods" solve and coupling,
        plotting - produces plots
        model_data -  contains information about the model """
    
    MNAME = "MainProblem"
    
    model = gmsh.model
    factory = gmsh.model.occ
    model.add(MNAME)
    
    densities = kwargs.pop('densities', {})

    pos = kwargs.pop("coil_position", (0.0, 0.01))
    angle = kwargs.pop("coil_angle", 0)
    
    holes = [pos, (0.0,-0.01)]
    angles = [(*pos, angle), (0.0, 0.0, 0.0)]
    
    geo.main.create(gmsh, holes, angles, densities)
    factory.synchronize()
    model.mesh.generate(2)    
    
    mesh = npyfem.from_gmsh(gmsh)
    
    domain = mesh.define_domain([AIR])
    Kmedium = utils.compute_stiffness_matrix(domain, nu0)
    
    boundary_nodes = mesh.global_nodes_in([BDRY])
        
    I1 = npyfem.dirichletbc.ones_to_diag_matrix(boundary_nodes, mesh.ndofs)
    O1 = npyfem.dirichletbc.zero_rowcols_mapping(boundary_nodes, mesh.ndofs)
    
    Ko = O1*Kmedium*O1 + I1

    Li = rmodel['Li']
    Lb = rmodel['Lb']    
    
    flux_Bbc = rmodel['flux_Bbc']
    flux_I = rmodel['flux_I']    

    M = rmodel['M']
    permute = rmodel['permute']
    to_full = rmodel['to_full']
        
    boundary_slotu = mesh.define_domain([SLOTU])
    boundary_slotd = mesh.define_domain([SLOTD])
    
    nodesu_uvw, bdomu, uncutu = solve_parametric_coordinates(boundary_slotu)
    nodesd_uvw, bdomd, uncutd = solve_parametric_coordinates(boundary_slotd)
    
    # Select number of lagrange multipliers to be min(Ncpl_main, Ncpl_sub)
    nLag = min(boundary_slotu.ndofs, boundary_slotd.ndofs, 
               rmodel['nodes_uvw'].shape[0])
    
    #nLag = kwargs.pop("nlag", nLag)
      
    #%%
    Mu_ = compute_nudft_analyt(bdomu, uncutu, nodesu_uvw, K=nLag)
    Md_ = compute_nudft_analyt(bdomd, uncutd, nodesd_uvw, K=nLag)
            
    all_nodeids = mesh.get_global_nodeids()
    bslotu_nodeids = mesh.get_global_nodeids([SLOTU])
    bslotd_nodeids = mesh.get_global_nodeids([SLOTD])
    
    Bmu = boundary_mapping(all_nodeids, bslotu_nodeids)
    Bmd = boundary_mapping(all_nodeids, bslotd_nodeids)
    

    ## NEXT ASSEMBLE THE TOTAL SYSTEM AND SOLVE   

    # Do some "preconditioning" by scaling M matrices to contain 
    # appriximately same size terms as Ko and L
    #scale = np.max(np.abs(Ko))/np.max(np.abs(Mc))
    # no need to be too fancy
    scale = 1e6
    
    # Pick only nLag "frequencies" or rows to the mortar block
    # This is limited by the SPARSER side of the mortar boundary
    Mc = scale*M[:nLag,:]
    Mu = scale*Mu_[:nLag,:]
    Md = scale*Md_[:nLag,:]
    
    MBu = Mu @ Bmu
    MBd = Md @ Bmd
    
    MBuH = MBu.conj().T
    MBdH = MBd.conj().T
    McH = Mc.conj().T

    rSglob = sps.bmat([[  Ko,  None, None,  MBuH,  MBdH],
                       [None,     Lb, None,  -McH,  None],
                       [None,  None,    Lb,  None,  -McH],                       
                       [ MBu,   -Mc, None,  None,  None],
                       [ MBd,  None,  -Mc,  None,  None],
                       ], format='csr')
    
#    print(f"Ko max {np.max(np.abs(Ko))}")
#    print(f"L max {np.max(np.abs(L))}")
#    
#    print(f"MBu max {np.max(np.abs(MBu))}")
#    print(f"Mc max {np.max(np.abs(Mc))}")
#    condn = sps.linalg.norm(rSglob)*sps.linalg.norm(sps.linalg.inv(rSglob))
#    print(f"Cond: {condn}")

    Fo = np.zeros((Ko.shape[0],1))
    Ou = sps.csr_matrix((Mc.shape[0], 1))
    Ol = sps.csr_matrix((Mc.shape[0], 1))

    def solve_reduced(i1, i2):
        # This function solves the system for currents i1 and i2
        rFglob = sps.bmat([[Fo], [-i1*Li[:,None]], [-i2*Li[:,None]], [Ou], [Ol]]).tocsr()
        redaglob = spsolve(rSglob, rFglob)
        return redaglob
    
    ndofs = Ko.shape[0]   
    
    ncdofs = flux_Bbc.shape[1]
    
    def coupling_coefficient():
        
        #t1 = timer()
        a1 = solve_reduced(6,0)
        a2 = solve_reduced(0,6)
        
        
        # self inductances  are affected by the current
        L1 = np.real(flux_Bbc@a1[ndofs:ndofs+ncdofs] + flux_I*6)[0]
        L2 = np.real(flux_Bbc@a2[ndofs+ncdofs:ndofs+2*ncdofs] + flux_I*6 )[0]

        # these inductances don't have any current present
        M12 = np.real(flux_Bbc@a1[ndofs+ncdofs:ndofs+2*ncdofs])[0]
        M21 = np.real(flux_Bbc@a2[ndofs:ndofs+ncdofs])[0]
        
        k = M12/np.sqrt(L1*L2)
        
        # check that mutual inductance is the same both ways      
        assert np.linalg.norm(M12-M21)/np.linalg.norm(M12) < 0.01

        return CouplingData(L1, M12, M21, L2, k)
            
    
    emb1 = geo.sub.embedding(pos, angle)
    emb2 = geo.sub.embedding((0, -0.01), 0)
        
    def plotting(i1,i2, **kwargs):    
        # this plots the solution and domains for currents i1, i2
        
        solution = solve_reduced(i1, i2)
        
        aglob = np.real(solution)
        
        sub_mesh = rmodel['mesh']
        
        main_domain = mesh.define_domain([AIR])
#        figsizes = (8,3) 
        figsizes = (12,9)
        
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
                          [((2,1), 'green')]) 
       
        mesh_linewidth = 0.5
        if not kwargs.get('solution_only', False):
            ### main domain plot
            fig = plt.figure(10, figsize=figsizes)
            fig.clear()
            ax = plt.gca()
            main_domain.plot_mesh(ax, unit_scale=1000, color='black', 
                                  linewidth=mesh_linewidth)
            ax.set_aspect('equal')
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
            fig.savefig('figs/maindomain.pdf')
            
            ### Combined plot
            fig = plt.figure(11, figsize=figsizes)
            fig.clear()
            ax = plt.gca()
            main_domain.plot_mesh(ax, unit_scale=1000, color='black', 
                                  linewidth=mesh_linewidth)
            coil_domain.plot_mesh(ax, unit_scale=1000, colors=subcolors1,
                                  linewidth=mesh_linewidth, pmap=emb1)
            coil_domain.plot_mesh(ax, unit_scale=1000, colors=subcolors2,
                                  linewidth=mesh_linewidth, pmap=emb2)
            ax.set_aspect('equal')
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
            fig.savefig('figs/combined.pdf')
            plt.show()
            
            ### Subdomain plot
            fig = plt.figure(12, figsize=figsizes)
            fig.clear()
            ax = plt.gca()
            coil_domain.plot_mesh(ax, unit_scale=1000, colors=subcolors1,
                                  linewidth=mesh_linewidth, pmap=emb1)
            ax.set_aspect('equal')
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
            fig.savefig('figs/sub-domain.pdf')
            plt.show()
        
        ### Solution plot
        coilu_bc = aglob[mesh.ndofs:mesh.ndofs+ncdofs]
        coill_bc = aglob[mesh.ndofs+ncdofs:mesh.ndofs+2*ncdofs]
        
        coilu = permute.T @ to_full @ np.concatenate([[i1], coilu_bc])
        coill = permute.T @ to_full @ np.concatenate([[i2], coill_bc])
        
        glob_min = min(np.min(coill), np.min(coilu))
        glob_max = max(np.max(coill), np.max(coilu))
        
        fig = plt.figure(44, figsize=figsizes)
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
        
        domain.plot_contour(ax, aglob, levels=levls, linewidths=mesh_linewidth,
                            colors='k', unit_scale=1000)   
        coil_domain.plot_contour(ax, coilu, levels=levls, pmap=emb1,
                                 linewidths=mesh_linewidth, 
                                 colors='k', unit_scale=1000)
        coil_domain.plot_contour(ax, coill, levels=levls, pmap=emb2,
                                 linewidths=mesh_linewidth, 
                                 colors='k', unit_scale=1000)
        
        domain.plot_mesh(ax, linewidth=mesh_linewidth, color='black', 
                         markersize=6, alpha=0.3, unit_scale=1000)
        coil_domain.plot_mesh(ax, pmap=emb1, linewidth=mesh_linewidth,
                      color='black', markersize=6, alpha=0.3, unit_scale=1000)
        coil_domain.plot_mesh(ax,pmap=emb2, linewidth=mesh_linewidth,
                      color='black', markersize=6, alpha=0.3, unit_scale=1000)
        
        ax.set_title(r'Reduced $A_z$ in $\frac{Vs}{m}$ with $1\,$A in both coils')
        ax.set_ylabel('mm')
        ax.set_xlabel('mm')
        fig.tight_layout()
        fig.colorbar(c[0], ax=ax, format='%.3e')
        
        fig.savefig('figs/combined_solution.pdf')
        plt.show()

        
        
    nnodes_u = len(bslotu_nodeids)
    nnodes_d = len(bslotd_nodeids)
    model_data = {"num_nodes": mesh.ndofs,
                  "num_cplnodes_u": nnodes_u,
                  "num_cplnodes_d": nnodes_d,
                  "lagrange_multipliers": Mc.shape[0],
                  "total_dimension": rSglob.shape[0],
                  "total_nonzeros": rSglob.nnz}


    return Solver(solve_reduced, coupling_coefficient), plotting, model_data

