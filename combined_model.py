import deps

#from mordecomp import gmsh_tricks
#gmsh = gmsh_tricks.gmsh

import numpy as np
import scipy.sparse as sps

from scipy.sparse.linalg import spsolve
#from scikits.umfpack import spsolve

import utils

from timeit import default_timer as timer

import npyfem
from npyfem import dirichletbc as dbc

from coupling import make_mortar_assembler, boundary_mapping
import geometry as geo

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from datatypes import CouplingData, Solver

MNAME = "MainProblem"

nu0 = 1/(np.pi*4e-7)

AIR = (2,1)
AIR_COILS = (2,2)
UWIND_LEFT = [(2,x) for x in range(100, 106)]
UWIND_RIGHT = [(2,x) for x in range(200, 206)]
LWIND_LEFT = [(2,x) for x in range(300, 306)]
LWIND_RIGHT = [(2,x) for x in range(400, 406)]
BDRY = (1,6)
    

def make_combined_model(gmsh, **kwargs):
 
    gmsh.clear()
    gmsh.model.add(MNAME)

    densities = kwargs.pop('densities', {})

    pos = kwargs.pop("coil_position", (0.0, 0.01))
    angle = kwargs.pop("coil_angle", 0)
    
    # holes must be given as coordinates of the center
    geo.combined.create(gmsh, [pos, (0, -0.01)], 
                              [(*pos, angle), (0.0, -0.01, 0)],
                              densities=densities)
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
        
    mesh = npyfem.from_gmsh(gmsh)
    
    MAGNETIC = [AIR, AIR_COILS]+UWIND_LEFT+UWIND_RIGHT+LWIND_LEFT+LWIND_RIGHT
    
    magnetic = mesh.define_domain(MAGNETIC)
    
    uwindl = mesh.define_domain(UWIND_LEFT)
    uwindr = mesh.define_domain(UWIND_RIGHT)
    lwindl = mesh.define_domain(LWIND_LEFT)
    lwindr = mesh.define_domain(LWIND_RIGHT)
    
    Kmedium = utils.compute_stiffness_matrix(magnetic, nu0)
    
    Ccu = 1/utils.compute_area(uwindl)*utils.compute_c(uwindl) \
          - 1/utils.compute_area(uwindr)*utils.compute_c(uwindr)
    
    
    Ccl = 1/utils.compute_area(lwindl)*utils.compute_c(lwindl) \
          - 1/utils.compute_area(lwindr)*utils.compute_c(lwindr)
    
    boundary_nodes = mesh.global_nodes_in([BDRY])
        
    I1 = dbc.ones_to_diag_matrix(boundary_nodes, mesh.ndofs)
    O1 = dbc.zero_rowcols_mapping(boundary_nodes, mesh.ndofs)
    
    Ko = O1*Kmedium*O1 + I1
    Fo = np.zeros((Kmedium.shape[0],1))
    
    #print(f"Nonzero combined: {Ko.nnz}")

    
    def solve(i1, i2):
    
        Rhs = Fo + Ccu*i1 + Ccl*i2
        Lhs = Ko
        return spsolve(Lhs, Rhs)

    def coupling_coefficient():
        #t1 = timer()
        a1 = solve(6,0)
        a2 = solve(0,6)
        
        L1 = (Ccu.T@a1)[0]
        M12 = (Ccl.T@a1)[0]
        M21 = (Ccu.T@a2)[0]
        L2 = (Ccl.T@a2)[0]
        
        k = M12/np.sqrt(L1*L2)
        
        #t2 = timer()
        
        #print(f"Traditional cc solve: {t2-t1}")
        
        assert np.linalg.norm(M12-M21)/np.linalg.norm(M12) < 0.01
        
        return CouplingData(L1, M12, M21, L2, k)
        
    
    
    def plotting(sol, **kwargs):
        if not kwargs.get("solution_only", False):
            fig = plt.figure(59, figsize=(12,9))
            fig.clear()
            ax = fig.add_subplot(111)
            magnetic.plot_mesh(ax,linewidth=0.5,unit_scale=1000)
            ax.set_aspect('equal')
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
            plt.savefig("figs/traditional.pdf")
        
        glob_min = np.min(sol)
        glob_max = np.max(sol)
        
        fig = plt.figure(47, figsize=(12,6))
        fig.clear()
        ax = plt.gca()
        ax.set_aspect('equal')
        levls = np.linspace(glob_min,
                            glob_max,
                            20)
        
        c = magnetic.plot_contourf(ax, sol, levels=levls, unit_scale=1000)
        
        magnetic.plot_contour(ax, sol, levels=levls, linewidths=0.5,
                              colors='k', unit_scale=1000)   
        magnetic.plot_mesh(ax, linewidth=0.5, color='black', markersize=6,
                           alpha=0.3, unit_scale=1000)
        
        ax.set_title(r'$A_z$ in $\frac{Vs}{m}$ with $1\,$A in both coils')
        ax.set_ylabel('mm')
        ax.set_xlabel('mm')
        fig.tight_layout()
        fig.colorbar(c[0], ax=ax, format='%.3e')
        plt.savefig("figs/trad_solution.pdf")
        plt.show()
        

        
    model_data = {"full_dimension": Ko.shape[0],
                  "total_nonzeros": Ko.nnz}
    
    return Solver(solve, coupling_coefficient), plotting, model_data
