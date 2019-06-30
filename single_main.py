import deps

from matplotlib import pyplot as plt

import geometry

import gmsh
import numpy as np
import scipy.sparse as sps

from timeit import default_timer as clock


#from fourier_sub_sinu import build_reduced_model
from lagrange_sub import build_reduced_model
from lagrange_decomposed_model import make_decomposed_model
from combined_model import make_combined_model

gmsh.initialize()

# cpl_density=0.0025 is close to what the tracmoditional model has

# 0.002 is height/5, this is the smallest common length 
# 0.005 is height/2, this is the second closest common length
# 0.03 density means 10 coupling nodes

#densities = {'density_windings': 0.0003,
#             # density in the coupling boundary in sub domain
#             'density_coupling_sub': 0.003,
#             # density in coupling boundary in main domain
#             'density_coupling_main': 0.005,
#             # density at "infinity"
#             'density_inf': 0.02 }

densities = {'density_windings': 0.0003,
             # density in the coupling boundary in sub domain
             'density_coupling_sub': 0.003,
             # density in coupling boundary in main domain
             'density_coupling_main': 0.003,
             # density at "infinity"
             'density_inf': 0.02 }

def dcfg(**kwargs):
    d = {k: kwargs.pop(k, densities[k]) for k, v in densities.items()}
    assert not kwargs, f"Unexpected arguments in kwargs {list(kwargs.keys())}"
    
    return d

blen = geometry.sub.width*2 + geometry.sub.height*2
cpl_nodes_target = 80

cpl_density_coil = blen/cpl_nodes_target
cpl_density_air = blen/80


positions = [(0.000, 0.01),
             (0.0, 0.01)]

angle = 0.0/180*np.pi


density_config_rmodel = densities
#density_config_rmodel = dcfg(density_coupling_sub = cpl_density_coil)

t1 = clock()
rmodel, pp_data = build_reduced_model(gmsh, 
                                      densities=density_config_rmodel,
                                      #poddim=50
                                      )
t2 = clock()
print(f"Rom took: {t2-t1} seconds")


#%%
resc_ = []
runtimesc_ = []
resd_ = []
runtimesd_ = []

postions =  np.linspace(-0.15,0.15, 1)

densconf_cmodel = dcfg(density_coupling_sub = cpl_density_coil)
t1 = clock()
for pos in postions:
    cmodel, cplotter, model_data = make_combined_model(gmsh, 
                                                       coil_position=(pos, 0.01),
                                                       coil_angle=angle,
                                                       densities=densconf_cmodel)
#    t1 = clock()
    resc_.append(cmodel.coupling())
#    t2 = clock()
#    runtimesc_.append(t2-t1)
t2 = clock()
print(f"Traditional took avg: {(t2-t1)/100}")

densconf_dmodel = dcfg(density_coupling_main=cpl_density_air)

t1 = clock()
for pos in postions:
    #%%
    
    
    dmodel, dplotter, dmodel_data = make_decomposed_model(gmsh, rmodel, 
                                                          coil_position=(pos, 0.01),
                                                          coil_angle=angle,
                                                          densities=densconf_dmodel)
                                                          
                                                          
                                                          
                                                          #nlag=8                                                          

#    t1 = clock()
    resd_.append(dmodel.coupling())
#    t2 = clock()
#    runtimesd_.append(t2-t1)
t2 = clock()
print(f"Decomposed took avg: {(t2-t1)/101}")

refcpl = np.array(resc_)[:,-1]
cpld = np.array(resd_)[:,-1]

error = 100*np.divide(np.linalg.norm(refcpl-cpld), np.linalg.norm(refcpl)) 

    
#res = np.array([resc_, resd_])
#
#plt.plot(postions, res[:,0,-1])
#plt.plot(postions, res[:,1,-1])

#%%
#t1 = clock()
#cplc = cmodel.coupling() 
#t2 = clock()
#print(f"Full took: {t2-t1} seconds")
#
#t1 = clock()
#cpld = dmodel.coupling()
#t2 = clock()
#print(f"Reduced took: {t2-t1} seconds")
#                  
#dsolution = dmodel.solve(1,0)

#csolution = cmodel.solve(1,0)
###%%
#dplotter(1,0, solution_only=True)
#cplotter(csolution, solution_only=True)

#print("{:.02}".format((cpld.k-cplc.k)/(cplc.k)*100))

#decompplott(decomp.solve(0,1))

#solution = decomp_solver.solve(1,-1)
#decompplt(solution)
#
#sol = combined_solver.solve(1,-1)
#plt(sol)

#a = decomp.coupling()

#b = combined.coupling()

