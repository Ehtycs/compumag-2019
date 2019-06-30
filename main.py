import deps
import os

from matplotlib import pyplot as plt

import gmsh
import numpy as np
import scipy.sparse as sps

from timeit import default_timer as clock

import geometry

from lagrange_sub import build_reduced_model
from lagrange_decomposed_model import make_decomposed_model
from combined_model import make_combined_model

gmsh.initialize()


densities = {# density in the coupling boundary in sub domain
             'density_coupling_sub': 0.003,
             # density in the windings in "static case"
             'density_windings': 0.0005,
             # density in the windings in dynamic case
#             'density_windings': 0.0002,
             # density in coupling boundary in main domain
             'density_coupling_main': 0.005,
             # density at "infinity"
             'density_inf': 0.03 }


def dcfg(**kwargs):
    d = {k: kwargs.pop(k, densities[k]) for k, v in densities.items()}
    assert not kwargs, f"Unexpected arguments in kwargs {list(kwargs.keys())}"
    
    return d

# cpl_density=0.0025 is close to what the traditional model has

# 0.002 is height/5, this is the smallest common length 
# 0.005 is height/2, this is the second closest common length

# boundary length, to calculate the cpl density to get specific 
# number of elements
blen = geometry.sub.width*2 + geometry.sub.height*2
cpl_nodes_targets = [20, 40, 60, 80, 100]
angle_inds = [0,1,2]

positions = np.linspace(-0.15, 0.15, 101)
angles = [x/180*np.pi for x in [-7.5, 0, 7.5]]

for cpl_nodes_target in cpl_nodes_targets:
    
    rdensfcg = dcfg(density_coupling_sub=blen/cpl_nodes_target)
    rmodel, pp_data = build_reduced_model(gmsh, densities=rdensfcg)
    #gmsh.fltk.run()
    
    cpldim = pp_data["cpl_dimension"]
    np.savez(f'results/results_preprocessing_{cpldim}.npz', **pp_data)


    for angle_ind in angle_inds:
        
        print(f"Simulating angle {angle_ind} with {cpl_nodes_target} cpl nodes")


        # dont change the position count or I'll kick your ass

        angle = angles[angle_ind]#7.5/180*np.pi
    
        cpl_node_targets = [int(x) 
                            for x in np.linspace(8, cpl_nodes_target, 
                                                 int(cpl_nodes_target/5))] 
                            
        bdensities = [blen/x for x in reversed(cpl_node_targets)]
#        nLags = [int(x) 
#                 for x in np.linspace(8, cpl_nodes_target, 
#                                      int(cpl_nodes_target/10))]
        
        #%%
        
        cpld_ = []
        runtimesd_ = []
        cpl_nodes_ = []
        tot = len(bdensities)
        for indd, dens in enumerate(bdensities) :
            print(f"Density: {indd+1}/{tot}")
#            print(f"Nlags: {nLag} {indd+1}/{tot}")
            
            density_cfg = dcfg(density_coupling_main=dens)
#            density_cfg = dcfg(density_coupling_main=(blen/cpl_nodes_target))
            
            for ind, p in enumerate(positions):  
                if(ind % 10 == 0):
                    print(f"Decomposed: {ind+1} / {len(positions)}")

                solver,_ , mdata = make_decomposed_model(gmsh, rmodel, 
                                              coil_position=(p, 0.01),
                                              coil_angle=angle,
                                              densities=density_cfg,
                                              #nlag=nLag
                                              ) 
                t1 = clock()
                k = solver.coupling()[-1] 
                t2 = clock()
                runtimesd_.append(t2-t1)
                cpld_.append(k)
        
            cpl_nodes_.append(mdata['num_cplnodes_d'])
        
        
        
#        cpld = np.stack(cpld_).reshape((len(bdensities), len(positions))) 
#        runtimesd = np.array(runtimesd_).reshape((len(bdensities), len(positions)))
#        cpl_nodes = np.array(cpl_nodes_)
        
        cpld = np.stack(cpld_).reshape((tot, len(positions))) 
        runtimesd = np.array(runtimesd_).reshape((tot, len(positions)))
        cpl_nodes = np.array(cpl_nodes_)
        
        np.savez(f"results/results_decomp_{cpldim}_{angle_ind}.npz", 
                 angle=angle, 
                 positions=positions, 
                 cpld=cpld, runtimesd=runtimesd,
                 cpl_nodes=cpl_nodes,
                 #nlags=nLags
                 )
        
        
        
        #%% Run 
        full_order_file = f"results/results_traditional_{cpldim}_{angle_ind}.npz"
        if not os.path.isfile(full_order_file):
            cpl_ = []
            runtimes_ = []
            
            for ind, p in enumerate(positions):  
                if(ind % 10 == 0):
                    print(f"Traditional: {ind+1} / {len(positions)}")
                
                solver,_ , mdata = make_combined_model(gmsh, 
                                              coil_position=(p, 0.01),
                                              coil_angle=angle,
                                              densities=rdensfcg)
                t1 = clock()
                k = solver.coupling()[-1] 
                t2 = clock()
                runtimes_.append(t2-t1)
                cpl_.append(k)
            
            cpl = np.stack(cpl_)
            runtimes = np.array(runtimes_)
            
            np.savez(full_order_file,
                     angle=angle, 
                     positions=positions, 
                     cpl=cpl, runtimes=runtimes)
        else:
            print("Full order solution already exists.")


