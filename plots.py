import deps

import numpy as np
import scipy.sparse as sps

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#
#trad0 = np.load("results/results_traditional_0.npz")
#decomp0 = np.load("results/results_decomp_0.npz")
#trad0 = np.load("results/results_traditional_62_0.npz")
#decomp0 = np.load("results/results_decomp_62_0.npz")
#decomp082 = np.load("results/results_decomp_82_0.npz")
#decomp0122 = np.load("results/results_decomp_122_0.npz")
#decomp0162 = np.load("results/results_decomp_162_0.npz")

#trad0 = np.load("results/results_traditional_2.npz")
#decomp0 = np.load("results/results_decomp_2.npz")

#decomp = [decomp0]#, decomp082, decomp0122, decomp0162]

#foldr = "results_denser"
foldr = "run1"
#foldr = "total_time"
#foldr = "."

#foldr = "run2"

def decomp_loader(nodes, angle):
#    return np.load(f"results/results_denser/results_decomp_{nodes}_{angle}.npz")
    return np.load(f"results/{foldr}/results_decomp_{nodes}_{angle}.npz")
#    return np.load(f"results/results_decomp_{nodes}_{angle}.npz")

def trad_loader(nodes, angle):
    return np.load(f"results/{foldr}/results_traditional_{nodes}_{angle}.npz")

#    return np.load(f"results/results_traditional_{nodes}_{angle}.npz")
#    return np.load(f"results/results_denser/results_traditional_{nodes}_{angle}.npz")


def get(it,field):
    return [d[field] for d in it]

decomp = [decomp_loader(22,0),
          decomp_loader(42,0),
          decomp_loader(62,0),
          decomp_loader(82,0),
#          decomp_loader(102,0)
          ]

decomp1 = [decomp_loader(22,1)        
        ]

trad = [trad_loader(22,0),
        trad_loader(42,0),
        trad_loader(62,0),
        trad_loader(82,0),
#        trad_loader(102,0)
        ]

trad1 = [trad_loader(82,1)]

positions = trad[0]['positions']
#densities = decomp0['densities']
#cpl_nodes = decomp0['cpl_nodes']
cpl_nodes = get(decomp, 'cpl_nodes')
#nlags = get(decomp, "nlags")
#cpl_nodes = get(decomp, 'lagrange_multipliers')

cpl = get(trad, 'cpl')
runtimes =  get(trad, 'runtimes')

cpld = get(decomp, 'cpld')
runtimesd = get(decomp, 'runtimesd')

cpld1 = get(decomp1, 'cpld')
cpl1 = get(trad1, 'cpl')
#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 11}

#matplotlib.rc('font', **font)
#%%
linwidth = 2.0

fig = plt.figure(2, figsize=(5,3))
fig.clear()
ax = fig.add_subplot(211)
ax.set_ylabel("Cpl. coeff \n $ \\alpha=7.5^{\circ}$")
#for k in cpld[0][0]:
ax.plot(1000*positions, cpl[-1], linewidth=linwidth)
ax.plot(1000*positions, cpld[-1][-1], linestyle='--', linewidth=linwidth)
ax.axhline(0, linewidth=1.0, linestyle='--', color='black')
plt.setp(ax.get_xticklabels(), visible=False)
ax.legend(['reference',
           '{} DOF'.format(cpl_nodes[-1][-1])])

tcks = ax.get_yticks()

ax2 = fig.add_subplot(212)

#ax.legend([r"$\alpha={}^\circ$".format(i) for i in [0]])

ax2.set_ylabel('Cpl. coeff. \n $ \\alpha=0^{\circ}$')

ax2.plot(1000*positions, cpl1[-1], linewidth=linwidth)
ax2.plot(1000*positions, cpld1[-1][-1], linestyle='--', linewidth=linwidth)
ax2.axhline(0, linewidth=1.0, linestyle='--', color='black')
ax2.legend(['reference', 
           '{} DOF'.format(cpl_nodes[-1][-1])])
ax2.set_xlabel("Displacement (mm)")
ax2.set_yticks(tcks[1:-1])
plt.subplots_adjust(hspace=0.1, wspace=0.5)
fig.savefig('figs/coupling_coefficients.pdf', bbox_inches='tight')
plt.show()
#%%

refcpl = cpl[-1]
refruntime = runtimes[-1]

errors = [100*np.divide(np.linalg.norm(refcpl-cpld_, axis=1), np.linalg.norm(refcpl)) 
          for  cpld_ in cpld]
#speedup = [np.divide(np.sum(runtimes_), np.sum(runtimesd_, axis=1))
#            for runtimes_, runtimesd_ in zip(runtimes, runtimesd)]
speedup = [np.divide(np.sum(refruntime), np.sum(runtimesd_, axis=1))
            for runtimesd_ in runtimesd]

#%%

#lims = np.arange(0,5)
#densities = densities[lims]
#errors = errors[lims]
#speedup = speedup[lims]

#ticks = np.arange(0, np.max(cpl_nodes), 50)
#sticks = np.arange(np.min(speedup), np.max(speedup), 10)

#cpl_nodes = nlags

#fig = plt.figure(41, figsize=(5,3))
#fig.clear()
#axu = fig.add_subplot(211)
#for cpln, spd in zip(cpl_nodes, speedup):
#    axu.plot(cpln, spd, linewidth=2)
#axu.set_ylabel("Speedup")
##plt.xticks(ticks)
##plt.yticks(sticks)
#axd = fig.add_subplot(212)
#for cpln, err in zip(cpl_nodes, errors):
#    axd.plot(cpln, err, linewidth=2)
#axd.set_ylabel("Error / (%)")
#axd.set_xlabel("Coupling nodes")
#axu.legend([f'{x} DOF' for x in [22,42,62,82]])
#plt.ylim(0,5)
#plt.setp(axu.get_xticklabels(), visible=False)
#
#plt.subplots_adjust(hspace=0.1, wspace=0.5)
##plt.xticks(ticks)
#
#fig.savefig("figs/speed_and_error.pdf")
#plt.show()


#%%

fig = plt.figure(51, figsize=(6,4))
fig.clear()
axu = fig.add_subplot(111)
for err, spd in zip(errors, speedup):
    axu.plot(spd, err, linewidth=2, marker='o')
axu.set_ylabel("Error / (%)")
axu.set_xlabel("Speedup")
plt.legend([f'{x} DOF' for x in [22,42,62,82]])

# annotate 


datasets = list(zip(cpl_nodes, speedup, errors))

nds, xs, ys = datasets[0]
for nds, x, y in zip(nds,xs,ys):
    #axu.scatter(x,y, marker='o')
    axu.annotate(f'{nds}', xy=(x,y), xytext=(x-0.5, y-0.1),
                 color='blue')

nds, xs, ys = datasets[1]
lst =  list(zip(nds,xs,ys))
selected = lst[0:2] + lst[3:4] + lst[4:]
for nds, x, y in selected:
    #axu.scatter(x,y, marker='o')
    axu.annotate(f'{nds}', xy=(x,y), xytext=(x-0.2, y+0.5),
                 color='orange')

nds, xs, ys = datasets[2]
lst = list(zip(nds,xs,ys))
selected = lst[8:]
for nds, x, y in selected:
    #axu.scatter(x,y, marker='o')
    axu.annotate(f'{nds}', xy=(x,y), xytext=(x-0.2, y+0.5),
                 color='green')
#nds, x, y = selected[-2]
#axu.annotate(f'{nds}', xy=(x,y), xytext=(x, y),
#                 color='green')

#nds, x, y = selected[-1]
#axu.annotate(f'{nds}', xy=(x,y), xytext=(x, y),
#                 color='green')
#
nds, xs, ys = datasets[3]
lst = list(zip(nds,xs,ys))
selected = lst[11:]
for nds, x, y in selected:
    #axu.scatter(x,y, marker='o')
    axu.annotate(f'{nds}', xy=(x,y), xytext=(x+0.05, y-0.5),
                 color='red')


#for cplns, err, spd in zip(cpl_nodes, errors, speedup):
#    iterable = list(reversed(list(zip(cplns, spd, err))))
#    selected = iterable[0:2]+iterable[3:4]+iterable[5::5]
#    for nds, x, y in selected:
#        #axu.scatter(x,y, marker='o')
#        axu.annotate(f'{nds}', xy=(x,y), xytext=(x*1.01, y*1.01),
#                     backgroundcolor='white')


fig.savefig("figs/error_vs_speed_annotated.pdf")
plt.show()

#%%
#main_cpl_dim = decomposed_loader('cpl_dimension').squeeze(-1)[:,0]
#main_full_dim = decomposed_loader('full_dimension')[:,0]
#sub_cpl_dim = preprocess_loader['cpl_dimension']
#full_dim = np.mean(combined_loader("full_dimension"))
#sub_full_dim = preprocess_loader['full_dimension']
#
##%% Plot 
#
## pick any density index for plotting the coupling coefficient
#
#density_index = 0
##angle_indices = [0,1,2]
#angle_indices = [0]
#
##font = {'family' : 'normal',
##        'weight' : 'normal',
##        'size'   : 11}
#
##matplotlib.rc('font', **font)
#
#fig = plt.figure(2, figsize=(5,2))
#fig.clear()
#ax = fig.add_subplot(111)
#ax.set_xlabel("Displacement (mm)")
#ax.set_ylabel("Coupling coefficient")
#for k in k_decomp[density_index, angle_indices]:
#    ax.plot(1000*positions, k)
#    
#ax.legend([r"$\alpha={}^\circ$".format(i) for i in [-7.5, 0, 7.5]])
#fig.savefig('figs/coupling_coefficients.pdf', bbox_inches='tight')
#plt.show()
#
##%%
##
##fig = plt.figure(3)
##fig.clear()
##ax = fig.add_subplot(111)
##ax.set_xlabel("displacement (mm)")
##ax.set_ylabel("k")
##for k in k_decomp[1:2, 0]:
##    ax.plot(1000*positions, k)
##for k in k_comb[0:1]:
##    ax.plot(1000*positions, k)
##    
###ax.legend(["${}^\circ$ angle".format(i) for i in [-7.5, 0]])
###fig.savefig('cpl_coeff_traditional.pdf')
##%%
#k_err = np.divide(np.linalg.norm(k_comb[None,...] - k_decomp, axis=-1),
#                  np.linalg.norm(k_comb, axis=-1)[None,...])*100
#
#k_err = np.max(k_err, axis=-1)                  
#                  
#densdiffs = main_cpl_dim/sub_cpl_dim
#
#fig = plt.figure(4, figsize=(5,3))
#fig.clear()
#ax = fig.add_subplot(211)
##ax.set_title("RPD error limits of all runs")
#ax.set_ylabel("Relative error (%)")
##for dens, k in zip(densdiffs.T, k_err.T):
#ax.plot(densdiffs[2:], k_err[2:])
#plt.xticks(np.arange(0.2, 1.2, 0.2))
#plt.yticks([0,1,2,3], ['     {}'.format(k) for k in [0,1,2,3]])
#plt.setp(ax.get_xticklabels(), visible=False)
#plt.subplots_adjust(hspace=0.1, wspace=0.5)
##fig.savefig("figs/error_vs_densdiff.pdf")
##plt.show()
##
##%%
#comb_total_runtimes = np.average(np.sum(comb_times, axis=-1), axis=-1)
#decomp_total_runtimes = np.average(np.sum(decomp_times, axis=-1), axis=-1)
#pre_process_runtime = preprocess_loader['runtime']
#
#speedup = np.divide(np.mean(decomp_total_runtimes, axis=-1)+pre_process_runtime,
#                    np.mean(comb_total_runtimes))
#speedups = np.divide(decomp_total_runtimes+pre_process_runtime,
#                    np.mean(comb_total_runtimes))
#
##fig = plt.figure(5)
##fig.clear()
#ax = fig.add_subplot(212)
##ax.set_title("RPD error limits of all runs")
#ax.set_ylabel("Speedup ratio $t_r/t_f$")
#ax.plot(densdiffs[2:], speedups[2:])
#plt.xticks(np.arange(0.2, 1.2, 0.2))
#plt.yticks([0.15, 0.25, 0.35])
#ax.set_xlabel(r"Boundary density ratio $d_m/d_s$")
##plt.yticks(np.arange(0., 0.35, 0.05))
#plt.subplots_adjust(hspace=0.5)
#plt.savefig("figs/error_speedup_vs_densdiff.pdf", bbox_inches='tight')
#plt.show()
#
##%%
#stop()
#print("")
#print("Times for combined model:")
#print("Dofs:    {}".format(full_dim))
#print("Total:   {} sec".format(comb_total_runtimes))
#print("")
#
#print("Times for decomposed model:")
#print("Preprocessing: {} sec".format(pre_process_runtime))
#print("Reduced dim  : {} ".format(sub_cpl_dim+1))
#print("Cpl dofs:      {}".format(sub_cpl_dim))
#
#
#
#for cpldim, dim, t, e in zip(main_cpl_dim, main_full_dim, 
#                          decomp_total_runtimes, k_err):
#    print("Total:         {} sec".format(t))
#    print("Total w. pp:   {} sec".format(t+pre_process_runtime))
#    print("Cpl dofs:      {}".format(cpldim))
#    totaldofs = dim+2*sub_cpl_dim+2*(sub_cpl_dim+1)
#    print("Total dofs:    {}".format(totaldofs))
#    print("Reduction:     {} %".format(((full_dim -totaldofs)/full_dim*100)[0]))
#    print("Error:         {} %".format(e))
#    
#    print("")
#
##
##plt.savefig("k_vs_displacement_combined.pdf")
##plt.show()
##
##
##fig = plt.figure(2)
##fig.clear()
##ax = fig.add_subplot(111)
##ax.set_title("Cpl. coeff. from decomposed")
##ax.set_xlabel("displacement / mm")
##ax.set_ylabel("k")
##ax.plot(posdecomp, kdecomp)
##ax.plot(poscomb, kcomb)
##
##
##plt.savefig("k_vs_displacement_decomposed.pdf")
##plt.show()
##
##
### Relative percent difference
##relerr = 2*(kcomb-kdecomp)/(np.abs(kcomb)+np.abs(kdecomp))*100
##
##fig = plt.figure(2)
##fig.clear()
##ax = fig.add_subplot(111)
##ax.set_title("RPD error in k vs. position")
##ax.set_ylabel("error (%)")
##ax.set_xlabel("displacement (mm)")
##ax.plot(posdecomp, relerr)
##plt.savefig("relerr_in_k.pdf")
##plt.show()
