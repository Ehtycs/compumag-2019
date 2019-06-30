import deps 

import numpy as np
import scipy as sc
import scipy.sparse as sps
import scipy.sparse.linalg
import scipy.spatial

from npyfem import dirichletbc as dbc

from timeit import default_timer as timer

from itertools import product

from matplotlib import pyplot as plt

runtime_sparse = []
nnz_sparse = []
for nnodes in [int(np.sqrt(x)) for x in [1e2, 1e3, 1e4, 5e4, 1e5, 2e5, 3e5]]:
    print(f"For {nnodes**2} nodes")
    cord = np.array(list(product(np.linspace(0,1,nnodes), np.linspace(0,1,nnodes))))
    print("Meshing")
    tri = sc.spatial.Delaunay(cord)
    
    #plt.triplot(cord[:,0], cord[:,1], tri.simplices)
    
    print("Building stiffmat")
    gnop = tri.simplices
    
    arr = np.ones((*gnop.shape, 3), dtype=int).reshape(-1)*0.001
    
    index_i = (gnop[:,None,:]*np.ones((1,gnop.shape[1],1), dtype=int)).reshape(-1)
    index_j = (gnop[:,None,:]*np.ones((1,gnop.shape[1],1), dtype=int)).swapaxes(-1,-2).reshape(-1)
    
    
    K0 = sps.csr_matrix((arr, (index_i, index_j)))
    F0 = sps.csc_matrix((K0.shape[0],1))
    
    #plt.spy(K)
    
    print("Setting boundary conditions")
    dir1, = np.where(cord[:,0] == 0)
    dir2, = np.where(cord[:,0] == 1)

    O1 = dbc.zero_rowcols_mapping(dir1, K0.shape[0])
    O2 = dbc.zero_rowcols_mapping(dir2, K0.shape[0])
    
    I1 = dbc.ones_to_diag_matrix(dir1, K0.shape[0])
    I2 = dbc.ones_to_diag_matrix(dir2, K0.shape[0])
    
    vals1 = np.zeros(dir1.shape)
    vals2 = np.ones(dir2.shape)
    
    Q1 = dbc.coupling(K0, dir1, vals1)
    Q2 = dbc.coupling(K0, dir2, vals2)
    
    O = O1*O2
    I = I1+I2
    
    K = O*K0*O + I
        
    F = F0-Q1-Q2
    F[dir1] = vals1[:,None]
    F[dir2] = vals2[:,None]
    
    Lhs = K.toarray()
    Rhs = F.toarray()
    print(f"solving... nonzeros: {K.nnz}")
    t1 = timer()
#    sol = sps.linalg.spsolve(K, F)
    sol = np.linalg.solve(Lhs, Rhs)
    t2 = timer()
    runtime_sparse.append(t2-t1)
    print("done!")
    nnz_sparse.append(K.nnz)
    
sparse_runtimes = np.array(runtime_sparse)
sparse_nnzs = np.array(nnz_sparse)

#%%
runtime_dense = []
nnz_dense = []
for nnodes in [int(np.sqrt(x)) for x in [10, 1e2, 1e3, 1e4, 2e4]]:
    print(f"For {nnodes**2} nodes")
    cord = np.array(list(product(np.linspace(0,1,nnodes), np.linspace(0,1,nnodes))))
    print("Meshing")
    tri = sc.spatial.Delaunay(cord)
    
    #plt.triplot(cord[:,0], cord[:,1], tri.simplices)
    
    print("Building stiffmat")
    gnop = tri.simplices
    
    arr = np.ones((*gnop.shape, 3), dtype=int).reshape(-1)*0.001
    
    index_i = (gnop[:,None,:]*np.ones((1,gnop.shape[1],1), dtype=int)).reshape(-1)
    index_j = (gnop[:,None,:]*np.ones((1,gnop.shape[1],1), dtype=int)).swapaxes(-1,-2).reshape(-1)
    
    
    K0 = sps.csr_matrix((arr, (index_i, index_j)))
    F0 = sps.csc_matrix((K0.shape[0],1))
    
    #plt.spy(K)
    
    print("Setting boundary conditions")
    dir1, = np.where(cord[:,0] == 0)
    dir2, = np.where(cord[:,0] == 1)

    O1 = dbc.zero_rowcols_mapping(dir1, K0.shape[0])
    O2 = dbc.zero_rowcols_mapping(dir2, K0.shape[0])
    
    I1 = dbc.ones_to_diag_matrix(dir1, K0.shape[0])
    I2 = dbc.ones_to_diag_matrix(dir2, K0.shape[0])
    
    vals1 = np.zeros(dir1.shape)
    vals2 = np.ones(dir2.shape)
    
    Q1 = dbc.coupling(K0, dir1, vals1)
    Q2 = dbc.coupling(K0, dir2, vals2)
    
    O = O1*O2
    I = I1+I2
    
    K = O*K0*O + I
    
    F = F0-Q1-Q2
    F[dir1] = vals1[:,None]
    F[dir2] = vals2[:,None]
    
    Lhs = K.toarray()
    Rhs = F.toarray()
    print(f"solving... nonzeros: {K.nnz}")
    t1 = timer()
#    sol = sps.linalg.spsolve(K, F)
    sol = np.linalg.solve(Lhs, Rhs)
    t2 = timer()
    runtime_dense.append(t2-t1)
    print("done!")
    nnz_dense.append(K.nnz)
    
dense_runtimes = np.array(runtime_dense)
dense_nnzs = np.array(nnz_dense)
#%%

plt.figure()
plt.plot(sparse_nnzs, sparse_runtimes)

plt.figure()
plt.plot(dense_nnzs, dense_runtimes)
#
#a = 5/nnzs[-1]**2
#def fun(x):
#    return a*x**2
    
#plt.figure()
#plt.plot(nnzs, [fun(x) for x in nnzs])

#conclusion: looks like O(n^2) where n is nnz(K)

#cbar = plt.tricontourf(cord[:,0], cord[:,1], tri.simplices, sol)
#plt.colorbar(cbar)
