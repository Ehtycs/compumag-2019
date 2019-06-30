import numpy as np

t_float = np.float64
t_int = np.int64
t_uint = np.uint64

# Standard scalar basis implementation
# One can provide a different basis by implementing a module offering
# the needed functions and feeding it to Integrator class constructor

# This basis is a a "special case" because this is anyway needed in
# isoparametric mapping
"""
* * * * For Lines * * * *
"""

# 2- Node Line

def lines2(t):
    return np.array((1-t,t),ndmin=4).swapaxes(2,3).swapaxes(0,2)

def d_lines2(t):
    return np.array([-1,1])*np.ones((t.size,1,1,1))

# 3- Node Line

def lines3(t):
    return np.array( ( (2*t-1)*(t-1),(2*t-1)*t,1-(2*t-1)*(2*t-1) ),
                    ndmin=4).swapaxes(2,3).swapaxes(0,2)

def d_lines3(t):
    return np.array((4*t-3,4*t-1,4*(1-2*t)),
                    ndmin=4).swapaxes(2,3).swapaxes(0,2)


"""
* * * * For Triangles * * * *
"""

# 3 - Node Triangle

def triangles3(t):
    return np.array((1-t[:,0]-t[:,1],
                     t[:,0],
                     t[:,1]),ndmin=4, dtype=t_float).swapaxes(2,3).swapaxes(0,2)

def d_triangles3(t):
    return np.array([[-1,1,0], [-1,0,1]], dtype=t_float)*np.ones((t.shape[0],1,1,1))

# 6 - Node Triangle

def triangles6(t):
    return np.array(( ( 1-t[:,0]-t[:,1])*(1-2*t[:,0]-2*t[:,1]) ,
                      t[:,0]*(2*t[:,0]-1) ,
                      t[:,1]*(2*t[:,1]-1) ,
                      4*t[:,0]*(1-t[:,0]-t[:,1]) ,
                      4*t[:,0]*t[:,1] ,
                      4*t[:,1]*(1-t[:,0]-t[:,1])
                        ),ndmin=4, dtype=t_float).swapaxes(2,3).swapaxes(0,2)

def d_triangles6(t):
    return np.array(( ( 4*t[:,0]+4*t[:,1]-3 , 4*t[:,0]+4*t[:,1]-3 ) ,
                     ( 4*t[:,0]-1 , np.zeros(t.shape[0]) ) ,
                     ( np.zeros(t.shape[0]) , 4*t[:,1]-1 ) ,
                     ( 4*(1-t[:,1]-2*t[:,0]), -4*t[:,0] ) ,
                     ( 4*t[:,1] , 4*t[:,0]  ) ,
                     ( -4*t[:,1] , 4*(1-2*t[:,1]-t[:,0]) ) ,
                        ),ndmin=4, dtype=t_float).swapaxes(0,3).swapaxes(1,3)


"""
* * * * For Quadrangles * * * *
"""

# 4 - Node Quadrangle
def quadrangles4(t):
#shape is of the form (intpoints,elements=1,1,dof)
    return np.array(( 1/4*(1-t[:,0])*(1-t[:,1]),
                   1/4*(1+t[:,0])*(1-t[:,1]),
                   1/4*(1+t[:,0])*(1+t[:,1]),
                   1/4*(1-t[:,0])*(1+t[:,1]) ),
                   ndmin=4, dtype=t_float).swapaxes(2,3).swapaxes(0,2)

def d_quadrangles4(t):
    #d_shape of the form (intpoints,elements=1,coordinates,dof)
    return np.array([[-1/4,1/4,1/4,-1/4],
                    [-1/4,-1/4,1/4,1/4]], dtype=t_float)*np.ones((t.shape[0],1,1,1))

# 9 - Node Quadrangle
def quadrangles9(t):
    return np.array((  1/4*t[:,0]*t[:,1]*(1-t[:,0])*(1-t[:,1]),
                       -1/4*t[:,0]*t[:,1]*(1+t[:,0])*(1-t[:,1]),
                        1/4*t[:,0]*t[:,1]*(1+t[:,0])*(1+t[:,1]),
                       -1/4*t[:,0]*t[:,1]*(1-t[:,0])*(1+t[:,1]),
                       -1/2*t[:,1]*(1-t[:,0]*t[:,0])*(1-t[:,1]),
                        1/2*t[:,0]*(1+t[:,0])*(1-t[:,1]*t[:,1]),
                        1/2*t[:,1]*(1-t[:,0]*t[:,0])*(1+t[:,1]),
                       -1/2*t[:,0]*(1-t[:,0])*(1-t[:,1]*t[:,1]),
                        (1-t[:,0]*t[:,0])*(1-t[:,1]*t[:,1]) ),
                        ndmin=4, dtype=t_float).swapaxes(2,3).swapaxes(0,2)
def d_quadrangles9(t):
    return np.array(( (  1/4*t[:,1]*(1-t[:,1])*(1-2*t[:,0]) ,  1/4*t[:,0]*(1-t[:,0])*(1-2*t[:,1]) ) ,
                         ( -1/4*t[:,1]*(1-t[:,1])*(1+2*t[:,0]) , -1/4*t[:,0]*(1+t[:,0])*(1-2*t[:,1]) ) ,
                         (  1/4*t[:,1]*(1+t[:,1])*(1+2*t[:,0]) ,  1/4*t[:,0]*(1+t[:,0])*(1+2*t[:,1]) ) ,
                         ( -1/4*t[:,1]*(1+t[:,1])*(1-2*t[:,0]) , -1/4*t[:,0]*(1-t[:,0])*(1+2*t[:,1]) ) ,
                         (          t[:,0]*t[:,1]*(1  -t[:,1]) , -1/2*(1-t[:,0]*t[:,0])*(1-2*t[:,1]) ) ,
                         (  1/2*(1-t[:,1]*t[:,1])*(1+2*t[:,0]) , -t[:,1]*t[:,0]*(1+t[:,0])           ) ,
                         (         -t[:,0]*t[:,1]*(1  +t[:,1]) ,  1/2*(1-t[:,0]*t[:,0])*(1+2*t[:,1]) ) ,
                         ( -1/2*(1-t[:,1]*t[:,1])*(1-2*t[:,0]) ,  t[:,1]*t[:,0]*(1-t[:,0])           ) ,
                         (         -2*t[:,0]*(1-t[:,1]*t[:,1]) , -2*t[:,1]*(1-t[:,0]*t[:,0])  )
                                                        ),ndmin=4, dtype=t_float).swapaxes(0,3).swapaxes(1,3)
