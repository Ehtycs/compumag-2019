from sympy import *

u = Symbol('u', real=True, nonnegative=True)
k = Symbol('k', real=True, positive=True, integer=True)
x1 = Symbol('x1', real=True, nonnegative=True)
L = Symbol('L', real=True, positive=True)

ak1 = integrate((1-u)*cos(2*pi*k*(x1+L*u))*L, (u,0,1))
bk1 = integrate((1-u)*sin(2*pi*k*(x1+L*u))*L, (u,0,1))

ak2 = integrate(u*cos(2*pi*k*(x1+L*u))*L, (u,0,1))
bk2 = integrate(u*sin(2*pi*k*(x1+L*u))*L, (u,0,1))




