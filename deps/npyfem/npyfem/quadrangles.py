import numpy as np
from .element_group import (ElementGroup, IntegrationPoints, t_uint, t_float)
from .mesh_exceptions import ElementNotImplementedExeption

geom_coeff = 4

class Quadrangles(ElementGroup):
#Note that the reference element is defined in [-1,1]x[-1,1] !!!
    dim = 2
    #geometric_coefficient = 4

    def get_integration_points(self, degree):
        # from http://paulino.ce.gatech.edu/courses/cee570/2014/Class_notes/CEE570_ppt31.pdf
        if degree == 1:
            # one point quadrature
            intpoints = np.array([[0, 0]])
            intweights = geom_coeff*np.array([1])
        elif degree == 2 :
            # four point quadrature
            intpoints = (
                np.sqrt(3)/3*np.array([[-1 , -1],
                                       [ 1 , -1],
                                       [ 1 ,  1],
                                       [-1 ,  1]  ]) )
            intweights = geom_coeff*1/4*np.array([1, 1, 1, 1])
        elif degree == 3:
            # nine point quadrature
            intpoints = (np.sqrt(15)/5*
                             np.array([[-1 , -1],
                                       [ 1 , -1],
                                       [ 1 ,  1],
                                       [-1 ,  1],
                                       [ 0 , -1],
                                       [ 1 ,  0],
                                       [ 0 ,  1],
                                       [-1 ,  0],
                                       [ 0 ,  0]  ]) )
            intweights = geom_coeff*np.array([25/324,25/324,25/324,25/324,40/324,40/324,40/324,40/324,64/324])
        else:
            raise ElementNotImplementedExeption("Gauss integration of degree "
                                                +str(degree)+" for Quadrangles "
                                                "not implemented")

        return IntegrationPoints(intpoints, intweights)


class Quadrangles4(Quadrangles):

    _nodepoints = np.array([[ -1 , -1],
                            [  1 , -1],
                            [  1 ,  1],
                            [ -1 ,  1]  ])
    @staticmethod
    def eval_basis(basis, points):
        return basis.quadrangles4(points)

    @staticmethod
    def eval_d_basis(basis, points):
        return basis.d_quadrangles4(points)

class Quadrangles9(Quadrangles):

    _nodepoints=np.array( [ [ -1 , -1 ],
                            [  1 , -1 ],
                            [  1 ,  1 ],
                            [ -1 ,  1 ],
                            [  0 , -1 ],
                            [  1 ,  0 ],
                            [  0 ,  1 ],
                            [ -1 ,  0 ],
                            [  0 ,  0 ]  ] )

    @staticmethod
    def eval_basis(basis, points):
        return basis.quadrangles9(points)

    @staticmethod
    def eval_d_basis(basis, points):
        return basis.d_quadrangles9(points)
