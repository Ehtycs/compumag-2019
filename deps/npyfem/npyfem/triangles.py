import numpy as np
from .element_group import (ElementGroup, IntegrationPoints, t_uint, t_float)
from .mesh_exceptions import ElementNotImplementedExeption

geom_coeff = 0.5

class Triangles(ElementGroup):

    dim = 2
    # geometric_coefficient = 0.5

    def get_integration_points(self, degree):
        # from http://math2.uncc.edu/~shaodeng/TEACHING/math5172/Lectures/Lect_15.PDF
        if degree == 1:
            intpoints = np.array([[1/3, 1/3]])
            intweights =  geom_coeff*np.array([1])
        elif degree == 2:
            intpoints = np.array([[1/6 , 1/6],
                              [4/6 , 1/6],
                              [1/6 , 4/6]])
            intweights = geom_coeff*np.array([1/3, 1/3, 1/3])
        elif degree == 3:
            intpoints = np.array([[1/3 , 1/3],
                              [1/5 , 3/5],
                              [1/5 , 1/5],
                              [3/5 , 1/5]])
            intweights = geom_coeff*np.array([-27/48, 25/48, 25/48, 25/48])
        else:
            raise ElementNotImplementedExeption("Gauss integration of degree "
                                                +str(degree)+" for Triangles "
                                                "not implemented")
        return IntegrationPoints(intpoints, intweights)

class Triangles3(Triangles):

    _nodepoints = np.array([[0,0],[1,0],[0,1]])

    @staticmethod
    def eval_basis(basis, points):
        return basis.triangles3(points)

    @staticmethod
    def eval_d_basis(basis, points):
        return basis.d_triangles3(points)

class Triangles6(Triangles):

    _nodepoints = np.array([[0,0],[1,0],[0,1],[0.5,0],[0.5,0.5],[0,0.5]])

    @staticmethod
    def eval_basis(basis, points):
        return basis.triangles6(points)

    @staticmethod
    def eval_d_basis(basis, points):
        return basis.d_triangles6(points)
