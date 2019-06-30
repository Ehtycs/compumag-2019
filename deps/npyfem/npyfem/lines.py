import numpy as np
from .element_group import (ElementGroup, IntegrationPoints, t_uint, t_float)
from .mesh_exceptions import ElementNotImplementedExeption

class Lines(ElementGroup):
#Note that the reference element is defined in [0,1]!!!
    dim = 1

    def get_integration_points(self, degree):
        if degree == 1:
            intpoints = np.array([0.5])
            intweights = np.array([1])
        elif degree == 2:
            intpoints  = np.array([0.788675, 0.211325])
            intweights  = np.array([0.5, 0.5])
        elif degree == 3:
            intpoints  = np.array([0.887298, 0.500000, 0.112702])
            intweights  = np.array([0.2778, 0.4444, 0.2778])
        else:
            raise ElementNotImplementedExeption("Gauss integration of degree "
                                                + str(degree) + " for Lines"
                                                       "not implemented")
        return IntegrationPoints(intpoints, intweights)

class Lines2(Lines):

    _nodepoints = np.array([0, 1])

    @staticmethod
    def eval_basis(basis, points):
        return basis.lines2(points)

    @staticmethod
    def eval_d_basis(basis, points):
        return basis.d_lines2(points)

class Lines3(Lines):

    _nodepoints = np.array([0, 1, 0.5])

    @staticmethod
    def eval_basis(basis, points):
        return basis.lines3(points)

    @staticmethod
    def eval_d_basis(basis, points):
        return basis.d_lines3(points)
