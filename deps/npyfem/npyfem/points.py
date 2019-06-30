import numpy as np
from .element_group import (ElementGroup, t_uint, t_float)
from .mesh_exceptions import ElementNotImplementedExeption

class Points(ElementGroup):

    dim = 0
    nodel = 1
