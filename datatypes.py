from collections import namedtuple

ReducedModelData = namedtuple('ReducedModelData',['psi',
                                                  'redK',
                                                  'redCc',
                                                  'mortar_C',
                                                  'nodes_xyz',
                                                  'nodes_uvw',
                                                  'redBsc',
                                                  'mesh'])

ReducedModelDataFourier = namedtuple('ReducedModelData',['psi',
                                                         'redK',
                                                         'redCc',
                                                         'nudft_matrix',
                                                         'nodes_xyz',
                                                         'nodes_uvw',
                                                         'redBsc',
                                                         'mesh'])

PreProcessingData = namedtuple('PreProcessingData', ['full_dimension',
                                                     'cpl_dimension',
                                                     'runtime',
                                                     'reduced_dimension'])

CouplingData = namedtuple("CouplingData", 'L1 M12 M21 L2 k')


FullModelData = namedtuple('FullModelData', ['K','Cc', 'Bsc', 'mesh'])

Solver = namedtuple("Solver", 'solve coupling')

DecomposedModel = namedtuple('DecomposedModel', ['mesh',
                                                 'coil_pos',
                                                 'angle',
                                                 'submesh'])

from types import FunctionType
