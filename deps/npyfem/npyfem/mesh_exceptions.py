""" Lets keep all exceptions here."""


class InvalidMeshException(Exception):
    """ Thrown if mesh has something cheesy going on """


class ElementNotImplementedExeption(InvalidMeshException):
    """ For example an element type which is not implemented """


class MeshParseException(Exception):
    """Generic exception when mesh parsing goes wrong"""


class NodeParseException(MeshParseException):
    """Parsing of a node failed"""


class ElementParseException(MeshParseException):
    """Parsing of an element failed"""


class MshFileException(MeshParseException):
    """Something was wrong with the meshfile"""
