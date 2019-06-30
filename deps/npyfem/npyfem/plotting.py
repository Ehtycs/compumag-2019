from .basis_functions import scalarbasis as sbasis

""" Generic functions for plotting of npstructs and fields in matplotlib """

def plot_mesh(ax, field_or_domain, **kwargs):
    return field_or_domain.plot_mesh(ax, **kwargs)

def plot_colormap(ax, field, pmap=None, **kwargs):
    return field.plot_colormap(ax, pmap, **kwargs)

def plot_quiver(ax, field, **kwargs):
    field.plot_quiver(ax, **kwargs)

def plot_contour(ax, domain, F, **kwargs):
    domain.plot_contour(ax, F, **kwargs)

def plot_contourf(ax, domain, F, **kwargs):
    return domain.plot_contourf(ax, F, **kwargs)
