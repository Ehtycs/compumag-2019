""" In this test we integrate the volume form sqrt(g) dV over the mesh domain
which should yield the area of the mesh.

This tests the system as a whole, from mesh construction to integration and
assembly of the global system (on a basic level). """

from context import npyfem, open_resource


import unittest
import numpy as np
import meshio as mio

from npyfem import from_gmsh, from_mio

simple2d = "resources/simple2d.msh"
simple2d_2ord = "resources/simple2d_2ord.msh"
beast2d = "resources/beast_2d.msh"
beast2d_2ord = "resources/beast_2d_2ord.msh"

def sum_area(arr):
    return np.sum(np.sum(arr))

class TestIntegrationNpyfemBackend(unittest.TestCase):

    # change the default precision from 7 to 6 decimals
    def assertAlmostEqual(self, v1, v2, precision=6):
        return super().assertAlmostEqual(v1, v2, precision)

    def assertArrEq(self, nparr1, nparr2):
        self.assertTrue(np.array_equal(nparr1, nparr2))

    # helper method
    def integrate_mesh_area(self, file, deg):
        # integrate sqrt(g) over the mesh
        msh = from_mio(open_resource(mio.read, file))
        domain = msh.define_domain([1])
        jac = domain.jacobians(deg)
        g = jac@jac.T
        detg = g.det()
        sqrt_detg = detg.unary_map(lambda x: np.sqrt(x)[..., None, None])

        return sum_area(sqrt_detg.integrate().assemble())

    # actual testcases
    def test_integrate_mesh_areas_simple2d_1ord(self):
        res1 = self.integrate_mesh_area(simple2d, 1)
        self.assertAlmostEqual(1.0, res1)
        res2 = self.integrate_mesh_area(simple2d, 2)
        self.assertAlmostEqual(1.0, res2)


    def test_integrate_mesh_areas_simple2d_2ord(self):
        res1 = self.integrate_mesh_area(simple2d_2ord, 1)
        self.assertAlmostEqual(1.0, res1)
        res2 = self.integrate_mesh_area(simple2d_2ord, 2)
        self.assertAlmostEqual(1.0, res2)

    def test_integrate_mesh_areas_beast2d_1ord(self):
        res1 = self.integrate_mesh_area(beast2d, 1)
        self.assertAlmostEqual(3.0, res1)
        res2 = self.integrate_mesh_area(beast2d, 2)
        self.assertAlmostEqual(3.0, res2)

    def test_integrate_mesh_areas_beast2d_2ord(self):
        res1 = self.integrate_mesh_area(beast2d_2ord, 1)
        self.assertAlmostEqual(3.0, res1)
        res2 = self.integrate_mesh_area(beast2d_2ord, 2)
        self.assertAlmostEqual(3.0, res2)

class TestIntegrationGmshBackend(TestIntegrationNpyfemBackend):

    """ Test integration in beast when using gmsh as a backend.
    Quick'n'dirty version
    """

    def __init__(self, *args,**kwargs):
        # check if gmsh is available
        try:
            import gmsh
            self.gmsh = gmsh
        except (ImportError, ModuleNotFoundError, OSError):
            #oopsie woopsie, SEEMS like gmsh is not available
            # at least it cant be loaded...
            self.gmsh = None
        super().__init__(*args, **kwargs)

    def setUp(self):
        if(self.gmsh is None):
            self.skipTest("Gmsh backend unit tests need gmsh python "
                          "API and gmsh library, which SEEM to be "
                          "unavailable (at least can't be loaded)")


    # helper method, specialized for gmsh backend
    def integrate_mesh_area(self, file, deg):
        # integrate sqrt(g) over the mesh
        self.gmsh.initialize()
        open_resource(self.gmsh.open, file)
        msh = from_gmsh(self.gmsh, backend='gmsh')
        domain = msh.define_domain([(2,1)])
        jac = domain.jacobians(deg)
        g = jac@jac.T
        detg = g.det()
        sqrt_detg = detg.unary_map(lambda x: np.sqrt(x)[..., None, None])

        return sum_area(sqrt_detg.integrate().assemble())

class TestIntegrationGmsh(TestIntegrationGmshBackend):

    """ Test integration in beast when using gmsh to read the mesh but
    npyfem as a backend.
    Quick'n'dirty version
    """

    def integrate_mesh_area(self, file, deg):
        # integrate sqrt(g) over the mesh
        self.gmsh.initialize()
        self.gmsh.open(file)
        msh = from_gmsh(self.gmsh)
        # This should cause an explosion if we try to read something
        # from GMSH after this point
        self.gmsh.finalize()

        domain = msh.define_domain([(2,1)])
        jac = domain.jacobians(deg)
        g = jac@jac.T
        detg = g.det()
        sqrt_detg = detg.unary_map(lambda x: np.sqrt(x)[..., None, None])

        return sum_area(sqrt_detg.integrate().assemble())


if __name__ == '__main__':
    unittest.main()
