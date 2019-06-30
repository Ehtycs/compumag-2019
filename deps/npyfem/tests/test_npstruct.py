""" In this test we integrate the volume form sqrt(g) dV over the mesh domain
which should yield the area of the mesh.

This tests the system as a whole, from mesh construction to integration and
assembly of the global system (on a basic level). """

from context import npyfem

import unittest
import numpy as np

from npyfem.npstruct import NPStruct


class TestNPStruct(unittest.TestCase):

    # change the default precision from 7 to 6 decimals
    def assertAlmostEqual(self, v1, v2, precision=6):
        return super().assertAlmostEqual(v1, v2, precision)

    def assertArrEq(self, nparr1, nparr2):
        self.assertTrue(np.array_equal(nparr1, nparr2))

    def setUp(self):
        
        self.egs1 = [1,2,3]
        d1 = np.array([[1,2,3],[4,5,6],[7,8,9]])[None,None,...]            
        datam1 = {eg: d1 for eg in self.egs1}
        self.m1 = NPStruct(datam1, None, None)

        d1 = np.array([[1,2,3]])[None,None,...]            
        datav1 = {eg: d1 for eg in self.egs1}
        self.v1 = NPStruct(datav1, None, None)

    def test_transpose_vector_multiplication(self):
        
        mult1 = self.v1.T @ self.v1
        mult2 = self.v1 @ self.v1.T
        
        res1 = np.array([[1,2,3],[2,4,6], [3,6,9]])[None,None,...]
        data1 = {1: res1, 2: res1, 3: res1}

        for i in self.egs1:
            self.assertTrue(np.all(mult1.data[i] == data1[i]))

        
        res2 = np.array([[14]])[None,None,...]
        data2 = {1: res2, 2: res2, 3: res2}
        
        
        for i in self.egs1:
            self.assertTrue(np.all(mult2.data[i] == data2[i]))
        
if __name__ == '__main__':
    unittest.main()
