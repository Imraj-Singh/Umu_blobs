import unittest
import torch
from Umu_blobs import Umu_blobs_2D, Umu_blobs_3D

class TestUmuBlobs(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.N_r = 100
        self.N_b = 100
        self.r_coords_2D = torch.randn(self.N_r, 1, 2, device='cuda', requires_grad=False, dtype=torch.float64)
        self.U_2D = torch.randn(1, self.N_b, 3, device='cuda', requires_grad=True, dtype=torch.float64)
        self.mu_2D = torch.randn(1, self.N_b, 2, device='cuda', requires_grad=True, dtype=torch.float64)
        self.c_2D = torch.randn(1, self.N_b, 1, device='cuda', requires_grad=True, dtype=torch.float64)
        self.r_coords_3D = torch.randn(self.N_r, 1, 3, device='cuda', requires_grad=False, dtype=torch.float64)
        self.U_3D = torch.randn(1, self.N_b, 6, device='cuda', requires_grad=True, dtype=torch.float64)
        self.mu_3D = torch.randn(1, self.N_b, 3, device='cuda', requires_grad=True, dtype=torch.float64)
        self.c_3D = torch.randn(1, self.N_b, 1, device='cuda', requires_grad=True, dtype=torch.float64)

    def test_gradcheck_2D(self):
        gradcheck_result = torch.autograd.gradcheck(Umu_blobs_2D.apply, (self.r_coords_2D, self.U_2D, self.mu_2D, self.c_2D))
        print(f'2D Gradcheck result: {gradcheck_result}')
        self.assertTrue(gradcheck_result)

    def test_gradcheck_3D(self):
        gradcheck_result = torch.autograd.gradcheck(Umu_blobs_3D.apply, (self.r_coords_3D, self.U_3D, self.mu_3D, self.c_3D))
        print(f'3D Gradcheck result: {gradcheck_result}')
        self.assertTrue(gradcheck_result)
        
if __name__ == '__main__':
    unittest.main()
