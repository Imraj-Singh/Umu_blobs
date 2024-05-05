import torch
from Umu_blobs.cuda_extension import Umu_blobs_2D_forward_cuda, Umu_blobs_2D_backward_cuda, Umu_blobs_3D_forward_cuda, Umu_blobs_3D_backward_cuda

class Umu_blobs_2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_coords, U, mu, c):
        ctx.save_for_backward(r_coords, U, mu, c)
        return Umu_blobs_2D_forward_cuda(r_coords, U, mu, c)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        r_coords, U, mu, c = ctx.saved_tensors
        grad_U, grad_mu, grad_c = Umu_blobs_2D_backward_cuda(grad_output, r_coords, U, mu, c)
        return None, grad_U, grad_mu, grad_c
    
class Umu_blobs_3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_coords, U, mu, c):
        ctx.save_for_backward(r_coords, U, mu, c)
        return Umu_blobs_3D_forward_cuda(r_coords, U, mu, c)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        r_coords, U, mu, c = ctx.saved_tensors
        grad_U, grad_mu, grad_c = Umu_blobs_3D_backward_cuda(grad_output, r_coords, U, mu, c)
        return None, grad_U, grad_mu, grad_c