#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void Umu_blobs_2D_forward_kernel(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> value, 
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> r_coords,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> U_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> mu_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c_blobs,
                                            const int N_b) {
    // value size [N_r, 1, 1] (value)
    // r_coords size [N_r, 1, 2] (r_x, r_y)
    // U size [1, N_b, 3] (U_x, U_y, U_xy)
    // mu size [1, N_b, 2] (mu_x, mu_y)
    // c size [1, N_b, 1] (c)
    // N_b = number of blobs
    const int r_coord_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_coord_index >= value.size(0)) {
        return;
    }
    for (int i = 0; i < N_b; i++) {
        const scalar_t tmp_r_mu_x = r_coords[r_coord_index][0][0] - mu_blobs[0][i][0];
        const scalar_t tmp_r_mu_y = r_coords[r_coord_index][0][1] - mu_blobs[0][i][1];
        const scalar_t tmp_1 = U_blobs[0][i][0] * tmp_r_mu_x + U_blobs[0][i][2] * tmp_r_mu_y;
        const scalar_t tmp_2 = U_blobs[0][i][1] * tmp_r_mu_y;
        value[r_coord_index][0][0] += c_blobs[0][i][0] * exp(-.5 * (tmp_1 * tmp_1 + tmp_2 * tmp_2));
        delete &tmp_r_mu_x, &tmp_r_mu_y, &tmp_1, &tmp_2;
    };
}

template <typename scalar_t>
__global__ void Umu_blobs_2D_backward_kernel(const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> r_coords,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> U_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> mu_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c_blobs,
                                            torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> U_blobs_grad,
                                            torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> mu_blobs_grad,
                                            torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c_blobs_grad,
                                            const int N_r) {
    // grad_output size [N_r, 1, 1] (grad_output)
    // r_coords size [N_r, 1, 2] (r_x, r_y)
    // U size [1, N_b, 3] (U_x, U_y, U_xy)
    // mu size [1, N_b, 2] (mu_x, mu_y)
    // c size [1, N_b, 1] (c)
    // U_grad size [1, N_b, 3] (U_x, U_y, U_xy)
    // mu_grad size [1, N_b, 2] (mu_x, mu_y)
    // c_grad size [1, N_b, 1] (c)
    // N_r = number of r_coords
    const int blob_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (blob_index >= c_blobs.size(1)) {
        return;
    }
    for (int i = 0; i < N_r; i++) {
        const scalar_t tmp_r_mu_x = r_coords[i][0][0] - mu_blobs[0][blob_index][0];
        const scalar_t tmp_r_mu_y = r_coords[i][0][1] - mu_blobs[0][blob_index][1];
        const scalar_t tmp_1 = U_blobs[0][blob_index][0] * tmp_r_mu_x + U_blobs[0][blob_index][2] * tmp_r_mu_y;
        const scalar_t tmp_2 = U_blobs[0][blob_index][1] * tmp_r_mu_y;
        const scalar_t tmp_exponent = exp(-.5 * (tmp_1 * tmp_1 + tmp_2 * tmp_2)) * grad_output[i][0][0];
        const scalar_t tmp_c_exponent = c_blobs[0][blob_index][0] * tmp_exponent;
        // U_grad = -cU(r-mu)(r-mu)^Texp(-.5||U(r-mu)||^2)
        U_blobs_grad[0][blob_index][0] -= (tmp_r_mu_x * tmp_1) * tmp_c_exponent;
        U_blobs_grad[0][blob_index][1] -= (tmp_r_mu_y * tmp_2) * tmp_c_exponent;
        U_blobs_grad[0][blob_index][2] -= (tmp_r_mu_y * tmp_1) * tmp_c_exponent;
        // mu_grad = cU^TU(r-mu)exp(-.5||U(r-mu)||^2)
        mu_blobs_grad[0][blob_index][0] += (U_blobs[0][blob_index][0] * tmp_1) * tmp_c_exponent;
        mu_blobs_grad[0][blob_index][1] += (U_blobs[0][blob_index][2] * tmp_1 + U_blobs[0][blob_index][1] * tmp_2) * tmp_c_exponent;
        // c_grad = exp(-.5||U(r-mu)||^2)
        c_blobs_grad[0][blob_index][0] += tmp_exponent;
        delete &tmp_r_mu_x, &tmp_r_mu_y, &tmp_1, &tmp_2, &tmp_exponent, &tmp_c_exponent;
    };
}

template <typename scalar_t>
__global__ void Umu_blobs_3D_forward_kernel(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> value, 
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> r_coords,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> U_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> mu_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c_blobs,
                                            const int N_r,
                                            const int N_b) {
    // value size [N_r, 1, 1] (value)
    // r_coords size [N_r, 1, 3] (r_x, r_y, r_z)
    // U size [1, N_b, 6] (U_x, U_y, U_z, U_xy, U_xz, U_yz)
    // mu size [1, N_b, 3] (mu_x, mu_y, mu_z)
    // c size [1, N_b, 1] (c)
    // N_r = number of r_coords
    // N_b = number of blobs
    const int r_coord_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (r_coord_index >= N_r) {
        return;
    }
    for (int i = 0; i < N_b; i++) {
        const scalar_t tmp_r_mu_x = r_coords[r_coord_index][0][0] - mu_blobs[0][i][0];
        const scalar_t tmp_r_mu_y = r_coords[r_coord_index][0][1] - mu_blobs[0][i][1];
        const scalar_t tmp_r_mu_z = r_coords[r_coord_index][0][2] - mu_blobs[0][i][2];
        const scalar_t tmp_1 = U_blobs[0][i][0]*tmp_r_mu_x + U_blobs[0][i][3]*tmp_r_mu_y + U_blobs[0][i][4]*tmp_r_mu_z;
        const scalar_t tmp_2 = U_blobs[0][i][1]*tmp_r_mu_y + U_blobs[0][i][5]*tmp_r_mu_z;
        const scalar_t tmp_3 = U_blobs[0][i][2]*tmp_r_mu_z;
        value[r_coord_index][0][0] += c_blobs[0][i][0] * exp(-.5 * (tmp_1 * tmp_1 + tmp_2 * tmp_2 + tmp_3 * tmp_3));
        delete &tmp_r_mu_x, &tmp_r_mu_y, &tmp_r_mu_z, &tmp_1, &tmp_2, &tmp_3;
    };
}

template <typename scalar_t>
__global__ void Umu_blobs_3D_backward_kernel(const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> r_coords,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> U_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> mu_blobs,
                                            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c_blobs,
                                            torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> U_blobs_grad,
                                            torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> mu_blobs_grad,
                                            torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c_blobs_grad,
                                            const int N_r,
                                            const int N_b) {
    
    // grad_output size [N_r, 1, 1] (grad_output)
    // r_coords size [N_r, 1, 3] (r_x, r_y, r_z)
    // U size [1, N_b, 6] (U_x, U_y, U_z, U_xy, U_xz, U_yz)
    // mu size [1, N_b, 3] (mu_x, mu_y, mu_z)
    // c size [1, N_b, 1] (c)
    // U_grad size [1, N_b, 6] (U_x, U_y, U_z, U_xy, U_xz, U_yz)
    // mu_grad size [1, N_b, 3] (mu_x, mu_y, mu_z)
    // c_grad size [1, N_b, 1] (c)
    // N_r = number of r_coords
    // N_b = number of blobs
    const int blob_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (blob_index >= N_b) {
        return;
    }
    for (int i = 0; i < N_r; i++) {
        const scalar_t tmp_r_mu_x = r_coords[i][0][0] - mu_blobs[0][blob_index][0];
        const scalar_t tmp_r_mu_y = r_coords[i][0][1] - mu_blobs[0][blob_index][1];
        const scalar_t tmp_r_mu_z = r_coords[i][0][2] - mu_blobs[0][blob_index][2];
        const scalar_t tmp_1 = U_blobs[0][blob_index][0] * tmp_r_mu_x + U_blobs[0][blob_index][3] * tmp_r_mu_y + U_blobs[0][blob_index][4] * tmp_r_mu_z;
        const scalar_t tmp_2 = U_blobs[0][blob_index][1] * tmp_r_mu_y + U_blobs[0][blob_index][5] * tmp_r_mu_z;
        const scalar_t tmp_3 = U_blobs[0][blob_index][2] * tmp_r_mu_z;
        const scalar_t tmp_exponent = exp(-.5 * (tmp_1 * tmp_1 + tmp_2 * tmp_2 + tmp_3 * tmp_3)) * grad_output[i][0][0];
        const scalar_t tmp_c_exponent = c_blobs[0][blob_index][0] * tmp_exponent;
        // U_grad = -cU(r-mu)(r-mu)^Texp(-.5||U(r-mu)||^2)
        U_blobs_grad[0][blob_index][0] -= (tmp_r_mu_x * tmp_1) * tmp_c_exponent;
        U_blobs_grad[0][blob_index][1] -= (tmp_r_mu_y * tmp_2) * tmp_c_exponent;
        U_blobs_grad[0][blob_index][2] -= (tmp_r_mu_z * tmp_3) * tmp_c_exponent;
        U_blobs_grad[0][blob_index][3] -= (tmp_r_mu_y * tmp_1) * tmp_c_exponent;
        U_blobs_grad[0][blob_index][4] -= (tmp_r_mu_z * tmp_1) * tmp_c_exponent;
        U_blobs_grad[0][blob_index][5] -= (tmp_r_mu_z * tmp_2) * tmp_c_exponent;
        // mu_grad = cU^TU(r-mu)exp(-.5||U(r-mu)||^2)
        mu_blobs_grad[0][blob_index][0] += (U_blobs[0][blob_index][0] * tmp_1) * tmp_c_exponent;
        mu_blobs_grad[0][blob_index][1] += (U_blobs[0][blob_index][3] * tmp_1 + U_blobs[0][blob_index][1] * tmp_2) * tmp_c_exponent;
        mu_blobs_grad[0][blob_index][2] += (U_blobs[0][blob_index][4] * tmp_1 + U_blobs[0][blob_index][5] * tmp_2 + U_blobs[0][blob_index][2] * tmp_3) * tmp_c_exponent;
        // c_grad = exp(-.5||U(r-mu)||^2)
        c_blobs_grad[0][blob_index][0] += tmp_exponent;
        //delete &tmp_r_mu_x, &tmp_r_mu_y, &tmp_r_mu_z, &tmp_1, &tmp_2, &tmp_3, &tmp_exponent, &tmp_c_exponent;
    };
}

torch::Tensor Umu_blobs_2D_forward(torch::Tensor r_coords, torch::Tensor U, torch::Tensor mu, torch::Tensor c) {
    CHECK_INPUT(r_coords);
    CHECK_INPUT(U);
    CHECK_INPUT(mu);
    CHECK_INPUT(c);
    // r_coords size [N_r, 1, 2] (r_x, r_y)
    // U size [1, N_b, 3] (U_x, U_y, U_xy)
    // mu size [1, N_b, 2] (mu_x, mu_y)
    // c size [1, N_b, 1] (c)
    const int threads = 1024;
    const int blocks = (r_coords.size(0) + threads - 1) / threads;
    auto value = torch::zeros({r_coords.size(0),1,1}, r_coords.options());
    AT_DISPATCH_FLOATING_TYPES(c.type(), "Umu_blobs_2D_forward_cuda", ([&] {
    Umu_blobs_2D_forward_kernel<<<blocks, threads>>>(value.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    r_coords.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    U.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    mu.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    c.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    c.size(1));
    }));
    return value;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Umu_blobs_2D_backward(torch::Tensor grad_output, 
                        torch::Tensor r_coords, 
                        torch::Tensor U, 
                        torch::Tensor mu, 
                        torch::Tensor c) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(r_coords);
    CHECK_INPUT(U);
    CHECK_INPUT(mu);
    CHECK_INPUT(c);
    // grad_output size [N_r, 1, 1] (grad_output)
    // r_coords size [N_r, 1, 3] (r_x, r_y)
    // U size [1, N_b, 3] (U_x, U_y, U_xy)
    // mu size [1, N_b, 2] (mu_x, mu_y)
    // c size [1, N_b, 1] (c)
    const int threads = 1024;
    const int blocks = (c.size(1) + threads - 1) / threads;
    auto U_grad = torch::zeros_like(U);
    auto mu_grad = torch::zeros_like(mu);
    auto c_grad = torch::zeros_like(c);
    AT_DISPATCH_FLOATING_TYPES(c.type(), "Umu_blobs_2D_backward_cuda", ([&] {
    Umu_blobs_2D_backward_kernel<<<blocks, threads>>>(grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    r_coords.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    U.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    mu.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    c.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    U_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    mu_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    c_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    r_coords.size(0));
    }));
    return std::make_tuple(U_grad, mu_grad, c_grad);
}


torch::Tensor Umu_blobs_3D_forward(torch::Tensor r_coords, torch::Tensor U, torch::Tensor mu, torch::Tensor c) {
    CHECK_INPUT(r_coords);
    CHECK_INPUT(U);
    CHECK_INPUT(mu);
    CHECK_INPUT(c);
    // r_coords size [N_r, 1, 3] (r_x, r_y, r_z)
    // U size [1, N_b, 6] (U_x, U_y, U_z, U_xy, U_xz, U_yz)
    // mu size [1, N_b, 3] (mu_x, mu_y, mu_z)
    // c size [1, N_b, 1] (c)
    const int threads = 1024;
    const int blocks = (r_coords.size(0) + threads - 1) / threads;
    auto value = torch::zeros({r_coords.size(0),1,1}, r_coords.options());
    AT_DISPATCH_FLOATING_TYPES(c.type(), "Umu_blobs_3D_forward_cuda", ([&] {
    Umu_blobs_3D_forward_kernel<<<blocks, threads>>>(value.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    r_coords.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    U.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    mu.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    c.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    r_coords.size(0),
                                                    c.size(1));
    }));
    return value;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Umu_blobs_3D_backward(torch::Tensor grad_output, 
                        torch::Tensor r_coords, 
                        torch::Tensor U, 
                        torch::Tensor mu, 
                        torch::Tensor c) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(r_coords);
    CHECK_INPUT(U);
    CHECK_INPUT(mu);
    CHECK_INPUT(c);
    // grad_output size [N_r, 1, 1] (grad_output)
    // r_coords size [N_r, 1, 3] (r_x, r_y, r_z)
    // U size [1, N_b, 6] (U_x, U_y, Uz, U_xy, U_xz, U_yz)
    // mu size [1, N_b, 3] (mu_x, mu_y, mu_z)
    // c size [1, N_b, 1] (c)
    // if c.type() is float then use 1024 threads, if c.type() is double then use 512 threads
    int threads;
    if (c.type().scalarType() == at::ScalarType::Float) {
        threads = 1024;
    } else {
        threads = 768;
    }
    const int blocks = (c.size(1) + threads - 1) / threads;
    auto U_grad = torch::zeros_like(U);
    auto mu_grad = torch::zeros_like(mu);
    auto c_grad = torch::zeros_like(c);
    AT_DISPATCH_FLOATING_TYPES(c.type(), "Umu_blobs_3D_backward_cuda", ([&] {
    Umu_blobs_3D_backward_kernel<<<blocks, threads>>>(grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    r_coords.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    U.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    mu.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    c.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    U_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    mu_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    c_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                                    r_coords.size(0),
                                                    c.size(1));
    }));
    return std::make_tuple(U_grad, mu_grad, c_grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("Umu_blobs_2D_forward_cuda", &Umu_blobs_2D_forward, "Umu_blobs_2D forward");
    m.def("Umu_blobs_2D_backward_cuda", &Umu_blobs_2D_backward, "Umu_blobs_2D backward");
    m.def("Umu_blobs_3D_forward_cuda", &Umu_blobs_3D_forward, "Umu_blobs_3D forward");
    m.def("Umu_blobs_3D_backward_cuda", &Umu_blobs_3D_backward, "Umu_blobs_3D backward");
}