# Compile the CUDA Code: 1_fdtd_cuda_by_abdullah.cu

%%shell
# export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
nvcc -O3 -o fdtd_cuda fdtd_cuda.cu