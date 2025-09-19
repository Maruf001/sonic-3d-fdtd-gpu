# Compile the CUDA Code: 1i_fdtd_cuda-mixed_v1.cu

%%shell
# export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
nvcc -O3 -o fdtd_mixed fdtd_mixed.cu