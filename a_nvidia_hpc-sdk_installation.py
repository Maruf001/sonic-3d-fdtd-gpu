%%shell
# wget https://developer.download.nvidia.com/hpc-sdk/25.7/nvhpc_2025_257_Linux_x86_64_cuda_12.9.tar.gz
# tar xpzf nvhpc_2025_257_Linux_x86_64_cuda_12.9.tar.gz
export NVHPC_SILENT=true
export NVHPC_INSTALL_DIR=/opt/nvidia/hpc_sdk
export NVHPC_INSTALL_TYPE=single
cd nvhpc_2025_257_Linux_x86_64_cuda_12.9 && ./install
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
nvc++ --version  # Verify installation