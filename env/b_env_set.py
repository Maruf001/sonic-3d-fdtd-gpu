%%bash
set -euo pipefail

# Work on local SSD (fast)
cd /content

# ---- Adjust this path if your folder name differs ----
SRC="/content/drive/MyDrive/Colab Notebooks/HPC_FDTD/nvhpc_2025_257_Linux_x86_64_cuda_12.9.tar.gz"
# ------------------------------------------------------

# Copy the archive from Drive to local disk
cp "$SRC" /content/

# Faster untar (pigz optional)
apt-get -y update >/dev/null
apt-get -y install pigz >/dev/null || true
if command -v pigz >/dev/null 2>&1; then
  tar -I pigz -xpf nvhpc_2025_257_Linux_x86_64_cuda_12.9.tar.gz
else
  tar -xpf nvhpc_2025_257_Linux_x86_64_cuda_12.9.tar.gz
fi

# Install into a path with no spaces
export NVHPC_SILENT=true
export NVHPC_INSTALL_TYPE=single
export NVHPC_INSTALL_DIR=/content/nvhpc
/content/nvhpc_2025_257_Linux_x86_64_cuda_12.9/install

# Put compilers on PATH (for this shell)
export PATH=/content/nvhpc/Linux_x86_64/25.7/compilers/bin:$PATH

# Verify
which nvc++
nvc++ --version
