# Compile the OpenACC Code: 1_pde-fdtd_openacc_by_ziyi-yin.cpp

%%shell
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:$PATH
nvc++ -acc -Minfo=accel -o fdtd_openacc fdtd_openacc.cpp