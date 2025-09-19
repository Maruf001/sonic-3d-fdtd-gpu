// Mixed Precision CUDA Code : v1

%%writefile fdtd_mixed.cu
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// # ----------------------------------------
// # Tested on A100 GPU, v6e1 TPU, L4 GPU 
// #  A100 GPU gives highest performance gain. 
// # ----------------------------------------

#define CHECK_CUDA(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(1); \
  } \
}

struct dataobj {
  void *__restrict data;
  int *size;
  unsigned long nbytes;
  unsigned long *npsize;
  unsigned long *dsize;
  int *hsize;
  int *hofs;
  int *oofs;
  void *dmap;
};

struct profiler {
  double section0;
  double section1;
};

#define START(S) struct timeval start_##S, end_##S; gettimeofday(&start_##S, NULL);
#define STOP(S, T) gettimeofday(&end_##S, NULL); T->S += (double)(end_##S.tv_sec - start_##S.tv_sec) + (double)(end_##S.tv_usec - start_##S.tv_usec) / 1000000;

__device__ void atomicAdd_half(half *address, half val) {
  unsigned int * address_as_ui = reinterpret_cast<unsigned int *>(address);
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    half new_val = __float2half(__half2float(*reinterpret_cast<half *>(&old)) + __half2float(val));
    old = atomicCAS(address_as_ui, assumed, *reinterpret_cast<unsigned int *>(&new_val));
  } while (assumed != old);
}

__global__ void update_kernel(half * __restrict__ u, const half * __restrict__ m, int full_x, int full_y, int full_z, float dt, float h_x, float h_y, float h_z, int t0, int t1, int t2, int x_m, int x_M, int y_m, int y_M, int z_m, int z_M) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + x_m;
  int y = blockIdx.y * blockDim.y + threadIdx.y + y_m;
  int z = blockIdx.z * blockDim.z + threadIdx.z + z_m;

  if (x > x_M || y > y_M || z > z_M) return;

  half r1 = __float2half(1.0f / (dt * dt));
  half r2 = __float2half(1.0f / (h_x * h_x));
  half r3 = __float2half(1.0f / (h_y * h_y));
  half r4 = __float2half(1.0f / (h_z * h_z));

  int offset = 4;
  int idx = t0 * full_x * full_y * full_z + (x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset);
  half r5 = __float2half(-2.50f) * u[idx];

  half lap_x = r5 + __float2half(-8.33333333e-2f) * (u[idx - 2 * full_y * full_z] + u[idx + 2 * full_y * full_z]) + __float2half(1.33333333f) * (u[idx - full_y * full_z] + u[idx + full_y * full_z]);
  half lap_y = r5 + __float2half(-8.33333333e-2f) * (u[idx - 2 * full_z] + u[idx + 2 * full_z]) + __float2half(1.33333333f) * (u[idx - full_z] + u[idx + full_z]);
  half lap_z = r5 + __float2half(-8.33333333e-2f) * (u[idx - 2] + u[idx + 2]) + __float2half(1.33333333f) * (u[idx - 1] + u[idx + 1]);

  half mul = (__float2half(-2.0f) * r1 * u[idx] + r1 * u[t1 * full_x * full_y * full_z + (x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset)]) * m[(x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset)];

  int out_idx = t2 * full_x * full_y * full_z + (x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset);
  u[out_idx] = __float2half(dt * dt) * (r2 * lap_x + r3 * lap_y + r4 * lap_z - mul) / m[(x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset)];
}

__global__ void update_kernel_shared(half * __restrict__ u, const half * __restrict__ m, int full_x, int full_y, int full_z, float dt, float h_x, float h_y, float h_z, int t0, int t1, int t2, int x_m, int x_M, int y_m, int y_M, int z_m, int z_M) {
  extern __shared__ half sh_u[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;
  int bz = blockIdx.z * blockDim.z;

  int x = bx + tx + x_m;
  int y = by + ty + y_m;
  int z = bz + tz + z_m;

  if (x > x_M || y > y_M || z > z_M) return;

  int offset = 4;
  int lx = blockDim.x + 4;
  int ly = blockDim.y + 4;

  int local_idx = (tx + 2) + (ty + 2) * lx + (tz + 2) * lx * ly;
  int global_idx = t0 * full_x * full_y * full_z + (x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset);

  sh_u[local_idx] = u[global_idx];

  if (tx < 2) sh_u[tx + (ty + 2) * lx + (tz + 2) * lx * ly] = (bx + tx >= x_m) ? u[global_idx - 2 * full_y * full_z] : __float2half(0.0f);
  if (tx >= blockDim.x - 2) sh_u[tx + 4 + (ty + 2) * lx + (tz + 2) * lx * ly] = (bx + tx + 2 <= x_M) ? u[global_idx + 2 * full_y * full_z] : __float2half(0.0f);
  if (ty < 2) sh_u[(tx + 2) + ty * lx + (tz + 2) * lx * ly] = (by + ty >= y_m) ? u[global_idx - 2 * full_z] : __float2half(0.0f);
  if (ty >= blockDim.y - 2) sh_u[(tx + 2) + (ty + 4) * lx + (tz + 2) * lx * ly] = (by + ty + 2 <= y_M) ? u[global_idx + 2 * full_z] : __float2half(0.0f);
  if (tz < 2) sh_u[(tx + 2) + (ty + 2) * lx + tz * lx * ly] = (bz + tz >= z_m) ? u[global_idx - 2] : __float2half(0.0f);
  if (tz >= blockDim.z - 2) sh_u[(tx + 2) + (ty + 2) * lx + (tz + 4) * lx * ly] = (bz + tz + 2 <= z_M) ? u[global_idx + 2] : __float2half(0.0f);

  __syncthreads();

  half r1 = __float2half(1.0f / (dt * dt));
  half r2 = __float2half(1.0f / (h_x * h_x));
  half r3 = __float2half(1.0f / (h_y * h_y));
  half r4 = __float2half(1.0f / (h_z * h_z));

  half r5 = __float2half(-2.50f) * sh_u[local_idx];

  half lap_x = r5 + __float2half(-8.33333333e-2f) * (sh_u[local_idx - 2 * lx * ly] + sh_u[local_idx + 2 * lx * ly]) + __float2half(1.33333333f) * (sh_u[local_idx - lx * ly] + sh_u[local_idx + lx * ly]);
  half lap_y = r5 + __float2half(-8.33333333e-2f) * (sh_u[local_idx - 2 * lx] + sh_u[local_idx + 2 * lx]) + __float2half(1.33333333f) * (sh_u[local_idx - lx] + sh_u[local_idx + lx]);
  half lap_z = r5 + __float2half(-8.33333333e-2f) * (sh_u[local_idx - 2] + sh_u[local_idx + 2]) + __float2half(1.33333333f) * (sh_u[local_idx - 1] + sh_u[local_idx + 1]);

  int m_idx = (x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset);
  half mul = (__float2half(-2.0f) * r1 * sh_u[local_idx] + r1 * u[t1 * full_x * full_y * full_z + (x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset)]) * m[m_idx];

  int out_idx = t2 * full_x * full_y * full_z + (x + offset) * full_y * full_z + (y + offset) * full_z + (z + offset);
  u[out_idx] = __float2half(dt * dt) * (r2 * lap_x + r3 * lap_y + r4 * lap_z - mul) / m[m_idx];
}

__global__ void source_kernel(const float * __restrict__ src, const float * __restrict__ src_coords, const half * __restrict__ m, half * __restrict__ u, int time, float h_x, float h_y, float h_z, float o_x, float o_y, float o_z, int p_src_m, int p_src_M, int x_m, int x_M, int y_m, int y_M, int z_m, int z_M, int t2, int full_x, int full_y, int full_z, int num_src) {
  int p_src = blockIdx.x + p_src_m;
  if (p_src > p_src_M) return;

  for (int rsrcx = 0; rsrcx <= 1; ++rsrcx) {
    for (int rsrcy = 0; rsrcy <= 1; ++rsrcy) {
      for (int rsrcz = 0; rsrcz <= 1; ++rsrcz) {
        int posx = floorf((-o_x + src_coords[p_src * 3 + 0]) / h_x);
        int posy = floorf((-o_y + src_coords[p_src * 3 + 1]) / h_y);
        int posz = floorf((-o_z + src_coords[p_src * 3 + 2]) / h_z);

        float px = -floorf((-o_x + src_coords[p_src * 3 + 0]) / h_x) + (-o_x + src_coords[p_src * 3 + 0]) / h_x;
        float py = -floorf((-o_y + src_coords[p_src * 3 + 1]) / h_y) + (-o_y + src_coords[p_src * 3 + 1]) / h_y;
        float pz = -floorf((-o_z + src_coords[p_src * 3 + 2]) / h_z) + (-o_z + src_coords[p_src * 3 + 2]) / h_z;

        int rx = rsrcx + posx;
        int ry = rsrcy + posy;
        int rz = rsrcz + posz;

        if (rx >= x_m - 1 && ry >= y_m - 1 && rz >= z_m - 1 && rx <= x_M + 1 && ry <= y_M + 1 && rz <= z_M + 1) {
          float r0 = 1.0e-2f * (rsrcx * px + (1 - rsrcx) * (1 - px)) * (rsrcy * py + (1 - rsrcy) * (1 - py)) * (rsrcz * pz + (1 - rsrcz) * (1 - pz)) * src[time * num_src + (p_src - p_src_m)] / __half2float(m[(posx + 4) * full_y * full_z + (posy + 4) * full_z + (posz + 4)]);
          atomicAdd_half(&u[t2 * full_x * full_y * full_z + (rx + 4) * full_y * full_z + (ry + 4) * full_z + (rz + 4)], __float2half(r0));
        }
      }
    }
  }
}

int Kernel(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const float dt, const float h_x, const float h_y, const float h_z, const float o_x, const float o_y, const float o_z, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int deviceid, const int devicerm, struct profiler *timers, bool use_shared) {
  cudaSetDevice(deviceid);

  half *d_u, *d_m;
  float *d_src, *d_src_coords;

  size_t half_nbytes = u_vec->nbytes / 2;
  CHECK_CUDA(cudaMalloc(&d_u, half_nbytes));
  half *h_u = (half*)malloc(half_nbytes);
  float *f_u = (float*)u_vec->data;
  for (size_t i = 0; i < u_vec->nbytes / sizeof(float); ++i) h_u[i] = __float2half(f_u[i]);
  CHECK_CUDA(cudaMemcpy(d_u, h_u, half_nbytes, cudaMemcpyHostToDevice));
  free(h_u);

  size_t m_half_nbytes = m_vec->nbytes / 2;
  CHECK_CUDA(cudaMalloc(&d_m, m_half_nbytes));
  half *h_m = (half*)malloc(m_half_nbytes);
  float *f_m = (float*)m_vec->data;
  for (size_t i = 0; i < m_vec->nbytes / sizeof(float); ++i) h_m[i] = __float2half(f_m[i]);
  CHECK_CUDA(cudaMemcpy(d_m, h_m, m_half_nbytes, cudaMemcpyHostToDevice));
  free(h_m);

  CHECK_CUDA(cudaMalloc(&d_src, src_vec->nbytes));
  CHECK_CUDA(cudaMemcpy(d_src, src_vec->data, src_vec->nbytes, cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_src_coords, src_coords_vec->nbytes));
  CHECK_CUDA(cudaMemcpy(d_src_coords, src_coords_vec->data, src_coords_vec->nbytes, cudaMemcpyHostToDevice));

  int full_x = u_vec->size[1];
  int full_y = u_vec->size[2];
  int full_z = u_vec->size[3];

  dim3 block(8, 8, 8);

  dim3 grid((x_M - x_m + block.x -1) / block.x, (y_M - y_m + block.y -1) / block.y, (z_M - z_m + block.z -1) / block.z);

  int num_src = p_src_M - p_src_m + 1;

  cudaEvent_t event_start, event_stop;
  CHECK_CUDA(cudaEventCreate(&event_start));
  CHECK_CUDA(cudaEventCreate(&event_stop));

  for (int time = time_m, t0 = time % 3, t1 = (time + 2) % 3, t2 = (time + 1) % 3; time <= time_M; time += 1, t0 = time % 3, t1 = (time + 2) % 3, t2 = (time + 1) % 3) {
    CHECK_CUDA(cudaEventRecord(event_start));
    if (use_shared) {
      size_t shared_size = (block.x + 4) * (block.y + 4) * (block.z + 4) * sizeof(half);
      update_kernel_shared<<<grid, block, shared_size>>>(d_u, d_m, full_x, full_y, full_z, dt, h_x, h_y, h_z, t0, t1, t2, x_m, x_M, y_m, y_M, z_m, z_M);
    } else {
      update_kernel<<<grid, block>>>(d_u, d_m, full_x, full_y, full_z, dt, h_x, h_y, h_z, t0, t1, t2, x_m, x_M, y_m, y_M, z_m, z_M);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(event_stop));
    CHECK_CUDA(cudaEventSynchronize(event_stop));
    float ms0;
    CHECK_CUDA(cudaEventElapsedTime(&ms0, event_start, event_stop));
    timers->section0 += ms0 / 1000.0;

    CHECK_CUDA(cudaEventRecord(event_start));
    if (src_vec->size[0] * src_vec->size[1] > 0 && num_src > 0) {
      dim3 src_block(1);
      dim3 src_grid(num_src);
      source_kernel<<<src_grid, src_block>>>(d_src, d_src_coords, d_m, d_u, time, h_x, h_y, h_z, o_x, o_y, o_z, p_src_m, p_src_M, x_m, x_M, y_m, y_M, z_m, z_M, t2, full_x, full_y, full_z, num_src);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaEventRecord(event_stop));
    CHECK_CUDA(cudaEventSynchronize(event_stop));
    float ms1;
    CHECK_CUDA(cudaEventElapsedTime(&ms1, event_start, event_stop));
    timers->section1 += ms1 / 1000.0;
  }

  CHECK_CUDA(cudaEventDestroy(event_start));
  CHECK_CUDA(cudaEventDestroy(event_stop));

  half *h_out = (half*)malloc(half_nbytes);
  CHECK_CUDA(cudaMemcpy(h_out, d_u, half_nbytes, cudaMemcpyDeviceToHost));
  float *f_out = (float*)u_vec->data;
  for (size_t i = 0; i < u_vec->nbytes / sizeof(float); ++i) f_out[i] = __half2float(h_out[i]);
  free(h_out);

  if (devicerm) {
    CHECK_CUDA(cudaFree(d_u));
    CHECK_CUDA(cudaFree(d_m));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_src_coords));
  }

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <grid_size> <use_shared:0/1>" << std::endl;
    return 1;
  }
  int n = std::atoi(argv[1]);
  bool use_shared = std::atoi(argv[2]);
  int halo = 4;
  int nx = n, ny = n, nz = n;
  int x_m = 0, x_M = nx - 1;
  int y_m = 0, y_M = ny - 1;
  int z_m = 0, z_M = nz - 1;
  float dt = 0.001f, h_x = 0.01f, h_y = 0.01f, h_z = 0.01f;
  float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;
  int time_m = 0, time_M = 1000;
  int p_src_m = 0, p_src_M = 0;
  int deviceid = 0, devicerm = 1;

  dataobj u_vec;
  u_vec.size = new int[4]{3, nx + 2 * halo, ny + 2 * halo, nz + 2 * halo};
  u_vec.nbytes = sizeof(float) * u_vec.size[0] * u_vec.size[1] * u_vec.size[2] * u_vec.size[3];
  u_vec.data = aligned_alloc(64, u_vec.nbytes);
  std::memset(u_vec.data, 0, u_vec.nbytes);

  dataobj m_vec;
  m_vec.size = new int[3]{nx + 2 * halo, ny + 2 * halo, nz + 2 * halo};
  m_vec.nbytes = sizeof(float) * m_vec.size[0] * m_vec.size[1] * m_vec.size[2];
  m_vec.data = aligned_alloc(64, m_vec.nbytes);
  float *m_flat = (float *)m_vec.data;
  for (size_t i = 0; i < m_vec.nbytes / sizeof(float); ++i) m_flat[i] = 1.0f;

  dataobj src_vec;
  src_vec.size = new int[2]{time_M + 1, p_src_M - p_src_m + 1};
  src_vec.nbytes = sizeof(float) * src_vec.size[0] * src_vec.size[1];
  src_vec.data = aligned_alloc(64, src_vec.nbytes);
  float *src_flat = (float *)src_vec.data;
  for (size_t i = 0; i < src_vec.nbytes / sizeof(float); ++i) src_flat[i] = 1.0f;

  dataobj src_coords_vec;
  src_coords_vec.size = new int[2]{p_src_M - p_src_m + 1, 3};
  src_coords_vec.nbytes = sizeof(float) * src_coords_vec.size[0] * src_coords_vec.size[1];
  src_coords_vec.data = aligned_alloc(64, src_coords_vec.nbytes);
  float *src_coords_flat = (float *)src_coords_vec.data;
  src_coords_flat[0] = (nx / 2) * h_x;
  src_coords_flat[1] = (ny / 2) * h_y;
  src_coords_flat[2] = (nz / 2) * h_z;

  struct profiler timers = {0.0, 0.0};

  Kernel(&m_vec, &src_vec, &src_coords_vec, &u_vec, x_M, x_m, y_M, y_m, z_M, z_m, dt, h_x, h_y, h_z, o_x, o_y, o_z, p_src_M, p_src_m, time_M, time_m, deviceid, devicerm, &timers, use_shared);

  std::cout << "Section0 (Main Compute): " << timers.section0 << " s" << std::endl;
  std::cout << "Section1 (Source Injection): " << timers.section1 << " s" << std::endl;
  std::cout << "Total Time: " << timers.section0 + timers.section1 << " s" << std::endl;

  free(u_vec.data); delete[] u_vec.size;
  free(m_vec.data); delete[] m_vec.size;
  free(src_vec.data); delete[] src_vec.size;
  free(src_coords_vec.data); delete[] src_coords_vec.size;

  return 0;
}