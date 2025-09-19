%%writefile fdtd_openacc.cpp
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <openacc.h>

// # --------------------------------------------------
// # This code is written by Dr. Ziyi Yin from KronosAI
// # --------------------------------------------------


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

extern "C" int Kernel(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const float dt, const float h_x, const float h_y, const float h_z, const float o_x, const float o_y, const float o_z, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int deviceid, const int devicerm, struct profiler *timers);

int Kernel(struct dataobj *__restrict m_vec, struct dataobj *__restrict src_vec, struct dataobj *__restrict src_coords_vec, struct dataobj *__restrict u_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const float dt, const float h_x, const float h_y, const float h_z, const float o_x, const float o_y, const float o_z, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int deviceid, const int devicerm, struct profiler *timers) {
  acc_init(acc_device_nvidia);
  if (deviceid != -1) {
    acc_set_device_num(deviceid, acc_device_nvidia);
  }
  float (*__restrict m)[m_vec->size[1]][m_vec->size[2]] __attribute__((aligned(64))) = (float (*)[m_vec->size[1]][m_vec->size[2]]) m_vec->data;
  float (*__restrict src)[src_vec->size[1]] __attribute__((aligned(64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*__restrict src_coords)[src_coords_vec->size[1]] __attribute__((aligned(64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*__restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
#pragma acc enter data copyin(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
#pragma acc enter data copyin(m[0:m_vec->size[0]][0:m_vec->size[1]][0:m_vec->size[2]])
#pragma acc enter data copyin(src[0:src_vec->size[0]][0:src_vec->size[1]])
#pragma acc enter data copyin(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]])
  float r1 = 1.0F / (dt * dt);
  float r2 = 1.0F / (h_x * h_x);
  float r3 = 1.0F / (h_y * h_y);
  float r4 = 1.0F / (h_z * h_z);
  for (int time = time_m, t0 = (time) % (3), t1 = (time + 2) % (3), t2 = (time + 1) % (3); time <= time_M; time += 1, t0 = (time) % (3), t1 = (time + 2) % (3), t2 = (time + 1) % (3)) {
    START(section0)
#pragma acc parallel loop collapse(3) present(m, u)
    for (int x = x_m; x <= x_M; x += 1) {
      for (int y = y_m; y <= y_M; y += 1) {
        for (int z = z_m; z <= z_M; z += 1) {
          float r5 = -2.50F * u[t0][x + 4][y + 4][z + 4];
          u[t2][x + 4][y + 4][z + 4] = dt * dt * (r2 * (r5 + (-8.33333333e-2F) * (u[t0][x + 2][y + 4][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 1.333333330F * (u[t0][x + 3][y + 4][z + 4] + u[t0][x + 5][y + 4][z + 4])) + r3 * (r5 + (-8.33333333e-2F) * (u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 6][z + 4]) + 1.333333330F * (u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 5][z + 4])) + r4 * (r5 + (-8.33333333e-2F) * (u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6]) + 1.333333330F * (u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5])) - (-2.0F * r1 * u[t0][x + 4][y + 4][z + 4] + r1 * u[t1][x + 4][y + 4][z + 4]) * m[x + 4][y + 4][z + 4]) / m[x + 4][y + 4][z + 4];
        }
      }
    }
    STOP(section0, timers)
    START(section1)
    if (src_vec->size[0] * src_vec->size[1] > 0 && p_src_M - p_src_m + 1 > 0) {
#pragma acc parallel loop collapse(4) present(m, src, src_coords, u)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1) {
        for (int rsrcx = 0; rsrcx <= 1; rsrcx += 1) {
          for (int rsrcy = 0; rsrcy <= 1; rsrcy += 1) {
            for (int rsrcz = 0; rsrcz <= 1; rsrcz += 1) {
              int posx = static_cast<int>(std::floor((-o_x + src_coords[p_src][0]) / h_x));
              int posy = static_cast<int>(std::floor((-o_y + src_coords[p_src][1]) / h_y));
              int posz = static_cast<int>(std::floor((-o_z + src_coords[p_src][2]) / h_z));
              float px = -std::floor((-o_x + src_coords[p_src][0]) / h_x) + (-o_x + src_coords[p_src][0]) / h_x;
              float py = -std::floor((-o_y + src_coords[p_src][1]) / h_y) + (-o_y + src_coords[p_src][1]) / h_y;
              float pz = -std::floor((-o_z + src_coords[p_src][2]) / h_z) + (-o_z + src_coords[p_src][2]) / h_z;
              if (rsrcx + posx >= x_m - 1 && rsrcy + posy >= y_m - 1 && rsrcz + posz >= z_m - 1 && rsrcx + posx <= x_M + 1 && rsrcy + posy <= y_M + 1 && rsrcz + posz <= z_M + 1) {
                float r0 = 1.0e-2F * (rsrcx * px + (1 - rsrcx) * (1 - px)) * (rsrcy * py + (1 - rsrcy) * (1 - py)) * (rsrcz * pz + (1 - rsrcz) * (1 - pz)) * src[time][p_src] / m[posx + 4][posy + 4][posz + 4];
#pragma acc atomic update
                u[t2][rsrcx + posx + 4][rsrcy + posy + 4][rsrcz + posz + 4] += r0;
              }
            }
          }
        }
      }
    }
    STOP(section1, timers)
  }
#pragma acc exit data copyout(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
#pragma acc exit data delete(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]]) if (devicerm)
#pragma acc exit data delete(m[0:m_vec->size[0]][0:m_vec->size[1]][0:m_vec->size[2]]) if (devicerm)
#pragma acc exit data delete(src[0:src_vec->size[0]][0:src_vec->size[1]]) if (devicerm)
#pragma acc exit data delete(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]]) if (devicerm)
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <grid_size> (e.g., 128 for 128^3 grid)" << std::endl;
    return 1;
  }
  int n = std::atoi(argv[1]);  // Grid size (n x n x n inner points)
  int halo = 4;
  int nx = n, ny = n, nz = n;
  int x_m = 0, x_M = nx - 1;
  int y_m = 0, y_M = ny - 1;
  int z_m = 0, z_M = nz - 1;
  float dt = 0.001f, h_x = 0.01f, h_y = 0.01f, h_z = 0.01f;
  float o_x = 0.0f, o_y = 0.0f, o_z = 0.0f;
  int time_m = 0, time_M = 100;  // 100 time steps for benchmarking
  int p_src_m = 0, p_src_M = 0;  // One source
  int deviceid = 0, devicerm = 1;

  // Allocate u (4D: time buffers x (nx+8) x (ny+8) x (nz+8))
  dataobj u_vec;
  u_vec.size = new int[4]{3, nx + 2 * halo, ny + 2 * halo, nz + 2 * halo};
  size_t u_bytes = sizeof(float) * u_vec.size[0] * u_vec.size[1] * u_vec.size[2] * u_vec.size[3];
  u_vec.data = aligned_alloc(64, u_bytes);
  std::memset(u_vec.data, 0, u_bytes);

  // Allocate m (3D: (nx+8) x (ny+8) x (nz+8))
  dataobj m_vec;
  m_vec.size = new int[3]{nx + 2 * halo, ny + 2 * halo, nz + 2 * halo};
  size_t m_bytes = sizeof(float) * m_vec.size[0] * m_vec.size[1] * m_vec.size[2];
  m_vec.data = aligned_alloc(64, m_bytes);
  float *m_flat = (float *)m_vec.data;
  for (size_t i = 0; i < m_bytes / sizeof(float); ++i) m_flat[i] = 1.0f;

  // Allocate src (2D: (time_M+1) x (p_src_M+1))
  dataobj src_vec;
  src_vec.size = new int[2]{time_M + 1, p_src_M + 1};
  size_t src_bytes = sizeof(float) * src_vec.size[0] * src_vec.size[1];
  src_vec.data = aligned_alloc(64, src_bytes);
  float *src_flat = (float *)src_vec.data;
  for (size_t i = 0; i < src_bytes / sizeof(float); ++i) src_flat[i] = 1.0f;

  // Allocate src_coords (2D: (p_src_M+1) x 3)
  dataobj src_coords_vec;
  src_coords_vec.size = new int[2]{p_src_M + 1, 3};
  size_t src_coords_bytes = sizeof(float) * src_coords_vec.size[0] * src_coords_vec.size[1];
  src_coords_vec.data = aligned_alloc(64, src_coords_bytes);
  float (*src_coords)[3] = (float (*)[3])src_coords_vec.data;
  src_coords[0][0] = (nx / 2) * h_x;  // Center source
  src_coords[0][1] = (ny / 2) * h_y;
  src_coords[0][2] = (nz / 2) * h_z;

  struct profiler timers = {0.0, 0.0};

  // Call kernel
  Kernel(&m_vec, &src_vec, &src_coords_vec, &u_vec, x_M, x_m, y_M, y_m, z_M, z_m, dt, h_x, h_y, h_z, o_x, o_y, o_z, p_src_M, p_src_m, time_M, time_m, deviceid, devicerm, &timers);

  // Output timings
  std::cout << "Section0 (Main Compute): " << timers.section0 << " s" << std::endl;
  std::cout << "Section1 (Source Injection): " << timers.section1 << " s" << std::endl;
  std::cout << "Total Time: " << timers.section0 + timers.section1 << " s" << std::endl;

  // Cleanup
  free(u_vec.data); delete[] u_vec.size;
  free(m_vec.data); delete[] m_vec.size;
  free(src_vec.data); delete[] src_vec.size;
  free(src_coords_vec.data); delete[] src_coords_vec.size;

  return 0;
}