#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include <cuda_runtime.h>
#include <chrono>

__device__ vec3 color(const ray& r, hitable **world) {
   hit_record rec;
   if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
      return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
   }
   else {
      vec3 unit_direction = unit_vector(r.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
   }
}
__global__ void render(vec3 *fb, int max_x, int max_y, hitable **world){

   vec3 lower_left_corner(-2.0, -1.0, -1.0);
   vec3 horizontal(4.0, 0.0, 0.0);
   vec3 vertical(0.0, 2.0, 0.0);
   vec3 origin(0.0, 0.0, 0.0);

   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
   float u = float(i) / float(max_x);
   float v = float(j) / float(max_y);
   ray r(origin, lower_left_corner + u*horizontal + v*vertical);
   fb[pixel_index] = color(r, world);
}


 __global__ void create_world(hitable **list, hitable **world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(list) = new sphere(vec3(0,0,-1), 0.5);
        *(list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *(list+2) = new sphere(vec3(1, 0,-1), 0.5);
        *(list+3) = new sphere(vec3(1, 0.2, 1), 0.5);
        *world    = new hitable_list(list,3);
    }
}

__global__ void free_world(hitable **list, hitable **world) {
    delete *(list);
    delete *(list+1);
    delete *(list+2);
    delete *(list+3);
    delete *world;
 }

int main() {
    using namespace std::chrono;
    int nx = 200;
    int ny = 100;
    int tx = 8;
    int ty = 8;

    int buffer_size = nx*ny*sizeof(vec3);

    vec3* buffer;

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    cudaMallocManaged((void **)&buffer, buffer_size);

    hitable **list;
    cudaMalloc((void **)&list, 4*sizeof(hitable *));
    hitable **world;
    cudaMalloc((void **)&world, sizeof(hitable *));
    create_world<<<1,1>>>(list,world);
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    render<<<blocks, threads>>>(buffer, nx, ny, world);

    cudaDeviceSynchronize();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*buffer[pixel_index].r());
            int ig = int(255.99*buffer[pixel_index].g());
            int ib = int(255.99*buffer[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cerr << "Tempo: " << time_span.count() << " segundos.";
    std::cerr << std::endl;
    
    
    cudaDeviceSynchronize();
    free_world<<<1,1>>>(list,world);
    cudaGetLastError();
    cudaFree(list);
    cudaFree(world);
    cudaFree(buffer);

    return 0;
}
