#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include <cuda_runtime.h>

__global__ void render(vec3 *fb, int max_x, int max_y,vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **world)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
   float u = float(i) / float(max_x);
   float v = float(j) / float(max_y);
   ray r(origin, lower_left_corner + u*horizontal + v*vertical);
   fb[pixel_index] = color(r, world);
}

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

 __global__ void create_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hitable_list(d_list,2);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
 }

int main() {
    int nx = 200;
    int ny = 100;

    int buffer_size = nx*ny*3*sizeof(float)

    float* buffer;

    cudaMallocManaged((void **)&buffer, buffer_size);


    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertial(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    // hitable *list[2];
    // list[0] = new sphere(vec3(0,0,-1), 0.5);
    // list[1] = new sphere(vec3(0,-100.5,-1), 100);
    // hitable *world = new hitable_list(list,2);

    hitable **d_list;
    cudaMalloc((void **)&d_list, 2*sizeof(hitable *));
    hitable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hitable *));
    create_world<<<1,1>>>(d_list,d_world);
    cudaDeviceSynchronize();

    render<<<blocks, threads>>>(fb, nx, ny,
        vec3(-2.0, -1.0, -1.0),
        vec3(4.0, 0.0, 0.0),
        vec3(0.0, 2.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        d_world);

    cudaDeviceSynchronize();
    
    // std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    // for (int j = ny-1; j >= 0; j--) {
    //     for (int i = 0; i < nx; i++) {
    //         float u = float(i) / float(nx);
    //         float v = float(j) / float(ny);
    //         ray r(origin, lower_left_corner + u*horizontal + v*vertial);
    //         vec3 col = color(r, world);
    //         int ir = int(255.99*col[0]);
    //         int ig = int(255.99*col[1]);
    //         int ib = int(255.99*col[2]);
    //         std::cout << ir << " " << ig << " " << ib << "\n";
    //     }
    // }

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*3*nx + i*3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    
    
    cudaDeviceSynchronize();
    free_world<<<1,1>>>(d_list,d_world);
    cudaGetLastError();
    cudaFree(d_list);
    cudaFree(d_world);
    cudaFree(fb);
}
