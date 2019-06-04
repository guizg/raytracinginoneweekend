#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include <cuda_runtime.h>
#include <chrono>

// faz o caminho do raio de luz simulado para descobrir a cor dos pixels
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

// Função principal, executada em todas a threads, que usa a função color para descobrir e preencher a cor de cada pixel
__global__ void render(vec3 *fb, int max_x, int max_y, hitable **world){

   vec3 lower_left_corner(-2.0, -1.0, -1.0);
   vec3 horizontal(4.0, 0.0, 0.0);
   vec3 vertical(0.0, 2.0, 0.0);
   vec3 origin(0.0, 0.0, 0.0);
   
   // são usadas variáveis do CUDA para descobrir qual pixel devemos calcular
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return; // como as dimensões da matriz de threads podem ser um pouco maiores que as da imagem, esse if nos previne de escrever onde nao devemos
   int pixel_index = j*max_x + i;
   float u = float(i) / float(max_x);
   float v = float(j) / float(max_y);
   ray r(origin, lower_left_corner + u*horizontal + v*vertical);
   fb[pixel_index] = color(r, world);
}

// Inicializador. Executada em um kernel exclusivo na GPU, ela instancia as esferas e o mundo
 __global__ void create_world(hitable **list, hitable **world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(list) = new sphere(vec3(0,0,-1), 0.5);
        *(list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *(list+2) = new sphere(vec3(1, 0.2,-1), 0.5);
        *(list+3) = new sphere(vec3(-1, 0.2, -1), 0.5);
        *world    = new hitable_list(list,4);
    }
}

// Tambem executada em um kernel exclusivo, ela libera a memória instanciada na função acima
__global__ void free_world(hitable **list, hitable **world) {
    delete *(list);
    delete *(list+1);
    delete *(list+2);
    delete *(list+3);
    delete *world;
 }

int main() {
    using namespace std::chrono;

    // dimensões da imagem
    int nx = 400;
    int ny = 200;

    //numero de linhas e colunas de threads nos blocos (estou usando sempre matrizes quadradas)
    int nthreads = 8;

    // descobre o tamanho necessário para o vetor que armazenará os pixels
    int buffer_size = nx*ny*sizeof(vec3);

    // vetores de pixels do host e do device
    vec3* h_buffer;
    vec3* d_buffer;


    // criação de blocos e threads para o kernel principal
    dim3 blocks(nx/nthreads+1,ny/nthreads+1);
    dim3 threads(nthreads,nthreads);

    // aloca o vetor de pixels na CPU
    h_buffer = (vec3 *)malloc(buffer_size);

    // aloca o vetor de pixels na GPU
    cudaMalloc((void**)&d_buffer, buffer_size);

    // cria e aloca a lista de hitables e o mundo na GPU
    hitable **list;
    cudaMalloc((void **)&list, 4*sizeof(hitable *));
    hitable **world;
    cudaMalloc((void **)&world, sizeof(hitable *));

    // lança o kernel para a instanciacao do mundo e esferas (1 bloco e 1 thread)
    create_world<<<1,1>>>(list,world);
    cudaDeviceSynchronize();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // lança o kernel principal, para fazer os cálculos de ray tracing e preencher o vetor de pixels na GPU
    render<<<blocks, threads>>>(d_buffer, nx, ny, world);

    cudaDeviceSynchronize();

    
    // copia o vetor de pixels da GPU para a CPU
    cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    // Escreve o arquivo de saída que descreve a imagem
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*h_buffer[pixel_index].r());
            int ig = int(255.99*h_buffer[pixel_index].g());
            int ib = int(255.99*h_buffer[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cerr << "Tempo: " << time_span.count() << " segundos.";
    std::cerr << std::endl;
    
    // Libera a memória
    cudaDeviceSynchronize();
    free_world<<<1,1>>>(list,world); // aqui usando um kernel de um bloco e uma thread
    cudaGetLastError();
    cudaFree(list);
    cudaFree(world);
    cudaFree(d_buffer);

    return 0;
}
