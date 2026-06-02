

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE 16


#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)


__global__ void matMulNaive(const float *d_A, const float *d_B,
                             float *d_C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += d_A[row * N + k] * d_B[k * N + col];
        d_C[row * N + col] = sum;
    }
}


__global__ void matMulTiled(const float *d_A, const float *d_B,
                             float *d_C, int N)
{
    /* Tiles en shared memory — cada bloque los llena cooperativamente */
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    /* Recorrer todos los tiles de la fila/columna */
    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {

        /* Cargar tile de A en shared memory (con guard de límite) */
        if (row < N && (t * TILE + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] = d_A[row * N + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        /* Cargar tile de B en shared memory (con guard de límite) */
        if (col < N && (t * TILE + threadIdx.y) < N)
            sB[threadIdx.y][threadIdx.x] = d_B[(t * TILE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        /* Esperar a que todos los threads del bloque hayan cargado el tile */
        __syncthreads();

        /* Multiplicar los tiles en shared memory (acceso rápido) */
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        /* Esperar antes de cargar el siguiente tile */
        __syncthreads();
    }

    if (row < N && col < N)
        d_C[row * N + col] = sum;
}


void matMulCPU(const float *h_A, const float *h_B,
               float *h_C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += h_A[i * N + k] * h_B[k * N + j];
            h_C[i * N + j] = sum;
        }
}


float measureKernel(void (*launchFn)(const float *, const float *,
                                     float *, int,
                                     cudaEvent_t, cudaEvent_t),
                    const float *d_A, const float *d_B, float *d_C, int N)
{
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    launchFn(d_A, d_B, d_C, N, evStart, evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    return ms;
}

/* Wrappers para los kernels que incluyen los eventos */
void launchNaive(const float *d_A, const float *d_B, float *d_C, int N,
                 cudaEvent_t evStart, cudaEvent_t evStop)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    CUDA_CHECK(cudaEventRecord(evStart));
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(evStop));
}

void launchTiled(const float *d_A, const float *d_B, float *d_C, int N,
                 cudaEvent_t evStart, cudaEvent_t evStop)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    CUDA_CHECK(cudaEventRecord(evStart));
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(evStop));
}


void runBenchmark(int N)
{
    printf("\n========================================\n");
    printf("  Matriz %d x %d\n", N, N);
    printf("========================================\n");

    size_t bytes = (size_t)N * N * sizeof(float);

    /* --- Memoria host --- */
    float *h_A     = (float *)malloc(bytes);
    float *h_B     = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Error: malloc falló para N = %d\n", N);
        return;
    }

    /* Inicializar matrices con valores aleatorios pequeños */
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f;
        h_B[i] = (float)(rand() % 10) / 10.0f;
    }

    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matMulCPU(h_A, h_B, h_C_cpu, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                  + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("CPU:             %8.2f ms\n", cpu_ms);

    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    
    float naive_ms = measureKernel(launchNaive, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verificar naïve */
    int errorsNaive = 0;
    for (int i = 0; i < N * N; i++)
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-3f) errorsNaive++;
    printf("GPU naïve:       %8.2f ms   (errores: %d)\n", naive_ms, errorsNaive);

    
    float tiled_ms = measureKernel(launchTiled, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verificar tiled */
    int errorsTiled = 0;
    for (int i = 0; i < N * N; i++)
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-3f) errorsTiled++;
    printf("GPU tiled:       %8.2f ms   (errores: %d)\n", tiled_ms, errorsTiled);

    printf("Speedup tiling:  %8.2fx  (tiled vs naïve)\n",
           naive_ms / tiled_ms);
    printf("Speedup vs CPU:  %8.2fx  (tiled vs CPU)\n",
           (float)cpu_ms / tiled_ms);

    /* Liberar memoria */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}


int main(void)
{
    printf("\n=== Benchmark matMul: CPU vs GPU naïve vs GPU tiled ===\n");
    printf("TILE_SIZE = %d\n", TILE);

    runBenchmark(512);
    runBenchmark(1024);

    printf("\nBenchmark completado.\n");
    return 0;
}
