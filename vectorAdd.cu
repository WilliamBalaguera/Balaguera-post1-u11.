/**
 * vectorAdd.cu
 * Suma de vectores en CUDA con benchmark CPU vs GPU
 * Arquitectura de Computadores — Unidad 11, Post-Contenido 1
 *
 * Compilar: nvcc -O2 -o vectorAdd src/vectorAdd.cu
 * Ejecutar: ./vectorAdd
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ---------------------------------------------------------------
 * Macro para verificar errores CUDA
 * --------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

/* ---------------------------------------------------------------
 * Kernel CUDA: cada thread calcula un elemento de C = A + B
 * --------------------------------------------------------------- */
__global__ void vectorAdd(const float *d_A, const float *d_B,
                          float *d_C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_C[idx] = d_A[idx] + d_B[idx];
}

/* ---------------------------------------------------------------
 * Suma de vectores en CPU (implementación de referencia)
 * --------------------------------------------------------------- */
void vectorAddCPU(const float *h_A, const float *h_B,
                  float *h_C, int n)
{
    for (int i = 0; i < n; i++)
        h_C[i] = h_A[i] + h_B[i];
}

/* ---------------------------------------------------------------
 * Función que ejecuta el benchmark para un N dado
 * --------------------------------------------------------------- */
void runBenchmark(int N)
{
    printf("\n========================================\n");
    printf("  N = %d (%.1f M elementos)\n", N, N / 1e6f);
    printf("========================================\n");

    size_t bytes = (size_t)N * sizeof(float);

    /* --- Memoria host --- */
    float *h_A     = (float *)malloc(bytes);
    float *h_B     = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Error: malloc falló para N = %d\n", N);
        return;
    }

    /* Inicializar vectores con valores conocidos */
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    /* -------------------------------------------------------
     * Benchmark CPU
     * ------------------------------------------------------- */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double cpu_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                  + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("CPU:             %8.2f ms\n", cpu_ms);

    /* -------------------------------------------------------
     * Benchmark GPU
     * ------------------------------------------------------- */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* Eventos para medir tiempo total (incluye memcpy) */
    cudaEvent_t evTotalStart, evTotalStop;
    CUDA_CHECK(cudaEventCreate(&evTotalStart));
    CUDA_CHECK(cudaEventCreate(&evTotalStop));

    /* Eventos para medir solo el kernel */
    cudaEvent_t evKernelStart, evKernelStop;
    CUDA_CHECK(cudaEventCreate(&evKernelStart));
    CUDA_CHECK(cudaEventCreate(&evKernelStop));

    /* Inicio tiempo total (incluye H->D) */
    CUDA_CHECK(cudaEventRecord(evTotalStart));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Lanzamiento del kernel */
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaEventRecord(evKernelStart));
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(evKernelStop));
    CUDA_CHECK(cudaEventSynchronize(evKernelStop));

    /* Verificar errores del kernel */
    CUDA_CHECK(cudaGetLastError());

    /* Copia D->H y fin tiempo total */
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evTotalStop));
    CUDA_CHECK(cudaEventSynchronize(evTotalStop));

    float gpu_kernel_ms = 0.0f, gpu_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, evKernelStart, evKernelStop));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_total_ms,  evTotalStart,  evTotalStop));

    printf("GPU kernel:      %8.2f ms\n", gpu_kernel_ms);
    printf("GPU total:       %8.2f ms  (incluye cudaMemcpy)\n", gpu_total_ms);
    printf("Speedup kernel:  %8.2fx\n", (float)(cpu_ms / gpu_kernel_ms));

    /* -------------------------------------------------------
     * Verificación de resultados
     * ------------------------------------------------------- */
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f)
            errors++;
    }
    printf("Errores:         %d\n", errors);

    /* Liberar memoria */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(evTotalStart));
    CUDA_CHECK(cudaEventDestroy(evTotalStop));
    CUDA_CHECK(cudaEventDestroy(evKernelStart));
    CUDA_CHECK(cudaEventDestroy(evKernelStop));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
}

/* ---------------------------------------------------------------
 * Main: ejecuta benchmark para N = 1M, 4M, 16M
 * --------------------------------------------------------------- */
int main(void)
{
    printf("\n=== Benchmark vectorAdd: CPU vs GPU ===\n");
    printf("BlockSize = 256 threads\n");

    runBenchmark(1 << 20);   /*  1 M elementos */
    runBenchmark(1 << 22);   /*  4 M elementos */
    runBenchmark(1 << 24);   /* 16 M elementos */

    printf("\nBenchmark completado.\n");
    return 0;
}
