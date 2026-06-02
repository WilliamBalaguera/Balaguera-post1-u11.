# Post-Contenido 1 — CUDA Benchmark CPU vs GPU
Arquitectura de Computadores — Unidad 11

**Universidad Francisco de Paula Santander — Ingeniería de Sistemas 2026**

---

## Descripción del Entorno

| Campo                  | Valor                        |
|------------------------|------------------------------|
| **GPU Model**          | Tesla T4                     |
| **CUDA Version**       | 13.0                         |
| **Driver Version**     | 580.82.07                    |
| **CUDA Toolkit**       | 12.8 (nvcc V12.8.93)         |
| **Memoria GPU**        | 15360 MiB                    |
| **Sistema Operativo**  | Ubuntu 22.04 (Google Colab)  |
| **Plataforma**         | Google Colab (GPU T4)        |

---

## Compilación

```bash
# Suma de vectores
nvcc -O2 -o vectorAdd src/vectorAdd.cu
./vectorAdd

# Multiplicación de matrices
nvcc -O2 -o matMul src/matMul.cu
./matMul
```

---

## Resultados — vectorAdd

| N (elementos)   | CPU (ms) | GPU kernel (ms) | GPU total con memcpy (ms) | Speedup kernel |
|-----------------|----------|-----------------|---------------------------|----------------|
| 1 M (1 048 576) | 2.03     | 107.27          | 113.46                    | 0.02x          |
| 4 M (4 194 304) | 9.50     | 0.19            | 19.18                     | 49.61x         |
| 16 M (16 777 216)| 39.63   | 0.76            | 74.58                     | 51.93x         |

> **Errores de verificación:** 0 en todos los casos.

---

## Resultados — matMul

| N    | CPU (ms) | GPU Naïve (ms) | GPU Tiled TILE=16 (ms) | Speedup tiling vs naïve | Speedup tiled vs CPU |
|------|----------|----------------|------------------------|-------------------------|----------------------|
| 512  | 302.18   | 32.54          | 0.75                   | 43.43x                  | 403.33x              |
| 1024 | 3572.80  | 9.20           | 5.84                   | 1.58x                   | 611.68x              |

> **Errores de verificación:** 0 en todos los casos.

---

## Análisis de Resultados

### ¿Por qué la GPU es más rápida que la CPU para N grande?

La GPU Tesla T4 posee miles de núcleos CUDA que operan en paralelo bajo el modelo SIMT (Single Instruction, Multiple Threads). En el kernel `vectorAdd`, cada thread procesa exactamente un elemento del arreglo de forma simultánea. Para N = 16 M elementos se lanzan ~65 536 bloques de 256 threads, procesando todos los elementos en paralelo. La CPU, en cambio, ejecuta el bucle de forma secuencial, lo que explica que para N = 16 M tarde 39.63 ms mientras el kernel GPU solo necesita 0.76 ms (speedup de 51.93x). Para N = 1 M el resultado es inverso: la CPU tarda apenas 2.03 ms porque los datos caben en caché, mientras que el kernel GPU tarda 107.27 ms debido al overhead de inicialización del primer lanzamiento CUDA (JIT compilation).

### ¿Por qué el tiempo total GPU (con memcpy) es mayor que la CPU?

La transferencia de datos entre la RAM del host y la VRAM del dispositivo a través del bus PCIe tiene un overhead fijo significativo. Para N = 16 M, el kernel tarda 0.76 ms pero el tiempo total (incluyendo los dos `cudaMemcpy`) sube a 74.58 ms, superando ampliamente los 39.63 ms de la CPU. Esto demuestra que la GPU solo es ventajosa cuando el cómputo es muy intenso (como en matMul con N = 1024, donde la CPU tarda 3572.80 ms vs 5.84 ms del kernel tiled). En aplicaciones reales se minimiza este costo manteniendo los datos en VRAM durante múltiples operaciones consecutivas.

---

## Capturas de Checkpoints

| Checkpoint | Descripción | Archivo |
|------------|-------------|---------|
| CP1 | vectorAdd compila y ejecuta, tiempos CPU vs GPU, Errores: 0 | `capturas/checkpoint1_vectorAdd.png` |
| CP2 | matMul tiled produce resultados correctos y tabla comparativa | `capturas/checkpoint2_matMul.png` |
| CP3 | Repositorio publicado en GitHub (público) | `capturas/checkpoint3_github.png` |

---

## Estructura del Repositorio

```
apellido-post1-u11/
├── README.md
├── src/
│   ├── vectorAdd.cu      # Kernel suma de vectores + benchmark CPU vs GPU
│   └── matMul.cu         # Kernel matMul naïve + tiled shared memory
└── capturas/
    ├── checkpoint1_vectorAdd.png
    ├── checkpoint2_matMul.png
    └── checkpoint3_github.png
```

---

## Historial de Commits

```
init: estructura inicial del proyecto CUDA
feat: kernel vectorAdd con benchmark CPU vs GPU
feat: matMul con tiling shared memory y benchmark comparativo
docs: README completo con tablas de resultados y análisis
```
