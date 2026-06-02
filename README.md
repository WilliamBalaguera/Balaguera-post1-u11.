# Programacion GPU con CUDA — Unidad 11

Benchmark de computacion paralela en GPU implementado en CUDA C, comparando el rendimiento de CPU contra GPU para dos problemas clasicos: suma de vectores y multiplicacion de matrices.

---

## Informacion del proyecto

| Campo           | Detalle                               |
|-----------------|---------------------------------------|
| Autor           | William Balaguera — 1152439           |
| Materia         | Arquitectura de Computadores          |
| Unidad          | 11 — Post-Contenido 1                 |
| Ano             | 2026                                  |
| Universidad     | Francisco de Paula Santander          |
| Lenguaje        | CUDA C (nvcc)                         |

---

## Descripcion

El proyecto contiene dos programas independientes que miden y comparan tiempos de ejecucion entre CPU y GPU para operaciones de algebra lineal. El objetivo es evidenciar el speedup que ofrece el paralelismo masivo de la GPU frente a la ejecucion secuencial en CPU, y ademas comparar el impacto del uso de shared memory en el rendimiento de kernels CUDA.

---

## Estructura del repositorio

```
Balaguera-post1-u11/
├── src/
│   ├── vectorAdd.cu      — Suma de vectores CPU vs GPU
│   └── matMul.cu         — Multiplicacion de matrices CPU vs GPU naive vs tiled
├── Capturas/
│   ├── checkpoint1.png   — Evidencia vectorAdd
│   └── checkpoint2.png   — Evidencia matMul
└── README.md
```

---

## Requisitos

- GPU NVIDIA compatible con CUDA
- CUDA Toolkit instalado (`nvcc` disponible en PATH)
- Sistema operativo Linux o Windows con drivers NVIDIA

---

## Compilacion y ejecucion

### vectorAdd — Suma de vectores

```bash
nvcc -O2 -o vectorAdd src/vectorAdd.cu
./vectorAdd
```

### matMul — Multiplicacion de matrices

```bash
nvcc -O2 -o matMul src/matMul.cu
./matMul
```

---

## Descripcion de los programas

### vectorAdd.cu

Calcula `C = A + B` sobre vectores de punto flotante. Compara tres mediciones para tamansos de 1M, 4M y 16M elementos:

- Tiempo de ejecucion en **CPU** (bucle secuencial)
- Tiempo del **kernel GPU** (sin contar transferencias de memoria)
- Tiempo **GPU total** (kernel + `cudaMemcpy` Host->Device y Device->Host)

Cada thread CUDA calcula un elemento del resultado. El tamano de bloque es de 256 threads.

| Tamano   | Metrica reportada         |
|----------|--------------------------|
| 1M elem  | CPU ms / GPU kernel ms / GPU total ms / Speedup |
| 4M elem  | CPU ms / GPU kernel ms / GPU total ms / Speedup |
| 16M elem | CPU ms / GPU kernel ms / GPU total ms / Speedup |

---

### matMul.cu

Calcula `C = A x B` para matrices cuadradas de N x N. Implementa tres versiones del algoritmo:

**CPU** — multiplicacion O(N³) secuencial, usada como referencia.

**GPU naive** — cada thread calcula un elemento de C accediendo directamente a memoria global. Sin optimizacion de localidad.

**GPU tiled** — divide las matrices en bloques de 16x16 (`TILE_SIZE = 16`) cargados cooperativamente en shared memory. Reduce los accesos a memoria global en un factor igual al tamano del tile, mejorando significativamente el rendimiento.

| Tamano    | CPU ms | GPU naive ms | GPU tiled ms | Speedup tiling | Speedup vs CPU |
|-----------|--------|-------------|-------------|----------------|----------------|
| 512 x 512 | —      | —           | —           | —              | —              |
| 1024 x 1024 | —    | —           | —           | —              | —              |

*Los valores se completan al ejecutar el benchmark en el hardware objetivo.*

---

## Detalles tecnicos

**Medicion de tiempo**
- CPU: `clock_gettime(CLOCK_MONOTONIC)` con resolucion en nanosegundos
- GPU: `cudaEvent_t` con `cudaEventElapsedTime`, que mide directamente en la GPU sin overhead del sistema operativo

**Verificacion de resultados**
Ambos programas comparan el resultado GPU contra el resultado CPU elemento a elemento con una tolerancia de `1e-4` (vectorAdd) y `1e-3` (matMul) para confirmar la correccion numerica.

**Manejo de errores**
Todas las llamadas CUDA pasan por la macro `CUDA_CHECK`, que imprime archivo, linea y mensaje de error antes de terminar el proceso si ocurre un fallo.

---

## Checkpoints verificados

- Checkpoint 1 — `vectorAdd`: kernel lanza correctamente, speedup GPU kernel visible frente a CPU, errores = 0
- Checkpoint 2 — `matMul`: version tiled supera a naive en speedup, errores = 0 en ambas versiones GPU

---

## Capturas de evidencia

Las capturas de los resultados se encuentran en la carpeta `Capturas/`:

| Archivo          | Contenido                              |
|------------------|----------------------------------------|
| checkpoint1.png  | Salida del benchmark vectorAdd         |
| checkpoint2.png  | Salida del benchmark matMul            |

---

## Licencia

Proyecto academico — Universidad Francisco de Paula Santander · 2026
