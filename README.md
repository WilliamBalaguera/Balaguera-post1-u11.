# Catálogo de Productos — Unidad 11

Refactorización de una API REST en Spring Boot aplicando principios SOLID, patrones DAO/DTO y manejo centralizado de errores con `@ControllerAdvice`.

---

## Información del proyecto

| Campo           | Detalle                               |
|-----------------|---------------------------------------|
| Autor           | William Balaguera — 1152439           |
| Materia         | Arquitectura de Computadores          |
| Unidad          | 11                                    |
| Año             | 2026                                  |
| Universidad     | Francisco de Paula Santander          |
| Lenguaje        | Java 17+ / Spring Boot                |

---

## Descripción

El proyecto implementa un CRUD completo sobre una entidad `Producto`, estructurado en capas bien definidas. El objetivo principal es aplicar de forma práctica los principios **SRP** y **DIP** de SOLID, junto con los patrones **DAO**, **DTO** y **Factory**, sobre una base de datos H2 en memoria.

---

## Arquitectura en capas

```
ProductoController  (@RestController /api/productos)
        |
        | depende de interfaz (DIP)
        v
ProductoService  (interfaz)
ProductoServiceImpl  (logica de negocio)
        |                        |
        v                        v
ProductoRepository          ProductoFactory
(DAO / JpaRepository)       toEntity / toResponseDTO
        |                        |
        v                        v
   Producto (Entity)     ProductoRequestDTO
                         ProductoResponseDTO

Manejo de errores transversal:
GlobalExceptionHandler (@RestControllerAdvice)
  |- RecursoNoEncontradoException  ->  404 + ApiError JSON
  |- MethodArgumentNotValidException  ->  400 + ApiError JSON
  |- Exception generica  ->  500 + ApiError JSON
```

---

## Requisitos

- Java 17 o superior
- Maven 3.9.x

---

## Compilacion y ejecucion

```bash
# Compilar el proyecto
mvn compile

# Iniciar la aplicacion
mvn spring-boot:run
```

La aplicacion queda disponible en `http://localhost:8080`.

La consola H2 se puede acceder en `http://localhost:8080/h2-console` con los siguientes datos:

| Campo     | Valor                  |
|-----------|------------------------|
| JDBC URL  | `jdbc:h2:mem:catalogodb` |
| Usuario   | `sa`                   |
| Contraseña | *(vacía)*             |

---

## Endpoints disponibles

| Metodo | URL                    | Descripcion               |
|--------|------------------------|---------------------------|
| GET    | `/api/productos`       | Listar todos los productos activos |
| GET    | `/api/productos/{id}`  | Buscar producto por ID    |
| POST   | `/api/productos`       | Crear nuevo producto      |
| PUT    | `/api/productos/{id}`  | Actualizar producto existente |
| DELETE | `/api/productos/{id}`  | Eliminar producto         |

---

## Ejemplos de respuesta

### POST /api/productos — Creacion exitosa (201)

```json
{
  "id": 1,
  "nombre": "Laptop",
  "precio": 3500000.0,
  "categoria": "ELECTRONICA"
}
```

### GET /api/productos/999 — Recurso no encontrado (404)

```json
{
  "status": 404,
  "error": "Not Found",
  "mensaje": "Producto con id 999 no encontrado.",
  "timestamp": "2026-01-01T10:00:00",
  "path": "/api/productos/999"
}
```

### POST /api/productos — Error de validacion (400)

```json
{
  "status": 400,
  "error": "Bad Request",
  "mensaje": "nombre: El nombre es obligatorio; precio: El precio debe ser mayor a cero",
  "timestamp": "2026-01-01T10:00:00",
  "path": "/api/productos"
}
```

---

## Principios SOLID aplicados

**SRP — Single Responsibility Principle**
Cada clase tiene una unica responsabilidad: el controlador gestiona HTTP, el servicio contiene la logica de negocio, el repositorio maneja el acceso a datos y la fabrica realiza la conversion entre entidades y DTOs.

**DIP — Dependency Inversion Principle**
`ProductoController` depende de la interfaz `ProductoService` y no de su implementacion concreta `ProductoServiceImpl`. Spring Boot inyecta la implementacion en tiempo de ejecucion.

---

## Patrones de diseno aplicados

**DAO (Data Access Object)**
`ProductoRepository` extiende `JpaRepository` y abstrae completamente el acceso a la base de datos, desacoplando la capa de persistencia del resto de la aplicacion.

**DTO (Data Transfer Object)**
Se usan dos DTOs diferenciados: `ProductoRequestDTO` recibe y valida los datos de entrada, mientras que `ProductoResponseDTO` controla los campos expuestos en la respuesta, sin exponer la entidad directamente.

**Factory**
`ProductoFactory` centraliza la logica de conversion entre la entidad `Producto` y sus DTOs, evitando que esa responsabilidad quede dispersa en el servicio o el controlador.

---

## Checkpoints verificados

- Checkpoint 1 — Arquitectura en capas con separacion de responsabilidades (SRP y DIP)
- Checkpoint 2 — Creacion de producto via POST retorna status 201 con `ProductoResponseDTO`
- Checkpoint 3 — `GlobalExceptionHandler` retorna errores estructurados en formato `ApiError` para 400, 404 y 500

---

## Capturas de evidencia

Las capturas de las pruebas realizadas se encuentran en los archivos `img.png`, `img_1.png` e `img_2.png` en la raiz del repositorio, correspondientes a los checkpoints 2 y 3.

---

## Licencia

Proyecto academico — Universidad Francisco de Paula Santander · 2026
