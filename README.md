# NeuroSim: Motor Denso de Redes Neuronales Optimizado en C++/CUDA

**NeuroSim** es un motor de redes neuronales densas diseñado desde cero en C++ con aceleración CUDA, enfocado en eficiencia computacional y utilidad real. Permite entrenar e inferir modelos densos sobre datasets estructurados o linealizables, con capacidad de integración directa desde Python mediante `pybind11`.

---

## ¿Qué es NeuroSim?

NeuroSim es un motor especializado en **redes neuronales densas (fully connected)**, con soporte completo para:

- Entrenamiento supervisado mediante backpropagation
- Inferencia rápida en CPU o GPU
- Configuración flexible de arquitectura y parámetros
- Uso de precisión reducida (`float16`) para acelerar cómputo en GPU
- Interfaz desde Python para uso práctico en ciencia de datos

Este motor busca demostrar cómo un backend optimizado puede alcanzar gran eficiencia sin recurrir a frameworks pesados como PyTorch o TensorFlow.

---

## ¿Qué lo hace diferente?

| Característica               | NeuroSim                         | PyTorch / TensorFlow            |
|-----------------------------|----------------------------------|---------------------------------|
| Implementación              | Desde cero en C++/CUDA           | En C++ con capas de abstracción|
| Autodiff                    | ❌ Derivadas analíticas manuales | ✅                             |
| Precisión reducida (`fp16`) | ✅ Opcional                       | ✅                             |
| Enfoque                     | Redes densas altamente optimizadas | General-purpose              |
| Visualización integrada     | Opcional                         | Sí, vía TensorBoard             |
| Código abierto y educativo  | ✅                                | ✅                             |

---

## ⚙️ Arquitectura del Proyecto

- `core/`
  - Implementación de redes densas, capas, funciones de activación y pérdida
- `cuda/`
  - Kernels CUDA para operaciones vectorizadas (forward y backward pass)
- `bindings/`
  - Enlace con Python usando `pybind11`
- `examples/`
  - Scripts de entrenamiento e inferencia desde Python
- `tests/`
  - Validaciones numéricas y pruebas de regresión

---

## Casos de uso

- Clasificación de imágenes (por ejemplo, MNIST, FashionMNIST)
- Clasificación tabular multiclase o binaria
- Modelos embebidos de inferencia rápida (IoT / edge computing)
- Experimentación científica (genómica, salud, sensores)

---

## Instalación

Requiere:

- NVIDIA GPU con soporte CUDA
- Python ≥ 3.8
- CMake ≥ 3.18
- pybind11

```bash
git clone https://github.com/andresgomez31/NeuroSim.git
cd NeuroSim
mkdir build && cd build
cmake ..
make -j$(nproc)
```

En Python:

```bash
import neurosim

model = neurosim.Model(...)
model.train(...)
preds = model.predict(...)
```

## En desarrollo futuro

- Entrenamiento multinodo
- Compilación a modelos estáticos para microcontroladores
- Simulador de retropropagación visual
- Modo embedded-only para inferencia portable

## Autor
Desarrollado por Andres Gomez Moreno, como proyecto de aprendizaje avanzado en computación de alto rendimiento, inteligencia artificial y diseño de software eficiente.
