# Explorando JAX: El Futuro de la Computación Numérica Acelerada

Este repositorio contiene una investigación y prueba de concepto sobre **JAX**, la librería de Google que está transformando el desarrollo de modelos de Machine Learning de alto rendimiento.

---

## 1. ¿Qué es JAX?

**JAX** es una librería de computación numérica desarrollada por Google Research que combina la interfaz familiar de **NumPy** con la potencia de la aceleración por hardware (GPU/TPU) y la diferenciación automática.

A diferencia de otros frameworks, JAX no es solo para Redes Neuronales; es una herramienta de **álgebra lineal acelerada** que sigue el paradigma de la **programación funcional**.

### Características Principales

* **Composibilidad de Transformaciones:** Permite encadenar operaciones complejas de forma eficiente.
* **Autograd de Alto Nivel:** Capacidad para calcular gradientes de funciones complejas, incluyendo derivadas de orden superior ($f''(x)$, $f'''(x)$).
* **Compilación XLA (Accelerated Linear Algebra):** JAX utiliza un compilador que optimiza las operaciones de Python para que se ejecuten a velocidades de código nativo.
* **Vectorización Automática (`vmap`):** Permite escribir código para un solo ejemplo y aplicarlo automáticamente a lotes (batches) sin cambiar la lógica.



---

## 2. Comparación: JAX vs. TensorFlow vs. PyTorch

| Característica | JAX | PyTorch | TensorFlow |
| :--- | :--- | :--- | :--- |
| **Paradigma** | Funcional / Inmutable | Imperativo / Orientado a Objetos | Declarativo e Imperativo |
| **Diferenciación** | Basada en funciones (`grad`) | Basada en grafos dinámicos (`Autograd`) | Basada en grafos / Keras |
| **Curva de Aprendizaje** | Alta (requiere mentalidad funcional) | Media (muy "Pythonic") | Media/Alta |
| **Ecosistema** | Modular y en crecimiento | Masivo y estándar en academia | El más maduro para producción industrial |
| **Velocidad** | Excepcional mediante XLA nativo | Muy buena | Excelente con optimizaciones de grafo |

---

## 3. Ecosistema de JAX

JAX es minimalista por diseño. Para construir modelos complejos, se apoya en una serie de librerías especializadas:

* **Flax:** La librería más popular para construir y entrenar redes neuronales sobre JAX.
* **Haiku (DeepMind):** Enfocada en un estilo orientado a objetos para quienes vienen de PyTorch/Sonnet.
* **Optax:** Una librería dedicada exclusivamente a optimizadores (Adam, SGD, etc.).
* **RLAX:** Herramientas para Aprendizaje por Refuerzo (Reinforcement Learning).
* **Chex:** Utilidades para realizar pruebas unitarias y validaciones en código JAX.

---

## 4. Bibliografía y Recursos

Para la elaboración de esta actividad y para profundizar en el uso de **JAX**, se han consultado las siguientes fuentes oficiales, tutoriales y documentación técnica:

### Documentación Oficial y Repositorios
* **[JAX Documentation (Official)](https://jax.readthedocs.io/):** La fuente principal de información, guías de usuario y referencia de la API.
* **[GitHub: Google/JAX](https://github.com/google/jax):** Repositorio oficial de código fuente, ejemplos y notas de las últimas versiones.
* **[XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla):** Documentación sobre el compilador que JAX utiliza para optimizar el rendimiento en hardware.

### Librerías del Ecosistema
* **[Haiku Documentation](https://dm-haiku.readthedocs.io/):** Librería de DeepMind para redes neuronales sobre JAX.
