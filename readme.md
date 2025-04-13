
# Sistema de Optimización de Rutas de Transporte con IA 🌐

---

## 📅 Descripción General
Este proyecto implementa un sistema de optimización de rutas de transporte mediante técnicas de **Inteligencia Artificial**:
- 🔄 **Aprendizaje No Supervisado** (clustering con **K-Means**)
- 📊 **Aprendizaje por Refuerzo** (algoritmo simple de **Q-Learning**)

El objetivo es encontrar rutas eficientes considerando distancias, tiempos de viaje, mantenimiento de tramos y congestiones.

---

## 👉 Requisitos Previos

| Requisito | Detalle |
|:---------|:-------|
| **Python** | Versión recomendada: **3.11.9** |
| **Librerías** | `numpy`, `networkx`, `scikit-learn`, `matplotlib` |

### Instalación de Librerías

```bash
pip install numpy networkx scikit-learn matplotlib
```

---

## 📂 Estructura del Proyecto

| Archivo | Descripción |
|:--------|:------------|
| `main.py` | Script principal del sistema de transporte IA. |
| `datos.json` | Archivo de entrada: estaciones, conexiones y reglas. |
| `Resultados.json` | Archivo de salida: rutas óptimas y tiempos estimados. |

---

## 🔧 Pasos de Ejecución

1. 📂 **Preparar el Archivo `datos.json`**

Ejemplo de estructura:

```json
{
  "estaciones": [
    {
      "id": "EST1",
      "lineas": ["L1", "L2"],
      "nombre": "Estación Central"
    }
  ],
  "conexiones": [
    {
      "origen": "EST1",
      "destino": "EST2",
      "tiempo": 10,
      "distancia": 5,
      "linea": "L1"
    }
  ],
  "reglas": [
    {
      "tipo": "mantenimiento_tramo",
      "origen": "EST1",
      "destino": "EST2"
    }
  ]
}
```

2. 🚀 **Ejecutar el Script**

```bash
python main.py
```

---

## 🔹 Características Principales

- 🎉 **Optimización de Rutas** con **Q-Learning**
- 🌍 **Agrupación de Conexiones** mediante **K-Means**
- 🛟 **Visualización de Clusters** usando **matplotlib**
- 🔢 **Consideración de Reglas Dinámicas** (mantenimiento y congestión)
- 📚 **Resultados Detallados** en formato JSON

---

## 📋 Salida Esperada

Tras la ejecución:
- `Resultados.json` contendrá:
  - Rutas óptimas entre combinaciones de estaciones.
  - Tiempos estimados de cada trayecto.
  - Información detallada de las estaciones.

Ejemplo de salida:

```json
{
  "origen": "EST1",
  "destino": "EST2",
  "ruta": ["EST1", "EST2"],
  "tiempo_total": {"horas": 0, "minutos": 12},
  "tiempos_tramos": [{"horas": 0, "minutos": 12}],
  "detalles_estaciones": [{"id": "EST1", "nombre": "Central"}, {"id": "EST2", "nombre": "Norte"}]
}
```

---

## 🔺 Personalización

Puedes editar `datos.json` para:
- Agregar 📉 nuevas estaciones.
- Modificar 🔗 conexiones o tiempos.
- Definir 🏫 reglas de mantenimiento o congestionamiento.

---

## 📚 Notas Adicionales
- Este proyecto es un **prototipo académico** orientado a aprender conceptos de IA aplicada.
- No está pensado para ser usado en sistemas de producción reales.
- Se recomienda experimentar con diferentes configuraciones para entender mejor el comportamiento del modelo.

---

## 🚀 Autor

**David Andrés Rincón López**

✨ Proyecto de Aprendizaje en Inteligencia Artificial Aplicada a Transporte Urbano ✨
