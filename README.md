# 🔮 Ajedrez con Inteligencia Artificial

Un proyecto de ajedrez que implementa un agente inteligente usando algoritmos de aprendizaje por refuerzo para jugar ajedrez de manera estratégica.

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de ajedrez con IA que se enfoca en **ganar partidas** en lugar de simplemente capturar piezas. El modelo utiliza un sistema de recompensas orientado a objetivos estratégicos como jaque, jaque mate y victoria general.

### Sistema de Recompensas
```python
MOVE = -1              # Penalización por movimiento (fomenta eficiencia)
CHECK_WIN = 10         # Recompensa por dar jaque
CHECK_LOSE = -10       # Penalización por recibir jaque
CHECK_MATE_WIN = 100   # Recompensa máxima por jaque mate
CHECK_MATE_LOSE = -100 # Penalización máxima por perder
```

## 🏗️ Arquitectura del Sistema

### Componentes Principales
- **Motor de Ajedrez**: Implementación de las reglas y lógica del juego
- **Agente IA**: Algoritmo de aprendizaje por refuerzo 
- **Interfaz Gráfica**: Aplicación web con Streamlit
- **Sistema de Entrenamiento**: Módulo para entrenar el modelo

### Tecnologías Utilizadas
- **Python 3.9**
- **Streamlit 1.46.1** (interfaz web)
- **python-chess 1.999** + **chess 1.11.2** (motor de ajedrez)
- **PyTorch 2.7.1** (deep learning)
- **OpenAI Gym 0.26.2** (entorno de aprendizaje por refuerzo)
- **NumPy, Pandas, Matplotlib** (análisis de datos)
- **OpenCV, Pygame** (visualización)
- **Scikit-learn** (machine learning)

## 🚀 Instalación y Configuración

### Requisitos
```bash
# Instalar desde requirements.txt
pip install -r requirements.txt

# O instalar las dependencias principales manualmente:
pip install streamlit==1.46.1
pip install python-chess==1.999
pip install chess==1.11.2
pip install torch==2.7.1
pip install numpy==1.24.3
pip install pygame==2.6.1
pip install opencv-python==4.11.0.86
pip install matplotlib==3.9.4
pip install pandas==2.3.0
pip install scikit-learn==1.6.1
pip install gym==0.26.2
```

### Crear Environment (Recomendado)
```bash
# Crear environment con conda
conda create -n chess_ai python=3.9
conda activate chess_ai
pip install -r requirements.txt

# O clonar el environment existente
conda create --name chess_ai --clone chess_fixed
```

### Ejecución
```bash
# Activar environment
conda activate chess_fixed

# Ejecutar la aplicación web
streamlit run app.py

# Alternativamente, ejecutar desde terminal
python main.py
```

### Generar requirements.txt (si necesitas actualizarlo)
```bash
conda activate chess_fixed
pip freeze > requirements.txt
```

## 🎮 Cómo Usar

### Interfaz Web
1. Ejecuta `streamlit run app.py`
2. Abre tu navegador en `http://localhost:8501`
3. **Nota**: La interfaz muestra el tablero como imagen, usa los controles de la terminal para realizar movimientos

### Terminal
- Utiliza la terminal para introducir movimientos
- El formato de movimientos sigue la notación algebraica estándar

## 🧠 Entrenamiento del Modelo

El modelo fue entrenado utilizando:
- **Enfoque**: Aprendizaje por refuerzo
- **Objetivo**: Maximizar victorias, no capturas
- **Recompensas**: Sistema balanceado que premia estrategia sobre táctica simple

### Proceso de Entrenamiento
1. Inicialización del agente
2. Juegos de práctica contra diferentes oponentes
3. Ajuste de parámetros basado en resultados
4. Validación del modelo entrenado

## 📊 Resultados

- El modelo demostró capacidad para priorizar movimientos estratégicos
- Enfoque en jaque mate sobre captura de piezas
- [Agregar métricas específicas de rendimiento]

## 🔧 Estructura del Código

```
proyecto/
├── app.py              # Aplicación Streamlit
├── main.py             # Ejecución por terminal
├── chess_engine.py     # Motor de ajedrez
├── ai_agent.py         # Agente de IA
├── training.py         # Módulo de entrenamiento
└── README.md          # Este archivo
```

## 🎯 Características Clave

- ✅ Sistema de recompensas orientado a victoria
- ✅ Interfaz web intuitiva
- ✅ Soporte para terminal
- ✅ Modelo entrenado con enfoque estratégico
- ✅ Evaluación basada en jaque mate, no en material

## 🚧 Limitaciones Conocidas

- La interfaz Streamlit muestra el tablero como imagen (no táctil)
- Los movimientos deben realizarse através de la terminal
- [Otras limitaciones identificadas]

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte de [contexto académico/profesional]. Las contribuciones principales incluyen:
- Implementación del sistema de recompensas estratégicas
- Desarrollo de la interfaz híbrida (web + terminal)
- Entrenamiento del modelo con enfoque en victoria

## 📝 Notas Adicionales

- El modelo no busca comer piezas sino ganar partidas
- La penalización por movimiento (-1) fomenta la eficiencia
- Las recompensas de jaque y jaque mate guían el aprendizaje estratégico

---

*Desarrollado con 🧠 y ♟️*
