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

### Crear Environment (Recomendado)
```bash
# Crear environment con conda
conda create -n chess_ai python=3.9.23
conda activate chess_ai
pip install -r requirements.txt

# O clonar el environment existente
conda create --name chess_ai --clone chess_fixed
```

### Requisitos
```bash
# Instalar desde requirements.txt
pip install -r requirements.txt

# O instalar las dependencias principales manualmente:
pip install streamlit==1.46.1
pip install python-chess==1.999
pip install chess==1.11.2
pip3 install torch torchvision torchaudio
pip install numpy==1.21.2
pip install pygame==2.1.0
pip install opencv-python==4.11.0.86
pip install matplotlib==3.9.4
pip install pandas==2.3.0
pip install scikit-learn==1.6.1
pip install gym==0.26.2
pip install numpydoc==1.5.0
pip install gym-notices==0.0.6
pip install matplotlib==3.5.1
pip install tqdm==4.62.3
pip install opencv-python==4.4.0.46

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
### Train de 2000 episodes
Para entrenar double agents, corrimos train, y en single agents, cambiamos el directorio de results y la librería a SingleAgents , con 2000 episodes, donde los archivos se guardarán en formato ppt en results/
```bash
python train.py
python train_single.py
```

### Evaluar modelos
Se generaron N partidas para evaluar los modelos. Para single_agents y double agents son 400 partidas, y para ambos, son 100 partidas
```bash
wins_white = 0
wins_black = 0
draws = 0
N = 400  # 400 partidas para single_agents_stats y double_agents stats, pero 100 para evaluate_models
```

Con el siguiente comando:
```bash
python evaluate_single_agent_stats.py # Estadísticas para Single agents
python evaluate_double_agent_stats.py  # estadísticas para double agents 
python evaluate_models_vs.py  # Estadísticas para partidas enfrentando ambos modelos
```


### Play_vs human en entorno local
```bash
python play_vs_human.py
```

### Entorno de streamlit:
Agregar este código al directorio raiz: 

```python
import streamlit as st
import pygame
import numpy as np
from PIL import Image
from learnings.ppo import PPO
from chess import Chess

# Configuración de la página
st.set_page_config(
    page_title="Ajedrez vs IA",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inicializar Pygame una sola vez
if 'pygame_initialized' not in st.session_state:
    pygame.init()
    st.session_state.pygame_initialized = True

# Variables globales para el screen
if 'screen' not in st.session_state:
    st.session_state.screen = pygame.display.set_mode((800, 800))

# Mapeo de nombres de piezas
def get_piece_names(human_color):
    return {
        0: "Vacío",
        1: f"Peón {'Blanco' if human_color == 0 else 'Negro'}",
        2: f"Alfil {'Blanco' if human_color == 0 else 'Negro'}",
        3: f"Caballo {'Blanco' if human_color == 0 else 'Negro'}",
        4: f"Torre {'Blanca' if human_color == 0 else 'Negra'}",
        5: f"Reina {'Blanca' if human_color == 0 else 'Negra'}",
        6: f"Rey {'Blanco' if human_color == 0 else 'Negro'}",
    }

# Función para reiniciar el juego
def reset_game():
    st.session_state.env = Chess(render_mode="human")
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.info = [{}, {}]
    st.session_state.output_log = []
    st.session_state.awaiting_input = False
    st.session_state.legal_moves = []
    st.session_state.move_count = 0
    st.session_state.game_log = []

# Inicialización de variables en sesión
if 'env' not in st.session_state:
    st.session_state.env = Chess(render_mode="human")
    st.session_state.human_color = 0
    st.session_state.model_color = 1
    st.session_state.model = None
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.info = [{}, {}]
    st.session_state.output_log = []
    st.session_state.awaiting_input = False
    st.session_state.legal_moves = []
    st.session_state.move_count = 0
    st.session_state.game_log = []

def render_board():
    """Renderiza el tablero y devuelve la imagen"""
    try:
        st.session_state.env.render()
        pygame.display.flip()
        surface = pygame.display.get_surface()
        data = pygame.surfarray.array3d(surface)
        img = Image.fromarray(np.transpose(data, (1, 0, 2)))
        return img
    except Exception as e:
        st.error(f"Error al renderizar el tablero: {e}")
        return None

# Título principal
st.title("♟️ Ajedrez vs IA")

# Selección de color inicial
if 'color_selected' not in st.session_state:
    st.markdown("## 🎯 Elige el color con el que deseas jugar")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("### Selecciona tu color:")
        
        col_white, col_black = st.columns(2)
        
        with col_white:
            if st.button("⚪ Blancas", use_container_width=True, type="primary"):
                st.session_state.human_color = 0
                st.session_state.model_color = 1
                try:
                    st.session_state.model = PPO.load("results/DoubleAgents/black_ppo.pt")
                    st.session_state.color_selected = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al cargar el modelo: {e}")
        
        with col_black:
            if st.button("⚫ Negras", use_container_width=True, type="secondary"):
                st.session_state.human_color = 1
                st.session_state.model_color = 0
                try:
                    st.session_state.model = PPO.load("results/DoubleAgents/white_ppo.pt")
                    st.session_state.color_selected = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al cargar el modelo: {e}")

else:
    # Juego en curso - Layout principal
    st.markdown("---")
    
    # Información del juego
    info_col1, info_col2, info_col3 = st.columns([1, 1, 1])
    
    with info_col1:
        st.metric("🤵 Humano", f"{'⚪ Blancas' if st.session_state.human_color == 0 else '⚫ Negras'}")
    
    with info_col2:
        st.metric("🎯 Movimientos", st.session_state.move_count)
    
    with info_col3:
        st.metric("🤖 IA", f"{'⚪ Blancas' if st.session_state.model_color == 0 else '⚫ Negras'}")
    
    st.markdown("---")
    
    # Layout principal del juego
    board_col, controls_col = st.columns([2, 1])
    
    with board_col:
        st.markdown("### 🏁 Tablero de Ajedrez")
        
        # Renderizar el tablero
        board_img = render_board()
        if board_img:
            st.image(board_img, width=500)
        else:
            st.error("No se pudo renderizar el tablero")
    
    with controls_col:
        if not st.session_state.done:
            env = st.session_state.env
            turn = env.turn
            
            # Información del turno actual
            current_player = "🤵 Humano" if turn == st.session_state.human_color else "🤖 IA"
            current_color = "⚪ Blancas" if turn == 0 else "⚫ Negras"
            
            st.markdown(f"### {current_player}")
            st.markdown(f"**Turno:** {current_color}")
            
            if turn == st.session_state.human_color:
                # Turno del humano
                state = env.get_state(turn)
                source_pos, all_actions, action_mask = env.get_all_actions(turn)
                
                legal_moves = [
                    (i, source_pos[i], all_actions[i])
                    for i in range(len(action_mask))
                    if action_mask[i]
                ]
                
                if not legal_moves:
                    st.warning("⚠️ No hay movimientos válidos.")
                    action = 0
                    rewards, done, info = env.step(action)
                    st.session_state.done = done
                    st.session_state.info = info
                    st.rerun()
                else:
                    st.markdown("### 🎯 Movimientos disponibles:")
                    
                    # Crear opciones de movimiento más legibles
                    piece_names = get_piece_names(st.session_state.human_color)
                    move_options = []
                    
                    for idx, (_, src, dst) in enumerate(legal_moves):
                        board_index = st.session_state.human_color
                        piece_code = env.board[board_index, src[0], src[1]]
                        piece_name = piece_names.get(piece_code, f"Desconocido({piece_code})")
                        move_desc = f"{idx}: {piece_name} ({src[1]}, {src[0]}) → ({dst[1]}, {dst[0]})"
                        move_options.append(move_desc)
                    
                    # Mostrar movimientos en un selectbox
                    with st.container():
                        st.markdown("**Selecciona tu movimiento:**")
                        for i, option in enumerate(move_options):
                            st.markdown(f"**{i}:** {option.split(':', 1)[1]}")
                    
                    st.markdown("---")
                    
                    # Input para elegir movimiento
                    move_input = st.text_input(
                        "Introduce el índice del movimiento:",
                        key="move_input",
                        placeholder="Ejemplo: 0, 1, 2..."
                    )
                    
                    col_execute, col_reset = st.columns(2)
                    
                    with col_execute:
                        if st.button("✅ Ejecutar", use_container_width=True, type="primary"):
                            if move_input.strip() != "":
                                try:
                                    choice = int(move_input)
                                    if 0 <= choice < len(legal_moves):
                                        action_idx, src, dst = legal_moves[choice]
                                        
                                        # Registrar el movimiento
                                        board_index = st.session_state.human_color
                                        piece_code = env.board[board_index, src[0], src[1]]
                                        piece_name = piece_names.get(piece_code, f"Desconocido({piece_code})")
                                        move_desc = f"Movimiento {st.session_state.move_count + 1}: {piece_name} ({src[1]}, {src[0]}) → ({dst[1]}, {dst[0]})"
                                        st.session_state.game_log.append(move_desc)
                                        st.session_state.move_count += 1
                                        
                                        # Ejecutar la acción
                                        rewards, done, info = env.step(action_idx)
                                        st.session_state.done = done
                                        st.session_state.info = info
                                        st.rerun()
                                    else:
                                        st.error("❌ Índice fuera de rango.")
                                except ValueError:
                                    st.error("❌ Entrada inválida. Introduce un número.")
                            else:
                                st.warning("⚠️ Introduce un índice válido.")
                    
                    with col_reset:
                        if st.button("🔄 Reiniciar", use_container_width=True, type="secondary"):
                            reset_game()
                            st.rerun()
                            
            else:
                # Turno de la IA
                st.markdown("### 🤖 La IA está pensando...")
                
                with st.spinner("Procesando movimiento..."):
                    try:
                        state = env.get_state(turn)
                        source_pos, all_actions, action_mask = env.get_all_actions(turn)
                        
                        action, _, _ = st.session_state.model.take_action(state, action_mask)
                        
                        # Registrar el movimiento de la IA
                        st.session_state.move_count += 1
                        st.session_state.game_log.append(f"Movimiento {st.session_state.move_count}: IA juega")
                        
                        rewards, done, info = env.step(action)
                        st.session_state.done = done
                        st.session_state.info = info
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error en el movimiento de la IA: {e}")
        else:
            # Juego terminado
            st.markdown("### 🎉 Resultado del juego")
            
            if "check_mate_win" in st.session_state.info[st.session_state.human_color]:
                st.success("🎉 **¡Ganaste! ¡Felicidades!**")
                st.balloons()
            elif "check_mate_win" in st.session_state.info[st.session_state.model_color]:
                st.error("🤖 **La IA ganó esta vez**")
            else:
                st.info("🤝 **Empate - ¡Buen juego!**")
            
            st.markdown("---")
            
            # Botones de acción final
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Nueva partida", use_container_width=True, type="primary"):
                    reset_game()
                    st.rerun()
            
            with col2:
                if st.button("🎯 Cambiar color", use_container_width=True, type="secondary"):
                    # Limpiar todo para empezar de nuevo
                    keys_to_clear = [key for key in st.session_state.keys()]
                    for key in keys_to_clear:
                        if key != 'pygame_initialized' and key != 'screen':
                            del st.session_state[key]
                    st.rerun()

# Sidebar con historial de movimientos
if 'color_selected' in st.session_state and st.session_state.game_log:
    with st.sidebar:
        st.markdown("### 📋 Historial de movimientos")
        for move in st.session_state.game_log[-10:]:  # Mostrar últimos 10 movimientos
            st.markdown(f"• {move}")

```
            



### Interfaz Web
1. Ejecuta `streamlit run streamlit2.py`
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


## 🎯 Características Clave

- ✅ Sistema de recompensas orientado a victoria
- ✅ Interfaz web intuitiva
- ✅ Soporte para terminal
- ✅ Modelo entrenado con enfoque estratégico
- ✅ Evaluación basada en jaque mate, no en material

## 🚧 Limitaciones Conocidas

- La interfaz Streamlit muestra el tablero como imagen (no táctil)
- Los movimientos deben realizarse através de la terminal


## 🤝 Contribuciones

Este proyecto fue desarrollado como parte del curso de Reforcement Learning. Las contribuciones principales incluyen:
- Implementación del sistema de recompensas estratégicas
- Desarrollo de la interfaz híbrida (web + terminal)
- Entrenamiento del modelo con enfoque en victoria

## 📝 Notas Adicionales

- El modelo no busca comer piezas sino ganar partidas
- La penalización por movimiento (-1) fomenta la eficiencia
- Las recompensas de jaque y jaque mate guían el aprendizaje estratégico

---

*Desarrollado con 🧠 y ♟️*
