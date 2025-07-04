import streamlit as st
import pygame
import numpy as np
from PIL import Image
from learnings.ppo import PPO
from chess import Chess

# Inicializar Pygame una sola vez
pygame.init()
screen = pygame.display.set_mode((800, 800))

# Inicialización de variables en sesión
if 'env' not in st.session_state:
    st.session_state.env = Chess(render_mode="human")
    st.session_state.human_color = 0
    st.session_state.model_color = 1
    st.session_state.model = PPO.load("results/DoubleAgents/black_ppo.pt")
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.info = [{}, {}]
    st.session_state.output_log = []
    st.session_state.awaiting_input = False
    st.session_state.legal_moves = []

def render_board():
    st.session_state.env.render()
    pygame.display.flip()  # <-- Esto asegura que el contenido se actualice
    surface = pygame.display.get_surface()
    data = pygame.surfarray.array3d(surface)
    img = Image.fromarray(np.transpose(data, (1, 0, 2)))
    st.image(img, caption="Tablero", width=800)


# Selección de color inicial
if 'color_selected' not in st.session_state:
    st.write("## Elige el color con el que deseas jugar")
    col1, col2 = st.columns(2)
    if col1.button("Blancas"):
        st.session_state.human_color = 0
        st.session_state.model_color = 1
        st.session_state.model = PPO.load("results/DoubleAgents/black_ppo.pt")
        st.session_state.color_selected = True
        st.rerun()
    if col2.button("Negras"):
        st.session_state.human_color = 1
        st.session_state.model_color = 0
        st.session_state.model = PPO.load("results/DoubleAgents/white_ppo.pt")
        st.session_state.color_selected = True
        st.rerun()
else:
    # Juego en curso
    render_board()

    if not st.session_state.done:
        env = st.session_state.env
        turn = env.turn
        state = env.get_state(turn)
        source_pos, all_actions, action_mask = env.get_all_actions(turn)

        if turn == st.session_state.human_color:
            st.write(f"### Turno del humano ({'blancas' if turn == 0 else 'negras'})")

            legal_moves = [
                (i, source_pos[i], all_actions[i])
                for i in range(len(action_mask))
                if action_mask[i]
            ]
            st.session_state.legal_moves = legal_moves

            if not legal_moves:
                st.session_state.output_log.append("No hay movimientos válidos.")
                action = 0
                st.session_state.awaiting_input = False
            else:
                st.write("### Movimientos posibles:")
                for idx, (_, src, dst) in enumerate(legal_moves):
                    st.write(f"{idx}: ({src[1]}, {src[0]}) → ({dst[1]}, {dst[0]})")

                st.session_state.awaiting_input = True

                move_input = st.text_input("Elige tu acción (índice) y presiona ENTER:", key="move")
                if move_input.strip() != "":
                    try:
                        choice = int(move_input)
                        if 0 <= choice < len(legal_moves):
                            action_idx, _, _ = legal_moves[choice]
                            action = action_idx
                            st.session_state.awaiting_input = False
                        else:
                            st.warning("Índice fuera de rango.")
                            st.stop()
                    except ValueError:
                        st.warning("Entrada inválida.")
                        st.stop()
                else:
                    st.stop()
        else:
            st.write(f"### El modelo ({'blancas' if turn == 0 else 'negras'}) juega...")
            action, _, _ = st.session_state.model.take_action(state, action_mask)

        rewards, done, info = env.step(action)
        st.session_state.done = done
        st.session_state.info = info
        st.rerun()
    else:
        if "check_mate_win" in st.session_state.info[st.session_state.human_color]:
            st.success("Ganó el humano")
        elif "check_mate_win" in st.session_state.info[st.session_state.model_color]:
            st.error("Ganó el modelo")
        else:
            st.info("Empate")

        if st.button("Jugar otra partida"):
            st.session_state.clear()
            st.rerun()
