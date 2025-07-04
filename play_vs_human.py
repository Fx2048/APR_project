from learnings.ppo import PPO
from chess import Chess
import numpy as np

def jugar_partida(human_color, model_color, model, env):
    piece_names = {
        0: "Vacío",
        1: f"Peón {'Blanco' if human_color == 0 else 'Negro'}",
        2: f"Alfil {'Blanco' if human_color == 0 else 'Negro'}",
        3: f"Caballo {'Blanco' if human_color == 0 else 'Negro'}",
        4: f"Torre {'Blanca' if human_color == 0 else 'Negra'}",
        5: f"Reina {'Blanca' if human_color == 0 else 'Negra'}",
        6: f"Rey {'Blanco' if human_color == 0 else 'Negro'}",
    }

    print(f"\nComienza la partida humano ({'blancas' if human_color == 0 else 'negras'}) vs modelo ({'negras' if model_color == 1 else 'blancas'})")

    state = env.reset()
    done = False
    info = [{}, {}]
    env.render()

    while not done:
        state = env.get_state(env.turn)
        source_pos, all_actions, action_mask = env.get_all_actions(env.turn)

        if env.turn == human_color:
            print(f"\nTurno del humano ({'blancas' if human_color == 0 else 'negras'})")

            legal_moves = [
                (i, source_pos[i], all_actions[i])
                for i in range(len(action_mask))
                if action_mask[i]
            ]

            if not legal_moves:
                print("No hay movimientos válidos.")
                action = 0
            else:
                print("Movimientos posibles:")
                for idx, (_, src, dst) in enumerate(legal_moves):
                    board_index = human_color
                    piece_code = env.board[board_index, src[0], src[1]]
                    piece_name = piece_names.get(piece_code, f"Desconocido({piece_code})")
                    print(f"{idx}: {piece_name} en ({src[1]}, {src[0]}) → ({dst[1]}, {dst[0]})")

                while True:
                    try:
                        choice = int(input("Elige tu acción (índice): "))
                        if 0 <= choice < len(legal_moves):
                            action_idx, src, dst = legal_moves[choice]
                            action = action_idx
                            break
                        else:
                            print("Índice fuera de rango.")
                    except ValueError:
                        print("Entrada inválida. Ingresa un número.")
        else:
            print(f"\nEl modelo ({'blancas' if model_color == 0 else 'negras'}) juega")
            action, _, _ = model.take_action(state, action_mask)

        rewards, done, info = env.step(action)
        env.render()

    # Resultado
    if "check_mate_win" in info[human_color]:
        print("Ganó el humano")
    elif "check_mate_win" in info[model_color]:
        print("Ganó el modelo")
    else:
        print("Empate")

# ===================== CICLO PRINCIPAL =====================
while True:
    # Elegir color del humano
    while True:
        color_input = input("¿Con qué color deseas jugar? (blancas / negras): ").strip().lower()
        if color_input in ["blancas", "blanca", "white"]:
            human_color = 0
            model_color = 1
            break
        elif color_input in ["negras", "negra", "black"]:
            human_color = 1
            model_color = 0
            break
        else:
            print("Entrada inválida. Escribe 'blancas' o 'negras'.")

    # Cargar modelo adecuado
    model_path = "results/DoubleAgents/black_ppo.pt" if model_color == 1 else "results/DoubleAgents/white_ppo.pt"
    model = PPO.load(model_path)

    # Crear nuevo entorno
    env = Chess(render_mode="human")

    # Jugar partida
    jugar_partida(human_color, model_color, model, env)

    # Preguntar si desea jugar otra vez
    while True:
        again = input("\n¿Deseas jugar otra partida? (y/n): ").strip().lower()
        if again == "y":
            break
        elif again == "n":
            print("Gracias por jugar. ¡Hasta pronto!")
            exit()
        else:
            print("Entrada inválida. Escribe 'y' o 'n'.")

