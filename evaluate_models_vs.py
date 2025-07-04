from learnings.ppo import PPO
from chess import Chess

white_model = PPO.load("results/DoubleAgents/white_ppo.pt")
black_model = PPO.load("results/DoubleAgents/black_ppo.pt")
single_model = PPO.load("results/SingleAgent/single_agent_ppo.pt.pt")

wins_double = 0
wins_single = 0
draws = 0
N = 100

for i in range(N):
    env = Chess(render_mode="rgb_array")
    state = env.reset()
    done = False

    while not done:
        state = env.get_state(env.turn)
        _, _, action_mask = env.get_all_actions(env.turn)

        if env.turn == 0:  # blancas = double agent (white)
            action, _, _ = white_model.take_action(state, action_mask)
        else:  # negras = single agent
            action, _, _ = single_model.take_action(state, action_mask)

        rewards, done, info = env.step(action)

    if "check_mate_win" in info[0]:
        wins_double += 1
    elif "check_mate_win" in info[1]:
        wins_single += 1
    else:
        draws += 1

print(f"En {N} partidas:")
print(f"- DoubleAgent (blancas) ganó: {wins_double}")
print(f"- SingleAgent (negras) ganó: {wins_single}")
print(f"- Empates: {draws}")
