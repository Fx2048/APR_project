from chess import Chess
from agents import SingleAgentChess, DoubleAgentsChess
from learnings.ppo import PPO

buffer_size = 32
if __name__ == "__main__":
    chess = Chess(window_size=512, max_steps=128, render_mode="rgb_array")
    chess.reset()

    ppo = PPO(
        chess,
        hidden_layers=(2048,) * 4,
        epochs=2,
        buffer_size=buffer_size * 2,
        batch_size=8
    )

    print(ppo.device)
    print(ppo)
    print("-" * 64)

    agent = SingleAgentChess(
        env=chess,
        learner=ppo,
        episodes=10,
        train_on=buffer_size,
        result_folder="results/SingleAgent",
    )
    agent.train(render_each=20, save_on_learn=True)
    agent.save()
    chess.close()
