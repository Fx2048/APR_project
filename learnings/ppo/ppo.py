import os
import gym
import numpy as np
import torch as T
import torch.optim as optim
import torch
from tqdm import tqdm
from buffer.ppo import BufferPPO
from buffer.episode import Episode

from learnings.base import Learning
from learnings.ppo.actor import Actor
from learnings.ppo.critic import Critic

# Optimización para CPU
torch.set_num_threads(os.cpu_count())  # Usar todos los núcleos disponibles

class PPO(Learning):
    def __init__(
        self,
        environment: gym.Env,
        hidden_layers: tuple[int],
        epochs: int,
        buffer_size: int,
        batch_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        learning_rate: float = 0.003,
    ) -> None:
        super().__init__(environment, epochs, gamma, learning_rate)

        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.buffer = BufferPPO(
            gamma=gamma,
            max_size=buffer_size,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
        )

        self.hidden_layers = hidden_layers
        self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
        self.critic = Critic(self.state_dim, hidden_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Forzar dispositivo CPU
        self.device = T.device('cpu')
        self.to(self.device)

    def take_action(self, state: np.ndarray, action_mask: np.ndarray):
        # Asegurar que los tensores estén en CPU
        state = T.tensor(state, dtype=T.float32).unsqueeze(0)
        action_mask = T.tensor(action_mask, dtype=T.float32).unsqueeze(0)
        
        with T.no_grad():  # Optimización: no calcular gradientes durante inferencia
            dist = self.actor(state, action_mask)
            action = dist.sample()
            probs = T.squeeze(dist.log_prob(action)).item()
            value = T.squeeze(self.critic(state)).item()
            action = T.squeeze(action).item()
        
        return action, probs, value

    def epoch(self):
        (
            states_arr,
            actions_arr,
            rewards_arr,
            goals_arr,
            old_probs_arr,
            values_arr,
            masks_arr,
            advantages_arr,
            batches,
        ) = self.buffer.sample()

        for batch in batches:
            # Crear tensores directamente en CPU con tipo de dato específico
            masks = T.tensor(masks_arr[batch], dtype=T.float32)
            values = T.tensor(values_arr[batch], dtype=T.float32)
            states = T.tensor(states_arr[batch], dtype=T.float32)
            actions = T.tensor(actions_arr[batch], dtype=T.float32)
            old_probs = T.tensor(old_probs_arr[batch], dtype=T.float32)
            advantages = T.tensor(advantages_arr[batch], dtype=T.float32)

            dist = self.actor(states, masks)
            critic_value = T.squeeze(self.critic(states))

            new_probs = dist.log_prob(actions)
            prob_ratio = (new_probs - old_probs).exp()

            weighted_probs = advantages * prob_ratio
            weighted_clipped_probs = (
                T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                * advantages
            )

            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            critic_loss = ((advantages + values - critic_value) ** 2).mean()
            total_loss = actor_loss + 0.5 * critic_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def learn(self):
        for epoch in tqdm(range(self.epochs), desc="PPO Learning...", ncols=64, leave=False):
            self.epoch()
        self.buffer.clear()

    def remember(self, episode: Episode):
        self.buffer.add(episode)

    def save(self, folder: str, name: str):
        # Asegurar que el modelo esté en CPU antes de guardar
        self.to('cpu')
        T.save(self, os.path.join(folder, f"{name}.pt"))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPO":
        # Solución simple: usar weights_only=False
        model = T.load(path, map_location='cpu', weights_only=False)
        model.device = T.device('cpu')
        model.actor.to('cpu')
        model.critic.to('cpu')
        return model

# env.render()