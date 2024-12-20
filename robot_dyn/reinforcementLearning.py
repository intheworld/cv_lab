import gymnasium as gym
from stable_baselines3 import PPO
import stable_baselines3
import matplotlib.pylab as plt
import numpy as np

'''
with gym.make("InvertedPendulum-v4", render_mode="human") as env:
    action = 0.0 * env.action_space.sample()
    observation, _ = env.reset()
    episode_return = 0.0
    for step in range(200):
        # action[0] = 5.0 * observation[1] + 0.3 * observation[0]
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        if terminated or truncated:
            observation, _ = env.reset()
    print(f"Return of the episode: {episode_return}")
'''

OBSERVATION_LEGEND = ("position", "pitch", "linear_velocity", "angular_velocity")


def rollout_from_env(env, policy):
    episode = []
    observation, _ = env.reset()
    episode.append(observation)
    for step in range(1000):
        action = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode.extend([action, reward, observation])
        if terminated or truncated:
            return episode
    return episode

def rollout(policy, show: bool = True):
    kwargs = {"render_mode": "human"} if show else {}
    with gym.make("InvertedPendulum-v4", **kwargs) as env:
        episode = rollout_from_env(env, policy)
    return episode

def pid_policy(observation: np.ndarray) -> np.ndarray:
    position, pitch, linear_velocity, angular_velocity = observation
    print(observation)
    pole_control = 70 * pitch + 5.4 * angular_velocity
    cart_control = - 2 * position - 0 * linear_velocity
    my_action_value: float = pole_control + cart_control
    print(f'action = {my_action_value}')
    return np.array([my_action_value])

episode = rollout(pid_policy, show = True)

observations = np.array(episode[::3])

plt.plot(observations)
plt.legend(OBSERVATION_LEGEND)
plt.show()

print(f"Return of the episode: {sum(episode[2::3])}")

### PPO

env = gym.make("InvertedPendulum-v4")
second_policy = PPO("MlpPolicy", env, verbose=1)
second_policy.learn(total_timesteps=1_000, progress_bar=False)


def policy_closure(policy):
    """Utility function to turn our policy instance into a function.
    Args:
        policy: Policy to turn into a function.
    Returns:
        Function from observation to policy action.
    """

    def policy_function(observation):
        action, _ = policy.predict(observation)
        return action

    return policy_function

episode = rollout(policy_closure(second_policy), show=True)

erudite_policy = PPO(
    "MlpPolicy",
    env,
    tensorboard_log="./inverted_pendulum_tensorboard/",
    verbose=0,
)

erudite_policy.learn(
    total_timesteps=1_000_000,
    progress_bar=False,
    tb_log_name="erudite",
)


input('Press ENTER to exit.')