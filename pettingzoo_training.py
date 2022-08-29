from stable_baselines3 import PPO
import pettingzoo_env_parallel
import pettingzoo_env
import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel


print("--------------------------------------------------------------")
# region parallel
env = pettingzoo_env_parallel.parallel_env()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)
# env.reset()
# for agent in env.agent_iter():
#     observation, reward, done, info = env.last()
#     action = policy(observation, agent)
#     env.step(action)
# endregion

# region regular
# env.reset()
# for agent in env.agent_iter():
#    obs, reward, done, info = env.last()
#    act = model.predict(obs, deterministic=True)[0] if not done else None
#    env.step(act)
#    env.render()
# endregion



