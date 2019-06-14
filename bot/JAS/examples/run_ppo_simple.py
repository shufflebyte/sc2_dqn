import ray
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.preprocessors import Preprocessor
import ray.rllib.agents.ppo.ppo as ppo



ray.init(
    local_mode=True
)
agent = ppo.PPOAgent(env="PongDeterministic-v4", config={
    "num_workers": 1,
    "sample_async": False,
    "sample_batch_size": 20,
})

while True:
    result = agent.train()
    print("training_iteration", result["training_iteration"])
    print("timesteps this iter", result["timesteps_this_iter"])
    print("timesteps_total", result["timesteps_total"])
    print("time_this_iter_s", result["time_this_iter_s"])
    print("time_total_s", result["time_total_s"])
    print("episode_reward_mean", result["episode_reward_mean"])
    print()