from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
import json
import argparse

from env import DynamicHalfCheetahEnv
from ppo import PolicyAnchoredPPO


parser = argparse.ArgumentParser()
parser.add_argument("--expt_id", type=str, required=True)

args = parser.parse_args()
# params
expt_id = args.expt_id
params = json.load(open(f"params/{expt_id}.json"))
expt_id = params["expt_id"]
max_steps = params["max_steps"]
env = DynamicHalfCheetahEnv(
    switch_after=params["switch_after"], xml_file=params["xml_file"]
)
env = TimeLimit(env, max_episode_steps=params["eps_length"])
env = Monitor(env)
env = make_vec_env(lambda: env, n_envs=1)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # log the anchor_pol_kl_coef value
        self.logger.record("anchor/anchor_pol_kl_coef", self.model.anchor_pol_kl_coef)
        # log td_counter
        self.logger.record("anchor/td_counter", self.model.td_counter)
        # log td_alpha
        self.logger.record("anchor/td_alpha", self.model.td_alpha)
        # log gp_threshold
        self.logger.record("anchor/gp_treshold", self.model.gp_threshold)
        # log env's switches
        self.logger.record("env/switches", self.model.env.get_attr("env")[0].switches)
        # log env's current xml file
        self.logger.record("env/xml_file", self.model.env.get_attr("env")[0].xml_file)

        return True


model = PolicyAnchoredPPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=max_steps,
    device="cuda",
    anchor_pol_kl_coef=params["anchor_pol_kl_coef"],
    td_alpha=params["td_alpha"],
    gp_threshold=params["gp_threshold"],
    tensorboard_log=f"./tbd/{expt_id}",
    eps_length=params["eps_length"],
)

model.load(f"ppo_half_cheetah_upright.zip")
model.learn(total_timesteps=params["total_timesteps"], callback=TensorboardCallback())
model.save(f".models/{expt_id}")
