from typing import Type, Optional, Union, Dict, Any, List, Tuple, ClassVar, TypeVar
import numpy as np
from collections import deque

from stable_baselines3.ppo import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
)
from stable_baselines3.common.buffers import RolloutBuffer

from gymnasium import spaces

import warnings


import torch as th
from torch.nn import functional as F

device = "cuda" if th.cuda.is_available() else "cpu"

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PolicyAnchoredPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        anchor_pol_sample_size: int = 300,
        anchor_pol_kl_coef: float = 0.1,
        gp_threshold: float = 0.5,
        gp_k: int = 5,
        td_alpha: float = 0.5,
        _init_setup_model: bool = True,
        eps_length: int = 512,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1 or (not normalize_advantage)
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.anchor_policy = None
        self.anchor_pol_sample_size = anchor_pol_sample_size
        self.anchor_pol_kl_coef = anchor_pol_kl_coef
        self.gp_threshold = (
            gp_threshold  # Reward threshold to consider a policy as "good"
        )
        self.gp_k = gp_k  # Max number of good policies to store
        self.good_policies: List[
            Tuple[ActorCriticPolicy, float]
        ] = []  # List of (policy, reward)
        self.td_alpha = td_alpha  # Sensitivity parameter for task change detection
        self.previous_rewards = []  # Buffer to store rewards of the previous training step
        self.td_counter = 0  # Counter to keep track of task changes
        self._ep_length = eps_length  # Length of the episode
        self._reward_grad_window = (
            100_000 + self._ep_length
        ) // self._ep_length  # Max length of the reward gradient buffer
        self.reward_grad_threshold = (
            -0.001
        )  # Threshold angle for the reward gradient (tan theta)
        self.episodic_rewards = deque(
            maxlen=self._reward_grad_window
        )  # Buffer to store rewards of the previous training steps
        self.anchor_policy_timestep = (
            -1
        )  # Timestep at which the anchor policy was recorded

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # KL div with anchor
                sampled_state_indices = th.randint(
                    0,
                    rollout_data.observations.shape[0],
                    (self.anchor_pol_sample_size,),
                )
                sampled_states = rollout_data.observations[sampled_state_indices]
                if self.anchor_policy is not None:
                    _, _, anchor_policy_log_probs = self.anchor_policy(sampled_states)
                    # anchor_policy_log_probs = th.log(anchor_policy_probs)

                anchor_policy_kl_div = th.tensor(0.0, device=self.device)
                if self.anchor_policy is not None:
                    # print("using Anchor")
                    _, _, curr_policy_log_probs = self.policy(sampled_states)
                    # curr_policy_log_probs = th.log(curr_policy_probs)
                    anchor_policy_probs = th.exp(anchor_policy_log_probs)
                    anchor_policy_kl_div = th.mean(
                        anchor_policy_probs
                        * (anchor_policy_log_probs - curr_policy_log_probs)
                    )

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = (
                    -th.min(policy_loss_1, policy_loss_2)
                    + self.anchor_pol_kl_coef * anchor_policy_kl_div
                )
                policy_loss = policy_loss.mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # task change detection starts ...
        if len(self.ep_info_buffer) != 0:
            current_rewards = [ep_info["r"] for ep_info in self.ep_info_buffer]
            accumulated_rewards = np.mean(current_rewards)

            self.episodic_rewards.append(accumulated_rewards)
            self.detect_task_change(current_rewards)

            self.previous_rewards = current_rewards

            self.update_good_policies(self.policy, accumulated_rewards)
        # task change detection ends ...

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record(
            "train/policy_loss_main_term",
            -th.min(policy_loss_1, policy_loss_2).mean().item(),
        )
        self.logger.record(
            "train/policy_loss_div_term",
            (self.anchor_pol_kl_coef * anchor_policy_kl_div).item(),
        )
        self.logger.record("train/anchor_kl_div", anchor_policy_kl_div.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def update_good_policies(
        self, policy: ActorCriticPolicy, reward: float, task_change: bool = False
    ):
        """
        Check if the policy is a 'good' policy based on the reward and update the list.
        """
        if reward >= self.gp_threshold or task_change:
            print("Saving good Policy...")
            self.good_policies.append((policy, reward, self.num_timesteps))

            # Sort based on rewards in descending order and keep only top k policies
            self.good_policies.sort(key=lambda x: x[2], reverse=True)
            self.good_policies = self.good_policies[: self.gp_k]

    def detect_task_change(self, current_rewards: List[float]):
        """
        Detect if the task/environment has changed based on reward decline.
        A task change is detected if the current rewards are significantly lower
        than the mean of the previous rewards (by alpha times standard deviation).
        """
        if len(self.previous_rewards) == 0:
            # No previous rewards available to compare, skip detection
            return

        # Calculate mean and standard deviation of previous rewards
        prev_mean = np.mean(self.previous_rewards)
        prev_std = np.std(self.previous_rewards)

        # Calculate mean of current rewards
        current_mean = np.mean(current_rewards)

        # Check if there's a sharp decline (current rewards < prev_mean - alpha * prev_std)
        if self.num_timesteps // 1e6 != (self.num_timesteps - self.n_steps) // 1e6:
            print(f"self.num_timesteps // 1e6:  {self.num_timesteps // 1e6}")
            print(f"self._ep_length:  {self._ep_length}")
            print(f"self.num_timesteps:  {self.num_timesteps}")
            print(
                f"(self.num_timesteps - self._ep_length) // 1e6:  {(self.num_timesteps - self._ep_length) // 1e6}"
            )
            # print("self.num_timesteps: ", self.num_timesteps)
            print(
                f"Task change detected! Current rewards: {current_mean:.2f}, Previous mean: {prev_mean:.2f}, Std: {prev_std:.2f}"
            )
            self.td_counter += 1
            self.anchor_policy = self.good_policies[0][0]
            # print("Policy Taken: ", self.good_policies[0][2])
            self.anchor_policy_timestep = self.good_policies[0][2]

    def get_good_policies(self):
        """
        Get the current list of good policies and their associated rewards.
        """
        return self.good_policies
