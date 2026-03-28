"""Unified trainer for stochastic flow policy PPO experiments."""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.envs import get_env_info, make_env, make_vec_env
from src.algorithms.ppo_awfm import PPOAWFM
from src.algorithms.ppo_cumulative import PPOCumulative
from src.algorithms.ppo_fisher import PPOFisher
from src.algorithms.ppo_hierarchical import PPOHierarchical
from src.algorithms.ppo_hierarchical_cumulative import PPOHierarchicalCumulative
from src.algorithms.ppo_holistic import PPOHolistic
from src.algorithms.ppo_intra_chain import PPOIntraChain
from src.algorithms.ppo_per_step import PPOPerStep
from src.algorithms.ppo_step_conditioned import PPOStepConditioned
from src.training.evaluate import evaluate_policy
from src.training.logger import Logger


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Unified trainer that orchestrates environment creation, agent setup,
    training loop, evaluation, logging, and checkpointing.

    Expected config keys::

        env_name: str               # e.g. "HalfCheetah-v5"
        seed: int                   # random seed
        total_timesteps: int        # total env steps to train

        policy:
            hidden_dim: int
            latent_dim: int
            K: int
            sigma_init: float
            learn_sigma: bool
            sigma_network: bool

        ppo:
            clip_eps: float
            gamma: float
            gae_lambda: float
            num_rollout_steps: int
            num_epochs: int
            batch_size: int
            lr: float
            vf_coef: float
            ent_coef: float
            max_grad_norm: float

        method:
            mode: str               # "holistic", "per_step_uniform",
                                    # "per_step_learned_global",
                                    # "per_step_state_dependent"

        eval:
            eval_freq: int          # evaluate every N timesteps
            num_eval_episodes: int

        logging:
            log_dir: str
            save_freq: int          # save checkpoint every N timesteps
            use_tensorboard: bool

    Args:
        config: Nested configuration dictionary (typically loaded from YAML).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Seed
        seed = config.get("seed", 0)
        _set_seed(seed)

        # Device
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Environment info
        env_name = config["env_name"]
        env_info = get_env_info(env_name)
        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.action_high = env_info["action_high"]

        # Create train and eval environments
        self.num_envs = config.get("num_envs", 1)
        if self.num_envs > 1:
            self.train_env = make_vec_env(env_name, self.num_envs, seed=seed)
        else:
            self.train_env = make_env(env_name, seed=seed)
        self.eval_env = make_env(env_name, seed=seed + 1000)

        # Build agent config
        policy_cfg = config.get("policy", {})
        ppo_cfg = config.get("ppo", {})
        method_cfg = config.get("method", {})
        mode = method_cfg.get("mode", "holistic")

        agent_config: Dict[str, Any] = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": policy_cfg.get("hidden_dim", 256),
            "latent_dim": policy_cfg.get("latent_dim") or self.action_dim,
            "K": policy_cfg.get("K", 4),
            "sigma_init": policy_cfg.get("sigma_init", 1.0),
            "learn_sigma": policy_cfg.get("learn_sigma", False),
            "sigma_network": policy_cfg.get("sigma_network", False),
            "gamma": ppo_cfg.get("gamma", 0.99),
            "gae_lambda": ppo_cfg.get("gae_lambda", 0.95),
            "clip_eps": ppo_cfg.get("clip_eps", 0.2),
            "vf_coef": ppo_cfg.get("vf_coef", 0.5),
            "ent_coef": ppo_cfg.get("ent_coef", 0.01),
            "max_grad_norm": ppo_cfg.get("max_grad_norm", 0.5),
            "num_steps": ppo_cfg.get("num_rollout_steps", 2048),
            "num_epochs": ppo_cfg.get("num_epochs", 10),
            "batch_size": ppo_cfg.get("batch_size", 64),
            "lr": ppo_cfg.get("lr", 3e-4),
            "device": str(self.device),
            "num_envs": self.num_envs,
        }

        # Create the appropriate PPO agent
        self.mode = mode
        if mode == "holistic":
            self.agent = PPOHolistic(agent_config)
        elif mode == "hierarchical":
            agent_config["clip_eps_scale"] = method_cfg.get("clip_eps_scale", 1.0)
            agent_config["correction_hidden_dim"] = method_cfg.get("correction_hidden_dim", 128)
            agent_config["correction_lr"] = method_cfg.get("correction_lr", ppo_cfg.get("lr", 3e-4))
            agent_config["delta_reg"] = method_cfg.get("delta_reg", 10.0)
            agent_config["delta_reg_init"] = method_cfg.get("delta_reg_init", None)
            agent_config["delta_reg_final"] = method_cfg.get("delta_reg_final", 0.1)
            agent_config["delta_reg_schedule"] = method_cfg.get("delta_reg_schedule", "constant")
            agent_config["asymmetric_clip"] = method_cfg.get("asymmetric_clip", False)
            self.agent = PPOHierarchical(agent_config)
        elif mode == "intra_chain":
            agent_config["clip_eps_scale"] = method_cfg.get("clip_eps_scale", 1.0)
            agent_config["intra_vf_coef"] = method_cfg.get("intra_vf_coef", 0.5)
            agent_config["intra_vf_lr"] = method_cfg.get("intra_vf_lr", ppo_cfg.get("lr", 3e-4))
            agent_config["intra_hidden_dim"] = method_cfg.get("intra_hidden_dim", 128)
            self.agent = PPOIntraChain(agent_config)
        elif mode == "cumulative":
            agent_config["clip_eps_scale"] = method_cfg.get("clip_eps_scale", 1.0)
            agent_config["eps_scaling"] = method_cfg.get("eps_scaling", "sqrt")
            self.agent = PPOCumulative(agent_config)
        elif mode == "hierarchical_cumulative":
            agent_config["clip_eps_scale"] = method_cfg.get("clip_eps_scale", 1.0)
            agent_config["correction_hidden_dim"] = method_cfg.get("correction_hidden_dim", 128)
            agent_config["correction_lr"] = method_cfg.get("correction_lr", ppo_cfg.get("lr", 3e-4))
            agent_config["delta_reg"] = method_cfg.get("delta_reg", 0.01)
            agent_config["eps_scaling"] = method_cfg.get("eps_scaling", "sqrt")
            self.agent = PPOHierarchicalCumulative(agent_config)
        elif mode in (
            "per_step_uniform",
            "per_step_learned_global",
            "per_step_state_dependent",
            "per_step_kl_inverse",
            "per_step_denoising_discount",
        ):
            weighting_map = {
                "per_step_uniform": "uniform",
                "per_step_learned_global": "learned_global",
                "per_step_state_dependent": "state_dependent",
                "per_step_kl_inverse": "kl_inverse",
                "per_step_denoising_discount": "denoising_discount",
            }
            agent_config["weighting_mode"] = weighting_map[mode]
            agent_config["weight_lr"] = ppo_cfg.get("weight_lr", ppo_cfg.get("lr", 3e-4))
            agent_config["weight_ent_coef"] = method_cfg.get("weight_ent_coef", 0.01)
            agent_config["clip_eps_scale"] = method_cfg.get("clip_eps_scale", 1.0)
            agent_config["asymmetric_clip"] = method_cfg.get("asymmetric_clip", False)
            if mode == "per_step_kl_inverse":
                agent_config["kl_beta"] = method_cfg.get("kl_beta", 5.0)
                agent_config["learn_kl_beta"] = method_cfg.get("learn_kl_beta", False)
            if mode == "per_step_denoising_discount":
                agent_config["gamma_denoise"] = method_cfg.get("gamma_denoise", 0.95)
            self.agent = PPOPerStep(agent_config)
        elif mode == "fisher_ppo":
            agent_config["clip_eps_scale"] = method_cfg.get("clip_eps_scale", 1.0)
            self.agent = PPOFisher(agent_config)
        elif mode == "step_conditioned":
            agent_config["clip_eps_scale"] = method_cfg.get("clip_eps_scale", 1.0)
            agent_config["intra_vf_coef"] = method_cfg.get("intra_vf_coef", 0.5)
            agent_config["intra_vf_lr"] = method_cfg.get("intra_vf_lr", ppo_cfg.get("lr", 3e-4))
            agent_config["intra_hidden_dim"] = method_cfg.get("intra_hidden_dim", 128)
            self.agent = PPOStepConditioned(agent_config)
        elif mode == "awfm":
            agent_config["beta"] = method_cfg.get("beta", 1.0)
            agent_config["top_frac"] = method_cfg.get("top_frac", 0.5)
            self.agent = PPOAWFM(agent_config)
        else:
            raise ValueError(
                f"Unknown method mode: {mode}. Expected one of: "
                "holistic, cumulative, hierarchical, hierarchical_cumulative, "
                "fisher_ppo, awfm, per_step_uniform, per_step_learned_global, "
                "per_step_state_dependent, per_step_kl_inverse, step_conditioned"
            )

        # Logging
        log_cfg = config.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "runs/default")
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = Logger(
            log_dir=self.log_dir,
            use_tensorboard=log_cfg.get("use_tensorboard", False),
        )
        self.save_freq = log_cfg.get("save_freq", 50000)

        # Eval config
        eval_cfg = config.get("eval", {})
        self.eval_freq = eval_cfg.get("eval_freq", 10000)
        self.num_eval_episodes = eval_cfg.get("num_eval_episodes", 10)

        # Training state
        self.total_timesteps = config.get("total_timesteps", 1_000_000)
        self.num_rollout_steps = ppo_cfg.get("num_rollout_steps", 2048)

    def run(self) -> None:
        """Execute the full training loop with periodic evaluation and checkpointing."""
        num_iterations = self.total_timesteps // self.num_rollout_steps

        # Reset agent env tracking state
        self.agent._last_obs = None
        self.agent._current_ep_reward = 0.0
        self.agent._current_ep_length = 0

        best_eval_return = -float("inf")

        for iteration in range(1, num_iterations + 1):
            # Linear LR annealing (reaches ~0 at end, never exactly 0 before last update)
            progress = (iteration - 1) / num_iterations
            self.agent.step_lr(progress)

            # Delta-reg annealing for hierarchical method
            if hasattr(self.agent, "step_delta_reg"):
                self.agent.step_delta_reg(progress)

            # Collect rollouts
            rollout_stats = self.agent.collect_rollouts(
                self.train_env, self.num_rollout_steps
            )

            # Update policy
            update_stats = self.agent.update()

            timestep = self.agent.total_steps

            # Log training metrics
            self.logger.log("train/mean_reward", rollout_stats["mean_reward"], timestep)
            self.logger.log("train/num_episodes", rollout_stats["num_episodes"], timestep)
            self.logger.log("train/mean_episode_length", rollout_stats["mean_episode_length"], timestep)
            self.logger.log("train/policy_loss", update_stats["policy_loss"], timestep)
            self.logger.log("train/value_loss", update_stats["value_loss"], timestep)
            self.logger.log("train/entropy", update_stats["entropy"], timestep)
            self.logger.log("train/clip_fraction", update_stats["clip_fraction"], timestep)
            self.logger.log("train/approx_kl", update_stats["approx_kl"], timestep)
            self.logger.log("train/iteration", iteration, timestep)
            self.logger.log("train/lr", self.agent.optimizer.param_groups[0]["lr"], timestep)
            if "intra_vf_loss" in update_stats:
                self.logger.log("train/intra_vf_loss", update_stats["intra_vf_loss"], timestep)
            if "delta_magnitude" in update_stats:
                self.logger.log("train/delta_magnitude", update_stats["delta_magnitude"], timestep)
                self.logger.log("train/delta_ratio", update_stats["delta_ratio"], timestep)
            if "effective_delta_reg" in update_stats:
                self.logger.log("train/effective_delta_reg", update_stats["effective_delta_reg"], timestep)
            if "mean_weight" in update_stats:
                self.logger.log("train/awfm_mean_weight", update_stats["mean_weight"], timestep)
                self.logger.log("train/awfm_max_weight", update_stats["max_weight"], timestep)

            # Log per-step clip fractions for per-step methods
            if "per_step_clip_fractions" in update_stats:
                pscf = update_stats["per_step_clip_fractions"]
                if isinstance(pscf, list):
                    for k_idx, cf in enumerate(pscf):
                        self.logger.log(f"train/clip_frac_step_{k_idx}", cf, timestep)

            # Log Fisher-scaled per-step clip epsilons
            if "per_step_eps" in update_stats:
                pse = update_stats["per_step_eps"]
                if isinstance(pse, list):
                    for k_idx, eps_val in enumerate(pse):
                        self.logger.log(f"train/fisher_eps_step_{k_idx}", eps_val, timestep)

            # Log learned weights for learned_global mode
            if self.mode == "per_step_learned_global" and hasattr(self.agent, "weight_net"):
                with torch.no_grad():
                    weights = torch.softmax(self.agent.weight_net.alpha, dim=0)
                    for k_idx in range(weights.shape[0]):
                        self.logger.log(
                            f"train/weight_step_{k_idx}",
                            weights[k_idx].item(),
                            timestep,
                        )

            self.logger.dump(timestep)

            # Print progress
            print(
                f"[Iter {iteration}/{num_iterations}] "
                f"steps={timestep} "
                f"reward={rollout_stats['mean_reward']:.2f} "
                f"pi_loss={update_stats['policy_loss']:.4f} "
                f"vf_loss={update_stats['value_loss']:.4f} "
                f"entropy={update_stats['entropy']:.4f} "
                f"kl={update_stats['approx_kl']:.4f}"
            )

            # Periodic evaluation
            if timestep % self.eval_freq < self.num_rollout_steps:
                eval_results = evaluate_policy(
                    env=self.eval_env,
                    policy=self.agent.policy,
                    num_episodes=self.num_eval_episodes,
                    deterministic=True,
                    device=self.device,
                )
                self.logger.log("eval/mean_return", eval_results["mean_return"], timestep)
                self.logger.log("eval/std_return", eval_results["std_return"], timestep)
                self.logger.dump(timestep)

                print(
                    f"  [Eval] mean_return={eval_results['mean_return']:.2f} "
                    f"std_return={eval_results['std_return']:.2f}"
                )

                # Save best model
                if eval_results["mean_return"] > best_eval_return:
                    best_eval_return = eval_results["mean_return"]
                    self._save_checkpoint("best")

            # Periodic checkpoint
            if timestep % self.save_freq < self.num_rollout_steps:
                self._save_checkpoint(f"step_{timestep}")

        # Final checkpoint
        self._save_checkpoint("final")

        # Cleanup
        self.logger.close()
        self.train_env.close()
        self.eval_env.close()

        print(f"Training complete. Logs saved to {self.log_dir}")

    def _save_checkpoint(self, tag: str) -> None:
        """Save a model checkpoint.

        Args:
            tag: Identifier for the checkpoint (e.g. "best", "step_100000", "final").
        """
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{tag}.pt")

        state = {
            "policy": self.agent.policy.state_dict(),
            "value_fn": self.agent.value_fn.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "total_steps": self.agent.total_steps,
            "config": self.config,
        }

        # Save weight network state for per-step methods
        if hasattr(self.agent, "weight_net"):
            state["weight_net"] = self.agent.weight_net.state_dict()

        # Save intra-chain value function for intra_chain method
        if hasattr(self.agent, "intra_vf"):
            state["intra_vf"] = self.agent.intra_vf.state_dict()

        # Save correction network for hierarchical method
        if hasattr(self.agent, "correction_net"):
            state["correction_net"] = self.agent.correction_net.state_dict()

        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        state = torch.load(path, map_location=self.device)
        self.agent.policy.load_state_dict(state["policy"])
        self.agent.value_fn.load_state_dict(state["value_fn"])
        self.agent.optimizer.load_state_dict(state["optimizer"])
        self.agent.total_steps = state["total_steps"]

        if "weight_net" in state and hasattr(self.agent, "weight_net"):
            self.agent.weight_net.load_state_dict(state["weight_net"])

        if "intra_vf" in state and hasattr(self.agent, "intra_vf"):
            self.agent.intra_vf.load_state_dict(state["intra_vf"])

        if "correction_net" in state and hasattr(self.agent, "correction_net"):
            self.agent.correction_net.load_state_dict(state["correction_net"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_value(s: str) -> Any:
    """Convert a CLI string to int, float, bool, or leave as str."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation (e.g. 'method.mode')."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train stochastic flow policy with PPO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Parse known args; everything else is treated as dotted overrides
    known, unknown = parser.parse_known_args()

    # Load YAML config
    with open(known.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides (e.g. --method.mode holistic --seed 42 --policy.K 8)
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg[2:]  # strip leading --
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = _parse_value(unknown[i + 1])
                i += 2
            else:
                # Flag with no value -> treat as True
                val = True
                i += 1
            _set_nested(config, key, val)
        else:
            i += 1

    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
