"""PPO configuration for the CTBT trajectory task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CTTrajectoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 4000
    save_interval = 250
    experiment_name = "crazyflie_ctbt_trajectory"
    logger = "tensorboard"
    empirical_normalization = False
    obs_groups = {
        "actor": ["current", "trajectory"],
        "critic": ["critic_current", "trajectory", "privileged"],
    }
    actor = RslRlCNNModelCfg(
        class_name="ctbt_drone.agents.trajectory_encoder.TrajectoryMLPModel",
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.25),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 32, 32],
            kernel_size=3,
            padding="zeros",
            global_pool="avg",
            activation="elu",
        ),
    )
    critic = RslRlCNNModelCfg(
        class_name="ctbt_drone.agents.trajectory_encoder.TrajectoryMLPModel",
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 32, 32],
            kernel_size=3,
            padding="zeros",
            global_pool="avg",
            activation="elu",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
