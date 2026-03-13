# run_bc_one_agent.py
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import polars as pl
import torch as th
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Repo-local imports (use from repo root) ---
from ship_env import ShipEnvironment, read_position_data, chunk_to_traj
from split_chunks import seperate_chunks
import utils
import det_bc  # provides reconstruct_policy + evaluate helper in IL-BC.py
from det_bc import evaluate_policy_with_finalinfo  # used by IL-BC.py


def build_env_from_repo_data():
    """
    Rebuild trajectories and create ShipEnvironment using the same
    data-prep steps described in the repo docs and IL-BC.py.
    """
    # Region of interest used across examples in the repo
    region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
    region_array = np.array([list(region_of_interest[k]) for k in ["LON", "LAT"]]).T

    # Load AIS positions (semicolon-separated CSV)
    position_file = "./raw_data/synthetic_ais_data.csv"
    if not os.path.exists(position_file):
        raise FileNotFoundError(
            f"Missing AIS data at {position_file}. Please keep the repo structure intact."
        )

    df = read_position_data(position_file)

    # Typical filters (as used in IL-BC.py / README)
    geo_filters = [
        (v[0] <= pl.col(k)) & (pl.col(k) <= v[1])
        for k, v in region_of_interest.items()
    ]
    heading_filter = (pl.col("HEADING").is_not_null()) & (pl.col("HEADING") != 511)
    status_filter = (pl.col("STATUS") == 0)
    # The README tutorials often filter by hour; here we keep the full-day window
    time_filter = (
        (pl.col("TIMESTAMP_UTC").dt.hour() >= 0)
        & (pl.col("TIMESTAMP_UTC").dt.hour() < 24)
    )
    df = df.filter(*(geo_filters + [time_filter, heading_filter, status_filter]))

    # Split into continuous chunks and interpolate to 10-second grid
    threshold = timedelta(minutes=30)
    inter_pol = timedelta(seconds=10)
    chunks : pl.DataFrame = seperate_chunks(df, threshold, minimum_sample_count=5)

    trajs, times, tw = [], [], []
    for j in range(len(chunks)):
        traj, t = chunk_to_traj(chunks[j], inter_pol, region_array)
        trajs.append(traj)
        times.append(t)
        tw.append((t[0], t[-1]))

    # Compute overlap sets for neighbor selection
    overlap_idx = utils.find_overlapping_intervals(tw)

    # Create the environment (defaults match training: unscaled actions in meters & radians)
    env = ShipEnvironment(
        ship_trajectories=trajs,
        ship_times=times,
        overlap_idx=overlap_idx,
        region_of_interest=region_of_interest,
        n_neighbor_agents=10,      # same order as training/eval examples
        # normalize_xy=False, max_steps=1000, second_perts=10, ...
        # leave other args at their defaults used during training
    )
    return env, len(trajs)


def wrap_with_video(env, video_dir="./videos"):
    """
    Preferred: Gymnasium's RecordVideo wrapper (as in README).
    If your local Gym complains about render_mode, you can also switch
    to SB3 VecVideoRecorder (see commented fallback below).
    """
    os.makedirs(video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_dir,
        episode_trigger=lambda ep_id: True,  # record the first episode
    )
    return env

    # --- Fallback (if RecordVideo complains about render mode) ---
    # from stable_baselines3.common.vec_env import VecVideoRecorder
    # vec = DummyVecEnv([lambda: env])
    # vec = VecVideoRecorder(
    #     vec, video_dir, record_video_trigger=lambda step: step == 0, video_length=2000
    # )
    # return vec


def load_bc_checkpoint():
    """
    Load the deterministic BC policy produced by IL-BC.py training.
    """
    ckpt = "./ckpoints/BC-deterministic-256-128hid-256batch-combMLPTanh-Maritime-Expert-v1.th"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found at:\n  {ckpt}\n"
            "Train with IL-BC.py or copy the .th file into ./ckpoints/ first."
        )
    policy = det_bc.reconstruct_policy(ckpt)
    policy.eval()
    return policy


def run_one_episode(ego_id=0):
    # Build env and wrap with video
    base_env, num_ships = build_env_from_repo_data()
    video_env = wrap_with_video(base_env, "./videos")

    # Vectorize because the repo's evaluate helper expects a VecEnv
    vec_env = DummyVecEnv([lambda: video_env])

    # Load trained policy
    policy = load_bc_checkpoint()

    # Reset with a specific ship as ego
    obs = vec_env.reset()

    # Use the repo's evaluation routine (keeps parity with IL-BC.py)
    # This will run exactly 1 episode and collect final info
    _, _, infos = evaluate_policy_with_finalinfo(
        policy, vec_env, n_eval_episodes=1, return_episode_rewards=True, render=False
    )

    # Print final metrics (same post-processing style as IL-BC.py)
    final = infos[0].copy()
    final.pop("TimeLimit.truncated", None)
    final.pop("terminal_observation", None)
    print("\n=== Final episode metrics for ego ship", ego_id, "===")
    for k, v in final.items():
        print(f"{k}: {v}")

    # Close envs
    vec_env.close()


if __name__ == "__main__":
    th.manual_seed(42)
    run_one_episode(ego_id=0)
