import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import minari
import polars as pl
from datetime import timedelta

from ship_env import ShipEnvironment, read_position_data, chunk_to_traj
import utils
import det_bc   # <- your deterministic BC module


# ------------------------------------------------------------
# 1. LOAD DATA & BUILD ENVIRONMENT (same logic as IL-BC)
# ------------------------------------------------------------
def build_environment():
    print("Loading AIS data and creating environment...")
    plan_start_time = 0
    plan_end_time = 24
    region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
    region_array = np.array([list(region_of_interest[k]) for k in ["LON", "LAT"]]).T

    data_folder = "./raw_data/"
    position_file = data_folder + "synthetic_ais_data.csv"
    df = read_position_data(position_file)

    geo_filters = [
        (v[0] <= pl.col(k)) & (pl.col(k) <= v[1])
        for k, v in region_of_interest.items()
    ]
    heading_filter = (pl.col("HEADING").is_not_null()) & (pl.col("HEADING") != 511)
    status_filter = (pl.col("STATUS") == 0)
    time_filter = (
        (pl.col("TIMESTAMP_UTC").dt.hour() >= plan_start_time) &
        (pl.col("TIMESTAMP_UTC").dt.hour() < plan_end_time)
    )

    df = df.filter(*(geo_filters + [time_filter, heading_filter, status_filter]))
    print("Filtered AIS data:", df)

    # Build trajectories
    threshold = timedelta(minutes=30)
    inter_pol = timedelta(seconds=10)
    num_samples = 5

    from split_chunks import seperate_chunks
    chunks = seperate_chunks(df, threshold, num_samples)
    chunks = chunks.sort("start_time")

    ship_trajs = []
    ship_times = []
    tw_list = []
    region_arr = region_array

    for j in range(len(chunks)):
        traj, time = chunk_to_traj(chunks[j], inter_pol, region_arr)
        ship_trajs.append(traj)
        ship_times.append(time)
        tw_list.append((time[0], time[-1]))

    overlap_idx = utils.find_overlapping_intervals(tw_list)
    max_ships = len(ship_trajs)

    # Create custom maritime environment
    env = ShipEnvironment(
        ship_trajs,
        ship_times,
        overlap_idx,
        region_of_interest,
        n_neighbor_agents=10
    )
    return env, len(ship_trajs)


# ------------------------------------------------------------
# 2. LOAD POLICY CHECKPOINT
# ------------------------------------------------------------
def load_policy():
    ckpt_path = "./ckpoints/BC-deterministic-256-128hid-256batch-combMLPTanh-Maritime-Expert-v1.th"
    print(f"Loading model from: {ckpt_path}")

    policy = det_bc.reconstruct_policy(ckpt_path)
    policy.eval()
    return policy


# ------------------------------------------------------------
# 3. RUN AGENT + RECORD VIDEO
# ------------------------------------------------------------
def run_agent():
    env, num_ships = build_environment()
    model = load_policy()

    # Wrap env with vector wrapper for stable baselines compatibility
    vec_env = DummyVecEnv([lambda: env])

    # Enable video recorder
    video_env = VecVideoRecorder(
        vec_env,
        video_folder="./videos/",
        record_video_trigger=lambda step: step == 0,
        video_length=2000
    )

    ego_id = 0  # which ship to imitate
    obs = video_env.reset()

    print("Running agent for 1 episode...")

    done = False
    truncated = False

    while not (done or truncated):
        # Policy expects dict observations
        with th.no_grad():
            action = model(obs)

        obs, reward, done, truncated, info = video_env.step(action)

    video_env.close()
    print("Video saved to ./videos/")



if __name__ == "__main__":
    run_agent()
