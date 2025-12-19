import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

script_dir = os.path.dirname(os.path.abspath(__file__))
problem1_folder = os.path.join(script_dir, "Problem 1")
problem2_folder = os.path.join(script_dir, "Problem 2")
problem3_folder = os.path.join(script_dir, "Problem 3")

PRB1_PATH = os.path.join(problem1_folder, "events.out.tfevents.1766105278.677a1d26499e.1464.0")
PRB2_PATH = os.path.join(problem2_folder, "events.out.tfevents.1766103354.d55c2424f1bd.13450.0")
PRB3_PATH = os.path.join(problem3_folder, "ray_simulation.xml")

# Select which file to plot
file_path = PRB1_PATH  # Change to PRB2_PATH if needed

# 1. Load the data
print(f"Loading {file_path}...")
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

event_acc = EventAccumulator(file_path)
event_acc.Reload()

# 2. Extract available tags
tags = event_acc.Tags()['scalars']
print("Found tags:", tags)

# 3. Select the correct tags for Reward and Length
reward_tag = 'Train/mean_reward'
length_tag = 'Train/mean_episode_length'

# Check if tags exist before plotting to avoid crashes
if reward_tag in tags and length_tag in tags:
    # Get reward data
    r_events = event_acc.Scalars(reward_tag)
    r_steps = [e.step for e in r_events]
    r_values = [e.value for e in r_events]

    # Get length data
    l_events = event_acc.Scalars(length_tag)
    l_steps = [e.step for e in l_events]
    l_values = [e.value for e in l_events]

    # --- MODIFIED PLOTTING SECTION ---
    # Create figure with 2 rows, 1 column
    # sharex=True ensures both plots share the same x-axis scale
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Mean Reward (Top)
    ax1.plot(r_steps, r_values, color='#1f77b4', linewidth=2)
    ax1.set_title(f'Mean Reward ({reward_tag})')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)

    # Plot Episode Length (Bottom)
    ax2.plot(l_steps, l_values, color='#ff7f0e', linewidth=2)
    ax2.set_title(f'Episode Length ({length_tag})')
    ax2.set_ylabel('Length')
    ax2.set_xlabel('Steps')  # X-label is only needed at the bottom
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print(f"Error: Could not find required tags '{reward_tag}' or '{length_tag}' in the file.")