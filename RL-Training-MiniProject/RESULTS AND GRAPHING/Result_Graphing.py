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

def get_data(tag):
    if tag in tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    else:
        print(f"Warning: Tag '{tag}' not found.")
        return [], []

file_path = PRB1_PATH

# 1. Load the data
print(f"Loading {file_path}...")
event_acc = EventAccumulator(file_path)
event_acc.Reload()

# 2. Extract available tags
tags = event_acc.Tags()['scalars']
print(tags)

# 3. Define Tags
reward_tag = 'Train/mean_reward'
length_tag = 'Train/mean_episode_length'
lin_track_tag = 'Episode_Reward/track_linear_velocity'
ang_track_tag = 'Episode_Reward/track_angular_velocity'
lin_err_tag = 'Metrics/twist/error_vel_xy'
ang_err_tag = 'Metrics/twist/error_vel_yaw'
slip_metric_tag = 'Metrics/slip_velocity_mean'

# 4. Extract Data Explicitly
# Mean Reward
r_events = event_acc.Scalars(reward_tag)
r_steps = [e.step for e in r_events]
r_values = [e.value for e in r_events]

# Episode Length
l_events = event_acc.Scalars(length_tag)
l_steps = [e.step for e in l_events]
l_values = [e.value for e in l_events]

# Linear Tracking Reward
lt_events = event_acc.Scalars(lin_track_tag)
lt_steps = [e.step for e in lt_events]
lt_values = [e.value for e in lt_events]

# Angular Tracking Reward
at_events = event_acc.Scalars(ang_track_tag)
at_steps = [e.step for e in at_events]
at_values = [e.value for e in at_events]

# Linear Velocity Error
le_events = event_acc.Scalars(lin_err_tag)
le_steps = [e.step for e in le_events]
le_values = [e.value for e in le_events]

# Angular Velocity Error
ae_events = event_acc.Scalars(ang_err_tag)
ae_steps = [e.step for e in ae_events]
ae_values = [e.value for e in ae_events]

# Slip Stuff
slip_events = event_acc.Scalars(slip_metric_tag)
slip_steps1 = [e.step for e in slip_events]
slip_values1 = [e.value for e in slip_events]


# --- FIGURE 1: Reward + Length ---
plt.figure(figsize=(10, 8))

# Subplot 1: Mean Reward
plt.subplot(2, 1, 1)
plt.plot(r_steps, r_values, color='#1f77b4', linewidth=2)
plt.title(f'Mean Reward')
plt.ylabel('Reward')
plt.grid(True, alpha=0.3)
# Hide x-labels for the top plot to avoid clutter
plt.gca().axes.xaxis.set_ticklabels([])

# Subplot 2: Episode Length
plt.subplot(2, 1, 2)
plt.plot(l_steps, l_values, color='#ff7f0e', linewidth=2)
plt.title(f'Episode Length')
plt.ylabel('Length')
plt.xlabel('Steps')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# --- FIGURE 2: Tracking Rewards + Metrics ---
plt.figure(figsize=(10, 8))

# Subplot 1: Tracking Rewards (Linear & Angular)
plt.subplot(2, 1, 1)
plt.plot(lt_steps, lt_values, label='Linear Velocity Reward', linewidth=2)
plt.plot(at_steps, at_values, label='Angular Velocity Reward', linewidth=2)
plt.title('Tracking Rewards')
plt.ylabel('Reward')
plt.legend()
plt.grid(True, alpha=0.3)
# Hide x-labels for the top plot
plt.gca().axes.xaxis.set_ticklabels([])

# Subplot 2: Velocity Errors (Linear & Angular)
plt.subplot(2, 1, 2)
plt.plot(le_steps, le_values, label='Linear Velocity Error', linewidth=2)
plt.plot(ae_steps, ae_values, label='Angular Velocity Error', linewidth=2)
plt.title('Velocity Tracking Errors')
plt.ylabel('Error')
plt.xlabel('Steps')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

#----------------------------------------------------------------------------------------------
# PROBLEM 2
file_path = PRB2_PATH

# 1. Load the data
event_acc = EventAccumulator(file_path)
event_acc.Reload()

# 2. Extract available tags

tags = event_acc.Tags()['scalars']
slip_events = event_acc.Scalars(slip_metric_tag)
slip_steps2 = [e.step for e in slip_events]
slip_values2 = [e.value for e in slip_events]

le_events = event_acc.Scalars(lin_err_tag)
le_steps2 = [e.step for e in le_events]
le_values2 = [e.value for e in le_events]

plt.figure(figsize=(10, 8))

# Subplot 1: Foot Slip Metric
plt.subplot(2, 1, 1)
plt.plot(slip_steps1, slip_values1, label='Without Critic Observations', linewidth=2)
plt.plot(slip_steps2, slip_values2, label='With Critic Observations', linewidth=2)
plt.title('Mean Slip Velocity')
plt.ylabel('Slip Velocity (m/s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().axes.xaxis.set_ticklabels([])

# Subplot 2: Tracking Performance (Re-plotted for context)
plt.subplot(2, 1, 2)
plt.plot(le_steps, le_values, label='Without Critic Observations', linewidth=2)
plt.plot(le_steps2, le_values2, label='With Critic Observations', linewidth=2)
plt.title('Linear Velocity Tracking Error')
plt.ylabel('Error')
plt.xlabel('Steps')
plt.legend()
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()