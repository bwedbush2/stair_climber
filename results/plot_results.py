import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np

## Pathing
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

### Searching for files
search_path = os.path.join(script_dir, "*.txt")
files = glob.glob(search_path)

if not files:
    print("Error: No .txt files were found.")
else:
    files.sort(key=os.path.getmtime, reverse=True)

    print("\nAvailable Simulation Results:")
    print("----------------------------")
    for i, file_path in enumerate(files):
        print(f"[{i}] {os.path.basename(file_path)}")

    user_input = input("\nSelect a file number: ").strip()
    choice = int(user_input) if user_input else 0

    if choice < 0 or choice >= len(files):
        print(f"Error: {choice} is not a valid selection.")
    else:
        selected_file = files[choice]
        fname = os.path.basename(selected_file)
        print(f"Loading: {fname}...")

        data = pd.read_csv(selected_file, skipinitialspace=True)
        time = data['Time']

        # NEW: Yaw series (optionally unwrap for cleaner plots across +/- pi)
        yaw = data['Yaw'].to_numpy()
        yaw_unwrapped = np.unwrap(yaw)

        ### Figure 1: XY path (optionally draw heading arrows sparsely)
        plt.figure(figsize=(8, 8))
        plt.plot(data['X'], data['Y'], color='black', linewidth=2, label='Robot Path')
        plt.scatter(data['X'].iloc[0], data['Y'].iloc[0], color='green', label='Start', zorder=5)
        plt.scatter(data['X'].iloc[-1], data['Y'].iloc[-1], color='red', label='End', zorder=5)

        # NEW: show yaw direction along the path with arrows (subsample to avoid clutter)
        step = max(1, len(data) // 25)  # ~25 arrows max
        xs = data['X'].to_numpy()[::step]
        ys = data['Y'].to_numpy()[::step]
        yaws = yaw[::step]
        arrow_len = 0.15  # meters
        plt.quiver(
            xs, ys,
            np.cos(yaws), np.sin(yaws),
            angles='xy', scale_units='xy', scale=1/arrow_len,
            width=0.01, alpha=0.7
        )

        plt.title(f"Spatial Trajectory\n{fname}")
        plt.xlabel("X Position (meters)")
        plt.ylabel("Y Position (meters)")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        ### Figure 2: Time histories (added a yaw subplot)
        fig, axs = plt.subplots(6, 1, figsize=(10, 19), sharex=True)
        plt.subplots_adjust(hspace=0.4)

        axs[0].plot(time, data['X'], color='blue', label='X Position')
        axs[0].plot(time, data['Y'], color='red', label='Y Position')
        axs[0].set_ylabel('Position (m)')
        axs[0].set_title('Global Coordinates Over Time')
        axs[0].legend(loc='upper right')

        # NEW: yaw subplot
        axs[1].plot(time, yaw, label='Yaw (raw)')
        axs[1].plot(time, yaw_unwrapped, linestyle='--', label='Yaw (unwrapped)')
        axs[1].set_ylabel('Rad')
        axs[1].set_title('Robot Yaw Over Time')
        axs[1].legend(loc='upper right')

        axs[2].plot(time, data['Drive_Fwd'], color='green')
        axs[2].set_ylabel('Command')
        axs[2].set_title('Forward Throttle Actuation')

        axs[3].plot(time, data['Drive_Turn'], color='orange')
        axs[3].set_ylabel('Command')
        axs[3].set_title('Steering Actuation')

        axs[4].plot(time, data['Climb'], color='purple')
        axs[4].set_ylabel('Value')
        axs[4].set_title('Bogie Actuation')

        axs[5].plot(time, data['Level_Bin'], color='brown')
        axs[5].set_ylabel('Rad')
        axs[5].set_xlabel('Time (seconds)')
        axs[5].set_title('Actuator: Bin Leveling')

        for ax in axs:
            ax.grid(True, alpha=0.3)

        print("Processing complete")
        plt.show()
