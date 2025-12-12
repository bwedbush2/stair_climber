import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

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


        ### FIgure 1
        plt.figure(figsize=(8, 8))
        plt.plot(data['X'], data['Y'], color='black', linewidth=2, label='Robot Path')
        plt.scatter(data['X'].iloc[0], data['Y'].iloc[0], color='green', label='Start', zorder=5)
        plt.scatter(data['X'].iloc[-1], data['Y'].iloc[-1], color='red', label='End', zorder=5)
        
        plt.title(f"Spatial Trajectory\n{fname}")
        plt.xlabel("X Position (meters)")
        plt.ylabel("Y Position (meters)")
        plt.axis('equal') 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        ### Figure 2
        fig, axs = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
        plt.subplots_adjust(hspace=0.4)

        axs[0].plot(time, data['X'], color='blue', label='X Position')
        axs[0].plot(time, data['Y'], color='red', label='Y Position')
        axs[0].set_ylabel('Position (m)')
        axs[0].set_title('Global Coordinates Over Time')
        axs[0].legend(loc='upper right')

        axs[1].plot(time, data['Drive_Fwd'], color='green')
        axs[1].set_ylabel('Command')
        axs[1].set_title('Forward Throttle Actuation')

        axs[2].plot(time, data['Drive_Turn'], color='orange')
        axs[2].set_ylabel('Command')
        axs[2].set_title('Steering Actuation')

        axs[3].plot(time, data['Climb'], color='purple')
        axs[3].set_ylabel('Value')
        axs[3].set_title('Bogie Actuation')

        axs[4].plot(time, data['Level_Bin'], color='brown')
        axs[4].set_ylabel('Rad')
        axs[4].set_xlabel('Time (seconds)')
        axs[4].set_title('Actuator: Bin Leveling')

        for ax in axs:
            ax.grid(True, alpha=0.3)
        
        print("Processing complete")
        plt.show()
