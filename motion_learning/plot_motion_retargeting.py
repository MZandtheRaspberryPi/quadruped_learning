import os

import matplotlib.pyplot as plt
import numpy as np

from ref_motion_utils import load_ref_motions


def main():
    file_dir = "/home/mz/quadruped_learning/"
    file_name = "pace.txt"
    motion_files = [file_name]
    num_motors = 8
    num_end_effectors = 4
    pos_size = 3
    orientation_size = 4

    for i in range(len(motion_files)):
        file_name = motion_files[i]
        file_path = os.path.join(file_dir, file_name)
        frames = load_ref_motions([file_path])[0]

        fig, axes = plt.subplots(
            nrows=num_motors, ncols=1, sharex=True, sharey=True)
        fig.suptitle(f"Joint Angles in {file_name}")

        time_steps = np.array(range(frames.shape[0]))
        for i in range(num_motors):
            motor_col_idx = pos_size + orientation_size + i
            axes[i].plot(time_steps, frames[:, motor_col_idx])
            axes[i].set_ylabel(f"{i}")
        fig.text(0.5, 0.04, 'timestep', ha='center')
        fig.text(0.01, 0.5, 'angle (rads)', va='center', rotation='vertical')
        plt.show()


if __name__ == "__main__":
    main()
