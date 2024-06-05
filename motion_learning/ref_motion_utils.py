import json
from typing import List, Union

from numpy.typing import NDArray
import numpy as np

from util import quat_norm, standardize_quat

POS_SIZE = 3
ROT_SIZE = 4

LOOP_MODE_KEY = "LoopMode"
FRAME_DURATION_KEY = "FrameDuration"
FRAMES_KEY = "Frames"
ENABLE_CYCLE_OFFSET_POSITION_KEY = "EnableCycleOffsetPosition"
ENABLE_CYCLE_OFFSET_ROTATION_KEY = "EnableCycleOffsetRotation"


def post_process_motion_frames(frames: NDArray):
    num_frames = frames.shape[0]
    if num_frames > 0:
        first_frame = frames[0]
        pos_start = first_frame[:POS_SIZE].copy()

        for i in range(num_frames):
            curr_frame = frames[i]

            root_pos = curr_frame[:POS_SIZE].copy()
            root_pos[0] -= pos_start[0]
            root_pos[1] -= pos_start[1]

            root_rot = curr_frame[POS_SIZE:POS_SIZE + ROT_SIZE].copy()
            root_rot = quat_norm(root_rot)
            root_rot = standardize_quat(root_rot)
            frames[i][:POS_SIZE] = root_pos
            frames[i][POS_SIZE:POS_SIZE + ROT_SIZE] = root_rot
    return frames

def expand_frames(frames, simsteps_per_frame: int):
    if simsteps_per_frame < 2 or not isinstance(simsteps_per_frame, int):
        raise ValueError
    num_original_frames = frames.shape[0]
    entries_per_frame = frames.shape[1]
    num_new_frames = num_original_frames * simsteps_per_frame
    new_frames = np.zeros((num_new_frames, entries_per_frame))
    for frame_idx in range(1, num_original_frames + 1):
        prev_frame = frames[frame_idx - 1]
        prev_rotation = prev_frame[POS_SIZE:POS_SIZE+ROT_SIZE]
        prev_position = prev_frame[:POS_SIZE]
        prev_angles = prev_frame[POS_SIZE + ROT_SIZE:]
        if frame_idx == num_original_frames:
            cur_frame = frames[0]
        else:
            cur_frame = frames[frame_idx]
        cur_rotation = cur_frame[POS_SIZE:POS_SIZE+ROT_SIZE]
        cur_position = cur_frame[:POS_SIZE]
        cur_angles = cur_frame[POS_SIZE + ROT_SIZE:]

        angles = np.linspace(prev_angles, cur_angles, simsteps_per_frame)
        positions = np.linspace(prev_position, cur_position, simsteps_per_frame)
        rotations = np.linspace(prev_rotation, cur_rotation, simsteps_per_frame)
        starting_frame_idx = (frame_idx - 1 )* simsteps_per_frame
        for i in range(rotations.shape[0]):
            rotations[i] = quat_norm(rotations[i])
            new_idx = starting_frame_idx + i
            new_frames[new_idx] = np.hstack([positions[i], rotations[i], angles[i]])
    # the last frame will match first frame, so drop it to close out the cycle
    return new_frames[:-1]



def load_motion_file(filename: str, sim_timestep: float = 0.005,
                     frame_duration_override: Union[float, None] = None):
    frames, frame_duration = None, None
    with open(filename, "r") as f:
        motion_json = json.load(f)

        loop_mode = motion_json[LOOP_MODE_KEY]
        frame_duration = float(motion_json[FRAME_DURATION_KEY])
        if frame_duration_override is not None:
            frame_duration = frame_duration_override

        if ENABLE_CYCLE_OFFSET_POSITION_KEY in motion_json:
            enable_cycle_offset_pos = bool(
                motion_json[ENABLE_CYCLE_OFFSET_POSITION_KEY])
        else:
            enable_cycle_offset_pos = False

        if ENABLE_CYCLE_OFFSET_ROTATION_KEY in motion_json:
            enable_cycle_offset_rot = bool(
                motion_json[ENABLE_CYCLE_OFFSET_ROTATION_KEY])
        else:
            enable_cycle_offset_rot = False

        frames = np.array(motion_json[FRAMES_KEY])
        frames = post_process_motion_frames(frames)

        # frame_vels = calc_frame_vels(frames)

        assert (frames.shape[0] > 0), "Must have at least 1 frame."
        assert (frames.shape[1] > POS_SIZE +
                ROT_SIZE), "Frames have too few degrees of freedom."
        # assert (frame_duration > 0), "Frame duration must be positive."
    num_frames = frames.shape[0]
    motion_total_time = num_frames * frame_duration
    num_timesteps_in_traj = motion_total_time / sim_timestep
    simsteps_per_frame = num_timesteps_in_traj // num_frames
    # return expand_frames(frames, int(simsteps_per_frame))
    return frames


def load_ref_motions(filenames: List[str]):
    num_files = len(filenames)
    if num_files == 0:
        raise ValueError("No reference motions specified.")

    motions = []
    for filename in filenames:
        frames = load_motion_file(filename)
        motions.append(frames)

    return motions

if __name__ == "__main__":
    expert_trajectory = load_motion_file("/home/mz/quadruped_learning/data_retargetted_motion/canter.txt", sim_timestep=0.005, frame_duration_override=0.04)
    print(expert_trajectory.shape)