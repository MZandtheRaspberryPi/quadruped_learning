import json
from typing import List

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


def load_motion_file(filename: str):
    with open(filename, "r") as f:
        motion_json = json.load(f)

        loop_mode = motion_json[LOOP_MODE_KEY]
        frame_duration = float(motion_json[FRAME_DURATION_KEY])

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
