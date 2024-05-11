import time

import sys
sys.path.append("/home/mz/quadruped_learning")

import numpy as np
import pybullet
import pybullet_data as pd
from pybullet_utils import transformations

from sim_config_bittle import SIM_MOTOR_IDS

INIT_POS = np.array([0, 0, 0.5])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0, aj=0, ak=0, axes="sxyz")

# DEFAULT_JOINT_POSE = np.array(
#     [0.7853145, -0.7853145, 0.7853145, -0.7853145, 0.7853145, -0.7853145, 0.7853145, -0.7853145, 0, 0, 0, 0])
DEFAULT_JOINT_POSE = np.array(
    [0.7853145, -0.7853145, 0.7853145, -0.7853145, 0.7853145, -0.7853145, 0.7853145, -0.7853145])

POS_SIZE = 3
ROT_SIZE = 4
FRAME_DURATION = 0.01667
FRAME_DURATION = 0.1


GROUND_URDF_FILENAME = "plane_implicit.urdf"
BITTLE_URDF_AND_PATH = "bittle/bittle_petoi.urdf"

MOCAP_DATA = [
    ["pace", "data/dog_walk00_joint_pos.txt", 162, 201],
    ["trot", "data/dog_walk03_joint_pos.txt", 448, 481],
    ["trot2", "data/dog_run04_joint_pos.txt", 630, 663],
    ["canter", "data/dog_run00_joint_pos.txt", 430, 459],
    ["left turn0", "data/dog_walk09_joint_pos.txt", 1085, 1124],
    ["right turn0", "data/dog_walk09_joint_pos.txt", 2404, 2450],
]

def get_joint_names(robot):
    num_joints = pybullet.getNumJoints(robot)
    id_names = [(j, pybullet.getJointInfo(robot, j)[1].decode()) for j in range(num_joints)]
    return id_names

def load_reference_data(reference_file: str, frame_start: int = None, frame_end: int = None):
    joint_pos_data = np.loadtxt(reference_file, delimiter=",")
    start_frame = 0 if (frame_start is None) else frame_start
    end_frame = joint_pos_data.shape[0] if (frame_end is None) else frame_end
    joint_pos_data = joint_pos_data[start_frame:end_frame]
    return joint_pos_data


def update_camera(robot):
    base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
    [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
    pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
    return


def get_root_pos(pose):
    return pose[0:POS_SIZE]


def get_root_rot(pose):
    return pose[POS_SIZE:(POS_SIZE + ROT_SIZE)]


def set_pose(robot, pose):
    num_joints = pybullet.getNumJoints(robot)
    root_pos = get_root_pos(pose)
    root_rot = get_root_rot(pose)
    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

    for j in range(num_joints):
        j_info = pybullet.getJointInfo(robot, j)
        j_state = pybullet.getJointStateMultiDof(robot, j)

        j_pose_idx = j_info[3]
        j_pose_size = len(j_state[0])
        j_vel_size = len(j_state[1])

        if (j_pose_size > 0):
            j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
            j_vel = np.zeros(j_vel_size)
            pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)

    return


def main():
    p = pybullet
    width, height = 1920, 1080
    width, height = 720, 480
    p.connect(
        p.GUI, options=f"--width={width} --height={height} --mp4=\"test.mp4\" --mp4fps=60")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    pybullet.setAdditionalSearchPath(pd.getDataPath())

    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -10)

    ground = pybullet.loadURDF(GROUND_URDF_FILENAME)
    robot = pybullet.loadURDF(
        BITTLE_URDF_AND_PATH, INIT_POS, INIT_ROT)

    # Set robot to default pose to bias knees in the right direction.
    # set_pose(robot, np.concatenate(
    #     [INIT_POS, INIT_ROT, DEFAULT_JOINT_POSE]))

    # pybullet.setJointMotorControlArray(robot,
    #                                    SIM_MOTOR_IDS,
    #                                    pybullet.POSITION_CONTROL,
    #                                    DEFAULT_JOINT_POSE)
    # for j_idx in range(len(SIM_MOTOR_IDS)):
    #     j_id = SIM_MOTOR_IDS[j_idx]
    #     j_angle = DEFAULT_JOINT_POSE[j_idx]
    #     j_vel = np.zeros(3)
    #     pybullet.resetJointStateMultiDof(robot, j_id, [j_angle], j_vel)
    pybullet.resetJointStateMultiDof(robot, SIM_MOTOR_IDS[3], [np.pi/4], np.zeros(3))
    pybullet.resetJointStateMultiDof(robot, SIM_MOTOR_IDS[2], [np.pi/4], np.zeros(3))

    joint_pose = pybullet.getJointInfo(robot, 68)
    joint_pose2 = pybullet.getJointInfo(robot, 63)

    # joint_pose = pybullet.calculateInverseKinematics2(robot, config.SIM_TOE_JOINT_IDS,
    #                                                 tar_toe_pos,
    #                                                 jointDamping=config.JOINT_DAMPING,
    #                                                 lowerLimits=joint_lim_low,
    #                                                 upperLimits=joint_lim_high,
    #                                                 restPoses=default_pose)


    for mocap_data in MOCAP_DATA:
        file_tag, file_path, start, end = mocap_data
        print(f"working on {file_tag}")
        reference_joint_poses = load_reference_data(file_path, start, end)
        print("hi")

        # p.removeAllUserDebugItems()
        # print("mocap_name=", mocap_motion[0])
        # joint_pos_data = load_ref_data(
        #     mocap_motion[1], mocap_motion[2], mocap_motion[3])

        # num_markers = joint_pos_data.shape[-1] // POS_SIZE
        # marker_ids = build_markers(num_markers)

        # retarget_frames = retarget_motion(robot, joint_pos_data)
        # output_motion(retarget_frames, f"{mocap_motion[0]}.txt")

        # f = 0
        # num_frames = joint_pos_data.shape[0]

        # for repeat in range(5*num_frames):
        # time_start = time.time()

        # f_idx = f % num_frames
        # print("Frame {:d}".format(f_idx))

        # ref_joint_pos = joint_pos_data[f_idx]
        # ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
        # ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

        # pose = retarget_frames[f_idx]

        # set_pose(robot, pose)
        # set_maker_pos(ref_joint_pos, marker_ids)

        for i in range(1000):
            if i % 50 == 0:
                print(i)
            time_start = time.time()
            update_camera(robot)
            p.configureDebugVisualizer(
                p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            p.stepSimulation()
            time_end = time.time()
            sleep_dur = FRAME_DURATION - (time_end - time_start)
            sleep_dur = max(0, sleep_dur)
            joint_state = p.getJointState(robot, 0)
            if i > 100:
                p.setJointMotorControlArray(
                    bodyIndex=robot,
                    jointIndices=[0],
                    controlMode=p.TORQUE_CONTROL,
                    forces=[0.2])

            time.sleep(sleep_dur)

        # f += 1

        # time_end = time.time()
        # sleep_dur = FRAME_DURATION - (time_end - time_start)
        # sleep_dur = max(0, sleep_dur)

        # time.sleep(sleep_dur)
        # # time.sleep(0.5) # jp hack
        # for m in marker_ids:
        # p.removeBody(m)
        # marker_ids = []

        pybullet.disconnect()

        return


if __name__ == "__main__":
    main()
