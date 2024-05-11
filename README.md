# quadruped_learning

## Motion re-targeting
We made a simple model of the Bittle robot using the universal robot description format (URDF) that is supported by tools like PyBullet. 

For retargeting the root orientation and position, we simply calculate orientation using pelvis, neck, shoulder, and hip locations in the reference motion. We take position by using pelvis and neck position. For joint positions, we first calculate target toe positions. We do this by reversing initial rotation and calculating heading for the robot using initial rotation and current rotation. We then get forward kinematics for the hip and the state using pybullet. We calc the difference in reference hip vs toe position. We take the simulated hip position and adjust for this delta. then we set the z height of simulated tar toe pos to the reference to pos, and then add the toe offset in the word. That is our targeted toe pos. We then use pybullet to calculate inverse kinematics to match all the toe positions.

