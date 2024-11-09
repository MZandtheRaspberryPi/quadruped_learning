#include "bittle_config.hpp"

/*
Example Servo Specs as of July 2024
Model
P1S
Operating Voltage	6.0V ~ 8.4V
Idle Current	200mA
Stall Current	1500mA
Stall Torque	3kg*cm
Control Type	Digital PWM
Signal range:	500~2500µs
Dead Band:	≤2µs
Operating Travel	270°
Operating Speed	0.07 sec/60°
Sensor	Potentiometer
Size	30 x 24 x 12mm
Weight	14g
Ball-bearing	1 bearing
Gear Material	Metal
Motor	Coreless
Connector wire	7mm/17mm
Spline count	25

*/

std::map<BittleJoint, uint8_t> get_bittle_joint_to_servo_num_mapping()
{
    std::map<BittleJoint, uint8_t> mapping;
    mapping[BittleJoint::BACK_LEFT_KNEE] = 0;
    mapping[BittleJoint::BACK_LEFT_SHOULDER] = 1;
    mapping[BittleJoint::BACK_RIGHT_SHOULDER] = 2;
    mapping[BittleJoint::BACK_RIGHT_KNEE] = 3;
    mapping[BittleJoint::FRONT_RIGHT_KNEE] = 4;
    mapping[BittleJoint::FRONT_RIGHT_SHOULDER] = 5;
    mapping[BittleJoint::FRONT_LEFT_SHOULDER] = 6;
    mapping[BittleJoint::FRONT_LEFT_KNEE] = 7;
    mapping[BittleJoint::HEAD_JOINT] = 8;
    return mapping;
}
std::map<BittleJoint, uint8_t> get_bittle_joint_to_pwm_pin_mapping()
{
    std::map<BittleJoint, uint8_t> mapping;
    mapping[BittleJoint::BACK_LEFT_KNEE] = 0;
    mapping[BittleJoint::BACK_LEFT_SHOULDER] = 1;
    mapping[BittleJoint::BACK_RIGHT_SHOULDER] = 6;
    mapping[BittleJoint::BACK_RIGHT_KNEE] = 7;
    mapping[BittleJoint::FRONT_RIGHT_KNEE] = 8;
    mapping[BittleJoint::FRONT_RIGHT_SHOULDER] = 9;
    mapping[BittleJoint::FRONT_LEFT_SHOULDER] = 14;
    mapping[BittleJoint::FRONT_LEFT_KNEE] = 15;
    mapping[BittleJoint::HEAD_JOINT] = 12;
    return mapping;
}

std::map<BittleJoint, uint8_t> get_bittle_joint_to_bittle_array_idx_mapping()
{
    std::map<BittleJoint, uint8_t> mapping;
    mapping[BittleJoint::BACK_LEFT_KNEE] = 15;
    mapping[BittleJoint::BACK_LEFT_SHOULDER] = 11;
    mapping[BittleJoint::BACK_RIGHT_SHOULDER] = 10;
    mapping[BittleJoint::BACK_RIGHT_KNEE] = 14;
    mapping[BittleJoint::FRONT_RIGHT_KNEE] = 13;
    mapping[BittleJoint::FRONT_RIGHT_SHOULDER] = 9;
    mapping[BittleJoint::FRONT_LEFT_SHOULDER] = 8;
    mapping[BittleJoint::FRONT_LEFT_KNEE] = 12;
    mapping[BittleJoint::HEAD_JOINT] = 0;
    return mapping;
}


ServoBoardConfig make_bittle_config()
{

    ServoBoardConfig bittle_config(BittleJoint::LENGTH, -120, 120, 0, false, 270, 500, 2500,
        150, 600, 240, 25000000);

    std::map<BittleJoint, uint8_t> joint_pwm_map = get_bittle_joint_to_pwm_pin_mapping();
    std::map<BittleJoint, uint8_t> joint_servo_num_map = get_bittle_joint_to_servo_num_mapping();
    std::map<BittleJoint, uint8_t> joint_bittle_idx_map = get_bittle_joint_to_bittle_array_idx_mapping();


    for (uint8_t i = 0; i < BittleJoint::LENGTH; i++)
    {
        BittleJoint joint = static_cast<BittleJoint>(i);
        uint8_t servo_num = joint_servo_num_map[joint];
        uint8_t pwm_num = joint_pwm_map[joint];
        bittle_config.set_servo_pwm_pin_num(servo_num, pwm_num);
        // if (joint == BittleJoint::BACK_LEFT_KNEE || joint == BittleJoint::BACK_RIGHT_KNEE ||
        //              joint == BittleJoint::FRONT_RIGHT_KNEE ||
        //          joint == BittleJoint::FRONT_LEFT_KNEE)
        // {
        //     bittle_config.set_servo_angular_range(servo_num, 180);
        // }

        uint8_t bittle_arr_idx = joint_bittle_idx_map[joint];
        bool is_servo_inverted = rotationDirection[bittle_arr_idx] == -1;
        float32_t min_angle = angleLimit[bittle_arr_idx][0];
        float32_t max_angle = angleLimit[bittle_arr_idx][1];
        float32_t mid_pos = middleShift[bittle_arr_idx];

        bittle_config.set_zero_position(servo_num, mid_pos);
        bittle_config.set_invert_servo_flag(servo_num, is_servo_inverted);
        bittle_config.set_min_angle(servo_num, min_angle);
        bittle_config.set_max_angle(servo_num, max_angle);

    }

    return bittle_config;

}

