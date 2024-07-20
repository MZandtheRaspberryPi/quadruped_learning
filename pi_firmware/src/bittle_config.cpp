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
ServoBoardConfig make_bittle_config()
{

    // ServoBoardConfig bittle_config(9,
    //                -120,
    //                 220,
    //                0,
    //                false,
    //                float32_t min_angle_to_command = -M_PI/2,
    //                float32_t max_angle_to_command = M_PI/2,
    //                uint16_t min_pulsewidth_to_command = 500,
    //                uint16_t max_pulsewidth_to_command = 2500);
    ServoBoardConfig bittle_config(9);
    return bittle_config;

}