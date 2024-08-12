#ifndef __BITTLE_CONFIG__
#define __BITTLE_CONFIG__

#include<map>

#include <servo_controller.hpp>

#define LOW_VOLTAGE 650
#define DEVICE_ADDRESS 0x54
#define BAUD_RATE 115200

static const uint8_t BITTLE_NUM_SERVOS = 9;

enum BittleJoint {
  BACK_LEFT_KNEE,
  BACK_LEFT_SHOULDER,
  BACK_RIGHT_SHOULDER,
  BACK_RIGHT_KNEE,
  FRONT_RIGHT_KNEE,
  FRONT_RIGHT_SHOULDER,
  FRONT_LEFT_SHOULDER,
  FRONT_LEFT_KNEE,
  HEAD_JOINT,
  // this has to go at the end so we can count the joints
  LENGTH
};

// below are constants pulled from bittle source code
// indexing these are tricky, look to function get_bittle_joint_to_bittle_array_idx_mapping for help

// https://www.petoi.camp/forum/software/about-calibration-state
// "There's some rescaling and calibration algorithm in the code before sending the neutral position to the servos."
// namely the middle point of the pwm signal, 1500, (500-2500), does not correspond to zero degrees using the bittle
// defaults because of this shifting below, and also calibration
static constexpr int8_t middleShift[] = { 0, 15, 0, 0,
                         -45, -45, -45, -45,
                         55, 55, -55, -55,
                         -55, -55, -55, -55 };
static constexpr int angleLimit[][2] = {
  { -120, 120 },
  { -30, 80 },
  { -120, 120 },
  { -120, 120 },
  { -90, 60 },
  { -90, 60 },
  { -90, 90 },
  { -90, 90 },
  { -200, 80 },
  { -200, 80 },
  { -80, 200 },
  { -80, 200 },
  { -80, 200 },
  { -80, 200 },
  { -80, 200 },
  { -80, 200 },
};

static constexpr int8_t rotationDirection[] = { 1, -1, 1, 1,
                               1, -1, 1, -1,
                               1, -1, -1, 1,
                               -1, 1, 1, -1 };

static constexpr uint8_t pwm_pin[] = { 12, 11, 4, 3,
                   13, 10, 5, 2,
                   14, 9, 6, 1,
                   15, 8, 7, 0 };

std::map<BittleJoint, uint8_t> get_bittle_joint_to_servo_num_mapping();
std::map<BittleJoint, uint8_t> get_bittle_joint_to_pwm_pin_mapping();
std::map<BittleJoint, uint8_t> get_bittle_joint_to_bittle_array_idx_mapping();
ServoBoardConfig make_bittle_config();

#endif