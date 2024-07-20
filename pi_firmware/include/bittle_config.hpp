#ifndef __BITTLE_CONFIG__
#define __BITTLE_CONFIG__

#include <servo_controller.hpp>

// enum mapping a bittle joint to which pin of the
// pca9685 chip
enum JointPin {
  BACK_LEFT_KNEE = 0,
  BACK_LEFT_SHOULDER = 1,
  BACK_RIGHT_SHOULDER = 6,
  BACK_RIGHT_KNEE = 7,
  FRONT_RIGHT_KNEE = 8,
  FRONT_RIGHT_SHOULDER = 9,
  FRONT_LEFT_SHOULDER = 14,
  FRONT_LEFT_KNEE = 15,
  HEAD_JOINT = 12
};

// enum mapping a bittle joint to the index in the petoi
// configs below
enum JointBittleIndex {
  HEAD_JOINT_IDX = 0,
  BACK_LEFT_KNEE_IDX = 15,
  BACK_LEFT_SHOULDER_IDX = 11,
  BACK_RIGHT_SHOULDER_IDX = 10,
  BACK_RIGHT_KNEE_IDX = 14,
  FRONT_RIGHT_KNEE_IDX = 13,
  FRONT_RIGHT_SHOULDER_IDX = 9,
  FRONT_LEFT_SHOULDER_IDX = 8,
  FRONT_LEFT_KNEE_IDX = 12
};

// https://www.petoi.camp/forum/software/about-calibration-state
// "There's some rescaling and calibration algorithm in the code before sending the neutral position to the servos."
// namely the middle point of the pwm signal, 1500, (500-2500), does not correspond to zero degrees using the bittle
// defaults because of this shifting below, and also calibration
int8_t middleShift[] = { 0, 15, 0, 0,
                         -45, -45, -45, -45,
                         55, 55, -55, -55,
                         -55, -55, -55, -55 };
int angleLimit[][2] = {
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

int8_t rotationDirection[] = { 1, -1, 1, 1,
                               1, -1, 1, -1,
                               1, -1, -1, 1,
                               -1, 1, 1, -1 };

uint8_t pwm_pin[] = { 12, 11, 4, 3,
                   13, 10, 5, 2,
                   14, 9, 6, 1,
                   15, 8, 7, 0 };
#define VOLTAGE_DETECTION_PIN A7
#define LOW_VOLTAGE 650
#define DEVICE_ADDRESS 0x54
#define BAUD_RATE 115200

ServoBoardConfig make_bittle_config();

#endif