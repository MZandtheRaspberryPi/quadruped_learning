#ifndef __BITTLE_CONFIG__
#define __BITTLE_CONFIG__

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

#endif