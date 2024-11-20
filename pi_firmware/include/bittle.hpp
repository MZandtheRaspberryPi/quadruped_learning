#include<ctype.h>
#include<string>
#include <i2c_linux.hpp>

#include <Adafruit_PWMServoDriver.h>
#include <MPU6050.hpp>
#include <servo_controller.hpp>
#include <utils.hpp>

#include "bittle_config.hpp"

struct ImuData {
    int16_t ax;
    int16_t ay;
    int16_t az;
    int16_t gx;
    int16_t gy;
    int16_t gz;
};

class Bittle {
  public:
    Bittle();
    ImuData get_sensor_data();
    bool set_servo_angle(const uint8_t &servo_num,
                         const float32_t &servo_angle,
                         bool debug = false);
  private:
    I2CLinuxAPI i2c_dev_;
    ServoBoardConfig servo_config_;
    Adafruit_PWMServoDriver_Wrapper motor_driver_;
    ServoController servo_controller_;
    MPU6050 accelgyro_;


};

