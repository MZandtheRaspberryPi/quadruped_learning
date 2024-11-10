#include<ctype.h>
#include<string>
#include <i2c_linux.hpp>
#include <Adafruit_PWMServoDriver.h>
#include <MPU6050.hpp>
#include <servo_controller.hpp>
#include <utils.hpp>

#include "bittle_config.hpp"

int main(int argc, char* argv[]) {

  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " servo_num angle" << std::endl;
    std::cout << "Example: " << argv[0] << " 1 0" << std::endl;
    return 1;
  }

  BittleJoint joint = static_cast<BittleJoint>(std::stoul(argv[1]));
  std::map<BittleJoint, uint8_t> joint_servo_num_map = get_bittle_joint_to_servo_num_mapping();
  uint8_t servo_num = joint_servo_num_map[joint];

  float32_t angle = std::stof(argv[2]);


  std::string i2c_name = "/dev/i2c-1";
  I2CLinuxAPI i2c_dev(i2c_name);
  i2c_dev.begin();
  ServoBoardConfig servo_config = make_bittle_config();
  std::string log_str = servo_config.to_string();
  std::cout << log_str << std::endl;

  Adafruit_PWMServoDriver_Wrapper motor_driver(PCA9685_I2C_ADDRESS, &i2c_dev);
  ServoController servo_controller = ServoController(&servo_config,
                                 &motor_driver,
                                 false);
 

  servo_controller.set_servo_angle(servo_num, angle);

  i2c_dev.close();
  std::cout << "joint num " << std::string(argv[1]) << " servo_num " << std::to_string(servo_num) <<
        " angle: " << std::to_string(angle) << std::endl;

}
