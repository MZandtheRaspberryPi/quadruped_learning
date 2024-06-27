#include<ctype.h>
#include<string>
#include <i2c_linux.hpp>
#include <Adafruit_PWMServoDriver.h>
#include <MPU6050.hpp>
#include <servo_controller.hpp>
#include <utils.hpp>

int main() {
  std::string i2c_name = "/dev/i2c-1";
  uint8_t mpu_address = MPU6050_DEFAULT_ADDRESS;
  I2CLinuxAPI i2c_dev(i2c_name);
  i2c_dev.begin();
  uint8_t num_servos = 8;
  ServoBoardConfig servo_config = ServoBoardConfig(num_servos);
  Adafruit_PWMServoDriver_Wrapper motor_driver(PCA9685_I2C_ADDRESS, &i2c_dev);
  ServoController servo_controller = ServoController(&servo_config,
                                 &motor_driver);
  MPU6050 accelgyro(mpu_address, &i2c_dev);
  if ( accelgyro.testConnection() )
    printf("MPU6050 connection test successful\n") ;
  else {
    fprintf( stderr, "MPU6050 connection test failed! something maybe wrong ...\n");
    return 1;
  }

  accelgyro.initialize();

  int16_t ax, ay, az, gx, gy, gz;
  ax = ay = az = gx = gy = gz = 0;
  uintmax_t sample_num = 5000;
  const uintmax_t samples_per_second = 200;
  const int64_t sleep_ms = 1000 / (int64_t)samples_per_second;
  servo_controller.set_servo_angle(0, 0);
  for (uintmax_t i = 0; i < sample_num; i++) {
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    if (i % 500 == 0) {
        std::cout << "ax: " << ax << " ay: " << ay << " az: " << az << std::endl;
    }
    if (i == sample_num / 2)
    {
          servo_controller.set_servo_angle(0, M_PI/8);
    }
    delay(sleep_ms);
  }
  i2c_dev.close();

}
