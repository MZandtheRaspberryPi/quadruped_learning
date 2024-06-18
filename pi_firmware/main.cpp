#include<ctype.h>
#include<string>
#include <i2c_linux.hpp>
#include <Adafruit_PWMServoDriver.h>
#include <MPU6050.hpp>
#include <utils.hpp>

int main() {
  std::string i2c_name = "/dev/i2c-1";
  uint8_t address = MPU6050_DEFAULT_ADDRESS;
  I2CLinuxAPI i2c_dev(i2c_name);
  i2c_dev.begin();

  Adafruit_PWMServoDriver motor_driver(PCA9685_I2C_ADDRESS, &i2c_dev);
  motor_driver.begin();
  MPU6050 accelgyro(address, &i2c_dev);
  if ( accelgyro.testConnection() ) 
    printf("MPU6050 connection test successful\n") ;
  else {
    fprintf( stderr, "MPU6050 connection test failed! something maybe wrong ...\n");
    return 1;
  }

  accelgyro.initialize();
  
  int16_t ax, ay, az, gx, gy, gz;
  ax = ay = az = gx = gy = gz = 0;
  const uintmax_t sample_num = 5000;
  const uintmax_t samples_per_second = 1000;
  const int64_t sleep_ms = 1000 / (int64_t)samples_per_second;
  uint16_t cur_servo_val = 4094 / 2;
  bool increase_angle = true;
  const uint16_t min_val = 1000;
  const uint16_t max_val = 3000;
  const uint16_t step_val = 1;
  const uint8_t pin_num = 0;
  for (uintmax_t i = 0; i < sample_num; i++) {
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    if (i % 500 == 0) {
        std::cout << "ax: " << ax << " ay: " << ay << " az: " << az << std::endl;
    }
    motor_driver.setPin(pin_num, cur_servo_val, false);
    if (increase_angle)
    {
        cur_servo_val += step_val;
    }
    else {
        cur_servo_val -= step_val;
    }
    if (cur_servo_val > max_val || cur_servo_val < min_val)
    {
        increase_angle = !increase_angle;
    }
    delay(sleep_ms);
  }
  i2c_dev.close();

}