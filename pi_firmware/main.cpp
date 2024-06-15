#include<ctype.h>
#include<string>
#include <pi_i2c.hpp>
#include <MPU6050.hpp>

int main() {
  std::string i2c_name = "/dev/i2c-3";
  uint8_t address = MPU6050_DEFAULT_ADDRESS;
  I2CBusRaspberryPi i2c_dev(i2c_name);
  MPU6050 accelgyro(address, &i2c_dev);
  // to do, test connection
  accelgyro.initialize();
}