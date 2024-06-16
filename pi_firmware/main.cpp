#include<ctype.h>
#include<string>
#include <pi_i2c.hpp>
#include <MPU6050.hpp>

int main() {
  std::string i2c_name = "/dev/i2c-1";
  uint8_t address = MPU6050_DEFAULT_ADDRESS;
  I2CBusRaspberryPi i2c_dev(i2c_name);
  i2c_dev.begin();

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
  for (uintmax_t i = 0; i < sample_num; i++) {
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    if (i % 500 == 0) {
        std::cout << "ax: " << ax << " ay: " << ay << " az: " << az << std::endl;
    }
    delay(sleep_ms);
  }
  i2c_dev.close();

}