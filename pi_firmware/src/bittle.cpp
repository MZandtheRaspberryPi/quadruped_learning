#include "bittle.hpp"

Bittle::Bittle() {
  std::string i2c_name = "/dev/i2c-1";
  uint8_t mpu_address = MPU6050_DEFAULT_ADDRESS;
  i2c_dev_(i2c_name);
  i2c_dev_.begin();
  servo_config_ = make_bittle_config();
  std::string log_str = servo_config.to_string();
  std::cout << log_str << std::endl;

  motor_driver_(PCA9685_I2C_ADDRESS, &i2c_dev_);
  servo_controller_ = ServoController(&servo_config,
                                 &motor_driver,
                                 false);
  accelgyro_(mpu_address, &i2c_dev_);
  if ( accelgyro_.testConnection() )
    printf("MPU6050 connection test successful\n") ;
  else {
    fprintf( stderr, "MPU6050 connection test failed! something maybe wrong ...\n");
    return 1;
  }

  accelgyro_.initialize();

}

Bittle::~Bittle()
{
    i2c_dev_.close();
}

ImuData Bittle::get_sensor_data()
{
    int16_t ax, ay, az, gx, gy, gz;
    ax = ay = az = gx = gy = gz = 0;
    
    accelgyro_.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    ImuData data = {ax, ay, az, gx, gy, gz};
    return data;
}


bool Bittle::set_servo_angle(const uint8_t &servo_num,
                        const float32_t &servo_angle,
                        bool debug)
{
    return servo_controller_.set_servo_angle(i, M_PI/8);
}

uint8_t Bittle:get_num_servos()
{
    return servo_config_.get_num_servos();
}

