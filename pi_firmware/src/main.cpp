#include<ctype.h>
#include<string>
#include <utils.hpp>

#include "bittle.hpp"

int main(int argc, char* argv[]) {
  Bittle bittle();

  uintmax_t sample_num = 5000;
  const uintmax_t samples_per_second = 200;
  const int64_t sleep_ms = 1000 / (int64_t)samples_per_second;
  uint64_t sensor_start, sensor_end, servo_start, servo_end, total_sensor, total_servo;
  total_sensor = total_servo = 0;
  servo_start = millis();
  for (uintmax_t i = 0; i < bittle.get_num_servos(); i++)
  {
      // bittle.set_servo_angle(i, 0);
  }
  servo_end = millis();
  total_servo += servo_end - servo_start;

  for (uintmax_t i = 0; i < sample_num; i++) {
    sensor_start = millis();
    ImuData data = bittle.get_sensor_data();
    sensor_end = millis();
    total_sensor += sensor_end - sensor_start;
    if (i % 500 == 0) {
        std::cout << "ax: " << data.ax << " ay: " << data.ay << " az: " << data.az << std::endl;
        std::cout << "gx: " << data.gx << " gy: " << data.gy << " gz: " << data.gz << std::endl;
    }
    if (i == sample_num / 2)
    {
      servo_start = millis();
      for (uintmax_t i = 0; i < bittle.get_num_servos(); i++)
      {
          // servo_controller.set_servo_angle(i, M_PI/8);
      }
      servo_end = millis();
      total_servo += servo_end - servo_start;
    }
    delay(sleep_ms);
  }
  
  std::cout << " sensor reads took " << 1.0 * total_sensor / sample_num << " ms on average" << std::endl;
  std::cout << " servo commands took " << 1.0 * total_servo / 2 << " ms on average" << std::endl;

}
