docker build -t bittle:2024-6-15 -f .\Dockerfile --progress=plain .

# setting up the bittle
on the Nyboard, on the side of the board facing the robot there is a switch. This switch should be toward the Bittle's head to be raspberry pi as i2cm aster.

i2cdetect -y 1

sudo docker run -it --device /dev/i2c-1 mzandtheraspberrypi/bittle:2024-6-15

# setting up the pi
1. Ensure that the line enable_uart=1 is uncommented in the /boot/firmware/config.txt file, it should be by default.
2. remove any reference to console in /boot/firmware/cmdline.txt, ie remove console=serial0,115200.
3. disable the serial console sudo systemctl stop serial-getty@ttyS0.service && sudo systemctl disable serial-getty@ttyS0.service
4. change i2c to operate at 400khz instead of the default 100khz. If we have a loop rate of 200hz, for motors We will be sending 8 motor commands, 16 bits per motor is 25.6kbps. If we read 6 values from the accelerometer, and each is 16 bits, and we read 200 times a second that is 19.2kbps. Total we are at about 50kbps.
5. install docker and other stuff by running `setup_pi.sh` in this repo and then rebooting the pi with something like `sudo reboot`.

# notes on the bittle joint angles
I installed my servos as per the Bittle setup instructions, with Nyboard v1.2. The installation directions are below, note that the numbers here do not correspond to PCA9685 pin numbers:  
![bittle_install](./assets/petoi_servo_installation.png)

When I tested my setup which is a Bittle with metal geared servos, Nyboard v1.2, I found the following joint/PCA9685 pin mappings:  
|joint_name|PCA9685_pin_num|direction_notes|
|---|---|---|
|back_left_knee|0|increasing pwm moves the foot towards body|
|back_left_shoulder|1|increasing pwm moves the knee away from body|
|back_right_shoulder|6|increasing pwm moves the knee toward the body|
|back_right_knee|7|increasing the pwm moves the foot away from body|
|front_right_knee|8|increasing the pwm moves the foot toward the body|
|front_right_shoulder|9|increasing the pwm moves the foot toward the body|
|front_left_shoulder|14|increasing the pwm moves the knee toward the body|
|front_left_knee|15|increasing the pwm moves the foot away from the body|
|head_joint|12|increasing the pwm moves the head to the left|
