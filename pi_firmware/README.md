docker build -t bittle:2024-6-15 -f .\Dockerfile --progress=plain .

# setting up the bittle
on the Nyboard, on the side of the board facing the robot there is a switch. This switch should be toward the Bittle's head to be raspberry pi as i2cm aster.

i2cdetect -y 1

sudo docker run -it --device /dev/i2c-1 mzandtheraspberrypi/bittle:2024-6-15

# setting up the pi
1. Ensure that the line enable_uart=1 is uncommented in the /boot/firmware/config.txt file, it should be by default.
2. remove any reference to console in /boot/firmware/cmdline.txt, ie remove console=serial0,115200.
3. disable the serial console sudo systemctl stop serial-getty@ttyS0.service && sudo systemctl disable serial-getty@ttyS0.service
4. change i2c to operate at 400khz instead of the default 100khz. We don't really need this, but i'll do it to give some future headroom in case we want to bump up the loop rate. We will be sending 8 motor commands, 16 bits per motor, 200 times a second is 25.6khz. If we read 6 values from the accelerometer, and each is 16 bits, and we read 200 times a second that is 19.2khz. Total we are at about 50khz.
5. install docker and other stuff by running `setup_pi.sh` in this repo and then rebooting the pi with something like `sudo reboot`.