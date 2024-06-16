docker build -t bittle:2024-6-15 -f .\Dockerfile --progress=plain .

# setting up the bittle
on the Nyboard, on the side of the board facing the robot there is a switch. This switch should be toward the Bittle's head to be raspberry pi as i2cm aster.

i2cdetect -y 1

sudo docker run -it --device /dev/i2c-1 mzandtheraspberrypi/bittle:2024-6-15