FROM ubuntu:22.04 AS build

RUN apt-get update
RUN apt-get install i2c-tools nano git libi2c-dev libboost-all-dev cmake build-essential g++ valgrind python3-pip -y

RUN mkdir /tmp/build
COPY pi_i2c /tmp/pi_i2c
COPY CMakeLists.txt /tmp/CMakeLists.txt
COPY include/ /tmp/include
COPY src/ /tmp/src/
WORKDIR /tmp/build
RUN cmake .. -DCOMPILE_WARNING_AS_ERROR=ON
RUN make
RUN ls -ltrh
RUN ls -ltrh pi_i2c

FROM ubuntu:22.04 AS deploy
RUN mkdir /quadruped
WORKDIR /quadruped
COPY --from=build /tmp/build/main /quadruped/main
COPY --from=build /tmp/build/pi_i2c/set_pwm /quadruped/set_pwm
COPY --from=build /tmp/build/pi_i2c/read_eeprom /quadruped/read_eeprom