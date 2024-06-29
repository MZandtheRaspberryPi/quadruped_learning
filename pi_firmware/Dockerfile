FROM ubuntu:22.04 as build

RUN apt-get update
RUN apt-get install i2c-tools nano git libi2c-dev libboost-all-dev cmake build-essential g++ -y

RUN mkdir /tmp/build
COPY pi_i2c /tmp/pi_i2c
COPY CMakeLists.txt /tmp/CMakeLists.txt
COPY main.cpp /tmp/
WORKDIR /tmp/build
RUN cmake .. -DCOMPILE_WARNING_AS_ERROR=ON
RUN make
RUN ls -ltrh
RUN ls -ltrh pi_i2c

FROM ubuntu:22.04
RUN mkdir /quadruped
WORKDIR /quadruped
COPY --from=build /tmp/build/main /quadruped/main
COPY --from=build /tmp/build/pi_i2c/set_angle /quadruped/set_angle
COPY --from=build /tmp/build/pi_i2c/set_pwm /quadruped/set_pwm