#
# Creator : Akio Lin
# Date    : 2020-01-08
# email   : akioolin@gmail.com
#

VERSION = 1.0
#TOOLCHAIN_PREFIX=/opt/aarch64-toolchains/linaro-7.4.1/bin/aarch64-linux-gnu

CPP=$(TOOLCHAIN_PREFIX)-g++
LD=$(TOOLCHAIN_PREFIX)-ld
AR=$(TOOLCHAIN_PREFIX)-ar
STRIP=$(TOOLCHAIN_PREFIX)-strip

OPTFLAGS = -O2 -mcpu=cortex-a53 -march=armv8-a
CPPFLAGS = -Wall -pipe -fPIC
IFLAGS = -I. -I./include
LDFLAGS = $(OPTFLAGS) -L./lib -ltensorflow-lite

LIB_NAME = libedgetpu_api.a

LIB_OBJS = \
    ./src/cpp/basic/basic_engine.o \
    ./src/cpp/basic/basic_engine_native.o \
    ./src/cpp/basic/edgetpu_resource_manager.o \
    ./src/cpp/classification/engine.o \
    ./src/cpp/detection/engine.o \
    ./src/cpp/learn/utils.o \
    ./src/cpp/learn/imprinting/engine.o \
    ./src/cpp/learn/imprinting/engine_native.o \
    ./src/cpp/fake_op.o \
    ./src/cpp/error_reporter.o \
    ./src/cpp/bbox_utils.o \
    ./src/cpp/version.o \
    ./src/cpp/examples/label_utils.o \
    ./src/cpp/examples/model_utils.o \
    ./src/cpp/utils.o \
    ./src/cpp/tools/tflite_graph_util.o \
    ./src/cpp/posenet/posenet_decoder.o \
    ./src/cpp/posenet/posenet_decoder_op.o \
    ./src/cpp/posenet/posenet_decoder_tflite_plugin.o


all: $(LIB_NAME) $(LIB_OBJS)

build: $(LIB_NAME) $(LIB_OBJS)

$(LIB_NAME): $(LIB_OBJS)
	$(AR) crs $(LIB_NAME) $(LIB_OBJS)

%.o: %.cc
	$(CPP) $(CPPFLAGS) $(OPTFLAGS) $(IFLAGS) $(DEFFLAGS) -o $@ -c $<

clean:
	rm -f *~ $(LIB_OBJS) $(LIB_NAME)
