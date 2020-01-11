#
# Creator : Akio Lin
# Date   : 2020-01-08
# email  : akioolin@gmail.com
#

VERSION = 1.0
TOOLCHAIN_PREFIX=

CC=$(TOOLCHAIN_PREFIX)-gcc
CPP=$(TOOLCHAIN_PREFIX)-g++
LD=$(TOOLCHAIN_PREFIX)-ld
STRIP=$(TOOLCHAIN_PREFIX)-strip

DEFFLAGS += -DVERSION_STR=\"$(VERSION)\" 

OPTFLAGS = -O2 -mcpu=cortex-a53 -march=armv8-a
CFLAGS = -Wall 
CPPFLAGS = -Wall -std=c++11
LDFLAGS = $(OPTFLAGS) -std=c++11
IFLAGS  = -I. -I./include
IFLAGS += -I./include/edgetpu
IFLAGS += -I./include/src/cpp
IFLAGS += -I./include/src/cpp/examples
LDLIBS  = -Wl,-rpath,/usr/local/lib 
LDLIBS += -L./lib
LDLIBS += -L/lib64/aarch64-linux-gnu
LDLIBS += -L/usr/lib/aarch64-linux-gnu
LDLIBS += -lpthread -lm -ldl
LDLIBS += -lusb-1.0
LDLIBS += -lglog
LDLIBS += -ledgetpu
LDLIBS += -ledgetpu_api
LDLIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs 
LDLIBS += -lopencv_highgui -lopencv_videoio
LDLIBS += -ltensorflow-lite
LDLIBS += -labsl_synchronization
LDLIBS += -labsl_base
LDLIBS += -labsl_spinlock_wait
LDLIBS += -labsl_malloc_internal
LDLIBS += -labsl_symbolize
LDLIBS += -labsl_stacktrace
LDLIBS += -labsl_debugging_internal
LDLIBS += -labsl_dynamic_annotations
LDLIBS += -labsl_demangle_internal
LDLIBS += -labsl_time
LDLIBS += -labsl_time_zone
LDLIBS += -labsl_graphcycles_internal
LDLIBS += -labsl_int128

APP = edgetpu_demo

APP_OBJS = \
    main.o 

all: $(APP_OBJS) $(APP)

build: $(APP_OBJS) $(APP)

$(APP): $(APP_OBJS)
	$(CPP) $(CFLAGS) $(LDFLAGS) -o $@ $(APP_OBJS) $(ADDON_OBJS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(IFLAGS) $(DEFFLAGS) -o $@ -c $<

%.o: %.cpp
	$(CPP) $(CPPFLAGS) $(OPTFLAGS) $(IFLAGS) $(DEFFLAGS) -o $@ -c $<

%.o: %.cc
	$(CPP) $(CPPFLAGS) $(OPTFLAGS) $(IFLAGS) $(DEFFLAGS) -o $@ -c $<

clean:
	rm -f *~ ./src/*~ ./include/*~ $(APP_OBJS) $(APP)
