OPENCV_DIR = `pkg-config --cflags opencv` -I/usr/include/opencv2
OPENCV_LIBS = `pkg-config --libs opencv`

all: hmatch spin_track

hmatch : hmatch.cpp
	g++ -Wall -ggdb $(OPENCV_DIR) $(OPENCV_LIBS) -o hmatch hmatch.cpp

spin_track : spin_track.cpp
	g++ -Wall -ggdb $(OPENCV_DIR) $(OPENCV_LIBS) -o spin_track spin_track.cpp
