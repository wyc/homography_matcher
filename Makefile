OPENCV_DIR = `pkg-config --cflags opencv` -I/usr/include/opencv2
OPENCV_LIBS = `pkg-config --libs opencv`

all: track_pose spin_track

track_pose : track_pose.cpp
	g++ -Wall -ggdb $(OPENCV_DIR) $(OPENCV_LIBS) -o track_pose track_pose.cpp

spin_track : spin_track.cpp
	g++ -Wall -ggdb $(OPENCV_DIR) $(OPENCV_LIBS) -o spin_track spin_track.cpp
