#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <utility>

using namespace cv;

int SLIDER_POS = 0;

int main(int argc, char** argv)
{
        namedWindow("Video", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO);
        namedWindow("Output", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO);

        Mat frame, img;
        int c;
        Mat ref_frame, ref_desc;
        std::vector<KeyPoint> ref_kp;
        int frame_idx;

        for (frame_idx = 1; frame_idx < argc; frame_idx++) {
                std::cout << "Showing frame #" << frame_idx << std::endl;

                frame = imread(argv[frame_idx]);
                if (frame.empty())
                        break;
                frame.copyTo(img);

                SurfFeatureDetector det(400);
                SurfDescriptorExtractor ext;
                Scalar kcolor = Scalar(255, 0, 0);
                if (SLIDER_POS % 5 == 0) {
                        frame.copyTo(ref_frame);
                        det.detect(ref_frame, ref_kp);
                        ext.compute(ref_frame, ref_kp, ref_desc);
                        for (size_t i = 0; i < ref_kp.size(); i++) {
                                circle(img, ref_kp[i].pt,
                                       1, kcolor, 1);
                        }
                } else {
                        std::vector<KeyPoint> kp;
                        std::vector<Point2f> ref_points, points;
                        Mat desc;
                        std::vector<DMatch> matches;
                        BFMatcher matcher(NORM_L2, false);

                        det.detect(frame, kp);
                        ext.compute(frame, kp, desc);
                        matcher.match(desc, ref_desc, matches);

                        for (size_t i = 0; i < matches.size(); i++) {
                                KeyPoint start = ref_kp[matches[i].trainIdx];
                                KeyPoint end = kp[matches[i].queryIdx];
                                ref_points.push_back(start.pt);
                                points.push_back(end.pt);
                        }
                        // Homography RANSAC Pruning
                        Mat hmask;
                        Mat hom = findHomography(ref_points, points, CV_RANSAC, 3.0, hmask);
                        size_t n, i;
                        for (n = 0, i = 0; i < points.size(); i++) {
                                if (hmask.at<bool>(i, 0)) {
                                        ref_points[n] = ref_points[i];
                                        points[n] = points[i];
                                        n++;
                                } else {
                                        line(frame, ref_points[i], points[i], CV_RGB(255, 0, 0));
                                }
                        }
                        ref_points.resize(n);
                        points.resize(n);

                        // Find the Transform Matrix
                        Mat fmask;
                        Mat T;
                        Mat F = findFundamentalMat(ref_points, points, CV_FM_LMEDS, 3.0, 0.99, fmask);
                        //std::cout << F << std::endl;
                        // Plot points and inliers
                        std::cout << points.size() << " SAMPLE SIZE" << std::endl;
                        for (size_t i = 0; i < points.size(); i++) {
                                if (fmask.at<bool>(i, 0)) {
                                        line(img, ref_points[i], points[i], CV_RGB(0, 255, 0));
                                        line(frame, ref_points[i], points[i], CV_RGB(0, 255, 0));
                                } else {
                                        line(frame, ref_points[i], points[i], CV_RGB(255, 255, 0));
                                }
                                circle(frame, points[i], 1, kcolor, 1);
                                circle(img, points[i], 1, kcolor, 1);

                        }


                }

                imshow("Video", frame);
                moveWindow("Output", frame.size().width + 4, 0);
                imshow("Output", img);

                SLIDER_POS++;

                if ((c = waitKey(33)) == 'q')
                        break;
        }

        return 0;
}

