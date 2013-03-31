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
#include <stdlib.h>

using namespace cv;

void length_filter_matches(const std::vector<DMatch>& matches_in,
                    std::vector<DMatch>& matches_out)
{
        double maxlen = 0, minlen = 1000;
        for (size_t i = 0; i < matches_in.size(); i++) {
                double len = matches_in[i].distance;
                if (len < minlen)
                        minlen = len;
                if (len > maxlen)
                        maxlen = len;
        }
        /*
        std::cout << "-- Max Len : " << maxlen << std::endl;
        std::cout << "-- Min Len : " << minlen << std::endl;
        */

        for (size_t i = 0; i < matches_in.size(); i++) {
                if (matches_in[i].distance < 3 * minlen) {
                        /*
                        std::cout << "pushing back match with distance "
                                << matches_in[i].distance << std::endl;
                        */
                        matches_out.push_back(matches_in[i]);
                }
        }
}

void find_homography_points(const std::vector<KeyPoint>& kp1,
                            const std::vector<KeyPoint>& kp2,
                            const std::vector<DMatch>& matches,
                            std::vector<Point2f>& pts1,
                            std::vector<Point2f>& pts2)
{
        std::vector<Point2f> tmp_pts1, tmp_pts2;
        for (size_t i = 0; i < matches.size(); i++) {
                tmp_pts1.push_back(kp1[matches[i].trainIdx].pt);
                tmp_pts2.push_back(kp2[matches[i].queryIdx].pt);
        }

        /* find homography & prune points */
        Mat hmask;
        Mat hom = findHomography(tmp_pts1, tmp_pts2, CV_RANSAC, 20.0, hmask);
        for (size_t i = 0; i < tmp_pts1.size(); i++) {
                if (hmask.at<bool>(i, 0)) {
                        pts1.push_back(tmp_pts1[i]);
                        pts2.push_back(tmp_pts2[i]);
                }
        }
}


int match(const char* path1, const char* path2)
{
        Mat im1, im2, imout;

        /* load our images */
        im1 = imread(path1);
        im2 = imread(path2);

        if (im1.rows > im2.rows) {
                im2.resize(im1.rows, 0);
        } else if (im2.rows > im1.rows) {
                im1.resize(im2.rows, 0);
        }
        hconcat(im1, im2, imout);

        /* find the feature points */
        SurfFeatureDetector det(400);
        SurfDescriptorExtractor ext;
        std::vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;

        det.detect(im1, kp1);
        ext.compute(im1, kp1, desc1);

        det.detect(im2, kp2);
        ext.compute(im2, kp2, desc2);



        /* match the feature points */
        FlannBasedMatcher matcher;
        std::vector<DMatch> matches;
        std::vector<DMatch> good_matches;
        std::vector<Point2f> pts1, pts2;
        matcher.match(desc2, desc1, matches);
        length_filter_matches(matches, good_matches);
        find_homography_points(kp1, kp2, good_matches, pts1, pts2);

        size_t i;
        for (i = 0; i < pts1.size(); i++) {
                Point2f pp1(pts1[i]);
                Point2f pp2(pts2[i].x + im1.cols, pts2[i].y);
                /*
                std::cout << pp1 << " to " << pp2 << std::endl;
                */
                line(imout, pp1, pp2,
                     CV_RGB(rand() % 256, rand() % 256, rand() % 256));
                Scalar kcolor = Scalar(255, 0, 0);
                circle(im1, pp1, 3, kcolor, 1);
                circle(im2, pp2, 3, kcolor, 1);
        }
        std::cout << path1 << ":" << path2
                  << " " << i << " true matches." << std::endl;

        namedWindow("Out", CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO);
        imshow("Out", imout);
        waitKey(0);

        return 0;
}

int main(int argc, char** argv)
{
        int q;
        q = match(argv[1], argv[2]);
        /*std::cout << "match quality: " << q << std::endl;*/
        q = q;
        return 0;
}

