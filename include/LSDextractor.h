/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/

#ifndef LSDEXTRACTOR_H
#define LSDEXTRACTOR_H

#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <eigen3/Eigen/Core>
#include <Eigen/Geometry>

namespace ORB_SLAM2 {
    class LineSegment {
        struct sort_lines_by_response {
            inline bool operator()(const cv::line_descriptor::KeyLine &a, const cv::line_descriptor::KeyLine &b) {
                return (a.response > b.response);
            }
        };

    public:
        LineSegment();

        ~LineSegment() = default;

        void ExtractLineSegment(const cv::Mat &img, std::vector<cv::line_descriptor::KeyLine> &keylines, cv::Mat &ldesc,
                                std::vector<Eigen::Vector3d> &keylineFunctions, float scale = 1.2, int numOctaves = 1);
    };
}


#endif //LSDEXTRACTOR_H
