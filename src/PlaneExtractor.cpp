/**
* This file is part of ManhattanSLAM.
*
* Copyright (C) 2021 Raza Yunus <razayunus31 at gmail dot com>
* For more information see <https://github.com/razayunus/ManhattanSLAM>
*
* ManhattanSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ManhattanSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ManhattanSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "PlaneExtractor.h"

using namespace std;
using namespace cv;
using namespace Eigen;

PlaneDetection::PlaneDetection() {}

PlaneDetection::~PlaneDetection() {
    cloud.vertices.clear();
    seg_img_.release();
    color_img_.release();
}

bool PlaneDetection::readColorImage(cv::Mat RGBImg) {
    color_img_ = RGBImg;
    if (color_img_.empty() || color_img_.depth() != CV_8U) {
        cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
        return false;
    }
    return true;
}

bool PlaneDetection::readDepthImage(const cv::Mat depthImg, const cv::Mat &K, const float &depthMapFactor) {
    cv::Mat depth_img = depthImg;
    if (depth_img.empty() || depth_img.depth() != CV_16U) {
        cout << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
        return false;
    }

    double width = ceil(depthImg.cols / 2.0);
    double height = ceil(depthImg.rows / 2.0);
    cloud.vertices.resize(height * width);
    cloud.verticesColour.resize(height * width);
    cloud.w = width;
    cloud.h = height;

    seg_img_ = cv::Mat(height, width, CV_8UC3);

    int rows = depth_img.rows, cols = depth_img.cols;
    int vertex_idx = 0;
    for (int i = 0; i < rows; i += 2) {
        for (int j = 0; j < cols; j += 2) {
            double z = (double) (depth_img.at<unsigned short>(i, j)) * depthMapFactor;
            if (_isnan(z)) {
                cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
                continue;
            }
            double x = ((double) j - K.at<float>(0, 2)) * z / K.at<float>(0, 0);
            double y = ((double) i - K.at<float>(1, 2)) * z / K.at<float>(1, 1);
            cloud.verticesColour[vertex_idx] = color_img_.at<cv::Vec3b>(i, j);
            cloud.vertices[vertex_idx++] = VertexType(x, y, z);
        }
    }
    return true;
}

void PlaneDetection::runPlaneDetection() {
    plane_filter.run(&cloud, &plane_vertices_, &seg_img_);
    plane_num_ = (int) plane_vertices_.size();
}
