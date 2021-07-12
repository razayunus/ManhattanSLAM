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

#ifndef PLANEEXTRACTOR_H
#define PLANEEXTRACTOR_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include "include/peac/AHCPlaneFitter.hpp"
#include <unordered_map>

typedef Eigen::Vector3d VertexType;
typedef cv::Vec3d VertexColour;

#ifdef __linux__
#define _isnan(x) isnan(x)
#endif

struct ImagePointCloud {
    std::vector<VertexType> vertices; // 3D vertices
    std::vector<VertexColour> verticesColour;
    int w, h;

    inline int width() const { return w; }

    inline int height() const { return h; }

    inline bool get(const int row, const int col, double &x, double &y, double &z) const {
        const int pixIdx = row * w + col;
        z = vertices[pixIdx][2];
        // Remove points with 0 or invalid depth in case they are detected as a plane
        if (z == 0 || std::_isnan(z)) return false;
        x = vertices[pixIdx][0];
        y = vertices[pixIdx][1];
        return true;
    }
};

class PlaneDetection {
public:
    ImagePointCloud cloud;
    ahc::PlaneFitter<ImagePointCloud> plane_filter;
    std::vector<std::vector<int>> plane_vertices_; // vertex indices each plane contains
    cv::Mat seg_img_; // segmentation image
    cv::Mat color_img_; // input color image
    int plane_num_;

public:
    PlaneDetection();

    ~PlaneDetection();

    bool readColorImage(cv::Mat RGBImg);

    bool readDepthImage(const cv::Mat depthImg, const cv::Mat &K, const float &depthMapFactor);

    void runPlaneDetection();

};


#endif //PLANEEXTRACTOR_H