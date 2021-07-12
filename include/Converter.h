/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CONVERTER_H
#define CONVERTER_H

#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include"Thirdparty/g2o/g2o/types/plane_3d.h"

namespace ORB_SLAM2 {

    class Converter {
    public:
        static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

        static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);

        static cv::Mat toCvMat(const g2o::SE3Quat &SE3);

        static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);

        static cv::Mat toCvVec(const Eigen::Matrix<double, 3, 1> &m);

        static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);

        static Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat &cvMat4);

        static std::vector<float> toQuaternion(const cv::Mat &M);

        static g2o::Plane3D toPlane3D(const cv::Mat &coe);
    };

}// namespace ORB_SLAM

#endif // CONVERTER_H
