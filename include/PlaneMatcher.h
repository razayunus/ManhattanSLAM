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

#ifndef PLANEMATCHER_H
#define PLANEMATCHER_H

#include "MapPlane.h"
#include "KeyFrame.h"
#include "Frame.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>

namespace ORB_SLAM2 {
    class PlaneMatcher {
    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        PlaneMatcher(float dTh = 0.1, float aTh = 0.86, float verTh = 0.08716, float parTh = 0.9962);

        int SearchMapByCoefficients(Frame &pF, const std::vector<MapPlane *> &vpMapPlanes);

    protected:
        float dTh, aTh, verTh, parTh;

        double PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud);
    };
}


#endif //PLANEMATCHER_H
