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

#include "PlaneMatcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace ORB_SLAM2 {
    PlaneMatcher::PlaneMatcher(float dTh, float aTh, float verTh, float parTh) : dTh(dTh), aTh(aTh), verTh(verTh),
                                                                                 parTh(parTh) {}

    int PlaneMatcher::SearchMapByCoefficients(Frame &pF, const std::vector<MapPlane *> &vpMapPlanes) {
        pF.mbNewPlane = false;

        int nmatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            float ldTh = dTh;
            float lverTh = verTh;
            float lparTh = parTh;

            bool found = false;
            int j = 0;
            for (auto vpMapPlane : vpMapPlanes) {
                if (vpMapPlane->isBad()) {
                    j++;
                    continue;
                }

                cv::Mat pW = vpMapPlane->GetWorldPos();

                float angle = pM.at<float>(0) * pW.at<float>(0) +
                              pM.at<float>(1) * pW.at<float>(1) +
                              pM.at<float>(2) * pW.at<float>(2);
                j++;

                // Associate plane
                if (angle > aTh) {
                    double dis = PointDistanceFromPlane(pM, vpMapPlane->mvPlanePoints);
                    if (dis < ldTh) {
                        ldTh = dis;
                        pF.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                        pF.mvpMapPlanes[i] = vpMapPlane;
                        found = true;
                        continue;
                    }
                }

                // Vertical planes
                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    pF.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    pF.mvpVerticalPlanes[i] = vpMapPlane;
                    continue;
                }

                // Parallel planes
                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    pF.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    pF.mvpParallelPlanes[i] = vpMapPlane;
                }
            }

            if (found) {
                nmatches++;
            }
        }

        return nmatches;
    }

    double PlaneMatcher::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud) {
        double res = 100;
        for (auto p : pointCloud->points) {
            double dis = abs(plane.at<float>(0, 0) * p.x +
                             plane.at<float>(1, 0) * p.y +
                             plane.at<float>(2, 0) * p.z +
                             plane.at<float>(3, 0));
            if (dis < res)
                res = dis;
        }
        return res;
    }

}
