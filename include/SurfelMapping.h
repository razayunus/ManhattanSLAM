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

/**
 * This code is adopted and modified from https://github.com/HKUST-Aerial-Robotics/DenseSurfelMapping.
 */

#ifndef SURFELMAPPING_H
#define SURFELMAPPING_H

#include "System.h"
#include "Map.h"
#include "Surfel.h"
#include "SurfelFusion.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>


namespace ORB_SLAM2 {
    typedef pcl::PointXYZRGB PointType;
    typedef pcl::PointCloud<PointType> PointCloud;

    struct PoseElement {
        std::vector<Surfel> attachedSurfels;
        std::vector<int> linkedPoseIndex;
        int pointsBeginIndex;
        int pointsPoseIndex;

        PoseElement() : pointsBeginIndex(-1), pointsPoseIndex(-1) {}
    };

    class SurfelMapping {
    public:

        SurfelMapping(Map *map, const string &strSettingsFile);

        void Run();

        pcl::PointCloud<pcl::PointSurfel>::Ptr Stop();

        void InsertKeyFrame(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat planeMembershipImg,
                            const cv::Mat &pose, const int referenceIndex);

    protected:

        bool CheckNewKeyFrames();

        void ProcessNewKeyFrame();

        void moveAddSurfels(int referenceIndex);

        void getAddRemovePoses(int rootIndex, std::vector<int> &poseToAdd, std::vector<int> &poseToRemove);

        void getDriftfreePoses(int rootIndex, std::vector<int> &driftfreePoses, int driftfreeRange);

        void fuseMap(cv::Mat image, cv::Mat depth, cv::Mat planeMembershipImg, Eigen::Matrix4f poseInput,
                     int referenceIndex);

        std::list<std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat, int>> mlNewKeyFrames;

        std::mutex mMutexNewKFs;
        bool mbStop;
        std::mutex mMutexStop;

        Map *mMap;

        SurfelFusion *mSurfelFusion;

        std::vector<PoseElement> posesDatabase;
        std::set<int> localSurfelsIndexs;
        int driftFreePoses;

        std::vector<int> pointcloudPoseIndex;
    };
}

#endif //SURFELMAPPING_H
