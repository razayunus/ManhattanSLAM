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

#include "SurfelMapping.h"

namespace ORB_SLAM2 {
    SurfelMapping::SurfelMapping(Map *map, const string &strSettingPath) : mMap(map),
                                                                           mbStop(false), driftFreePoses(10) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        int imgWidth = fSettings["Camera.width"];
        int imgHeight = fSettings["Camera.height"];

        float distanceFar = fSettings["Surfel.distanceFar"];
        float distanceNear = fSettings["Surfel.distanceNear"];

        mSurfelFusion = new SurfelFusion(imgWidth, imgHeight, fx, fy, cx, cy, distanceFar, distanceNear);
    }

    void SurfelMapping::Run() {
        while (true) {
            if (CheckNewKeyFrames()) {
                ProcessNewKeyFrame();
            }

            {
                unique_lock<mutex> lock(mMutexStop);
                if (mbStop) {
                    mbStop = false;
                    break;
                }
            }
        }
    }

    pcl::PointCloud<pcl::PointSurfel>::Ptr SurfelMapping::Stop() {
        unique_lock<mutex> lock(mMutexStop);
        mbStop = true;

        pcl::PointCloud<pcl::PointSurfel>::Ptr pointCloud(new pcl::PointCloud<pcl::PointSurfel>());
        for (int surfelIt = 0, surfel_end = mMap->mvLocalSurfels.size(); surfelIt < surfel_end; surfelIt++) {
            if (mMap->mvLocalSurfels[surfelIt].updateTimes < 5)
                continue;
            pcl::PointSurfel p;
            p.x = mMap->mvLocalSurfels[surfelIt].px;
            p.y = mMap->mvLocalSurfels[surfelIt].py;
            p.z = mMap->mvLocalSurfels[surfelIt].pz;
            p.r = mMap->mvLocalSurfels[surfelIt].r;
            p.g = mMap->mvLocalSurfels[surfelIt].g;
            p.b = mMap->mvLocalSurfels[surfelIt].b;
            p.normal_x = mMap->mvLocalSurfels[surfelIt].nx;
            p.normal_y = mMap->mvLocalSurfels[surfelIt].ny;
            p.normal_z = mMap->mvLocalSurfels[surfelIt].nz;
            p.radius = mMap->mvLocalSurfels[surfelIt].size * 1000;
            p.confidence = mMap->mvLocalSurfels[surfelIt].weight;
//            p.intensity = mMap->mvLocalSurfels[surfelIt].color;
            pointCloud->push_back(p);
        }

        for (int surfelIt = 0, surfel_end = mMap->mvInactiveSurfels.size(); surfelIt < surfel_end; surfelIt++) {
//            if(mMap->mvInactiveSurfels[surfelIt].updateTimes < 5)
//                continue;
            pcl::PointSurfel p;
            p.x = mMap->mvInactiveSurfels[surfelIt].px;
            p.y = mMap->mvInactiveSurfels[surfelIt].py;
            p.z = mMap->mvInactiveSurfels[surfelIt].pz;
            p.r = mMap->mvInactiveSurfels[surfelIt].r;
            p.g = mMap->mvInactiveSurfels[surfelIt].g;
            p.b = mMap->mvInactiveSurfels[surfelIt].b;
            p.normal_x = mMap->mvInactiveSurfels[surfelIt].nx;
            p.normal_y = mMap->mvInactiveSurfels[surfelIt].ny;
            p.normal_z = mMap->mvInactiveSurfels[surfelIt].nz;
            p.radius = mMap->mvInactiveSurfels[surfelIt].size * 1000;
            p.confidence = mMap->mvInactiveSurfels[surfelIt].weight;
//            p.intensity = mMap->mvLocalSurfels[surfelIt].color;
            pointCloud->push_back(p);
        }

        std::vector<ORB_SLAM2::MapPlane *> mapPlanes = mMap->GetAllMapPlanes();
        double radius = 0.1414 * 1000;

        for (auto pMP : mapPlanes) {
            auto &planePoints = pMP->mvPlanePoints->points;
            cv::Mat P3Dw = pMP->GetWorldPos();

            for (auto &planePoint : planePoints) {
                pcl::PointSurfel p;
                p.x = planePoint.x;
                p.y = planePoint.y;
                p.z = planePoint.z;
                p.r = planePoint.r;
                p.g = planePoint.g;
                p.b = planePoint.b;

                p.normal_x = P3Dw.at<float>(0);
                p.normal_y = P3Dw.at<float>(1);
                p.normal_z = P3Dw.at<float>(2);
                p.radius = radius;
                p.confidence = 1;
//                p.intensity = pMP->mRed + pMP->mBlue + pMP->mGreen / 255*3;

                pointCloud->push_back(p);
            }
        }

//        (*pointCloud) += (*inactive_pointcloud);

        return pointCloud;
    }

    void SurfelMapping::InsertKeyFrame(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat planeMembershipImg,
                                       const cv::Mat &pose, const int referenceIndex) {
        unique_lock<mutex> lock(mMutexNewKFs);
        mlNewKeyFrames.emplace_back(imRGB, imDepth, planeMembershipImg, pose, referenceIndex);
    }

    bool SurfelMapping::CheckNewKeyFrames() {
        unique_lock<mutex> lock(mMutexNewKFs);
        return (!mlNewKeyFrames.empty());
    }

    void SurfelMapping::ProcessNewKeyFrame() {
        std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat, int> frame;
        {
            unique_lock<mutex> lock(mMutexNewKFs);
            frame = mlNewKeyFrames.front();
            mlNewKeyFrames.pop_front();
        }

        cv::Mat image = std::get<0>(frame);
        cv::Mat depth = std::get<1>(frame);
        cv::Mat planeMembershipImg = std::get<2>(frame);
        cv::Mat pose = std::get<3>(frame);
        int relativeIndex = std::get<4>(frame);

        PoseElement poseElement;
        int index = posesDatabase.size();
        if (!posesDatabase.empty()) {
            poseElement.linkedPoseIndex.push_back(relativeIndex);
            posesDatabase[relativeIndex].linkedPoseIndex.push_back(index);
        }
        posesDatabase.push_back(poseElement);
        localSurfelsIndexs.insert(index);

        moveAddSurfels(relativeIndex);

        Eigen::Matrix4f poseEigen = Eigen::Matrix4f::Zero();
        poseEigen(0, 0) = pose.at<float>(0, 0);
        poseEigen(1, 0) = pose.at<float>(1, 0);
        poseEigen(2, 0) = pose.at<float>(2, 0);
        poseEigen(3, 0) = pose.at<float>(3, 0);
        poseEigen(0, 1) = pose.at<float>(0, 1);
        poseEigen(1, 1) = pose.at<float>(1, 1);
        poseEigen(2, 1) = pose.at<float>(2, 1);
        poseEigen(3, 1) = pose.at<float>(3, 1);
        poseEigen(0, 2) = pose.at<float>(0, 2);
        poseEigen(1, 2) = pose.at<float>(1, 2);
        poseEigen(2, 2) = pose.at<float>(2, 2);
        poseEigen(3, 2) = pose.at<float>(3, 2);
        poseEigen(0, 3) = pose.at<float>(0, 3);
        poseEigen(1, 3) = pose.at<float>(1, 3);
        poseEigen(2, 3) = pose.at<float>(2, 3);
        poseEigen(3, 3) = pose.at<float>(3, 3);

        fuseMap(image, depth, planeMembershipImg, poseEigen, relativeIndex);
    }

    void SurfelMapping::moveAddSurfels(int referenceIndex) {
        // Remove inactive surfels
        vector<int> posesToAdd;
        vector<int> posesToRemove;
        getAddRemovePoses(referenceIndex, posesToAdd, posesToRemove);

        if (posesToRemove.size() > 0) {
            int addedSurfelNum = 0;
            float sumUpdateTimes = 0.0;
            for (int inactiveIndex : posesToRemove) {
                posesDatabase[inactiveIndex].pointsBeginIndex = mMap->mvInactiveSurfels.size();
                posesDatabase[inactiveIndex].pointsPoseIndex = pointcloudPoseIndex.size();
                pointcloudPoseIndex.push_back(inactiveIndex);
                for (auto &localSurfel : mMap->mvLocalSurfels) {
                    if (localSurfel.updateTimes > 0 && localSurfel.lastUpdate == inactiveIndex) {
                        posesDatabase[inactiveIndex].attachedSurfels.push_back(localSurfel);

                        PointType p;
                        p.x = localSurfel.px;
                        p.y = localSurfel.py;
                        p.z = localSurfel.pz;
//                        p.intensity = localSurfel.color;
                        mMap->mvInactiveSurfels.push_back(localSurfel);

                        addedSurfelNum += 1;
                        sumUpdateTimes += localSurfel.updateTimes;

                        // Delete the surfel from the local point
                        localSurfel.updateTimes = 0;
                    }
                }
                localSurfelsIndexs.erase(inactiveIndex);
            }
            sumUpdateTimes = sumUpdateTimes / addedSurfelNum;
        }
        if (posesToAdd.size() > 0) {
            // Add indexs
            localSurfelsIndexs.insert(posesToAdd.begin(), posesToAdd.end());
            // Add surfels
            // Remove from inactivePointcloud
            std::vector<std::pair<int, int>> removeInfo;  // First, pointcloud start, pointcloud size, pointcloud pose index
            for (int addI = 0; addI < posesToAdd.size(); addI++) {
                int addIndex = posesToAdd[addI];
                int pointsPoseIndex = posesDatabase[addIndex].pointsPoseIndex;
                removeInfo.push_back(std::make_pair(pointsPoseIndex, addIndex));
            }
            std::sort(
                    removeInfo.begin(),
                    removeInfo.end(),
                    [](const std::pair<int, int> &first, const std::pair<int, int> &second) {
                        return first.first < second.first;
                    }
            );
            int removeBeginIndex = removeInfo[0].second;
            int removePointsSize = posesDatabase[removeBeginIndex].attachedSurfels.size();
            int removePoseSize = 1;
            for (int removeI = 1; removeI <= removeInfo.size(); removeI++) {
                bool needRemove = false;
                if (removeI == removeInfo.size())
                    needRemove = true;
                if (removeI < removeInfo.size()) {
                    if (removeInfo[removeI].first != (removeInfo[removeI - 1].first + 1))
                        needRemove = true;
                }
                if (!needRemove) {
                    int thisPoseIndex = removeInfo[removeI].second;
                    removePointsSize += posesDatabase[thisPoseIndex].attachedSurfels.size();
                    removePoseSize += 1;
                    continue;
                }

                int removeEndIndex = removeInfo[removeI - 1].second;

                vector<Surfel>::iterator beginPtr;
                vector<Surfel>::iterator endPtr;
                beginPtr = mMap->mvInactiveSurfels.begin() + posesDatabase[removeBeginIndex].pointsBeginIndex;
                endPtr = beginPtr + removePointsSize;
                mMap->mvInactiveSurfels.erase(beginPtr, endPtr);

                for (int pi = posesDatabase[removeEndIndex].pointsPoseIndex + 1;
                     pi < pointcloudPoseIndex.size(); pi++) {
                    posesDatabase[pointcloudPoseIndex[pi]].pointsBeginIndex -= removePointsSize;
                    posesDatabase[pointcloudPoseIndex[pi]].pointsPoseIndex -= removePoseSize;
                }

                pointcloudPoseIndex.erase(
                        pointcloudPoseIndex.begin() + posesDatabase[removeBeginIndex].pointsPoseIndex,
                        pointcloudPoseIndex.begin() + posesDatabase[removeEndIndex].pointsPoseIndex + 1
                );


                if (removeI < removeInfo.size()) {
                    removeBeginIndex = removeInfo[removeI].second;;
                    removePointsSize = posesDatabase[removeBeginIndex].attachedSurfels.size();
                    removePoseSize = 1;
                }
            }

            // Add the surfels into local
            for (int pi = 0; pi < posesToAdd.size(); pi++) {
                int pose_index = posesToAdd[pi];
                mMap->mvLocalSurfels.insert(
                        mMap->mvLocalSurfels.end(),
                        posesDatabase[pose_index].attachedSurfels.begin(),
                        posesDatabase[pose_index].attachedSurfels.end());
                posesDatabase[pose_index].attachedSurfels.clear();
                posesDatabase[pose_index].pointsBeginIndex = -1;
                posesDatabase[pose_index].pointsPoseIndex = -1;
            }
        }
    }

    void SurfelMapping::getAddRemovePoses(int rootIndex, vector<int> &poseToAdd, vector<int> &poseToRemove) {
        vector<int> driftfreePoses;
        getDriftfreePoses(rootIndex, driftfreePoses, driftFreePoses);
        poseToAdd.clear();
        poseToRemove.clear();
        // Get to add
        for (int i = 0; i < driftfreePoses.size(); i++) {
            int temp_pose = driftfreePoses[i];
            if (localSurfelsIndexs.find(temp_pose) == localSurfelsIndexs.end())
                poseToAdd.push_back(temp_pose);
        }
        // Get to remove
        for (auto i = localSurfelsIndexs.begin(); i != localSurfelsIndexs.end(); i++) {
            int temp_pose = *i;
            if (std::find(driftfreePoses.begin(), driftfreePoses.end(), temp_pose) == driftfreePoses.end()) {
                poseToRemove.push_back(temp_pose);
            }
        }
    }

    void SurfelMapping::getDriftfreePoses(int rootIndex, vector<int> &driftfreePoses, int driftfreeRange) {
        if (posesDatabase.size() < rootIndex + 1) {
            return;
        }
        vector<int> thisLevel;
        vector<int> nextLevel;
        thisLevel.push_back(rootIndex);
        driftfreePoses.push_back(rootIndex);
        // Get the drift
        for (int i = 1; i < driftfreeRange; i++) {
            for (auto thisIt = thisLevel.begin(); thisIt != thisLevel.end(); thisIt++) {
                for (auto linkedIt = posesDatabase[*thisIt].linkedPoseIndex.begin();
                     linkedIt != posesDatabase[*thisIt].linkedPoseIndex.end();
                     linkedIt++) {
                    bool alreadySaved = (find(driftfreePoses.begin(), driftfreePoses.end(), *linkedIt) !=
                                         driftfreePoses.end());
                    if (!alreadySaved) {
                        nextLevel.push_back(*linkedIt);
                        driftfreePoses.push_back(*linkedIt);
                    }
                }
            }
            thisLevel.swap(nextLevel);
            nextLevel.clear();
        }
    }

    void SurfelMapping::fuseMap(cv::Mat image, cv::Mat depth, cv::Mat planeMembershipImg, Eigen::Matrix4f poseInput,
                                int referenceIndex) {
        vector<Surfel> newSurfels;
        mSurfelFusion->fuseInitializeMap(
                referenceIndex,
                image,
                depth,
                planeMembershipImg,
                poseInput,
                mMap->mvLocalSurfels,
                newSurfels
        );

        // Get the deleted surfel index
        vector<int> deletedIndex;
        for (int i = 0; i < mMap->mvLocalSurfels.size(); i++) {
            if (mMap->mvLocalSurfels[i].updateTimes == 0)
                deletedIndex.push_back(i);
        }

        // Add new initialized surfels
        int addSurfelNum = 0;
        for (int i = 0; i < newSurfels.size(); i++) {
            if (newSurfels[i].updateTimes != 0) {
                Surfel this_surfel = newSurfels[i];
                if (deletedIndex.size() > 0) {
                    mMap->mvLocalSurfels[deletedIndex.back()] = this_surfel;
                    deletedIndex.pop_back();
                } else
                    mMap->mvLocalSurfels.push_back(this_surfel);
                addSurfelNum += 1;
            }
        }
        // Remove deleted surfels
        while (deletedIndex.size() > 0) {
            mMap->mvLocalSurfels[deletedIndex.back()] = mMap->mvLocalSurfels.back();
            deletedIndex.pop_back();
            mMap->mvLocalSurfels.pop_back();
        }
    }
}