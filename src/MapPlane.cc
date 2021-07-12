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

#include "MapPlane.h"

#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {
    long unsigned int MapPlane::nNextId = 0;
    mutex MapPlane::mGlobalMutex;

    MapPlane::MapPlane(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap) :
            mnFirstKFid(pRefKF->mnId), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
            mvPlanePoints(new PointCloud()), mpMap(pMap), nObs(0),
            mbBad(false) {
        mnId = nNextId++;

        Pos.copyTo(mWorldPos);

        rand();
        mRed = rand() % 256;
        mBlue = rand() % 256;
        mGreen = rand() % 256;
    }

    void MapPlane::AddObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;
        nObs++;
    }

    void MapPlane::AddVerObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mVerObservations.count(pKF))
            return;
        mVerObservations[pKF] = idx;
    }

    void MapPlane::AddParObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mParObservations.count(pKF))
            return;
        mParObservations[pKF] = idx;
    }

    void MapPlane::EraseObservation(KeyFrame *pKF) {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF)) {
                mObservations.erase(pKF);
                nObs--;

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                if (nObs <= 2)
                    bBad = true;
            }
        }

        if (bBad) {
            SetBadFlag();
        }
    }

    void MapPlane::EraseVerObservation(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mVerObservations.count(pKF)) {
            mVerObservations.erase(pKF);
        }
    }

    void MapPlane::EraseParObservation(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mParObservations.count(pKF)) {
            mParObservations.erase(pKF);
        }
    }

    map<KeyFrame *, size_t> MapPlane::GetObservations() {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapPlane::Observations() {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    int MapPlane::GetIndexInKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    cv::Mat MapPlane::GetWorldPos() {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    bool MapPlane::isBad() {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void MapPlane::SetBadFlag() {
        map<KeyFrame *, size_t> obs, verObs, parObs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
            verObs = mVerObservations;
            mVerObservations.clear();
            parObs = mParObservations;
            mParObservations.clear();
        }
        for (auto &ob : obs) {
            KeyFrame *pKF = ob.first;
            pKF->EraseMapPlaneMatch(ob.second);
        }
        for (auto &verOb : verObs) {
            KeyFrame *pKF = verOb.first;
            pKF->EraseMapVerticalPlaneMatch(verOb.second);
        }
        for (auto &parOb : parObs) {
            KeyFrame *pKF = parOb.first;
            pKF->EraseMapParallelPlaneMatch(parOb.second);
        }

        mpMap->EraseMapPlane(this);
    }

    void MapPlane::IncreaseVisible(int n) {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapPlane::IncreaseFound(int n) {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapPlane::GetFoundRatio() {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    void MapPlane::UpdateCoefficientsAndPoints() {
        PointCloud::Ptr combinedPoints(new PointCloud());
        map<KeyFrame *, size_t> observations = GetObservations();
        for (auto &observation : observations) {
            KeyFrame *frame = observation.first;
            int id = observation.second;

            PointCloud::Ptr points(new PointCloud());
            pcl::transformPointCloud(frame->mvPlanePoints[id], *points, Converter::toMatrix4d(frame->GetPoseInverse()));

            *combinedPoints += *points;
        }

        pcl::VoxelGrid<PointT> voxel;
        voxel.setLeafSize(0.2, 0.2, 0.2);

        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(combinedPoints);
        voxel.filter(*coarseCloud);

        mvPlanePoints = coarseCloud;
    }

    void MapPlane::UpdateCoefficientsAndPoints(ORB_SLAM2::Frame &pF, int id) {

        PointCloud::Ptr combinedPoints(new PointCloud());

        Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(pF.mTcw);
        pcl::transformPointCloud(pF.mvPlanePoints[id], *combinedPoints, T.inverse().matrix());

        *combinedPoints += *mvPlanePoints;

        pcl::VoxelGrid<PointT> voxel;
        voxel.setLeafSize(0.2, 0.2, 0.2);

        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(combinedPoints);
        voxel.filter(*coarseCloud);

        mvPlanePoints = coarseCloud;
    }
}