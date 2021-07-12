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

#ifndef MAPPLANE_H
#define MAPPLANE_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"
#include "Converter.h"

#include <opencv2/core/core.hpp>
#include <mutex>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/exceptions.h>


namespace ORB_SLAM2 {
    class KeyFrame;

    class Frame;

    class Map;

    class MapPlane {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
    public:
        MapPlane(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);

        cv::Mat GetWorldPos();

        void IncreaseVisible(int n = 1);

        void IncreaseFound(int n = 1);

        float GetFoundRatio();

        void SetBadFlag();

        bool isBad();

        void AddObservation(KeyFrame *pKF, int idx);

        void AddParObservation(KeyFrame *pKF, int idx);

        void AddVerObservation(KeyFrame *pKF, int idx);

        void EraseObservation(KeyFrame *pKF);

        void EraseVerObservation(KeyFrame *pKF);

        void EraseParObservation(KeyFrame *pKF);

        std::map<KeyFrame *, size_t> GetObservations();

        int Observations();

        int GetIndexInKeyFrame(KeyFrame *pKF);

        void UpdateCoefficientsAndPoints();

        void UpdateCoefficientsAndPoints(Frame &pF, int id);

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        int nObs;

        static std::mutex mGlobalMutex;

        // Used for visualization
        int mRed;
        int mGreen;
        int mBlue;

        PointCloud::Ptr mvPlanePoints;

        // Tracking counters
        int mnVisible;
        int mnFound;

    protected:
        cv::Mat mWorldPos;

        std::map<KeyFrame *, size_t> mObservations;
        std::map<KeyFrame *, size_t> mParObservations;
        std::map<KeyFrame *, size_t> mVerObservations;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;

        KeyFrame *mpRefKF;

        bool mbBad;

        Map *mpMap;
    };
}
#endif //MAPPLANE_H
