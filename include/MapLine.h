/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/

#ifndef MAPLINE_H
#define MAPLINE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/core/core.hpp>
#include <mutex>
#include <eigen3/Eigen/Core>
#include <map>

namespace ORB_SLAM2 {
    class KeyFrame;

    class Map;

    class Frame;

    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    class MapLine {
    public:
        MapLine(Vector6d &Pos, KeyFrame *pRefKF, Map *pMap);

        MapLine(Vector6d &Pos, Map *pMap, Frame *pFrame, const int &idxF);

        Vector6d GetWorldPos();

        Eigen::Vector3d GetNormal();

        int Observations();


        void AddObservation(KeyFrame *pKF, size_t idx);

        void EraseObservation(KeyFrame *pKF);

        bool IsInKeyFrame(KeyFrame *pKF);

        void SetBadFlag();

        bool isBad();

        void Replace(MapLine *pML);

        MapLine *GetReplaced();

        void IncreaseVisible(int n = 1);

        void IncreaseFound(int n = 1);

        float GetFoundRatio();

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateAverageDir();

        float GetMinDistanceInvariance();

        float GetMaxDistanceInvariance();

        int PredictScale(const float &currentDist, const float &logScaleFactor);

    public:
        long unsigned int mnId; //Global ID for MapLine
        static long unsigned int nNextId;
        const long int mnFirstKFid;
        int nObs;

        // Variables used by the tracking
        float mTrackProjX1;
        float mTrackProjY1;
        float mTrackProjX2;
        float mTrackProjY2;
        int mnTrackScaleLevel;
        float mTrackViewCos;
        bool mbTrackInView;

        long unsigned int mnTrackReferenceForFrame;

        long unsigned int mnLastFrameSeen;

        // Variables used by local mapping
        long unsigned int mnFuseCandidateForKF;

        static std::mutex mGlobalMutex;

    public:
        Vector6d mWorldPos;
        Eigen::Vector3d mStart3D;
        Eigen::Vector3d mEnd3D;

        // KeyFrames observing the line and associated index in keyframe
        std::map<KeyFrame *, size_t> mObservations;

        Eigen::Vector3d mNormalVector;

        cv::Mat mLDescriptor;

        KeyFrame *mpRefKF;

        //Tracking counters
        int mnVisible;
        int mnFound;

        // Bad flag , we don't currently erase MapPoint from memory
        bool mbBad;
        MapLine *mpReplaced;

        float mfMinDistance;
        float mfMaxDistance;

        Map *mpMap;

        mutex mMutexPos;
        mutex mMutexFeatures;
    };

}


#endif //MAPLINE_H
