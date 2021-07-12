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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"
#include "MapLine.h"
#include <mutex>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

namespace ORB_SLAM2 {

    class Map;

    class MapPoint;

    class Frame;

    class MapPlane;

    class KeyFrameDatabase;

    class MapLine;

    class KeyFrame {
    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);

        // Pose functions
        void SetPose(const cv::Mat &Tcw);

        cv::Mat GetPose();

        cv::Mat GetPoseInverse();

        cv::Mat GetCameraCenter();

        cv::Mat GetRotation();

        cv::Mat GetTranslation();


        // Bag of Words Representation
        void ComputeBoW();

        // Covisibility graph functions
        void AddConnection(KeyFrame *pKF, const int &weight);

        void EraseConnection(KeyFrame *pKF);

        void UpdateConnections();

        void UpdateBestCovisibles();

        std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();

        std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N);

        std::vector<KeyFrame *> GetCovisiblesByWeight(const int &w);

        int GetWeight(KeyFrame *pKF);

        // Spanning tree functions
        void AddChild(KeyFrame *pKF);

        void EraseChild(KeyFrame *pKF);

        void ChangeParent(KeyFrame *pKF);

        std::set<KeyFrame *> GetChilds();

        KeyFrame *GetParent();

        // MapPoint observation functions
        void AddMapPoint(MapPoint *pMP, const size_t &idx);

        void EraseMapPointMatch(const size_t &idx);

        void ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP);

        std::vector<MapPoint *> GetMapPointMatches();

        int TrackedMapPoints(const int &minObs);

        MapPoint *GetMapPoint(const size_t &idx);

        // KeyPoint functions
        std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;

        std::vector<size_t> GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2,
                                           const float &r, const int minLevel = -1, const int maxLevel = -1) const;

        cv::Mat UnprojectStereo(int i);

        // Image
        bool IsInImage(const float &x, const float &y) const;

        // Enable/Disable bad flag changes
        void SetNotErase();

        // Set/check bad flag
        void SetBadFlag();

        bool isBad();

        static bool weightComp(int a, int b) {
            return a > b;
        }

        static bool lId(KeyFrame *pKF1, KeyFrame *pKF2) {
            return pKF1->mnId < pKF2->mnId;
        }


        // MapLine observation functions,自己添加的，仿照MapPoint
        void AddMapLine(MapLine *pML, const size_t &idx);

        void EraseMapLineMatch(const size_t &idx);

        void ReplaceMapLineMatch(const size_t &idx, MapLine *pML);

        std::vector<MapLine *> GetMapLineMatches();

        MapLine *GetMapLine(const size_t &idx);

        // The following variables are accesed from only 1 thread or never change (no mutex needed).
    public:

        static long unsigned int nNextId;
        long unsigned int mnId;

        const double mTimeStamp;

        // Grid (to speed up feature matching)
        const int mnGridCols;
        const int mnGridRows;
        const float mfGridElementWidthInv;
        const float mfGridElementHeightInv;

        // Variables used by the tracking
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnFuseTargetForKF;

        // Variables used by the keyframe database
        long unsigned int mnRelocQuery;
        int mnRelocWords;
        float mRelocScore;

        // Calibration parameters
        const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;


        // Number of KeyPoints
        const int N;
        // KeyPoints, stereo coordinate and descriptors (all associated by an index)
        const std::vector<cv::KeyPoint> mvKeys;
        const std::vector<cv::KeyPoint> mvKeysUn;
        const std::vector<float> mvuRight; // negative value for monocular points
        const std::vector<float> mvDepth; // negative value for monocular points
        const cv::Mat mDescriptors;
        const std::vector<cv::line_descriptor::KeyLine> mvKeyLines;
        const cv::Mat mLineDescriptors;

        //BoW
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // Pose relative to parent (this is computed when bad flag is activated)
        cv::Mat mTcp;

        // Scale
        const int mnScaleLevels;
        const float mfScaleFactor;
        const float mfLogScaleFactor;
        const std::vector<float> mvScaleFactors;
        const std::vector<float> mvLevelSigma2;
        const std::vector<float> mvInvLevelSigma2;

        // Image bounds and calibration
        const int mnMinX;
        const int mnMinY;
        const int mnMaxX;
        const int mnMaxY;
        const cv::Mat mK;

        //For PointCloud
        std::vector<PointCloud> mvPlanePoints;
        std::vector<cv::Mat> mvPlaneCoefficients;
        int mnPlaneNum;
        std::vector<MapPlane *> mvpMapPlanes;
        std::vector<MapPlane *> mvpParallelPlanes;
        std::vector<MapPlane *> mvpVerticalPlanes;

        void AddMapPlane(MapPlane *pMP, const int &idx);

        void EraseMapPlaneMatch(const int &idx);

        void EraseMapVerticalPlaneMatch(const int &idx);

        void EraseMapParallelPlaneMatch(const int &idx);

        std::vector<MapPlane *> GetMapPlaneMatches();

    protected:

        // SE3 Pose and camera center
        cv::Mat Tcw;
        cv::Mat Twc;
        cv::Mat Ow;

        // MapPoints associated to keypoints
        std::vector<MapPoint *> mvpMapPoints;
        // MapLines associated to keylines
        std::vector<MapLine *> mvpMapLines;
        // BoW
        KeyFrameDatabase *mpKeyFrameDB;
        ORBVocabulary *mpORBvocabulary;

        // Grid over the image to speed up feature matching
        std::vector<std::vector<std::vector<size_t> > > mGrid;

        std::map<KeyFrame *, int> mConnectedKeyFrameWeights;
        std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;
        std::vector<int> mvOrderedWeights;

        // Spanning Tree
        bool mbFirstConnection;
        KeyFrame *mpParent;
        std::set<KeyFrame *> mspChildrens;

        // Bad flags
        bool mbNotErase;
        bool mbBad;

        float mHalfBaseline; // Only for visualization

        Map *mpMap;

        std::mutex mMutexPose;
        std::mutex mMutexConnections;
        std::mutex mMutexFeatures;
    };

} //namespace ORB_SLAM

#endif // KEYFRAME_H
