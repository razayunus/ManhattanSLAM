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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>

#include "LSDextractor.h"
#include "MapLine.h"
#include "3DLineExtractor.h"
#include <fstream>

#include "MapPlane.h"
#include "PlaneExtractor.h"

#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/voxel_grid.h>

namespace ORB_SLAM2 {
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    class MapPoint;

    class KeyFrame;

    class MapLine;

    class MapPlane;

    class Frame {

    public:
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        Frame();

        // Copy constructor.
        Frame(const Frame &frame);

        // Constructor for RGB-D cameras.
        Frame(const cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
              ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
              const float &thDepth, const float &depthMapFactor, const float mfDisTh);

        // Extract ORB on the image.
        void ExtractORB(const cv::Mat &im);

        // Extract line features.
        void ExtractLSD(const cv::Mat &im);

        void ExtractPlanes(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat &K, const float &depthMapFactor);

        void GetLineDepth(const cv::Mat &imDepth);

        // Compute Bag of Words representation.
        void ComputeBoW();

        // Set the camera pose.
        void SetPose(cv::Mat Tcw);

        // Computes rotation, translation and camera center matrices from the camera pose.
        void UpdatePoseMatrices();

        // Returns the camera center.
        inline cv::Mat GetCameraCenter() {
            return mOw.clone();
        }

        // Returns inverse of rotation
        inline cv::Mat GetRotationInverse() {
            return mRwc.clone();
        }

        // Check if a MapPoint is in the frustum of the camera
        // and fill variables of the MapPoint to be used by the tracking
        bool isInFrustum(MapPoint *pMP, float viewingCosLimit);

        bool isInFrustum(MapLine *pML, float viewingCosLimit);

        // Compute the cell of a keypoint (return false if outside the grid)
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1,
                                         const int maxLevel = -1) const;

        vector<size_t> GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2,
                                      const float &r, const int minLevel = -1, const int maxLevel = -1) const;

        // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
        void ComputeStereoFromRGBD(const cv::Mat &imDepth);

        // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
        cv::Mat UnprojectStereo(const int &i);

        Vector6d Obtain3DLine(const int &i, const cv::Mat &imDepth);

        cv::Mat ComputePlaneWorldCoeff(const int &idx);

        bool MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud);

    public:
        // Vocabulary used for relocalization.
        ORBVocabulary *mpORBvocabulary;

        // Feature extractor. The right is used only in the stereo case.
        ORBextractor *mpORBextractorLeft;
        // line feature extractor, 自己添加的
        LineSegment *mpLineSegment;
        // Frame timestamp.
        double mTimeStamp;

        // Calibration matrix and OpenCV distortion parameters.
        cv::Mat mK;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;
        cv::Mat mDistCoef;

        // Stereo baseline multiplied by fx.
        float mbf;

        // Stereo baseline in meters.
        float mb;

        // Threshold close/far points. Close points are inserted from 1 view.
        float mThDepth;

        // Number of KeyPoints and KeyLines.
        int N;
        int NL;
        // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
        // In the stereo case, mvKeysUn is redundant as images must be rectified.
        // In the RGB-D case, RGB images can be distorted.
        std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
        std::vector<cv::KeyPoint> mvKeysUn;

        // Corresponding stereo coordinate and depth for each keypoint.
        // "Monocular" keypoints have a negative value.
        std::vector<float> mvuRight;
        std::vector<float> mvDepth;
        std::vector<std::pair<float, float>> mvDepthLine;
        // Bag of Words Vector structures.
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // ORB descriptor, each row associated to a keypoint.
        cv::Mat mDescriptors, mDescriptorsRight;

        // MapPoints associated to keypoints, NULL pointer if no association.
        std::vector<MapPoint *> mvpMapPoints;

        // Flag to identify outlier associations.
        std::vector<bool> mvbOutlier;

        cv::Mat mLdesc;
        std::vector<cv::line_descriptor::KeyLine> mvKeylinesUn;
        std::vector<Eigen::Vector3d> mvKeyLineFunctions;

        std::vector<bool> mvbLineOutlier;
        std::vector<MapLine *> mvpMapLines;

        // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        // Camera pose.
        cv::Mat mTcw;
        cv::Mat mTwc;


        // Current and Next Frame id.
        static long unsigned int nNextId;
        long unsigned int mnId;

        // Reference Keyframe.
        KeyFrame *mpReferenceKF;

        // Scale pyramid info.
        int mnScaleLevels;
        float mfScaleFactor;
        float mfLogScaleFactor;
        std::vector<float> mvScaleFactors;
        std::vector<float> mvInvScaleFactors;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        // Undistorted Image Bounds (computed once).
        static float mnMinX;
        static float mnMaxX;
        static float mnMinY;
        static float mnMaxY;

        static bool mbInitialComputations;

        std::vector<PointCloud> mvPlanePoints;
        std::vector<cv::Mat> mvPlaneCoefficients;
        std::vector<MapPlane *> mvpMapPlanes;
        std::vector<MapPlane *> mvpParallelPlanes;
        std::vector<MapPlane *> mvpVerticalPlanes;
        // Flag to identify outlier planes new planes.
        std::vector<bool> mvbPlaneOutlier;
        std::vector<bool> mvbParPlaneOutlier;
        std::vector<bool> mvbVerPlaneOutlier;
        int mnPlaneNum;
        bool mbNewPlane; // used to determine a keyframe

        PlaneDetection planeDetector;
        float mfDisTh;

    private:

        // Undistort keypoints given OpenCV distortion parameters.
        // Only for the RGB-D case. Stereo must be already rectified!
        // (called in the constructor).
        void UndistortKeyPoints();

        // Computes image bounds for the undistorted image (called in the constructor).
        void ComputeImageBounds(const cv::Mat &imLeft);

        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        void AssignFeaturesToGrid();

        // Rotation, translation and camera center
        cv::Mat mRcw;
        cv::Mat mtcw;
        cv::Mat mRwc;
        cv::Mat mOw; //==mtwc
    };

}// namespace ORB_SLAM

#endif // FRAME_H