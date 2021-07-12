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

#include "Frame.h"
#include "Converter.h"

#include <thread>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame() {}

//Copy Constructor
    Frame::Frame(const Frame &frame)
            : mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
              mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
              mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
              mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
              mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
              mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
              mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
              mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
              mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
              mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
              mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
              mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn),
              mvpMapLines(frame.mvpMapLines),
              mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions),
              mvDepthLine(frame.mvDepthLine), mvPlaneCoefficients(frame.mvPlaneCoefficients),
              mbNewPlane(frame.mbNewPlane),
              mvpMapPlanes(frame.mvpMapPlanes), mnPlaneNum(frame.mnPlaneNum), mvbPlaneOutlier(frame.mvbPlaneOutlier),
              mvpParallelPlanes(frame.mvpParallelPlanes), mvpVerticalPlanes(frame.mvpVerticalPlanes),
              mvPlanePoints(frame.mvPlanePoints), mfDisTh(frame.mfDisTh) {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }

    Frame::Frame(const cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
                 ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth, const float &depthMapFactor, const float mfDisTh)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
              mfDisTh(mfDisTh) {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        cv::Mat imDepthScaled;
        if (depthMapFactor != 1 || imDepth.type() != CV_32F) {
            imDepth.convertTo(imDepthScaled, CV_32F, depthMapFactor);
        }

        thread threadPoints(&ORB_SLAM2::Frame::ExtractORB, this, imGray);
        thread threadLines(&ORB_SLAM2::Frame::ExtractLSD, this, imGray);
        thread threadPlanes(&ORB_SLAM2::Frame::ExtractPlanes, this, imRGB, imDepth, K, depthMapFactor);
        threadPoints.join();
        threadLines.join();
        threadPlanes.join();

        N = mvKeys.size();
        NL = mvKeylinesUn.size();

        mnPlaneNum = mvPlanePoints.size();
        mvpMapPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpParallelPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpVerticalPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvbPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbVerPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbParPlaneOutlier = vector<bool>(mnPlaneNum, false);

        GetLineDepth(imDepthScaled);

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoFromRGBD(imDepthScaled);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvpMapLines = vector<MapLine *>(NL, static_cast<MapLine *>(NULL));
        mvbOutlier = vector<bool>(N, false);
        mvbLineOutlier = vector<bool>(NL, false);


        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    void Frame::AssignFeaturesToGrid() {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractLSD(const cv::Mat &im) {
        mpLineSegment->ExtractLineSegment(im, mvKeylinesUn, mLdesc, mvKeyLineFunctions);

    }

    void Frame::ExtractORB(const cv::Mat &im) {
        (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
    }

    void Frame::GetLineDepth(const cv::Mat &imDepth) {
        mvDepthLine = std::vector<std::pair<float, float>>(mvKeylinesUn.size(), make_pair(-1.0f, -1.0f));

        for (int i = 0; i < mvKeylinesUn.size(); ++i) {
            mvDepthLine[i] = std::make_pair(imDepth.at<float>(mvKeylinesUn[i].startPointY, mvKeylinesUn[i].startPointX),
                                            imDepth.at<float>(mvKeylinesUn[i].endPointY, mvKeylinesUn[i].endPointX));
        }
    }

    void Frame::SetPose(cv::Mat Tcw) {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices() {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;

        mTwc = cv::Mat::eye(4, 4, mTcw.type());
        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));
    }

    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - mOw;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit) {
        pML->mbTrackInView = false;

        Vector6d P = pML->GetWorldPos();

        cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
        cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

        const cv::Mat SPc = mRcw * SP + mtcw;
        const float &SPcX = SPc.at<float>(0);
        const float &SPcY = SPc.at<float>(1);
        const float &SPcZ = SPc.at<float>(2);

        const cv::Mat EPc = mRcw * EP + mtcw;
        const float &EPcX = EPc.at<float>(0);
        const float &EPcY = EPc.at<float>(1);
        const float &EPcZ = EPc.at<float>(2);

        if (SPcZ < 0.0f || EPcZ < 0.0f)
            return false;

        const float invz1 = 1.0f / SPcZ;
        const float u1 = fx * SPcX * invz1 + cx;
        const float v1 = fy * SPcY * invz1 + cy;

        if (u1 < mnMinX || u1 > mnMaxX)
            return false;
        if (v1 < mnMinY || v1 > mnMaxY)
            return false;

        const float invz2 = 1.0f / EPcZ;
        const float u2 = fx * EPcX * invz2 + cx;
        const float v2 = fy * EPcY * invz2 + cy;

        if (u2 < mnMinX || u2 > mnMaxX)
            return false;
        if (v2 < mnMinY || v2 > mnMaxY)
            return false;


        const float maxDistance = pML->GetMaxDistanceInvariance();
        const float minDistance = pML->GetMinDistanceInvariance();

        const cv::Mat OM = 0.5 * (SP + EP) - mOw;
        const float dist = cv::norm(OM);

        if (dist < minDistance || dist > maxDistance)
            return false;


        Vector3d Pn = pML->GetNormal();
        cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));
        const float viewCos = OM.dot(pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

        pML->mbTrackInView = true;
        pML->mTrackProjX1 = u1;
        pML->mTrackProjY1 = v1;
        pML->mTrackProjX2 = u2;
        pML->mTrackProjY2 = v2;
        pML->mnTrackScaleLevel = nPredictedLevel;
        pML->mTrackViewCos = viewCos;

        return true;
    }


    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels) {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    vector<size_t>
    Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                          const int minLevel, const int maxLevel) const {
        vector<size_t> vIndices;

        vector<KeyLine> vkl = this->mvKeylinesUn;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

        for (size_t i = 0; i < vkl.size(); i++) {
            KeyLine keyline = vkl[i];

            float distance = (0.5 * (x1 + x2) - keyline.pt.x) * (0.5 * (x1 + x2) - keyline.pt.x) +
                             (0.5 * (y1 + y2) - keyline.pt.y) * (0.5 * (y1 + y2) - keyline.pt.y);
            if (distance > r * r)
                continue;

            float slope = (y1 - y2) / (x1 - x2) - keyline.angle;
            if (slope > r * 0.01)
                continue;

            if (bCheckLevels) {
                if (keyline.octave < minLevel)
                    continue;
                if (maxLevel >= 0 && keyline.octave > maxLevel)
                    continue;
            }

            vIndices.push_back(i);
        }

        return vIndices;
    }


    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }


    void Frame::ComputeBoW() {
        if (mBowVec.empty()) {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints() {
        if (mDistCoef.at<float>(0) == 0.0) {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
        if (mDistCoef.at<float>(0) != 0.0) {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

        } else {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0) {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i) {
        const float z = mvDepth[i];
        if (z > 0) {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return cv::Mat();
    }

    Vector6d Frame::Obtain3DLine(const int &i, const cv::Mat &imDepth) {
        double len = cv::norm(mvKeylinesUn[i].getStartPoint() - mvKeylinesUn[i].getEndPoint());

        vector<cv::Point3d> pts3d;
        // iterate through a line
        double numSmp = (double) min((int) len, 100); //number of line points sampled

        pts3d.reserve(numSmp);

        for (int j = 0; j <= numSmp; ++j) {
            // use nearest neighbor to querry depth value
            // assuming position (0,0) is the top-left corner of image, then the
            // top-left pixel's center would be (0.5,0.5)
            cv::Point2d pt = mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
                             mvKeylinesUn[i].getEndPoint() * (j / numSmp);
            if (pt.x < 0 || pt.y < 0 || pt.x >= imDepth.cols || pt.y >= imDepth.rows) continue;
            int row, col; // nearest pixel for pt
            if ((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) { // boundary issue
                col = max(int(pt.x - 1), 0);
                row = max(int(pt.y - 1), 0);
            } else {
                col = int(pt.x);
                row = int(pt.y);
            }

            float d = -1;
            if (imDepth.at<float>(row, col) <= 0.01) { // no depth info
                continue;
            } else {
                d = imDepth.at<float>(row, col);
            }
            cv::Point3d p;

            p.z = d;
            p.x = (col - cx) * p.z * invfx;
            p.y = (row - cy) * p.z * invfy;

            pts3d.push_back(p);

        }

        if (pts3d.size() < 10.0)
            return static_cast<Vector6d>(NULL);

        RandomLine3d tmpLine;
        vector<RandomPoint3d> rndpts3d;
        rndpts3d.reserve(pts3d.size());

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                0, fy, cy,
                0, 0, 1);

        // compute uncertainty of 3d points
        for (auto &j : pts3d) {
            rndpts3d.push_back(compPt3dCov(j, K, 1));
        }
        // using ransac to extract a 3d line from 3d pts
        tmpLine = extract3dline_mahdist(rndpts3d);

        if (tmpLine.pts.size() / len > 0.4 && cv::norm(tmpLine.A - tmpLine.B) > 0.02) {
            //this line is reliable

            Vector6d line3D;
            line3D << tmpLine.A.x, tmpLine.A.y, tmpLine.A.z, tmpLine.B.x, tmpLine.B.y, tmpLine.B.z;

            cv::Mat Ac = (Mat_<float>(3, 1) << line3D(0), line3D(1), line3D(2));
            cv::Mat A = mRwc * Ac + mOw;
            cv::Mat Bc = (Mat_<float>(3, 1) << line3D(3), line3D(4), line3D(5));
            cv::Mat B = mRwc * Bc + mOw;
            line3D << A.at<float>(0, 0), A.at<float>(1, 0), A.at<float>(2, 0),
                    B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0);
            return line3D;
        } else {
            return static_cast<Vector6d>(NULL);
        }
    }

    void
    Frame::ExtractPlanes(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat &K, const float &depthMapFactor) {
        planeDetector.readColorImage(imRGB);
        planeDetector.readDepthImage(imDepth, K, depthMapFactor);
        planeDetector.runPlaneDetection();

        for (int i = 0; i < planeDetector.plane_num_; i++) {
            auto &indices = planeDetector.plane_vertices_[i];
            PointCloud::Ptr inputCloud(new PointCloud());
            for (int j : indices) {
                PointT p;
                p.x = (float) planeDetector.cloud.vertices[j][0];
                p.y = (float) planeDetector.cloud.vertices[j][1];
                p.z = (float) planeDetector.cloud.vertices[j][2];
                p.r = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][0]);
                p.g = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][1]);
                p.b = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][2]);

                inputCloud->points.push_back(p);
            }

            auto extractedPlane = planeDetector.plane_filter.extractedPlanes[i];
            double nx = extractedPlane->normal[0];
            double ny = extractedPlane->normal[1];
            double nz = extractedPlane->normal[2];
            double cx = extractedPlane->center[0];
            double cy = extractedPlane->center[1];
            double cz = extractedPlane->center[2];

            float d = (float) -(nx * cx + ny * cy + nz * cz);

            pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize(0.2, 0.2, 0.2);

            PointCloud::Ptr coarseCloud(new PointCloud());
            voxel.setInputCloud(inputCloud);
            voxel.filter(*coarseCloud);

            cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);

            bool valid = MaxPointDistanceFromPlane(coef, coarseCloud);

            if (!valid) {
                continue;
            }

            mvPlanePoints.push_back(*coarseCloud);
            mvPlaneCoefficients.push_back(coef);
        }
    }

    cv::Mat Frame::ComputePlaneWorldCoeff(const int &idx) {
        cv::Mat temp;
        cv::transpose(mTcw, temp);
        return temp * mvPlaneCoefficients[idx];
    }

    bool Frame::MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud) {
        bool erased = false;
        double threshold = 0.04;
        int i = 0;
        auto &points = pointCloud->points;

        for (auto &p : points) {
            double absDis = abs(plane.at<float>(0) * p.x +
                                plane.at<float>(1) * p.y +
                                plane.at<float>(2) * p.z +
                                plane.at<float>(3));

            if (absDis > mfDisTh)
                return false;

            i++;
        }

        if (points.size() < 4) {
            return false;
        }
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(mfDisTh);

        seg.setInputCloud(pointCloud);
        seg.segment(*inliers, *coefficients);

        float oldVal = plane.at<float>(3);
        float newVal = coefficients->values[3];

        cv::Mat oldPlane = plane.clone();

        plane.at<float>(0) = coefficients->values[0];
        plane.at<float>(1) = coefficients->values[1];
        plane.at<float>(2) = coefficients->values[2];
        plane.at<float>(3) = coefficients->values[3];

        if ((newVal < 0 && oldVal > 0) || (newVal > 0 && oldVal < 0)) {
            plane = -plane;
        }

        return true;
    }
} //namespace ORB_SLAM
