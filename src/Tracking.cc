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


#include "Tracking.h"

#include "ORBmatcher.h"

#include "Optimizer.h"
#include "PnPsolver.h"


using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                       KeyFrameDatabase *pKFDB, const string &strSettingPath) :
            mState(NO_IMAGES_YET), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
            mpKeyFrameDB(pKFDB), mpSystem(pSys), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer),
            mpMap(pMap), mnLastRelocFrameId(0) {
// Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        int img_width = fSettings["Camera.width"];
        int img_height = fSettings["Camera.height"];

        cout << "img_width = " << img_width << endl;
        cout << "img_height = " << img_height << endl;

        initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3, 3), mK, Size(img_width, img_height), CV_32F,
                                mUndistX, mUndistY);

        cout << "mUndistX size = " << mUndistX.size << endl;
        cout << "mUndistY size = " << mUndistY.size << endl;

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

// Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

// Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;

        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (fabs(mDepthMapFactor) < 1e-5)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;

        // Load plane parameters

        float mfDThRef = fSettings["Plane.AssociationDisRef"];
        float mfAThRef = fSettings["Plane.AssociationAngRef"];

        float mfVerTh = fSettings["Plane.VerticalThreshold"];
        float mfParTh = fSettings["Plane.ParallelThreshold"];

        mfMFVerTh = fSettings["Plane.MFVerticalThreshold"];
        mfDisTh = fSettings["Plane.DistanceThreshold"];

        fullManhattanFound = false;

        // Initialize matchers
        mpLineMatcher = new LSDmatcher();
        mpPlaneMatcher = new PlaneMatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Load optimiter parameters

        double angleInfo = fSettings["Plane.AngleInfo"];
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = fSettings["Plane.DistanceInfo"];
        disInfo = disInfo * disInfo;
        double parInfo = fSettings["Plane.ParallelInfo"];
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = fSettings["Plane.VerticalInfo"];
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = fSettings["Plane.Chi"];
        double planeChiVP = fSettings["Plane.VPChi"];

        mpOptimizer = new Optimizer(angleInfo, disInfo, parInfo, verInfo, planeChi, planeChiVP, mfAThRef, mfParTh);
    }


    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetSurfelMapper(SurfelMapping *pSurfelMapper) {
        mpSurfelMapper = pSurfelMapper;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }

    cv::Mat Tracking::GrabImage(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp) {
        mImRGB = imRGB;
        mImGray = imRGB;
        mImDepth = imD;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        mCurrentFrame = Frame(mImRGB, mImGray, mImDepth, timestamp, mpORBextractor, mpORBVocabulary, mK,
                              mDistCoef, mbf, mThDepth, mDepthMapFactor, mfDisTh);

        if (mDepthMapFactor != 1 || mImDepth.type() != CV_32F) {
            mImDepth.convertTo(mImDepth, CV_32F, mDepthMapFactor);
        }

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track() {

        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED) {
            StereoInitialization();
            mpSurfelMapper->InsertKeyFrame(mImGray.clone(), mImDepth.clone(),
                                           mCurrentFrame.planeDetector.plane_filter.membershipImg.clone(),
                                           mCurrentFrame.mTwc.clone(), 0);

            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        } else {
            bool bOK = false;
            bool bManhattan = false;
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking) {

                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                } else {
                    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                }

                mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

                bManhattan = DetectManhattan();

                if (bManhattan) {
                    // Translation (only) estimation
                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        bOK = TranslationEstimation();
                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                        }
                    } else {
                        bOK = TranslationWithMotionModel();

                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                            bOK = TranslationEstimation();
                            if (!bOK) {
                                mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                            }
                        }
                    }
                }

                if (!bOK) {
                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        bOK = TrackReferenceKeyFrame();
                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                        }
                    } else {
                        bOK = TrackWithMotionModel();
                        if (!bOK) {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                            bOK = TrackReferenceKeyFrame();
                            if (!bOK) {
                                mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                            }
                        }
                    }
                }
            } else {
                // Localization Mode: Local Mapping is deactivated

                if (mState == LOST) {
                    bOK = Relocalization();
                } else {
                    if (!mbVO) {
                        // In last frame we tracked enough MapPoints in the map

                        if (!mVelocity.empty()) {
                            mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                        } else {
                            mCurrentFrame.SetPose(mLastFrame.mTcw);
                        }

                        mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

                        bManhattan = DetectManhattan();

                        if (bManhattan) {
                            // Translation (only) estimation
                            if (!mVelocity.empty()) {
                                bOK = TranslationWithMotionModel();

                                if (!bOK) {
                                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                                    bOK = TranslationEstimation();
                                    if (!bOK) {
                                        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                                    }
                                }
                            } else {
                                bOK = TranslationEstimation();
                                if (!bOK) {
                                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                                }
                            }
                        }

                        if (!bOK) {
                            if (!mVelocity.empty()) {
                                bOK = TrackWithMotionModel();
                                if (!bOK) {
                                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                                    bOK = TrackReferenceKeyFrame();
                                    if (!bOK) {
                                        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
                                    }
                                }
                            } else {
                                bOK = TrackReferenceKeyFrame();
                                if (!bOK) {
                                    mCurrentFrame.SetPose(mLastFrame.mTcw);
                                }
                            }
                        }
                    } else {
                        // In last frame we tracked mainly "visual odometry" points.

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint *> vpMPsMM;
                        vector<bool> vbOutMM;
                        cv::Mat TcwMM;
                        if (!mVelocity.empty()) {
                            mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

                            mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

                            bManhattan = DetectManhattan();

                            if (bManhattan) {
                                bOKMM = TranslationWithMotionModel();
                            }
                            if (!bOKMM) {
                                bOKMM = TrackWithMotionModel();
                            }

                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.mTcw.clone();
                        }
                        bOKReloc = Relocalization();

                        if (bOKMM && !bOKReloc) {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;

                            if (mbVO) {
                                for (int i = 0; i < mCurrentFrame.N; i++) {
                                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        } else if (bOKReloc) {
                            mbVO = false;
                        }

                        bOK = bOKReloc || bOKMM;
                    }
                }
            }

            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if (!mbOnlyTracking) {
                if (bOK) {
                    bOK = TrackLocalMap();
                } else {
                    bOK = Relocalization();
                }
            } else {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if (bOK && !mbVO)
                    bOK = TrackLocalMap();
            }

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            // Update drawer
            mpFrameDrawer->Update(this);

            //Update Planes
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                if (pMP) {
                    pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
                } else if (!mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mbNewPlane = true;
                }
            }

            if (bOK) {
                // Update motion model
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else
                    mVelocity = cv::Mat();

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    if (pML)
                        if (pML->Observations() < 1) {
                            mCurrentFrame.mvbLineOutlier[i] = false;
                            mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                for (list<MapLine *>::iterator lit = mlpTemporalLines.begin(), lend = mlpTemporalLines.end();
                     lit != lend; lit++) {
                    MapLine *pML = *lit;
                    delete pML;
                }
                mlpTemporalPoints.clear();
                mlpTemporalLines.clear();

                // Check if we need to insert a new keyframe
                if (NeedNewKeyFrame()) {
                    CreateNewKeyFrame();

                    int referenceIndex = 0;
                    double timeDiff = 1e9;
                    vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
                    for (int i = 0; i < vpKFs.size(); i++) {
                        double diff = fabs(vpKFs[i]->mTimeStamp - mpReferenceKF->mTimeStamp);
                        if (diff < timeDiff) {
                            referenceIndex = i;
                            timeDiff = diff;
                        }
                    }

                    mpSurfelMapper->InsertKeyFrame(mImGray.clone(), mImDepth.clone(),
                                                   mCurrentFrame.planeDetector.plane_filter.membershipImg.clone(),
                                                   mCurrentFrame.mTwc.clone(), referenceIndex);
                }

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    if (mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if (mState == LOST) {
                if (mpMap->KeyFramesInMap() <= 5) {
                    cout << "Track lost soon after initialisation, reseting..." << endl;
                    mpSystem->Reset();
                    return;
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (!mCurrentFrame.mTcw.empty()) {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

    }

    void Tracking::StereoInitialization() {
        if (mCurrentFrame.N > 50 || mCurrentFrame.NL > 15) {
            // Set Frame pose to the origin
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }


            for (int i = 0; i < mCurrentFrame.NL; i++) {

                pair<float, float> z = mCurrentFrame.mvDepthLine[i];

                if (z.first > 0 && z.second > 0) {
                    Vector6d line3D = mCurrentFrame.Obtain3DLine(i, mImDepth);
                    if (line3D == static_cast<Vector6d>(NULL)) {
                        continue;
                    }
                    MapLine *pNewML = new MapLine(line3D, pKFini, mpMap);
                    pNewML->AddObservation(pKFini, i);
                    pKFini->AddMapLine(pNewML, i);
                    pNewML->ComputeDistinctiveDescriptors();
                    pNewML->UpdateAverageDir();
                    mpMap->AddMapLine(pNewML);
                    mCurrentFrame.mvpMapLines[i] = pNewML;
                }
            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                MapPlane *pNewMP = new MapPlane(p3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
                mpMap->AddMapPlane(pNewMP);
                mCurrentFrame.mvpMapPlanes[i] = pNewMP;
            }

            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mvpLocalMapLines = mpMap->GetAllMapLines();

            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
            mpMap->SetReferenceMapLines(mvpLocalMapLines);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
        }
    }

    void Tracking::CheckReplacedInLastFrame() {
        for (int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP) {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }

        for (int i = 0; i < mLastFrame.NL; i++) {
            MapLine *pML = mLastFrame.mvpMapLines[i];

            if (pML) {
                MapLine *pReL = pML->GetReplaced();
                if (pReL) {
                    mLastFrame.mvpMapLines[i] = pReL;
                }
            }
        }
    }

    bool Tracking::DetectManhattan() {
        KeyFrame *pKFCandidate = nullptr;
        int maxScore = 0;
        cv::Mat pMFc1, pMFc2, pMFc3, pMFm1, pMFm2, pMFm3;
        fullManhattanFound = false;

        int id1, id2, id3 = -1;

        for (size_t i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            cv::Mat p3Dc1 = mCurrentFrame.mvPlaneCoefficients[i];
            MapPlane *pMP1 = mCurrentFrame.mvpMapPlanes[i];

            if (!pMP1 || pMP1->isBad()) {
                continue;
            }

            for (size_t j = i + 1; j < mCurrentFrame.mnPlaneNum; j++) {
                cv::Mat p3Dc2 = mCurrentFrame.mvPlaneCoefficients[j];
                MapPlane *pMP2 = mCurrentFrame.mvpMapPlanes[j];

                if (!pMP2 || pMP2->isBad()) {
                    continue;
                }

                float angle12 = p3Dc1.at<float>(0) * p3Dc2.at<float>(0) +
                                p3Dc1.at<float>(1) * p3Dc2.at<float>(1) +
                                p3Dc1.at<float>(2) * p3Dc2.at<float>(2);

                if (angle12 > mfMFVerTh || angle12 < -mfMFVerTh) {
                    continue;
                }

                for (size_t k = j + 1; k < mCurrentFrame.mnPlaneNum; k++) {
                    cv::Mat p3Dc3 = mCurrentFrame.mvPlaneCoefficients[k];
                    MapPlane *pMP3 = mCurrentFrame.mvpMapPlanes[k];

                    if (!pMP3 || pMP3->isBad()) {
                        continue;
                    }

                    float angle13 = p3Dc1.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc1.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc1.at<float>(2) * p3Dc3.at<float>(2);

                    float angle23 = p3Dc2.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc2.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc2.at<float>(2) * p3Dc3.at<float>(2);

                    if (angle13 > mfMFVerTh || angle13 < -mfMFVerTh || angle23 > mfMFVerTh || angle23 < -mfMFVerTh) {
                        continue;
                    }

                    KeyFrame *pKF = mpMap->GetManhattanObservation(pMP1, pMP2, pMP3);

                    if (!pKF) {
                        continue;
                    }

                    auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                    auto idx2 = pMP2->GetIndexInKeyFrame(pKF);
                    auto idx3 = pMP3->GetIndexInKeyFrame(pKF);

                    if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
                        continue;
                    }

                    int score = pKF->mvPlanePoints[idx1].size() +
                                pKF->mvPlanePoints[idx2].size() +
                                pKF->mvPlanePoints[idx3].size() +
                                mCurrentFrame.mvPlanePoints[i].size() +
                                mCurrentFrame.mvPlanePoints[j].size() +
                                mCurrentFrame.mvPlanePoints[k].size();

                    if (score > maxScore) {
                        maxScore = score;

                        pKFCandidate = pKF;
                        pMFc1 = p3Dc1;
                        pMFc2 = p3Dc2;
                        pMFc3 = p3Dc3;
                        pMFm1 = pKF->mvPlaneCoefficients[idx1];
                        pMFm2 = pKF->mvPlaneCoefficients[idx2];
                        pMFm3 = pKF->mvPlaneCoefficients[idx3];

                        id1 = pMP1->mnId;
                        id2 = pMP2->mnId;
                        id3 = pMP3->mnId;

                        fullManhattanFound = true;
                    }
                }

                KeyFrame *pKF = mpMap->GetPartialManhattanObservation(pMP1, pMP2);

                if (!pKF) {
                    continue;
                }

                auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                auto idx2 = pMP2->GetIndexInKeyFrame(pKF);

                if (idx1 == -1 || idx2 == -1) {
                    continue;
                }

                int score = pKF->mvPlanePoints[idx1].size() +
                            pKF->mvPlanePoints[idx2].size() +
                            mCurrentFrame.mvPlanePoints[i].size() +
                            mCurrentFrame.mvPlanePoints[j].size();

                if (score > maxScore) {
                    maxScore = score;

                    pKFCandidate = pKF;
                    pMFc1 = p3Dc1;
                    pMFc2 = p3Dc2;
                    pMFm1 = pKF->mvPlaneCoefficients[idx1];
                    pMFm2 = pKF->mvPlaneCoefficients[idx2];

                    id1 = pMP1->mnId;
                    id2 = pMP2->mnId;

                    fullManhattanFound = false;
                }
            }
        }

        if (pKFCandidate == nullptr) {
            return false;
        }

        if (!fullManhattanFound) {
            cv::Mat pMFc1n = (cv::Mat_<float>(3, 1) << pMFc1.at<float>(0), pMFc1.at<float>(1), pMFc1.at<float>(2));
            cv::Mat pMFc2n = (cv::Mat_<float>(3, 1) << pMFc2.at<float>(0), pMFc2.at<float>(1), pMFc2.at<float>(2));
            pMFc3 = pMFc1n.cross(pMFc2n);

            cv::Mat pMFm1n = (cv::Mat_<float>(3, 1) << pMFm1.at<float>(0), pMFm1.at<float>(1), pMFm1.at<float>(2));
            cv::Mat pMFm2n = (cv::Mat_<float>(3, 1) << pMFm2.at<float>(0), pMFm2.at<float>(1), pMFm2.at<float>(2));
            pMFm3 = pMFm1n.cross(pMFm2n);
        }

        cv::Mat MFc, MFm;
        MFc = cv::Mat::eye(cv::Size(3, 3), CV_32F);
        MFm = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        MFc.at<float>(0, 0) = pMFc1.at<float>(0);
        MFc.at<float>(1, 0) = pMFc1.at<float>(1);
        MFc.at<float>(2, 0) = pMFc1.at<float>(2);
        MFc.at<float>(0, 1) = pMFc2.at<float>(0);
        MFc.at<float>(1, 1) = pMFc2.at<float>(1);
        MFc.at<float>(2, 1) = pMFc2.at<float>(2);
        MFc.at<float>(0, 2) = pMFc3.at<float>(0);
        MFc.at<float>(1, 2) = pMFc3.at<float>(1);
        MFc.at<float>(2, 2) = pMFc3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFc) + 1) < 0.5) {
            MFc.at<float>(0, 2) = -pMFc3.at<float>(0);
            MFc.at<float>(1, 2) = -pMFc3.at<float>(1);
            MFc.at<float>(2, 2) = -pMFc3.at<float>(2);
        }

        cv::Mat Uc, Wc, VTc;

        cv::SVD::compute(MFc, Wc, Uc, VTc);

        MFc = Uc * VTc;

        MFm.at<float>(0, 0) = pMFm1.at<float>(0);
        MFm.at<float>(1, 0) = pMFm1.at<float>(1);
        MFm.at<float>(2, 0) = pMFm1.at<float>(2);
        MFm.at<float>(0, 1) = pMFm2.at<float>(0);
        MFm.at<float>(1, 1) = pMFm2.at<float>(1);
        MFm.at<float>(2, 1) = pMFm2.at<float>(2);
        MFm.at<float>(0, 2) = pMFm3.at<float>(0);
        MFm.at<float>(1, 2) = pMFm3.at<float>(1);
        MFm.at<float>(2, 2) = pMFm3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFm) + 1) < 0.5) {
            MFm.at<float>(0, 2) = -pMFm3.at<float>(0);
            MFm.at<float>(1, 2) = -pMFm3.at<float>(1);
            MFm.at<float>(2, 2) = -pMFm3.at<float>(2);
        }

        cv::Mat Um, Wm, VTm;

        cv::SVD::compute(MFm, Wm, Um, VTm);

        MFm = Um * VTm;

        cv::Mat Rwc = pKFCandidate->GetPoseInverse().rowRange(0, 3).colRange(0, 3) * MFm * MFc.t();
        manhattanRcw = Rwc.t();

        return true;
    }

    bool Tracking::TranslationEstimation() {

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = mpLineMatcher->SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        int planeMatches = mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        manhattanRcw.copyTo(mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3));

        mpOptimizer->TranslationOptimization(&mCurrentFrame);

        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        // Discard outliers
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;

                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;

            }
        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (nmatchesMap < 7) {
            return false;
        }

        return true;
    }

    bool Tracking::TranslationWithMotionModel() {
        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        // Project points seen in previous frame
        int th = 15;
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th);

        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        int lmatches = mpLineMatcher->SearchByProjection(mCurrentFrame, mLastFrame, th);

        if (nmatches < 40) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 4 * th);
        }

        int planeMatches = mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            return false;
        }

        manhattanRcw.copyTo(mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3));

        // Optimize frame pose with all matches
        mpOptimizer->TranslationOptimization(&mCurrentFrame);

        // Discard outliers
        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (nmatchesMap < 7) {
            return false;
        }

        return true;
    }

    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose());

        if (mnLastKeyFrameId == mLastFrame.mnId || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                bCreateNew = true;
            }

            if (bCreateNew) {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }


        // Create "visual odometry" MapLines
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vLineDepthIdx;
        vLineDepthIdx.reserve(mLastFrame.NL);
        int nLines = 0;
        for (int i = 0; i < mLastFrame.NL; i++) {
            pair<float, float> z = mLastFrame.mvDepthLine[i];

            if (z.first > 0 && z.second > 0) {
                bool bCreateNew = false;
                vLineDepthIdx.push_back(make_pair(min(z.first, z.second), i));
                MapLine *pML = mLastFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1) {
                    bCreateNew = true;
                }
                if (bCreateNew) {
                    Vector6d line3D = mLastFrame.Obtain3DLine(i, mImDepth);
                    if (line3D == static_cast<Vector6d>(NULL)) {
                        continue;
                    }
                    MapLine *pNewML = new MapLine(line3D, mpMap, &mLastFrame, i);

                    mLastFrame.mvpMapLines[i] = pNewML;

                    mlpTemporalLines.push_back(pNewML);
                    nLines++;
                } else {
                    nLines++;
                }

                if (nLines > 30)
                    break;

            }
        }
    }

    bool Tracking::TrackReferenceKeyFrame() {

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();
        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = mpLineMatcher->SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        int planeMatches = mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        float initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            return false;
        }

        mCurrentFrame.mvpMapLines = vpMapLineMatches;
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;

        mpOptimizer->PoseOptimization(&mCurrentFrame);

        // Discard outliers

        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;

            }
        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                    nDiscardPlane++;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        if (finalMatches < 7) {
            return false;
        }

        return true;
    }

    bool Tracking::TrackWithMotionModel() {
        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        // Project points seen in previous frame
        int th = 15;
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th);

        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        int lmatches = mpLineMatcher->SearchByProjection(mCurrentFrame, mLastFrame, th);

        // If few matches, uses a wider window search
        if (nmatches < 40) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
                 static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 4 * th);
        }

        int planeMatches = mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            return false;
        }

        // Optimize frame pose with all matches
        mpOptimizer->PoseOptimization(&mCurrentFrame);

        // Discard outliers
        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 6;
            return nmatches > 10;
        }

        if (finalMatches < 7) {
            return false;
        }

        return true;
    }

    bool Tracking::TrackLocalMap() {

        UpdateLocalMap();

        thread threadPoints(&Tracking::SearchLocalPoints, this);
        thread threadLines(&Tracking::SearchLocalLines, this);
        thread threadPlanes(&Tracking::SearchLocalPlanes, this);
        threadPoints.join();
        threadLines.join();
        threadPlanes.join();

        mpPlaneMatcher->SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        mpOptimizer->PoseOptimization(&mCurrentFrame);

        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                }

            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (!mCurrentFrame.mvbLineOutlier[i]) {
                    mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                }
            }
        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                } else {
                    mCurrentFrame.mvpMapPlanes[i]->IncreaseFound();
                    mnMatchesInliers++;
                }
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 20) {
            return false;
        }

        if (mnMatchesInliers < 7) {
            return false;
        } else
            return true;
    }


    bool Tracking::NeedNewKeyFrame() {
        if (mbOnlyTracking)
            return false;

// If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

// Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

// Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

// Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

// Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
// "total matches = matches to map + visual odometry matches"
// Visual odometry matches will become MapPoints if we insert a keyframe.
// This ratio measures how many MapPoints we could create if we insert a keyframe.
        int nMap = 0; //nTrackedClose
        int nTotal = 0;
        int nNonTrackedClose = 0;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                nTotal++;
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nMap++;
                else
                    nNonTrackedClose++;
            }
        }

        const float ratioMap = (float) nMap / fmax(1.0f, nTotal);

// Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        float thMapRatio = 0.35f;
        if (mnMatchesInliers > 300)
            thMapRatio = 0.20f;

// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
//Condition 1c: tracking is weak
        const bool c1c = (mnMatchesInliers < nRefMatches * 0.25 || ratioMap < 0.3f);
// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || ratioMap < thMapRatio) &&
                         mnMatchesInliers > 15);

        if (((c1a || c1b || c1c) && c2) || mCurrentFrame.mbNewPlane) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle) {
                return true;
            } else {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return true;
                else
                    return false;
            }
        }

        return false;
    }

    void Tracking::CreateNewKeyFrame() {
        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);

        for (int i = 0; i < mCurrentFrame.N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty()) {
            sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++) {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1) {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew) {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                } else {
                    nPoints++;
                }

                if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                    break;
            }
        }

        vector<pair<float, int>> vLineDepthIdx;
        vLineDepthIdx.reserve(mCurrentFrame.NL);

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            pair<float, float> z = mCurrentFrame.mvDepthLine[i];
            if (z.first > 0 && z.second > 0) {
                vLineDepthIdx.push_back(make_pair(min(z.first, z.second), i));
            }
        }

        if (!vLineDepthIdx.empty()) {
            sort(vLineDepthIdx.begin(), vLineDepthIdx.end());

            int nLines = 0;
            for (size_t j = 0; j < vLineDepthIdx.size(); j++) {
                int i = vLineDepthIdx[j].second;

                bool bCreateNew = false;

                MapLine *pMP = mCurrentFrame.mvpMapLines[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1) {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }

                if (bCreateNew) {
                    Vector6d line3D = mCurrentFrame.Obtain3DLine(i, mImDepth);
                    if (line3D == static_cast<Vector6d>(NULL)) {
                        continue;
                    }
                    MapLine *pNewML = new MapLine(line3D, pKF, mpMap);
                    pNewML->AddObservation(pKF, i);
                    pKF->AddMapLine(pNewML, i);
                    pNewML->ComputeDistinctiveDescriptors();
                    pNewML->UpdateAverageDir();
                    mpMap->AddMapLine(pNewML);
                    mCurrentFrame.mvpMapLines[i] = pNewML;
                    nLines++;
                } else {
                    nLines++;
                }

                if (nLines > 30)
                    break;
            }
        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
            if (mCurrentFrame.mvpParallelPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
            }
            if (mCurrentFrame.mvpVerticalPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
            }

            if (mCurrentFrame.mvpMapPlanes[i]) {
                mCurrentFrame.mvpMapPlanes[i]->AddObservation(pKF, i);
                continue;
            }

            if (mCurrentFrame.mvbPlaneOutlier[i]) {
                continue;
            }

            pKF->SetNotErase();

            cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
            MapPlane *pNewMP = new MapPlane(p3D, pKF, mpMap);
            pNewMP->AddObservation(pKF, i);
            pKF->AddMapPlane(pNewMP, i);
            pNewMP->UpdateCoefficientsAndPoints();
            mpMap->AddMapPlane(pNewMP);
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
    }

    void Tracking::SearchLocalPoints() {
// Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

// Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++; //将要match的
            }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 3;
            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::SearchLocalLines() {
        for (vector<MapLine *>::iterator vit = mCurrentFrame.mvpMapLines.begin(), vend = mCurrentFrame.mvpMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;
            if (pML) {
                if (pML->isBad()) {
                    *vit = static_cast<MapLine *>(NULL);
                } else {
                    pML->IncreaseVisible();
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    pML->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        for (vector<MapLine *>::iterator vit = mvpLocalMapLines.begin(), vend = mvpLocalMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;

            if (pML->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pML->isBad())
                continue;

            if (mCurrentFrame.isInFrustum(pML, 0.6)) {
                pML->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0) {
            int th = 1;

            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            mpLineMatcher->SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
        }
    }

    void Tracking::SearchLocalPlanes() {
        for (vector<MapPlane *>::iterator vit = mCurrentFrame.mvpMapPlanes.begin(), vend = mCurrentFrame.mvpMapPlanes.end();
             vit != vend; vit++) {
            MapPlane *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPlane *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                }
            }
        }
    }


    void Tracking::UpdateLocalMap() {
// This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

// Update
        UpdateLocalKeyFrames();

        UpdateLocalPoints();
        UpdateLocalLines();
    }

    void Tracking::UpdateLocalLines() {
        mvpLocalMapLines.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapLine *> vpMLs = pKF->GetMapLineMatches();

            for (vector<MapLine *>::const_iterator itML = vpMLs.begin(), itEndML = vpMLs.end();
                 itML != itEndML; itML++) {
                MapLine *pML = *itML;
                if (!pML)
                    continue;
                if (pML->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pML->isBad()) {
                    mvpLocalMapLines.push_back(pML);
                    pML->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad()) {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
// Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad()) {
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


// Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad()) {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad()) {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent) {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }

        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization() {
// Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

// Relocalization is performed when tracking is lost
// Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

// We perform first an ORB matching with each candidate
// If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

// Alternatively perform some iterations of P4P RANSAC
// Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nKFs; i++) {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty()) {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = mpOptimizer->PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood >= 50) {
                            nGood = mpOptimizer->PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50) {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nGood + nadditional >= 50) {
                                    nGood = mpOptimizer->PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50) {
                        bMatch = true;
                        break;
                    }
                }
            }

            if (!bMatch) {

            }
        }

        if (!bMatch) {
            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }

    }

    void Tracking::Reset() {
        mpViewer->RequestStop();

        cout << "System Reseting" << endl;
        while (!mpViewer->isStopped())
            usleep(3000);

// Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

// Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

// Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        mpViewer->Release();
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }


} //namespace ORB_SLAM
