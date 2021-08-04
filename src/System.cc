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



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <time.h>
#include <tinyply.h>

using namespace tinyply;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    System::System(const string &strVocFile, const string &strSettingsFile,
                   const bool bUseViewer) : mpViewer(static_cast<Viewer *>(NULL)), mbReset(false),
                                            mbActivateLocalizationMode(false),
                                            mbDeactivateLocalizationMode(false) {
        // Output welcome message
        cout << endl <<
             "ManhattanSLAM Copyright (C) 2021 Raza Yunus, Technical University of Munich (TUM)." << endl <<
             "This program comes with ABSOLUTELY NO WARRANTY;" << endl <<
             "This is free software, and you are welcome to redistribute it" << endl <<
             "under certain conditions. See LICENSE.txt." << endl << endl;

        //Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        // TO DO
        //float resolution = fsSettings["PointCloudMapping.Resolution"];
        //float resolution = 0.01;

        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        clock_t tStart = clock();
        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        //else
        //   bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
        if (!bVocLoad) {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Failed to open at: " << strVocFile << endl;
            exit(-1);
        }
        //printf("Vocabulary loaded in %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        cout << "Vocabulary loaded!" << endl << endl;
        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        //Create the Map
        mpMap = new Map();

        //Create Drawers. These are used by the Viewer
        mpFrameDrawer = new FrameDrawer(mpMap);
        mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

        //Initialize the Tracking thread
        //(it will live in the main thread of execution, the one that called this constructor)
        // TO DO
        //mpPointCloudMapping = make_shared<PointCloudMapping>( resolution );

        mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                                 mpMap, mpKeyFrameDatabase, strSettingsFile);

        //Initialize the Local Mapping thread and launch
        mpLocalMapper = new LocalMapping(mpMap, strSettingsFile);
        mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

        //Initialize the Surfel Mapping thread and launch
        mpSurfelMapper = new SurfelMapping(mpMap, strSettingsFile);
        mptSurfelMapping = new thread(&ORB_SLAM2::SurfelMapping::Run, mpSurfelMapper);

        //Initialize the Viewer thread and launch
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, strSettingsFile);
        if (bUseViewer) {
            mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, strSettingsFile);
            mptViewer = new thread(&Viewer::Run, mpViewer);
            mpTracker->SetViewer(mpViewer);
        }

        //Set pointers between threads
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpTracker->SetSurfelMapper(mpSurfelMapper);
    }


    cv::Mat System::Track(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp) {
        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode) {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped()) {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode) {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset) {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImage(im, depthmap, timestamp);

        return Tcw;
    }


    void System::ActivateLocalizationMode() {
        unique_lock<mutex> lock(mMutexMode);
        mbActivateLocalizationMode = true;
    }

    void System::DeactivateLocalizationMode() {
        unique_lock<mutex> lock(mMutexMode);
        mbDeactivateLocalizationMode = true;
    }

    void System::Reset() {
        unique_lock<mutex> lock(mMutexReset);
        mbReset = true;
    }

    void System::Shutdown() {
        mpLocalMapper->RequestFinish();

        pcl::PointCloud<pcl::PointSurfel>::Ptr pointCloud = mpSurfelMapper->Stop();
        saveSurfels(pointCloud);

        if (mpViewer) {
            mpViewer->RequestFinish();
            while (!mpViewer->isFinished())
                usleep(5000);

        }

        // Wait until all thread have effectively stopped
        while (!mpLocalMapper->isFinished()) {
            usleep(5000);
        }
        if (mpViewer)
            pangolin::BindToContext("ORB-SLAM2: Map Viewer");
    }

    void System::SaveTrajectoryTUM(const string &filename) {
        cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        list<bool>::iterator lbL = mpTracker->mlbLost.begin();
        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                     lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
            if (*lbL)
                continue;

            KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while (pKF->isBad()) {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            vector<float> q = Converter::toQuaternion(Rwc);

            f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " "
              << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
        f.close();
        cout << endl << "trajectory saved!" << endl;
    }


    void System::SaveKeyFrameTrajectoryTUM(const string &filename) {
        cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        //cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];



            // pKF->SetPose(pKF->GetPose()*Two);

            if (pKF->isBad())
                continue;
            cv::Mat R = pKF->GetRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1)
              << " " << t.at<float>(2)
              << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }

        f.close();
        cout << endl << "trajectory saved!" << endl;
    }

    struct float2 {
        float x, y;
    };
    struct float3 {
        float x, y, z;
    };
    struct float17 {
        float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q;
    };
    struct uchar3 {
        uint8_t x, y, z;
    };
    struct ushort1 {
        uint16_t x;
    };
    struct int2 {
        int x, y;
    };

    void System::saveSurfels(pcl::PointCloud<pcl::PointSurfel>::Ptr pointCloud) {
        std::filebuf fb_ascii;
        fb_ascii.open("Surfels.ply", std::ios::out);
        std::ostream outstream_ascii(&fb_ascii);
        if (outstream_ascii.fail()) throw std::runtime_error("failed to open Surfels.ply");

        auto &points = pointCloud->points;

        std::vector<float3> vertices, normals;
        std::vector<ushort1> labels;
        std::vector<uchar3> colours;
        std::vector<float2> props;

        for (auto &point : points) {
            if (std::isnan(point.x))
                continue;
            vertices.push_back({point.x, point.y, point.z});
            labels.push_back({1});
            normals.push_back({point.normal_x, point.normal_y, point.normal_z});
            colours.push_back({point.r, point.g, point.b});
            props.push_back({point.confidence, point.radius});
        }

        size_t pointCloudSize = vertices.size();

        std::vector<float17> cameraProp1;
        std::vector<int2> cameraProp2;
        std::vector<float2> cameraProp3;
        cameraProp1.push_back({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0});
        cameraProp2.push_back({(int) pointCloudSize, 1});
        cameraProp3.push_back({0, 0});

        PlyFile cube_file;

        cube_file.add_properties_to_element("vertex", {"x", "y", "z"},
                                            Type::FLOAT32, pointCloudSize, reinterpret_cast<uint8_t *>(vertices.data()),
                                            Type::INVALID, 0);

        cube_file.add_properties_to_element("vertex", {"nx", "ny", "nz"},
                                            Type::FLOAT32, pointCloudSize, reinterpret_cast<uint8_t *>(normals.data()),
                                            Type::INVALID, 0);

        cube_file.add_properties_to_element("vertex", {"red", "green", "blue"},
                                            Type::UINT8, pointCloudSize, reinterpret_cast<uint8_t *>(colours.data()),
                                            Type::INVALID, 0);

        cube_file.add_properties_to_element("vertex", {"alpha"},
                                            Type::UINT8, pointCloudSize, reinterpret_cast<uint8_t *>(labels.data()),
                                            Type::INVALID, 0);

        cube_file.add_properties_to_element("vertex", {"quality", "radius"},
                                            Type::FLOAT32, pointCloudSize, reinterpret_cast<uint8_t *>(props.data()),
                                            Type::INVALID, 0);

        cube_file.add_properties_to_element("camera", {"view_px",
                                                       "view_py",
                                                       "view_pz",
                                                       "x_axisx",
                                                       "x_axisy",
                                                       "x_axisz",
                                                       "y_axisx",
                                                       "y_axisy",
                                                       "y_axisz",
                                                       "z_axisx",
                                                       "z_axisy",
                                                       "z_axisz",
                                                       "focal",
                                                       "scalex",
                                                       "scaley",
                                                       "centerx",
                                                       "centery"},
                                            Type::FLOAT32, 1, reinterpret_cast<uint8_t *>(cameraProp1.data()),
                                            Type::INVALID, 0);

        cube_file.add_properties_to_element("camera", {"viewportx",
                                                       "viewporty"},
                                            Type::INT32, 1, reinterpret_cast<uint8_t *>(cameraProp2.data()),
                                            Type::INVALID, 0);

        cube_file.add_properties_to_element("camera", {"k1",
                                                       "k2"},
                                            Type::FLOAT32, 1, reinterpret_cast<uint8_t *>(cameraProp3.data()),
                                            Type::INVALID, 0);

        cube_file.write(outstream_ascii, false);

    }

} //namespace ORB_SLAM
