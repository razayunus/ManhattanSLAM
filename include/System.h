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


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "SurfelMapping.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"


namespace ORB_SLAM2 {

    class Viewer;

    class FrameDrawer;

    class Map;

    class Tracking;

    class LocalMapping;

    class SurfelMapping;

    class System {
    public:

        // Initialize the SLAM system. It launches the Local Mapping, Surfel Mapping and Viewer threads.
        System(const string &strVocFile, const string &strSettingsFile, const bool bUseViewer = true);

        // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
        // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
        // Input depthmap: Float (CV_32F).
        // Returns the camera pose (empty if tracking fails).
        cv::Mat Track(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

        // This stops local mapping thread (map building) and performs only camera tracking.
        void ActivateLocalizationMode();

        // This resumes local mapping thread and performs SLAM again.
        void DeactivateLocalizationMode();

        // Reset the system (clear map)
        void Reset();

        // All threads will be requested to finish.
        // It waits until all threads have finished.
        // This function must be called before saving the trajectory.
        void Shutdown();

        // Save camera trajectory in the TUM RGB-D dataset format.
        // Call first Shutdown()
        // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
        void SaveTrajectoryTUM(const string &filename);

        // Save keyframe poses in the TUM RGB-D dataset format.
        // Call first Shutdown()
        // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
        void SaveKeyFrameTrajectoryTUM(const string &filename);

        // TODO: Save/Load functions
        // SaveMap(const string &filename);
        // LoadMap(const string &filename);

    private:

        void saveSurfels(pcl::PointCloud<pcl::PointSurfel>::Ptr pointCloud);

        // ORB vocabulary used for place recognition and feature matching.
        ORBVocabulary *mpVocabulary;

        // KeyFrame database for place recognition (relocalization).
        KeyFrameDatabase *mpKeyFrameDatabase;


        // Map structure that stores the pointers to all KeyFrames and MapPoints.
        Map *mpMap;

        // Tracker. It receives a frame and computes the associated camera pose.
        // It also decides when to insert a new keyframe, create some new MapPoints and
        // performs relocalization if tracking fails.
        Tracking *mpTracker;

        // Local Mapper. It manages the local map and performs local bundle adjustment.
        LocalMapping *mpLocalMapper;

        // Surfel Mapper. It manages the surfel map.
        SurfelMapping *mpSurfelMapper;

        // The viewer draws the map and the current camera pose. It uses Pangolin.
        Viewer *mpViewer;

        FrameDrawer *mpFrameDrawer;
        MapDrawer *mpMapDrawer;

        // System threads: Local Mapping, Surfel Mapping, Viewer.
        // The Tracking thread "lives" in the main execution thread that creates the System object.
        std::thread *mptLocalMapping;
        std::thread *mptSurfelMapping;
        std::thread *mptViewer;

        // Reset flag
        std::mutex mMutexReset;
        bool mbReset;

        // Change mode flags
        std::mutex mMutexMode;
        bool mbActivateLocalizationMode;
        bool mbDeactivateLocalizationMode;
    };

}// namespace ORB_SLAM

#endif // SYSTEM_H
