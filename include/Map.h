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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>
#include <unordered_map>

#include <mutex>


#include "MapLine.h"
#include "Surfel.h"

#include "MapPlane.h"
#include <eigen3/Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

namespace ORB_SLAM2 {

    class MapPoint;

    class KeyFrame;

    class MapLine;

    class MapPlane;

    class Frame;

    struct PartialManhattanMapHash {
        size_t operator()(const std::pair<MapPlane *, MapPlane *> &key) const;
    };

    struct PartialManhattanMapEqual {
        bool operator()(const std::pair<MapPlane *, MapPlane *> &a, const std::pair<MapPlane *, MapPlane *> &b) const;
    };

    struct ManhattanMapHash {
        size_t operator()(const std::tuple<MapPlane *, MapPlane *, MapPlane *> &key) const;
    };

    struct ManhattanMapEqual {
        bool operator()(const std::tuple<MapPlane *, MapPlane *, MapPlane *> &a,
                        const std::tuple<MapPlane *, MapPlane *, MapPlane *> &b) const;
    };

    class Map {
    public:
        typedef std::pair<MapPlane *, MapPlane *> PartialManhattan;
        typedef std::tuple<MapPlane *, MapPlane *, MapPlane *> Manhattan;
        typedef std::unordered_map<PartialManhattan, KeyFrame *, PartialManhattanMapHash, PartialManhattanMapEqual> PartialManhattans;
        typedef std::unordered_map<Manhattan, KeyFrame *, ManhattanMapHash, ManhattanMapEqual> Manhattans;

        Map();

        void AddKeyFrame(KeyFrame *pKF);

        void AddMapPoint(MapPoint *pMP);

        void EraseMapPoint(MapPoint *pMP);

        void EraseKeyFrame(KeyFrame *pKF);

        void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);

        void AddMapLine(MapLine *pML);

        void EraseMapLine(MapLine *pML);

        void SetReferenceMapLines(const std::vector<MapLine *> &vpMLs);

        std::vector<KeyFrame *> GetAllKeyFrames();

        std::vector<MapPoint *> GetAllMapPoints();

        std::vector<MapPoint *> GetReferenceMapPoints();

        std::vector<MapLine *> GetAllMapLines();

        std::vector<MapLine *> GetReferenceMapLines();

        long unsigned int MapPointsInMap();

        long unsigned KeyFramesInMap();

        void clear();

        vector<KeyFrame *> mvpKeyFrameOrigins;

        std::mutex mMutexMapUpdate;

        // This avoid that two points are created simultaneously in separate threads (id conflict)
        std::mutex mMutexPointCreation;
        std::mutex mMutexLineCreation;

        void AddMapPlane(MapPlane *pMP);

        void EraseMapPlane(MapPlane *pMP);

        std::vector<MapPlane *> GetAllMapPlanes();

        void AddManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3, KeyFrame *pKF);

        KeyFrame *GetManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3);

        void AddPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, KeyFrame *pKF);

        KeyFrame *GetPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2);

        std::vector<Surfel> mvLocalSurfels;
        std::vector<Surfel> mvInactiveSurfels;

    protected:
        std::set<MapPoint *> mspMapPoints;

        std::set<MapLine *> mspMapLines;

        std::set<MapPlane *> mspMapPlanes;

        std::set<KeyFrame *> mspKeyFrames;

        PartialManhattans mmpPartialManhattanObservations;

        Manhattans mmpManhattanObservations;

        std::vector<MapPoint *> mvpReferenceMapPoints;
        std::vector<MapLine *> mvpReferenceMapLines;
        long unsigned int mnMaxKFid;

        std::mutex mMutexMap;
    };

} //namespace ORB_SLAM

#endif // MAP_H
