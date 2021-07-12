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

#include "Map.h"

#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {

    size_t PartialManhattanMapHash::operator()(const std::pair<MapPlane *, MapPlane *> &key) const {
        int id1, id2;
        if (key.first->mnId > key.second->mnId) {
            id1 = key.second->mnId;
            id2 = key.first->mnId;
        } else {
            id1 = key.first->mnId;
            id2 = key.second->mnId;
        }

        size_t hash = 0;
        hash += (71 * hash + id1) % 5;
        hash += (71 * hash + id2) % 5;
        return hash;
    }

    bool PartialManhattanMapEqual::operator()(const std::pair<MapPlane *, MapPlane *> &a,
                                              const std::pair<MapPlane *, MapPlane *> &b) const {
        MapPlane *pMP11, *pMP12, *pMP21, *pMP22;
        if (a.first->mnId > a.second->mnId) {
            pMP11 = a.second;
            pMP12 = a.first;
        } else {
            pMP11 = a.first;
            pMP12 = a.second;
        }

        if (b.first->mnId > b.second->mnId) {
            pMP21 = b.second;
            pMP22 = b.first;
        } else {
            pMP21 = b.first;
            pMP22 = b.second;
        }

        std::pair<MapPlane *, MapPlane *> p1 = std::make_pair(pMP11, pMP12);
        std::pair<MapPlane *, MapPlane *> p2 = std::make_pair(pMP21, pMP22);

        return p1 == p2;
    }

    size_t ManhattanMapHash::operator()(const std::tuple<MapPlane *, MapPlane *, MapPlane *> &key) const {
        vector<int> ids;
        ids.push_back(get<0>(key)->mnId);
        ids.push_back(get<1>(key)->mnId);
        ids.push_back(get<2>(key)->mnId);
        sort(ids.begin(), ids.end());

        size_t hash = 0;
        hash += (71 * hash + ids[0]) % 5;
        hash += (71 * hash + ids[1]) % 5;
        hash += (71 * hash + ids[2]) % 5;
        return hash;
    }

    bool ManhattanMapEqual::operator()(const std::tuple<MapPlane *, MapPlane *, MapPlane *> &a,
                                       const std::tuple<MapPlane *, MapPlane *, MapPlane *> &b) const {
        MapPlane *pMP11, *pMP12, *pMP13, *pMP21, *pMP22, *pMP23;

        pMP11 = get<0>(a);
        pMP12 = get<1>(a);
        pMP13 = get<2>(a);

        if (pMP11 > pMP12) {
            std::swap(pMP11, pMP12);
        }
        if (pMP12 > pMP13) {
            std::swap(pMP12, pMP13);
        }
        if (pMP11 > pMP12) {
            std::swap(pMP11, pMP12);
        }

        pMP21 = get<0>(b);
        pMP22 = get<1>(b);
        pMP23 = get<2>(b);

        if (pMP21 > pMP22) {
            std::swap(pMP21, pMP22);
        }
        if (pMP22 > pMP23) {
            std::swap(pMP22, pMP23);
        }
        if (pMP21 > pMP22) {
            std::swap(pMP21, pMP22);
        }

        std::tuple<MapPlane *, MapPlane *, MapPlane *> t1 = std::make_tuple(pMP11, pMP12, pMP13);
        std::tuple<MapPlane *, MapPlane *, MapPlane *> t2 = std::make_tuple(pMP21, pMP22, pMP23);

        return t1 == t2;
    }

    Map::Map() : mnMaxKFid(0) {
    }

    void Map::AddKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if (pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    void Map::AddMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::EraseMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::EraseKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    vector<KeyFrame *> Map::GetAllKeyFrames() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    vector<MapPoint *> Map::GetAllMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::MapPointsInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    vector<MapPoint *> Map::GetReferenceMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    void Map::clear() {
        for (auto mspMapPoint : mspMapPoints)
            delete mspMapPoint;
        for (auto mspMapLine : mspMapLines)
            delete mspMapLine;
        for (auto mspMapPlane : mspMapPlanes)
            delete mspMapPlane;

        for (auto mspKeyFrame : mspKeyFrames)
            delete mspKeyFrame;

        mspMapPlanes.clear();
        mspMapPoints.clear();
        mspKeyFrames.clear();
        mspMapLines.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpReferenceMapLines.clear();
        mvpKeyFrameOrigins.clear();
    }

    void Map::AddMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.insert(pML);
    }

    void Map::EraseMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.erase(pML);
    }

    void Map::SetReferenceMapLines(const std::vector<MapLine *> &vpMLs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapLines = vpMLs;
    }

    vector<MapLine *> Map::GetAllMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapLine *>(mspMapLines.begin(), mspMapLines.end());
    }

    vector<MapLine *> Map::GetReferenceMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapLines;
    }

    void Map::AddMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.insert(pMP);
    }

    void Map::EraseMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.erase(pMP);
    }

    vector<MapPlane *> Map::GetAllMapPlanes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPlane *>(mspMapPlanes.begin(), mspMapPlanes.end());
    }

    void Map::AddManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3, KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);

        Manhattan manhattan = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpManhattanObservations.count(manhattan) != 0)
            return;
        pKF->SetNotErase();
        mmpManhattanObservations[manhattan] = pKF;
    }

    KeyFrame *Map::GetManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, MapPlane *pMP3) {
        unique_lock<mutex> lock(mMutexMap);
        Manhattan manhattan = std::make_tuple(pMP1, pMP2, pMP3);
        if (mmpManhattanObservations.count(manhattan)) {
            return mmpManhattanObservations[manhattan];
        } else {
            return static_cast<KeyFrame *>(nullptr);
        }
    }

    void Map::AddPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2, KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);

        PartialManhattan manhattan = std::make_pair(pMP1, pMP2);
        if (mmpPartialManhattanObservations.count(manhattan) != 0)
            return;
        pKF->SetNotErase();
        mmpPartialManhattanObservations[manhattan] = pKF;
    }

    KeyFrame *Map::GetPartialManhattanObservation(MapPlane *pMP1, MapPlane *pMP2) {
        unique_lock<mutex> lock(mMutexMap);
        PartialManhattan manhattan = std::make_pair(pMP1, pMP2);
        if (mmpPartialManhattanObservations.count(manhattan)) {
            return mmpPartialManhattanObservations[manhattan];
        } else {
            return static_cast<KeyFrame *>(nullptr);
        }
    }

} //namespace ORB_SLAM
