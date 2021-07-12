/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/

#include "LSDmatcher.h"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2 {
    const int LSDmatcher::TH_HIGH = 100;
    const int LSDmatcher::TH_LOW = 50;

    LSDmatcher::LSDmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio) {
    }

    int LSDmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th) {
        int nmatches = 0;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat tlc = Rlw * twc + tlw;

        const bool bForward = tlc.at<float>(2) > CurrentFrame.mb;
        const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb;

        for (int i = 0; i < LastFrame.NL; i++) {
            MapLine *pML = LastFrame.mvpMapLines[i];

            if (!pML || pML->isBad() || LastFrame.mvbLineOutlier[i]) {
                continue;
            }

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = CurrentFrame.fx * SPcX * invz1 + CurrentFrame.cx;
            const float v1 = CurrentFrame.fy * SPcY * invz1 + CurrentFrame.cy;

            if (u1 < CurrentFrame.mnMinX || u1 > CurrentFrame.mnMaxX)
                continue;
            if (v1 < CurrentFrame.mnMinY || v1 > CurrentFrame.mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = CurrentFrame.fx * EPcX * invz2 + CurrentFrame.cx;
            const float v2 = CurrentFrame.fy * EPcY * invz2 + CurrentFrame.cy;

            if (u2 < CurrentFrame.mnMinX || u2 > CurrentFrame.mnMaxX)
                continue;
            if (v2 < CurrentFrame.mnMinY || v2 > CurrentFrame.mnMaxY)
                continue;

            int nLastOctave = LastFrame.mvKeylinesUn[i].octave;

            float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

            vector<size_t> vIndices;

            if (bForward)
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, nLastOctave);
            else if (bBackward)
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, 0, nLastOctave);
            else
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, nLastOctave - 1, nLastOctave + 1);

            if (vIndices.empty())
                continue;

            const cv::Mat desc = pML->GetDescriptor();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            for (unsigned long idx : vIndices) {
                if (CurrentFrame.mvpMapLines[idx])
                    if (CurrentFrame.mvpMapLines[idx]->Observations() > 0)
                        continue;

                const cv::Mat &d = CurrentFrame.mLdesc.row(idx);

                const int dist = DescriptorDistance(desc, d);

                if (dist < bestDist) {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = CurrentFrame.mvKeylinesUn[idx].octave;
                    bestIdx = idx;
                } else if (dist < bestDist2) {
                    bestLevel2 = CurrentFrame.mvKeylinesUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            if (bestDist <= TH_HIGH) {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                CurrentFrame.mvpMapLines[bestIdx] = pML;
                nmatches++;
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th) {
        int nmatches = 0;

        const bool bFactor = th != 1.0;

        for (auto pML : vpMapLines) {
            if (!pML || pML->isBad() || !pML->mbTrackInView)
                continue;

            const int &nPredictLevel = pML->mnTrackScaleLevel;

            float r = RadiusByViewingCos(pML->mTrackViewCos);

            if (bFactor)
                r *= th;

            vector<size_t> vIndices =
                    F.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2,
                                     r * F.mvScaleFactors[nPredictLevel], nPredictLevel - 1, nPredictLevel);

            if (vIndices.empty())
                continue;

            const cv::Mat MLdescriptor = pML->GetDescriptor();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            for (unsigned long idx : vIndices) {
                if (F.mvpMapLines[idx])
                    if (F.mvpMapLines[idx]->Observations() > 0)
                        continue;

                const cv::Mat &d = F.mLdesc.row(idx);

                const int dist = DescriptorDistance(MLdescriptor, d);

                if (dist < bestDist) {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeylinesUn[idx].octave;
                    bestIdx = idx;
                } else if (dist < bestDist2) {
                    bestLevel2 = F.mvKeylinesUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= TH_HIGH) {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                F.mvpMapLines[bestIdx] = pML;
                nmatches++;
            }
        }
        return nmatches;
    }

    int LSDmatcher::SearchByDescriptor(KeyFrame *pKF, Frame &currentF, vector<MapLine *> &vpMapLineMatches) {
        const vector<MapLine *> vpMapLinesKF = pKF->GetMapLineMatches();

        vpMapLineMatches = vector<MapLine *>(currentF.NL, static_cast<MapLine *>(NULL));

        int nmatches = 0;
        BFMatcher *bfm = new BFMatcher(NORM_HAMMING, false);
        Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = pKF->mLineDescriptors;
        ldesc2 = currentF.mLdesc;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        const float minRatio = 1.0f / 1.5f;
        lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th * 0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for (int i = 0; i < lmatches.size(); i++) {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][0].distance / lmatches[i][1].distance;
            if (dist_12 < minRatio) {
                MapLine *mapLine = vpMapLinesKF[qdx];

                if (mapLine) {
                    vpMapLineMatches[tdx] = mapLine;
                    nmatches++;
                }

            }
        }
        return nmatches;
    }

    int LSDmatcher::DescriptorDistance(const Mat &a, const Mat &b) {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    float LSDmatcher::RadiusByViewingCos(const float &viewCos) {
        if (viewCos > 0.998)
            return 5.0;
        else
            return 8.0;
    }

    int LSDmatcher::Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines, const float th) {
        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        cv::Mat Ow = pKF->GetCameraCenter();

        int nFused = 0;

        const int nLines = vpMapLines.size();

        // For each candidate MapPoint project and match
        for (int iML = 0; iML < nLines; iML++) {
            MapLine *pML = vpMapLines[iML];

            // Discard Bad MapLines and already found
            if (!pML || pML->isBad())
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX)
                continue;
            if (v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX)
                continue;
            if (v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if (OM.dot(pn) < 0.5 * dist)
                continue;

            const int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactor);

            const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1, v1, u2, v2, radius);

            if (vIndices.empty())
                continue;

            const cv::Mat dML = pML->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;

            for (unsigned long idx : vIndices) {
                const int &klLevel = pKF->mvKeyLines[idx].octave;

                if (klLevel < nPredictedLevel - 1 || klLevel > nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML, dKF);

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_LOW) {
                MapLine *pMLinKF = pKF->GetMapLine(bestIdx);
                if (pMLinKF) {
                    if (!pMLinKF->isBad()) {
                        if (pMLinKF->Observations() > pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                } else {
                    pML->AddObservation(pKF, bestIdx);
                    pKF->AddMapLine(pML, bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    void LSDmatcher::lineDescriptorMAD(vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const {
        vector<vector<DMatch>> matches_nn, matches_12;
        matches_nn = line_matches;
        matches_12 = line_matches;

        // estimate the NN's distance standard deviation
        double nn_dist_median;
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_dist_median = matches_nn[int(matches_nn.size() / 2)][0].distance;

        for (unsigned int i = 0; i < matches_nn.size(); i++)
            matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_mad = 1.4826 * matches_nn[int(matches_nn.size() / 2)][0].distance;

        // estimate the NN's 12 distance standard deviation
        double nn12_dist_median;
        sort(matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
        nn12_dist_median =
                matches_12[int(matches_12.size() / 2)][1].distance - matches_12[int(matches_12.size() / 2)][0].distance;
        for (unsigned int j = 0; j < matches_12.size(); j++)
            matches_12[j][0].distance = fabsf(matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
        sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
        nn12_mad = 1.4826 * matches_12[int(matches_12.size() / 2)][0].distance;
    }
}
