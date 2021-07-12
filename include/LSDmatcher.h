/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/

#ifndef LSDMATCHER_H
#define LSDMATCHER_H

#include <cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include "MapLine.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace ORB_SLAM2 {
    class LSDmatcher {

        struct compare_descriptor_by_NN_dist {
            inline bool operator()(const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b) {
                return (a[0].distance < b[0].distance);
            }
        };

        struct conpare_descriptor_by_NN12_dist {
            inline bool operator()(const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b) {
                return ((a[1].distance - a[0].distance) > (b[1].distance - b[0].distance));
            }
        };

        struct sort_descriptor_by_queryIdx {
            inline bool operator()(const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b) {
                return (a[0].queryIdx < b[0].queryIdx);
            }
        };

    public:
        static const int TH_HIGH, TH_LOW;

        LSDmatcher(float nnratio = 0.6, bool checkOri = true);

        int SearchByDescriptor(KeyFrame *pKF, Frame &currentF, std::vector<MapLine *> &vpMapLineMatches);

        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);

        int SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th = 3);

        int Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines, const float th = 3.0);

        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    protected:
        float RadiusByViewingCos(const float &viewCos);

        void lineDescriptorMAD(std::vector<std::vector<cv::DMatch>> matches, double &nn_mad, double &nn12_mad) const;

        float mfNNratio;
    };
}


#endif //LSDMATCHER_H
