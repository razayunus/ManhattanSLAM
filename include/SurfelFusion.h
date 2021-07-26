/**
* This file is part of ManhattanSLAM.
*
* Copyright (C) 2021 Raza Yunus <razayunus31 at gmail dot com>
* For more information see <https://github.com/razayunus/ManhattanSLAM>
*
* ManhattanSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ManhattanSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ManhattanSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * This code is adopted and modified from https://github.com/HKUST-Aerial-Robotics/DenseSurfelMapping.
 */

#ifndef SURFEL_FUSION_H
#define SURFEL_FUSION_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <Surfel.h>

#define ITERATION_NUM 3
#define THREAD_NUM 10
#define SP_SIZE 8
#define MAX_ANGLE_COS 0.1

#define HUBER_RANGE 0.4
#define BASELINE 0.5
#define DISPARITY_ERROR 4.0
#define MIN_TOLERATE_DIFF 0.1

class SurfelFusion {
private:

    struct SuperpixelSeed {
        float x, y;
        float size;
        float normX, normY, normZ;
        float posX, posY, posZ;
        float viewCos;
        float meanDepth;
        float meanIntensity;
        int r, g, b;
        bool fused;
        bool stable;
        bool use = true;
    };

    float fx, fy, cx, cy;
    int imageWidth, imageHeight;
    int spWidth, spHeight;
    float fuseFar, fuseNear;

    cv::Mat image;
    cv::Mat depth;
    cv::Mat planeMembershipImg;

    std::vector<double> spaceMap;
    std::vector<float> normMap;
    std::vector<SuperpixelSeed> superpixelSeeds;
    std::vector<int> superpixelIndex;

    std::vector<Surfel> *localSurfelsPtr;
    std::vector<Surfel> *newSurfelsPtr;

    void generateSuperPixels();

    void backProject(
            const float &u, const float &v, const float &depth, double &x, double &y, double &z);

    bool calculateCost(
            float &nodepthCost, float &depthCost,
            const float &pixelIntensity, const float &pixelInverseDepth,
            const int &x, const int &y,
            const int &spX, const int &spY);

    void updatePixelsKernel(int thread, int threadNum);

    void updatePixels();

    void updateSeedsKernel(
            int thread, int threadNum);

    void updateSeeds();

    void initializeSeedsKernel(
            int thread, int threadNum);

    void getHuberNorm(
            float &nx, float &ny, float &nz, float &nb,
            std::vector<float> &points);

    void initializeSeeds();

    void calculateSpacesKernel(int thread, int threadNum);

    void calculateSpDepthNormsKernel(int thread, int threadNum);

    void calculatePixelsNormsKernel(int thread, int threadNum);

    void calculateNorms();

    void fuseSurfelsKernel(
            const int thread, const int threadNum,
            const int referenceFrameIndex, const Eigen::Matrix4f &pose, const Eigen::Matrix4f &invPose);

    void initializeSurfels(
            const int referenceFrameIndex,
            const Eigen::Matrix4f &pose);

    void project(float &x, float &y, float &z, float &u, float &v);

    float getWeight(float &depth);

public:
    SurfelFusion(int width, int height,
                 float _fx, float _fy, float _cx, float _cy,
                 float _fuseFar, float _fuseNear);

    void fuseInitializeMap(
            const int referenceFrameIndex,
            const cv::Mat &inputImage,
            const cv::Mat &inputDepth,
            const cv::Mat &inputPlaneMembershipImg,
            const Eigen::Matrix4f &pose,
            std::vector<Surfel> &localSurfels,
            std::vector<Surfel> &newSurfels);
};

#endif //SURFEL_FUSION_H