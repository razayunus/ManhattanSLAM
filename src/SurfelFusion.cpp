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

#include "SurfelFusion.h"
#include <thread>
#include <cmath>

void SurfelFusion::initialize(
        int width, int height,
        float _fx, float _fy, float _cx, float _cy,
        float _fuseFar, float _fuseNear) {
    imageWidth = width;
    imageHeight = height;
    spWidth = imageWidth / SP_SIZE;
    spHeight = imageHeight / SP_SIZE;
    fx = _fx;
    fy = _fy;
    cx = _cx;
    cy = _cy;

    fuseFar = _fuseFar;
    fuseNear = _fuseNear;

    superpixelSeeds.resize(spWidth * spHeight);
    superpixelIndex.resize(imageWidth * imageHeight);
    spaceMap.resize(imageWidth * imageHeight * 3);
    normMap.resize(imageWidth * imageHeight * 3);
}

void SurfelFusion::fuseInitializeMap(
        int referenceFrameIndex,
        cv::Mat &inputImage,
        cv::Mat &inputDepth,
        cv::Mat &inputPlaneMembershipImg,
        Eigen::Matrix4f &camPose,
        std::vector<Surfel> &localSurfels,
        std::vector<Surfel> &newSurfels) {
    image = inputImage;
    depth = inputDepth;

    pose = camPose;
    planeMembershipImg = inputPlaneMembershipImg;

    localSurfelsPtr = &localSurfels;
    newSurfelsPtr = &newSurfels;

    generateSuperPixels();

    // Fuse
    std::vector<std::thread> threadPool;
    for (int i = 0; i < THREAD_NUM; i++) {
        std::thread this_thread(
                &SurfelFusion::fuseSurfelsKernel, this, i, THREAD_NUM,
                referenceFrameIndex);
        threadPool.push_back(std::move(this_thread));
    }
    for (auto &thread : threadPool)
        if (thread.joinable())
            thread.join();

    // Initialize
    initializeSurfels(referenceFrameIndex, pose);
}

void SurfelFusion::project(float &x, float &y, float &z, float &u, float &v) {
    u = x * fx / z + cx;
    v = y * fy / z + cy;
}

void SurfelFusion::backProject(
        const float &u, const float &v, const float &depth, double &x, double &y, double &z) {
    x = (u - cx) / fx * depth;
    y = (v - cy) / fy * depth;
    z = depth;
}

float SurfelFusion::getWeight(float &depth) {
    return std::min(1.0 / depth / depth, 1.0);
}

void SurfelFusion::getHuberNorm(
        float &nx, float &ny, float &nz, float &nb,
        std::vector<float> &points) {
    int pointNum = points.size() / 3;
    float sumX, sumY, sumZ;
    sumX = sumY = sumZ = 0.0;
    for (int i = 0; i < pointNum; i++) {
        sumX += points[i * 3];
        sumY += points[i * 3 + 1];
        sumZ += points[i * 3 + 2];
    }
    sumX /= pointNum;
    sumY /= pointNum;
    sumZ /= pointNum;
    nb = 0;
    for (int i = 0; i < pointNum; i++) {
        points[i * 3] -= sumX;
        points[i * 3 + 1] -= sumY;
        points[i * 3 + 2] -= sumZ;
    }
    for (int gnI = 0; gnI < 5; gnI++) {
        Eigen::Matrix4d hessian = Eigen::Matrix4d::Zero();
        Eigen::Vector4d jacobian = Eigen::Vector4d::Zero();
        for (int i = 0; i < pointNum; i++) {
            float residual = points[i * 3] * nx + points[i * 3 + 1] * ny + points[i * 3 + 2] * nz + nb;
            if (residual < HUBER_RANGE && residual > -1 * HUBER_RANGE) {
                jacobian(0) += 2 * residual * points[i * 3];
                jacobian(1) += 2 * residual * points[i * 3 + 1];
                jacobian(2) += 2 * residual * points[i * 3 + 2];
                jacobian(3) += 2 * residual;
                hessian(0, 0) += 2 * points[i * 3] * points[i * 3];
                hessian(0, 1) += 2 * points[i * 3] * points[i * 3 + 1];
                hessian(0, 2) += 2 * points[i * 3] * points[i * 3 + 2];
                hessian(0, 3) += 2 * points[i * 3];
                hessian(1, 0) += 2 * points[i * 3 + 1] * points[i * 3];
                hessian(1, 1) += 2 * points[i * 3 + 1] * points[i * 3 + 1];
                hessian(1, 2) += 2 * points[i * 3 + 1] * points[i * 3 + 2];
                hessian(1, 3) += 2 * points[i * 3 + 1];
                hessian(2, 0) += 2 * points[i * 3 + 2] * points[i * 3];
                hessian(2, 1) += 2 * points[i * 3 + 2] * points[i * 3 + 1];
                hessian(2, 2) += 2 * points[i * 3 + 2] * points[i * 3 + 2];
                hessian(2, 3) += 2 * points[i * 3 + 2];
                hessian(3, 0) += 2 * points[i * 3];
                hessian(3, 1) += 2 * points[i * 3 + 1];
                hessian(3, 2) += 2 * points[i * 3 + 2];
                hessian(3, 3) += 2;
            } else if (residual >= HUBER_RANGE) {
                jacobian(0) += HUBER_RANGE * points[i * 3];
                jacobian(1) += HUBER_RANGE * points[i * 3 + 1];
                jacobian(2) += HUBER_RANGE * points[i * 3 + 2];
                jacobian(3) += HUBER_RANGE;
            } else if (residual <= -1 * HUBER_RANGE) {
                jacobian(0) += -1 * HUBER_RANGE * points[i * 3];
                jacobian(1) += -1 * HUBER_RANGE * points[i * 3 + 1];
                jacobian(2) += -1 * HUBER_RANGE * points[i * 3 + 2];
                jacobian(3) += -1 * HUBER_RANGE;
            }
        }
        hessian(0, 0) += 5;
        hessian(1, 1) += 5;
        hessian(2, 2) += 5;
        hessian(3, 3) += 5;
        Eigen::Vector4d updateValue = hessian.inverse() * jacobian;
        nx -= updateValue(0);
        ny -= updateValue(1);
        nz -= updateValue(2);
        nb -= updateValue(3);
    }
    nb = nb - (nx * sumX + ny * sumY + nz * sumZ);
    float normLength = std::sqrt(nx * nx + ny * ny + nz * nz);
    nx /= normLength;
    ny /= normLength;
    nz /= normLength;
    nb /= normLength;
}

void SurfelFusion::fuseSurfelsKernel(
        int thread, int threadNum,
        int referenceFrameIndex) {

    std::vector<Surfel> &localSurfels = *localSurfelsPtr;
    Eigen::Matrix4f invPose = pose.inverse();

    int step = localSurfels.size() / threadNum;
    int beginIndex = step * thread;
    int endIndex = beginIndex + step;
    if (thread == threadNum - 1)
        endIndex = localSurfels.size();

    for (int i = beginIndex; i < endIndex; i++) {
        // remove unstable
        if (referenceFrameIndex - localSurfels[i].lastUpdate > 5 && localSurfels[i].updateTimes < 5) {
            localSurfels[i].updateTimes = 0;
            continue;
        }

        if (localSurfels[i].updateTimes == 0)
            continue;
        Eigen::Vector4f surfelPW;
        surfelPW(0) = localSurfels[i].px;
        surfelPW(1) = localSurfels[i].py;
        surfelPW(2) = localSurfels[i].pz;
        surfelPW(3) = 1.0;
        Eigen::Vector4f surfelPC = invPose * surfelPW;
        if (surfelPC(2) < fuseNear || surfelPC(2) > fuseFar)
            continue;
        Eigen::Vector3f normW;
        normW(0) = localSurfels[i].nx;
        normW(1) = localSurfels[i].ny;
        normW(2) = localSurfels[i].nz;
        Eigen::Vector3f normC;
        normC = invPose.block<3, 3>(0, 0) * normW;
        float projectU, projectV;
        project(surfelPC(0), surfelPC(1), surfelPC(2), projectU, projectV);
        int pUInt = projectU + 0.5;
        int pVInt = projectV + 0.5;
        if (pUInt < 1 || pUInt > imageWidth - 2 || pVInt < 1 || pVInt > imageHeight - 2)
            continue;
        if (surfelPC(2) < depth.at<float>(pVInt, pUInt) - 1.0) {
            localSurfels[i].updateTimes = 0;
            continue;
        }
        int spIndex = superpixelIndex[pVInt * imageWidth + pUInt];
        if (superpixelSeeds[spIndex].normX == 0 && superpixelSeeds[spIndex].normY == 0 &&
            superpixelSeeds[spIndex].normZ == 0)
            continue;
        if (superpixelSeeds[spIndex].viewCos < MAX_ANGLE_COS)
            continue;

        float cameraF = (fabs(fx) + fabs(fy)) / 2.0;
        float tolerateDiff =
                surfelPC(2) * surfelPC(2) / (BASELINE * cameraF) * DISPARITY_ERROR;
        tolerateDiff = tolerateDiff < MIN_TOLERATE_DIFF ? MIN_TOLERATE_DIFF : tolerateDiff;
        if (surfelPC(2) < superpixelSeeds[spIndex].meanDepth - tolerateDiff) {
            // localSurfels[i].updateTimes = 0;
            continue;
        }
        if (surfelPC(2) > superpixelSeeds[spIndex].meanDepth + tolerateDiff) {
            // localSurfels[i].updateTimes = 0;
            continue;
        }

        float normDiffCos = normC(0) * superpixelSeeds[spIndex].normX
                            + normC(1) * superpixelSeeds[spIndex].normY
                            + normC(2) * superpixelSeeds[spIndex].normZ;
        if (normDiffCos < MAX_ANGLE_COS) {
            localSurfels[i].updateTimes = 0;
            continue;
        }
        float oldWeigth = localSurfels[i].weight;
        float newWeight = getWeight(superpixelSeeds[spIndex].meanDepth);
        float sumWeight = oldWeigth + newWeight;
        Eigen::Vector4f spPC, spPW;
        spPC(0) = superpixelSeeds[spIndex].posX;
        spPC(1) = superpixelSeeds[spIndex].posY;
        spPC(2) = superpixelSeeds[spIndex].posZ;
        spPC(3) = 1.0;
        spPW = pose * spPC;
        float fusedPx = (localSurfels[i].px * oldWeigth + newWeight * spPW(0)) / sumWeight;
        float fusedPy = (localSurfels[i].py * oldWeigth + newWeight * spPW(1)) / sumWeight;
        float fusedPz = (localSurfels[i].pz * oldWeigth + newWeight * spPW(2)) / sumWeight;
        float fusedNx = normC(0) * oldWeigth + newWeight * superpixelSeeds[spIndex].normX;
        float fusedNy = normC(1) * oldWeigth + newWeight * superpixelSeeds[spIndex].normY;
        float fusedNz = normC(2) * oldWeigth + newWeight * superpixelSeeds[spIndex].normZ;
        double newNormLength = std::sqrt(fusedNx * fusedNx + fusedNy * fusedNy + fusedNz * fusedNz);
        fusedNx /= newNormLength;
        fusedNy /= newNormLength;
        fusedNz /= newNormLength;
        Eigen::Vector3f newNormC, newNormW;
        newNormC(0) = fusedNx;
        newNormC(1) = fusedNy;
        newNormC(2) = fusedNz;
        newNormW = pose.block<3, 3>(0, 0) * newNormC;
        localSurfels[i].px = fusedPx;
        localSurfels[i].py = fusedPy;
        localSurfels[i].pz = fusedPz;
        localSurfels[i].r = superpixelSeeds[spIndex].r;
        localSurfels[i].g = superpixelSeeds[spIndex].g;
        localSurfels[i].b = superpixelSeeds[spIndex].b;
        localSurfels[i].nx = newNormW(0);
        localSurfels[i].ny = newNormW(1);
        localSurfels[i].nz = newNormW(2);
        localSurfels[i].weight = sumWeight;
        localSurfels[i].color = superpixelSeeds[spIndex].meanIntensity;
        float newSize = superpixelSeeds[spIndex].size *
                        fabs(superpixelSeeds[spIndex].meanDepth / (cameraF * superpixelSeeds[spIndex].viewCos));
        if (newSize < localSurfels[i].size)
            localSurfels[i].size = newSize;
        localSurfels[i].lastUpdate = referenceFrameIndex;
        // if(localSurfels[i].updateTimes < 20)
        localSurfels[i].updateTimes += 1;
        superpixelSeeds[spIndex].fused = true;
    }
}

void SurfelFusion::initializeSurfels(
        int referenceFrameIndex,
        Eigen::Matrix4f pose) {
    std::vector<Surfel> &newSurfels = *newSurfelsPtr;
    newSurfels.clear();
    Eigen::Vector4f positionTempC, positionTempW;
    Eigen::Vector3f normTempC, normTempW;
    for (int i = 0; i < superpixelSeeds.size(); i++) {
        if (superpixelSeeds[i].meanDepth == 0)
            continue;
        if (superpixelSeeds[i].fused)
            continue;
        if (superpixelSeeds[i].viewCos < MAX_ANGLE_COS)
            continue;
        positionTempC(0) = superpixelSeeds[i].posX;
        positionTempC(1) = superpixelSeeds[i].posY;
        positionTempC(2) = superpixelSeeds[i].posZ;
        positionTempC(3) = 1.0;
        normTempC(0) = superpixelSeeds[i].normX;
        normTempC(1) = superpixelSeeds[i].normY;
        normTempC(2) = superpixelSeeds[i].normZ;
        if (normTempC(0) == 0 && normTempC(1) == 0 && normTempC(2) == 0)
            continue;
        positionTempW = pose * positionTempC;
        normTempW = pose.block<3, 3>(0, 0) * normTempC;

        Surfel newEle;
        newEle.px = positionTempW(0);
        newEle.py = positionTempW(1);
        newEle.pz = positionTempW(2);
        newEle.r = superpixelSeeds[i].r;
        newEle.g = superpixelSeeds[i].g;
        newEle.b = superpixelSeeds[i].b;
        newEle.nx = normTempW(0);
        newEle.ny = normTempW(1);
        newEle.nz = normTempW(2);
        float cameraF = (fabs(fx) + fabs(fy)) / 2.0;
        float newSize = superpixelSeeds[i].size *
                        fabs(superpixelSeeds[i].meanDepth / (cameraF * superpixelSeeds[i].viewCos));
        newEle.size = newSize;
        newEle.color = superpixelSeeds[i].meanIntensity;
        newEle.weight = getWeight(superpixelSeeds[i].meanDepth);
        newEle.updateTimes = 1;
        newEle.lastUpdate = referenceFrameIndex;
        newSurfels.push_back(newEle);
    }
}

bool SurfelFusion::calculateCost(
        float &nodepthCost, float &depthCost,
        const float &pixelIntensity, const float &pixelInverseDepth,
        const int &x, const int &y,
        const int &spX, const int &spY) {
    int spIndex = spY * spWidth + spX;
    nodepthCost = 0;
    float dist =
            (superpixelSeeds[spIndex].x - x) * (superpixelSeeds[spIndex].x - x) +
            (superpixelSeeds[spIndex].y - y) * (superpixelSeeds[spIndex].y - y);
    nodepthCost += dist / ((SP_SIZE / 2) * (SP_SIZE / 2));
    float intensityDiff = (superpixelSeeds[spIndex].meanIntensity - pixelIntensity);
    nodepthCost += intensityDiff * intensityDiff / 100.0;
    depthCost = nodepthCost;
    if (superpixelSeeds[spIndex].meanDepth > 0 && pixelInverseDepth > 0) {
        float inverseDepthDiff = 1.0 / superpixelSeeds[spIndex].meanDepth - pixelInverseDepth;
        depthCost += inverseDepthDiff * inverseDepthDiff * 400.0;
        // float inverseDepthDiff = superpixelSeeds[spIndex].meanDepth - 1.0/pixelInverseDepth;
        // depthCost += inverseDepthDiff * inverseDepthDiff * 400.0;
        return true;
    }
    return false;
}

void SurfelFusion::updatePixelsKernel(
        int thread, int threadNum) {
    int stepRow = imageHeight / threadNum;
    int startRow = stepRow * thread;
    int endRow = startRow + stepRow;
    if (thread == threadNum - 1)
        endRow = imageHeight;
    for (int rowI = startRow; rowI < endRow; rowI++)
        for (int colI = 0; colI < imageWidth; colI++) {
            if (planeMembershipImg.at<int>(rowI / 2, colI / 2) != -1) {
                continue;
            }
            if (superpixelSeeds[superpixelIndex[rowI * imageWidth + colI]].stable)
                continue;
            float myIntensity = image.at<uchar>(rowI, colI);
            float myInvDepth = 0.0;
            if (depth.at<float>(rowI, colI) > 0.01)
                myInvDepth = 1.0 / depth.at<float>(rowI, colI);
            int baseSpX = colI / SP_SIZE;
            int baseSpY = rowI / SP_SIZE;
            float minDistDepth = 1e6;
            int minSpIndexDepth = -1;
            float minDistNodepth = 1e6;
            int minSpIndexNodepth = -1;
            bool allHasDepth = true;
            for (int checkI = -1; checkI <= 1; checkI++)
                for (int checkJ = -1; checkJ <= 1; checkJ++) {
                    int checkSpX = baseSpX + checkI;
                    int checkSpY = baseSpY + checkJ;
                    int distSpX = fabs(checkSpX * SP_SIZE + SP_SIZE / 2 - colI);
                    int distSpY = fabs(checkSpY * SP_SIZE + SP_SIZE / 2 - rowI);
                    if (distSpX < SP_SIZE && distSpY < SP_SIZE &&
                        checkSpX >= 0 && checkSpX < spWidth &&
                        checkSpY >= 0 && checkSpY < spHeight) {
                        float distDepth, distNodepth;
                        allHasDepth &= calculateCost(
                                distNodepth,
                                distDepth,
                                myIntensity, myInvDepth,
                                colI, rowI, checkSpX, checkSpY);
                        if (distDepth < minDistDepth) {
                            minDistDepth = distDepth;
                            minSpIndexDepth = (baseSpY + checkJ) * spWidth + baseSpX + checkI;
                        }
                        if (distNodepth < minDistNodepth) {
                            minDistNodepth = distNodepth;
                            minSpIndexNodepth = (baseSpY + checkJ) * spWidth + baseSpX + checkI;
                        }
                    }
                }
            if (allHasDepth) {
                superpixelIndex[rowI * imageWidth + colI] = minSpIndexDepth;
                superpixelSeeds[minSpIndexDepth].stable = false;
            } else {
                superpixelIndex[rowI * imageWidth + colI] = minSpIndexNodepth;
                superpixelSeeds[minSpIndexNodepth].stable = false;
            }
        }
}

void SurfelFusion::updatePixels() {
    std::vector<std::thread> threadPool;
    for (int i = 0; i < THREAD_NUM; i++) {
        std::thread thisThread(&SurfelFusion::updatePixelsKernel, this, i, THREAD_NUM);
        threadPool.push_back(std::move(thisThread));
    }
    for (int i = 0; i < threadPool.size(); i++)
        if (threadPool[i].joinable())
            threadPool[i].join();
}

void SurfelFusion::updateSeedsKernel(
        int thread, int threadNum) {
    int step = superpixelSeeds.size() / threadNum;
    int beginIndex = step * thread;
    int endIndex = beginIndex + step;
    if (thread == threadNum - 1)
        endIndex = superpixelSeeds.size();
    for (int seedI = beginIndex; seedI < endIndex; seedI++) {
        if (!superpixelSeeds[seedI].use)
            continue;
        if (superpixelSeeds[seedI].stable)
            continue;
        int spX = seedI % spWidth;
        int spY = seedI / spWidth;
        int checkXBegin = spX * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        int checkYBegin = spY * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        int checkXEnd = checkXBegin + SP_SIZE * 2;
        int checkYEnd = checkYBegin + SP_SIZE * 2;
        checkXBegin = checkXBegin > 0 ? checkXBegin : 0;
        checkYBegin = checkYBegin > 0 ? checkYBegin : 0;
        checkXEnd = checkXEnd < imageWidth - 1 ? checkXEnd : imageWidth - 1;
        checkYEnd = checkYEnd < imageHeight - 1 ? checkYEnd : imageHeight - 1;
        float sumX = 0;
        float sumY = 0;
        float sumIntensity = 0.0;
        float sumIntensityNum = 0.0;
        float sumDepth = 0.0;
        float sumDepthNum = 0.0;
        std::vector<float> depthVector;
        for (int checkJ = checkYBegin; checkJ < checkYEnd; checkJ++)
            for (int checkI = checkXBegin; checkI < checkXEnd; checkI++) {
                int pixelIndex = checkJ * imageWidth + checkI;
                if (superpixelIndex[pixelIndex] == seedI) {
                    sumX += checkI;
                    sumY += checkJ;
                    sumIntensityNum += 1.0;
                    sumIntensity += image.at<uchar>(checkJ, checkI);
                    float checkDepth = depth.at<float>(checkJ, checkI);
                    if (checkDepth > 0.1) {
                        depthVector.push_back(checkDepth);
                        sumDepth += checkDepth;
                        sumDepthNum += 1.0;
                    }
                }
            }
        if (sumIntensityNum == 0)
            return;
        sumIntensity /= sumIntensityNum;
        sumX /= sumIntensityNum;
        sumY /= sumIntensityNum;
        float preIntensity = superpixelSeeds[seedI].meanIntensity;
        float preX = superpixelSeeds[seedI].x;
        float preY = superpixelSeeds[seedI].y;
        superpixelSeeds[seedI].meanIntensity = sumIntensity;
        superpixelSeeds[seedI].x = sumX;
        superpixelSeeds[seedI].y = sumY;
        cv::Vec3b rgb = image.at<cv::Vec3b>(sumY, sumX);
        superpixelSeeds[seedI].r = rgb[0];
        superpixelSeeds[seedI].g = rgb[1];
        superpixelSeeds[seedI].b = rgb[2];
        float updateDiff = fabs(preIntensity - sumIntensity) + fabs(preX - sumX) + fabs(preY - sumY);
        if (updateDiff < 0.2)
            superpixelSeeds[seedI].stable = true;
        if (sumDepthNum > 0) {
            float meanDepth = sumDepth / sumDepthNum;
            float sumA, sumB;
            for (int newtonI = 0; newtonI < 5; newtonI++) {
                sumA = sumB = 0;
                for (int pI = 0; pI < depthVector.size(); pI++) {
                    float residual = meanDepth - depthVector[pI];
                    if (residual < HUBER_RANGE && residual > -HUBER_RANGE) {
                        sumA += 2 * residual;
                        sumB += 2;
                    } else {
                        sumA += residual > 0 ? HUBER_RANGE : -1 * HUBER_RANGE;
                    }
                }
                float deltaDepth = -sumA / (sumB + 10.0);
                meanDepth = meanDepth + deltaDepth;
                if (deltaDepth < 0.01 && deltaDepth > -0.01)
                    break;
            }
            superpixelSeeds[seedI].meanDepth = meanDepth;
        } else {
            superpixelSeeds[seedI].meanDepth = 0.0;
        }
    }
}

void SurfelFusion::updateSeeds() {
    std::vector<std::thread> threadPool;
    for (int i = 0; i < THREAD_NUM; i++) {
        std::thread thisThread(&SurfelFusion::updateSeedsKernel, this, i, THREAD_NUM);
        threadPool.push_back(std::move(thisThread));
    }
    for (int i = 0; i < threadPool.size(); i++)
        if (threadPool[i].joinable())
            threadPool[i].join();
}

void SurfelFusion::initializeSeedsKernel(
        int thread, int threadNum) {
    int step = superpixelSeeds.size() / threadNum;
    int beginIndex = step * thread;
    int endIndex = beginIndex + step;
    if (thread == threadNum - 1)
        endIndex = superpixelSeeds.size();
    for (int seedI = beginIndex; seedI < endIndex; seedI++) {
        int spX = seedI % spWidth;
        int spY = seedI / spWidth;
        int imageX = spX * SP_SIZE + SP_SIZE / 2;
        int imageY = spY * SP_SIZE + SP_SIZE / 2;
        imageX = imageX < (imageWidth - 1) ? imageX : (imageWidth - 1);
        imageY = imageY < (imageHeight - 1) ? imageY : (imageHeight - 1);

        if (planeMembershipImg.at<int>(imageY / 2, imageX / 2) != -1) {
            superpixelSeeds[seedI].use = false;
            continue;
        }

        SuperpixelSeed thisSp;
        thisSp.x = imageX;
        thisSp.y = imageY;
        cv::Vec3b rgb = image.at<cv::Vec3b>(imageY, imageX);
        thisSp.r = rgb[0];
        thisSp.g = rgb[1];
        thisSp.b = rgb[2];
        thisSp.meanIntensity = image.at<uchar>(imageY, imageX);
        thisSp.fused = false;
        thisSp.stable = false;
        thisSp.meanDepth = depth.at<float>(imageY, imageX);
        if (thisSp.meanDepth < 0.01) {
            int checkXBegin = spX * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
            int checkYBegin = spY * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
            int checkXEnd = checkXBegin + SP_SIZE * 2;
            int checkYEnd = checkYBegin + SP_SIZE * 2;
            checkXBegin = checkXBegin > 0 ? checkXBegin : 0;
            checkYBegin = checkYBegin > 0 ? checkYBegin : 0;
            checkXEnd = checkXEnd < imageWidth - 1 ? checkXEnd : imageWidth - 1;
            checkYEnd = checkYEnd < imageHeight - 1 ? checkYEnd : imageHeight - 1;
            bool findDepth = false;
            for (int checkJ = checkYBegin; checkJ < checkYEnd; checkJ++) {
                for (int checkI = checkXBegin; checkI < checkXEnd; checkI++) {
                    float thisDepth = depth.at<float>(checkJ, checkI);
                    if (thisDepth > 0.01) {
                        thisSp.meanDepth = thisDepth;
                        findDepth = true;
                        break;
                    }
                }
                if (findDepth)
                    break;
            }
        }
        superpixelSeeds[seedI] = thisSp;
    }
}

void SurfelFusion::initializeSeeds() {
    std::vector<std::thread> threadPool;
    for (int i = 0; i < THREAD_NUM; i++) {
        std::thread thisThread(&SurfelFusion::initializeSeedsKernel, this, i, THREAD_NUM);
        threadPool.push_back(std::move(thisThread));
    }
    for (int i = 0; i < threadPool.size(); i++)
        if (threadPool[i].joinable())
            threadPool[i].join();
}

void SurfelFusion::calculateSpacesKernel(int thread, int threadNum) {
    int stepRow = imageHeight / threadNum;
    int startRow = stepRow * thread;
    int endRow = startRow + stepRow;
    if (thread == threadNum - 1)
        endRow = imageHeight;
    for (int rowI = startRow; rowI < endRow; rowI++)
        for (int colI = 0; colI < imageWidth; colI++) {
            int myIndex = rowI * imageWidth + colI;
            float myDepth = depth.at<float>(rowI, colI);
            double x, y, z;
            backProject(colI, rowI, myDepth, x, y, z);
            spaceMap[myIndex * 3] = x;
            spaceMap[myIndex * 3 + 1] = y;
            spaceMap[myIndex * 3 + 2] = z;
        }
}

void SurfelFusion::calculatePixelsNormsKernel(int thread, int threadNum) {
    int stepRow = imageHeight / threadNum;
    int startRow = stepRow * thread;
    startRow = startRow > 1 ? startRow : 1;
    int endRow = startRow + stepRow;
    if (thread == threadNum - 1)
        endRow = imageHeight - 1;
    for (int rowI = startRow; rowI < endRow; rowI++)
        for (int colI = 1; colI < imageWidth - 1; colI++) {
            int myIndex = rowI * imageWidth + colI;
            float myX, myY, myZ;
            myX = spaceMap[myIndex * 3];
            myY = spaceMap[myIndex * 3 + 1];
            myZ = spaceMap[myIndex * 3 + 2];
            float rightX, rightY, rightZ;
            rightX = spaceMap[myIndex * 3 + 3];
            rightY = spaceMap[myIndex * 3 + 4];
            rightZ = spaceMap[myIndex * 3 + 5];
            float downX, downY, downZ;
            downX = spaceMap[myIndex * 3 + imageWidth * 3];
            downY = spaceMap[myIndex * 3 + imageWidth * 3 + 1];
            downZ = spaceMap[myIndex * 3 + imageWidth * 3 + 2];
            if (myZ < 0.1 || rightZ < 0.1 || downZ < 0.1)
                continue;
            rightX = rightX - myX;
            rightY = rightY - myY;
            rightZ = rightZ - myZ;
            downX = downX - myX;
            downY = downY - myY;
            downZ = downZ - myZ;
            float normX, normY, normZ, normLength;
            normX = rightY * downZ - rightZ * downY;
            normY = rightZ * downX - rightX * downZ;
            normZ = rightX * downY - rightY * downX;
            normLength = std::sqrt(normX * normX + normY * normY + normZ * normZ);
            normX /= normLength;
            normY /= normLength;
            normZ /= normLength;
            float viewAngle = (normX * myX + normY * myY + normZ * myZ)
                              / std::sqrt(myX * myX + myY * myY + myZ * myZ);
            if (viewAngle > -MAX_ANGLE_COS && viewAngle < MAX_ANGLE_COS)
                continue;
            normMap[myIndex * 3] = normX;
            normMap[myIndex * 3 + 1] = normY;
            normMap[myIndex * 3 + 2] = normZ;
        }
}

void SurfelFusion::calculateSpDepthNormsKernel(int thread, int threadNum) {
    int step = superpixelSeeds.size() / threadNum;
    int beginIndex = step * thread;
    int endIndex = beginIndex + step;
    if (thread == threadNum - 1)
        endIndex = superpixelSeeds.size();
    for (int seedI = beginIndex; seedI < endIndex; seedI++) {
        int spX = seedI % spWidth;
        int spY = seedI / spWidth;
        int checkXBegin = spX * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        int checkYBegin = spY * SP_SIZE + SP_SIZE / 2 - SP_SIZE;
        std::vector<float> pixelDepth;
        std::vector<float> pixelNorms;
        std::vector<float> pixelPositions;
        std::vector<float> pixelInlierPositions;
        float validDepthNum = 0;
        float maxDist = 0;
        for (int checkJ = checkYBegin; checkJ < (checkYBegin + SP_SIZE * 2); checkJ++) {
            for (int checkI = checkXBegin; checkI < (checkXBegin + SP_SIZE * 2); checkI++) {
                int pixelIndex = checkJ * imageWidth + checkI;
                if (pixelIndex < 0 || pixelIndex >= superpixelIndex.size())
                    continue;
                if (superpixelIndex[pixelIndex] == seedI) {
                    float xDiff = checkI - superpixelSeeds[seedI].x;
                    float yDiff = checkJ - superpixelSeeds[seedI].y;
                    float dist = xDiff * xDiff + yDiff * yDiff;
                    if (dist > maxDist)
                        maxDist = dist;

                    float myDepth = depth.at<float>(checkJ, checkI);
                    if (myDepth > 0.05) {
                        pixelDepth.push_back(myDepth);
                        pixelNorms.push_back(normMap[pixelIndex * 3]);
                        pixelNorms.push_back(normMap[pixelIndex * 3 + 1]);
                        pixelNorms.push_back(normMap[pixelIndex * 3 + 2]);
                        validDepthNum += 1;
                        pixelPositions.push_back(spaceMap[pixelIndex * 3]);
                        pixelPositions.push_back(spaceMap[pixelIndex * 3 + 1]);
                        pixelPositions.push_back(spaceMap[pixelIndex * 3 + 2]);
                    }
                }
            }
        }
        if (validDepthNum < 16)
            continue;
        float meanDepth = superpixelSeeds[seedI].meanDepth;
        float normX, normY, normZ, normB;
        normX = normY = normZ = normB = 0.0;
        float inlierNum = 0;
        for (int pI = 0; pI < pixelDepth.size(); pI++) {
            float residual = meanDepth - pixelDepth[pI];
            if (residual < HUBER_RANGE && residual > -HUBER_RANGE) {
                normX += pixelNorms[pI * 3];
                normY += pixelNorms[pI * 3 + 1];
                normZ += pixelNorms[pI * 3 + 2];
                inlierNum += 1;
                // for test huber norm
                pixelInlierPositions.push_back(pixelPositions[pI * 3]);
                pixelInlierPositions.push_back(pixelPositions[pI * 3 + 1]);
                pixelInlierPositions.push_back(pixelPositions[pI * 3 + 2]);
            }
        }
        if (inlierNum / pixelDepth.size() < 0.8)
            continue;
        float normLength = std::sqrt(normX * normX + normY * normY + normZ * normZ);
        normX = normX / normLength;
        normY = normY / normLength;
        normZ = normZ / normLength;
        {
            // Robust norm
            float gnNx = normX;
            float gnNy = normY;
            float gnNz = normZ;
            float gnNb = normB;
            getHuberNorm(gnNx, gnNy, gnNz, gnNb, pixelInlierPositions);
            normX = gnNx;
            normY = gnNy;
            normZ = gnNz;
            normB = gnNb;
        }
        double avgX, avgY, avgZ;
        backProject(
                superpixelSeeds[seedI].x, superpixelSeeds[seedI].y, meanDepth,
                avgX, avgY, avgZ);
        {
            // Make sure the avgX, avgY, and avgZ are one the surfel
            float k = -1 * (avgX * normX + avgY * normY + avgZ * normZ) - normB;
            avgX += k * normX;
            avgY += k * normY;
            avgZ += k * normZ;
            meanDepth = avgZ;
        }
        float viewCos = -1.0 * (normX * avgX + normY * avgY + normZ * avgZ) /
                        std::sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ);
        if (viewCos < 0) {
            viewCos *= -1.0;
            normX *= -1.0;
            normY *= -1.0;
            normZ *= -1.0;
        }
        superpixelSeeds[seedI].normX = normX;
        superpixelSeeds[seedI].normY = normY;
        superpixelSeeds[seedI].normZ = normZ;
        superpixelSeeds[seedI].posX = avgX;
        superpixelSeeds[seedI].posY = avgY;
        superpixelSeeds[seedI].posZ = avgZ;
        superpixelSeeds[seedI].meanDepth = meanDepth;
        superpixelSeeds[seedI].viewCos = viewCos;
        superpixelSeeds[seedI].size = std::sqrt(maxDist);
    }
}

void SurfelFusion::calculateNorms() {
    std::vector<std::thread> threadPool;
    for (int i = 0; i < THREAD_NUM; i++) {
        std::thread thisThread(&SurfelFusion::calculateSpacesKernel, this, i, THREAD_NUM);
        threadPool.push_back(std::move(thisThread));
    }
    for (int i = 0; i < threadPool.size(); i++)
        if (threadPool[i].joinable())
            threadPool[i].join();
    threadPool.clear();

    for (int i = 0; i < THREAD_NUM; i++) {
        std::thread thisThread(&SurfelFusion::calculatePixelsNormsKernel, this, i, THREAD_NUM);
        threadPool.push_back(std::move(thisThread));
    }
    for (int i = 0; i < threadPool.size(); i++)
        if (threadPool[i].joinable())
            threadPool[i].join();
    threadPool.clear();

    for (int i = 0; i < THREAD_NUM; i++) {
        std::thread thisThread(&SurfelFusion::calculateSpDepthNormsKernel, this, i, THREAD_NUM);
        threadPool.push_back(std::move(thisThread));
    }
    for (int i = 0; i < threadPool.size(); i++)
        if (threadPool[i].joinable())
            threadPool[i].join();
    threadPool.clear();
}

void SurfelFusion::generateSuperPixels() {
    memset(superpixelSeeds.data(), 0, superpixelSeeds.size() * sizeof(SuperpixelSeed));
    std::fill(superpixelIndex.begin(), superpixelIndex.end(), 0);
    std::fill(normMap.begin(), normMap.end(), 0);
    initializeSeeds();

    for (int itI = 0; itI < ITERATION_NUM; itI++) {
        updatePixels();
        updateSeeds();
    }

    calculateNorms();
}