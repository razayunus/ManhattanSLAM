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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/plane_3d.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include <mutex>

#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;
using namespace g2o;

namespace ORB_SLAM2 {

    Optimizer::Optimizer(double angleInfo, double disInfo, double parInfo, double verInfo, double planeChi,
                         double planeChiVP, double aTh, double parTh) : angleInfo(angleInfo), disInfo(disInfo),
                                                                        parInfo(parInfo),
                                                                        verInfo(verInfo), planeChi(planeChi),
                                                                        planeChiVP(planeChiVP), aTh(aTh),
                                                                        parTh(parTh) {}

    int Optimizer::PoseOptimization(Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);

        vector<double> vMonoPointInfo(N, 1);
        vector<double> vSteroPointInfo(N, 1);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);


            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }

        const int NL = pFrame->NL;

        vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineEp;
        vector<size_t> vnIndexLineEdgeEp;
        vpEdgesLineEp.reserve(NL);
        vnIndexLineEdgeEp.reserve(NL);

        vector<double> vMonoStartPointInfo(NL, 1);
        vector<double> vMonoEndPointInfo(NL, 1);
        vector<double> vSteroStartPointInfo(NL, 1);
        vector<double> vSteroEndPointInfo(NL, 1);

        // Set MapLine vertices
        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    nInitialCorrespondences++;
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeLineProjectXYZOnlyPose *els = new EdgeLineProjectXYZOnlyPose();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity() * 1);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    els->Xw = pML->mWorldPos.head(3);
                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeLineProjectXYZOnlyPose *ele = new EdgeLineProjectXYZOnlyPose();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity() * 1);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    ele->Xw = pML->mWorldPos.tail(3);

                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
                }
            }
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;

        vector<g2o::EdgePlaneOnlyPose *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<g2o::EdgeParallelPlaneOnlyPose *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeVerticalPlaneOnlyPose *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgePlaneOnlyPose *e = new g2o::EdgePlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(sqrt(planeChi));

                    Isometry3D trans = static_cast<const VertexSE3Expmap *>(optimizer.vertex(0))->estimate();
                    cv::Mat Pc3D = pFrame->mvPlaneCoefficients[i];
                    Plane3D Pw3D = Converter::toPlane3D(pMP->GetWorldPos());
                    Vector4D Pw = Pw3D._coeffs;
                    Vector4D Pc;
                    Matrix3D R = trans.rotation();
                    Pc.head<3>() = R * Pw.head<3>();
                    Pc(3) = Pw(3) - trans.translation().dot(Pc.head<3>());

                    double angle = Pc(0) * Pc3D.at<float>(0) +
                                   Pc(1) * Pc3D.at<float>(1) +
                                   Pc(2) * Pc3D.at<float>(2);
                    if (angle < -aTh) {
                        Pw = -Pw;
                        Pw3D.fromVector(Pw);
                    }

                    e->Xw = Pw3D;

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

                    e->computeError();
                }
            }

            for (int i = 0; i < M; ++i) {
                // Add parallel planes
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeParallelPlaneOnlyPose *e = new g2o::EdgeParallelPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(sqrt(planeChiVP));

                    Isometry3D trans = static_cast<const VertexSE3Expmap *>(optimizer.vertex(0))->estimate();
                    cv::Mat Pc3D = pFrame->mvPlaneCoefficients[i];
                    Plane3D Pw3D = Converter::toPlane3D(pMP->GetWorldPos());
                    Vector4D Pw = Pw3D._coeffs;
                    Vector4D Pc;
                    Matrix3D R = trans.rotation();
                    Pc.head<3>() = R * Pw.head<3>();
                    Pc(3) = Pw(3) - trans.translation().dot(Pc.head<3>());

                    double angle = Pc(0) * Pc3D.at<float>(0) +
                                   Pc(1) * Pc3D.at<float>(1) +
                                   Pc(2) * Pc3D.at<float>(2);
                    if (angle < -parTh) {
                        Pw = -Pw;
                        Pw3D.fromVector(Pw);
                    }

                    e->Xw = Pw3D;

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                }
            }

            for (int i = 0; i < M; ++i) {
                // Add vertical planes
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeVerticalPlaneOnlyPose *e = new g2o::EdgeVerticalPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(sqrt(planeChiVP));

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                }
            }
        }

        if (nInitialCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;

        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutlier[idx] = false;
                    vMonoPointInfo[i] = 1.0 / sqrt(chi2);
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                    vSteroPointInfo[i] = 1.0 / sqrt(chi2);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeLineProjectXYZOnlyPose *e1 = vpEdgesLineSp[i];
                EdgeLineProjectXYZOnlyPose *e2 = vpEdgesLineEp[i];

                const size_t idx = vnIndexLineEdgeSp[i];

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }
                e1->computeError();
                e2->computeError();

                const float chi2_s = e1->chiline();
                const float chi2_e = e2->chiline();


                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                    vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
                    vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgePlaneOnlyPose *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeParallelPlaneOnlyPose *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > planeChiVP) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeVerticalPlaneOnlyPose *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > planeChiVP) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        pFrame->SetPose(Converter::toCvMat(SE3quat_recov));

        return nInitialCorrespondences - nBad;
    }

    int Optimizer::TranslationOptimization(ORB_SLAM2::Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        // Rotation
        cv::Mat R_cw = pFrame->mTcw.rowRange(0, 3).colRange(0, 3).clone();

        vector<g2o::EdgeSE3ProjectXYZOnlyTranslation *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);


        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyTranslation *e = new g2o::EdgeSE3ProjectXYZOnlyTranslation();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        cv::Mat Xc = R_cw * Xw;

                        e->Xc[0] = Xc.at<float>(0);
                        e->Xc[1] = Xc.at<float>(1);
                        e->Xc[2] = Xc.at<float>(2);


                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *e = new g2o::EdgeStereoSE3ProjectXYZOnlyTranslation();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        cv::Mat Xc = R_cw * Xw;

                        e->Xc[0] = Xc.at<float>(0);
                        e->Xc[1] = Xc.at<float>(1);
                        e->Xc[2] = Xc.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }

        const int NL = pFrame->NL;

        vector<EdgeLineProjectXYZOnlyTranslation *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeLineProjectXYZOnlyTranslation *> vpEdgesLineEp;
        vpEdgesLineEp.reserve(NL);

        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeLineProjectXYZOnlyTranslation *els = new EdgeLineProjectXYZOnlyTranslation();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity() * 1);//*vSteroStartPointInfo[i]);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    cv::Mat Xw = Converter::toCvVec(pML->mWorldPos.head(3));
                    cv::Mat Xc = R_cw * Xw;
                    els->Xc[0] = Xc.at<float>(0);
                    els->Xc[1] = Xc.at<float>(1);
                    els->Xc[2] = Xc.at<float>(2);

                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeLineProjectXYZOnlyTranslation *ele = new EdgeLineProjectXYZOnlyTranslation();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity() * 1);//vSteroEndPointInfo[i]);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    Xw = Converter::toCvVec(pML->mWorldPos.tail(3));
                    Xc = R_cw * Xw;
                    ele->Xc[0] = Xc.at<float>(0);
                    ele->Xc[1] = Xc.at<float>(1);
                    ele->Xc[2] = Xc.at<float>(2);


                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                }
            }
        }


        if (nInitialCorrespondences < 3) {
            return 0;
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;
        vector<g2o::EdgePlaneOnlyTranslation *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<g2o::EdgeParallelPlaneOnlyTranslation *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeVerticalPlaneOnlyTranslation *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgePlaneOnlyTranslation *e = new g2o::EdgePlaneOnlyTranslation();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    Isometry3D trans = static_cast<const VertexSE3Expmap *>(optimizer.vertex(0))->estimate();
                    cv::Mat Pc3D = pFrame->mvPlaneCoefficients[i];
                    Plane3D Pw3D = Converter::toPlane3D(pMP->GetWorldPos());
                    Vector4D Pw = Pw3D._coeffs;
                    Vector4D Pc;
                    Matrix3D R = trans.rotation();
                    Pc.head<3>() = R * Pw.head<3>();
                    Pc(3) = Pw(3) - trans.translation().dot(Pc.head<3>());

                    double angle = Pc(0) * Pc3D.at<float>(0) +
                                   Pc(1) * Pc3D.at<float>(1) +
                                   Pc(2) * Pc3D.at<float>(2);
                    if (angle < -aTh) {
                        Pw = -Pw;
                        Pw3D.fromVector(Pw);
                    }

                    Pw3D.rotateNormal(Converter::toMatrix3d(R_cw));
                    e->Xc = Pw3D;

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(sqrt(planeChi));

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);


                    e->computeError();
                }
            }
        }

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        int nLineBad = 0;
        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyTranslation *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            nLineBad = 0;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeLineProjectXYZOnlyTranslation *e1 = vpEdgesLineSp[i];
                EdgeLineProjectXYZOnlyTranslation *e2 = vpEdgesLineEp[i];

                const size_t idx = vnIndexLineEdgeSp[i];

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }

                const float chi2_s = e1->chiline();
                const float chi2_e = e2->chiline();

                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nLineBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgePlaneOnlyTranslation *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }
} //namespace ORB_SLAM
