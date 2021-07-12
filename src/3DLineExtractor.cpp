/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/

#include "3DLineExtractor.h"

using namespace std;
using namespace cv;

cv::Point3d projectPt3d2Ln3d(const cv::Point3d &P, const cv::Point3d &mid, const cv::Point3d &drct)
// project a 3d point P to a 3d line (represented with midpt and direction)
{
    cv::Point3d A = mid;
    cv::Point3d B = mid + drct;
    cv::Point3d AB = B - A;
    cv::Point3d AP = P - A;
    return A + (AB.dot(AP) / (AB.dot(AB))) * AB;
}

cv::Mat array2mat(double a[], int n) // inhomo mat
// n is the size of a[]
{
    return cv::Mat(n, 1, CV_64F, a);
}

cv::Point3d mat2cvpt3d(cv::Mat m) {
    if (m.cols * m.rows == 3)
        return cv::Point3d(m.at<double>(0),
                           m.at<double>(1),
                           m.at<double>(2));
    else
        cerr << "input matrix dimmension wrong!";
}

void computeLine3d_svd(const vector<RandomPoint3d> &pts, const vector<int> &idx, cv::Point3d &mean, cv::Point3d &drct)
// input: collinear 3d points with noise
// output: line direction vector and point
// method: linear equation, PCA
{
    int n = idx.size();
    mean = cv::Point3d(0, 0, 0);
    for (int i = 0; i < n; ++i) {
        mean = mean + pts[idx[i]].pos;
    }
    mean = mean * (1.0 / n);
    cv::Mat P(3, n, CV_64F);
    for (int i = 0; i < n; ++i) {
        //	pts[i].pos =  pts[i].pos - mean;
        //	cvpt2mat(pts[i].pos,0).copyTo(P.col(i));
        double pos[3] = {pts[idx[i]].pos.x - mean.x, pts[idx[i]].pos.y - mean.y, pts[idx[i]].pos.z - mean.z};
        array2mat(pos, 3).copyTo(P.col(i));
    }

    cv::SVD svd(P.t(), cv::SVD::MODIFY_A);  // FULL_UV is 60 times slower

    drct = mat2cvpt3d(svd.vt.row(0));
}


double depthStdDev(double d)
// standard deviation of depth d
// in meter
{
    double c1, c2, c3;

    c1 = 0.00273;//depth_stdev_coeff_c1;
    c2 = 0.00074;//depth_stdev_coeff_c2;
    c3 = -0.00058;//depth_stdev_coeff_c3;

    return c1 * d * d + c2 * d + c3;
}

RandomPoint3d compPt3dCov(cv::Point3d pt, cv::Mat K, double time_diff_sec) {
    RandomPoint3d rp;

    double f = K.at<double>(0, 0), // focal length
    cu = K.at<double>(0, 2),
            cv = K.at<double>(1, 2);

    //// opencv mat operation is slower than armadillo
    cv::Mat J0 = (cv::Mat_<double>(3, 3) << pt.z / f, 0, pt.x / pt.z,
            0, pt.z / f, pt.y / pt.z,
            0, 0, 1);

    cv::Mat cov_g_d0 = (cv::Mat_<double>(3, 3) << 1, 0, 0,
            0, 1, 0,
            0, 0, depthStdDev(pt.z) * depthStdDev(pt.z));
    cv::Mat cov0 = J0 * cov_g_d0 * J0.t();
    rp.pos = pt;

    cv::SVD svd(cov0);
    rp.U = svd.u.clone();
    rp.W = svd.w.clone();
    rp.W_sqrt[0] = sqrt(svd.w.at<double>(0));
    rp.W_sqrt[1] = sqrt(svd.w.at<double>(1));
    rp.W_sqrt[2] = sqrt(svd.w.at<double>(2));

    cv::Mat D = (cv::Mat_<double>(3, 3) << 1 / rp.W_sqrt[0], 0, 0,
            0, 1 / rp.W_sqrt[1], 0,
            0, 0, 1 / rp.W_sqrt[2]);

    cv::Mat du = D * rp.U.t();
    rp.DU[0] = du.at<double>(0, 0);
    rp.DU[1] = du.at<double>(0, 1);
    rp.DU[2] = du.at<double>(0, 2);
    rp.DU[3] = du.at<double>(1, 0);
    rp.DU[4] = du.at<double>(1, 1);
    rp.DU[5] = du.at<double>(1, 2);
    rp.DU[6] = du.at<double>(2, 0);
    rp.DU[7] = du.at<double>(2, 1);
    rp.DU[8] = du.at<double>(2, 2);

    return rp;
}

RandomLine3d extract3dline_mahdist(const vector<RandomPoint3d> &pts)
// extract a single 3d line from point clouds using ransac and mahalanobis distance
// input: 3d points and covariances
// output: inlier points, line parameters: midpt and direction
{
    int maxIterNo = min(10, int(pts.size() * (pts.size() - 1) * 0.5));
    double distThresh = 1.5;;//pt2line_mahdist_extractline; // meter
    // distance threshold should be adapted to line length and depth
    int minSolSetSize = 2;

    vector<int> indexes(pts.size());
    for (size_t i = 0; i < indexes.size(); ++i) indexes[i] = i;
    vector<int> maxInlierSet;
    RandomPoint3d bestA, bestB;
    for (int iter = 0; iter < maxIterNo; iter++) {
        vector<int> inlierSet;
        random_unique(indexes.begin(), indexes.end(), minSolSetSize);// shuffle
        const RandomPoint3d &A = pts[indexes[0]];

        const RandomPoint3d &B = pts[indexes[1]];
        //cout<<"A:"<<A.xyz[0]<<","<<A.xyz[1]<<","<<A.xyz[2]<<".B:"<<B.xyz[2]<<endl;

        if (cv::norm(B.pos - A.pos) < EPS) continue;
        for (size_t i = 0; i < pts.size(); ++i) {
            // compute distance to AB
            double dist = mah_dist3d_pt_line(pts[i], A.pos, B.pos);
            //cout<<"dist takes "<<dist<<endl;
            //cout<<"dist:A pos"<<A.pos.x<<","<<A.xyz[2]<<endl;
            //cout<<"Lineextractor: dist"<<dist<<endl;
            if (dist < distThresh) {
                inlierSet.push_back(i);
            }
        }
        if (inlierSet.size() > maxInlierSet.size()) {
            vector<RandomPoint3d> inlierPts(inlierSet.size());
            for (size_t ii = 0; ii < inlierSet.size(); ++ii)
                inlierPts[ii] = pts[inlierSet[ii]];
            if (verify3dLine(inlierPts, A.pos, B.pos)) {
                maxInlierSet = inlierSet;
                bestA = pts[indexes[0]];
                bestB = pts[indexes[1]];
            }
        }
        //cout<<"Lineextractor:"<<iter<<'\t'<<maxInlierSet.size()<<endl;
        if (maxInlierSet.size() > pts.size() * 0.6)
            break;
    }
    RandomLine3d rl;
    if (maxInlierSet.size() >= 2) {
        cv::Point3d m = (bestA.pos + bestB.pos) * 0.5, d = bestB.pos - bestA.pos;
        // optimize and reselect inliers
        // compute a 3d line using algebraic method
        while (true) {
            vector<int> tmpInlierSet;
            cv::Point3d tmp_m, tmp_d;
            computeLine3d_svd(pts, maxInlierSet, tmp_m, tmp_d);
            for (size_t i = 0; i < pts.size(); ++i) {
                if (mah_dist3d_pt_line(pts[i], tmp_m, tmp_m + tmp_d) < distThresh) {
                    tmpInlierSet.push_back(i);
                }
            }
            if (tmpInlierSet.size() > maxInlierSet.size()) {
                maxInlierSet = tmpInlierSet;
                m = tmp_m;
                d = tmp_d;
            } else
                break;
        }
        // find out two endpoints
        double minv = 100, maxv = -100;
        int idx_end1 = 0, idx_end2 = 0;
        for (size_t i = 0; i < maxInlierSet.size(); ++i) {
            double dproduct = (pts[maxInlierSet[i]].pos - m).dot(d);
            if (dproduct < minv) {
                minv = dproduct;
                idx_end1 = i;
            }
            if (dproduct > maxv) {
                maxv = dproduct;
                idx_end2 = i;
            }
        }
        rl.A = pts[maxInlierSet[idx_end1]].pos;
        rl.B = pts[maxInlierSet[idx_end2]].pos;
    }
    rl.pts.resize(maxInlierSet.size());
    for (size_t i = 0; i < maxInlierSet.size(); ++i) rl.pts[i] = pts[maxInlierSet[i]];
    return rl;
}

bool verify3dLine(const vector<RandomPoint3d> &pts, const cv::Point3d &A, const cv::Point3d &B)
// input: line AB, collinear points
// output: whether AB is a good representation for points
// method: divide AB (or CD, which is endpoints of the projected points on AB)
// into n sub-segments, detect how many sub-segments containing
// at least one point(projected onto AB), if too few, then it implies invalid line
{
    int nCells = 10; // number of cells
    int *cells = new int[nCells];
    double ratio = 0.7;
    for (int i = 0; i < nCells; ++i) cells[i] = 0;
    int nPts = pts.size();
    // find 2 extremities of points along the line direction
    double minv = 100, maxv = -100;
    int idx1 = 0, idx2 = 0;
    for (int i = 0; i < nPts; ++i) {
        if ((pts[i].pos - A).dot(B - A) < minv) {
            minv = (pts[i].pos - A).dot(B - A);
            idx1 = i;
        }
        if ((pts[i].pos - A).dot(B - A) > maxv) {
            maxv = (pts[i].pos - A).dot(B - A);
            idx2 = i;
        }
    }
    cv::Point3d C = projectPt3d2Ln3d(pts[idx1].pos, (A + B) * 0.5, B - A);
    cv::Point3d D = projectPt3d2Ln3d(pts[idx2].pos, (A + B) * 0.5, B - A);
    double cd = cv::norm(D - C);
    if (cd < EPS) {
        delete[] cells;
        return false;
    }
    for (int i = 0; i < nPts; ++i) {
        cv::Point3d X = pts[i].pos;
        double lambda = abs((X - C).dot(D - C) / cd / cd); // 0 <= lambd <=1
        if (lambda >= 1) {
            cells[nCells - 1] += 1;
        } else {
            cells[(unsigned int) floor(lambda * 10)] += 1;
        }
    }
    double sum = 0;
    for (int i = 0; i < nCells; ++i) {
        if (cells[i] > 0)
            sum = sum + 1;
    }

    delete[] cells;
    if (sum / nCells > ratio) {
        return true;
    } else {
        return false;
    }
}


double mah_dist3d_pt_line(const RandomPoint3d &pt, const cv::Point3d &q1, const cv::Point3d &q2)
// compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{
    if (pt.U.cols != 3) {
        cerr << "Error in mah_dist3d_pt_line: R matrix must be 3x3" << endl;
        return -1;
    }
    double out;
    double xa = q1.x, ya = q1.y, za = q1.z;
    double xb = q2.x, yb = q2.y, zb = q2.z;
    double c1 = pt.DU[0], c2 = pt.DU[1], c3 = pt.DU[2],
            c4 = pt.DU[3], c5 = pt.DU[4], c6 = pt.DU[5],
            c7 = pt.DU[6], c8 = pt.DU[7], c9 = pt.DU[8];
    double x1 = pt.pos.x, x2 = pt.pos.y, x3 = pt.pos.z;
    double term1 = (
            (c1 * (x1 - xa) + c2 * (x2 - ya) + c3 * (x3 - za)) * (c4 * (x1 - xb) + c5 * (x2 - yb) + c6 * (x3 - zb))
            - (c4 * (x1 - xa) + c5 * (x2 - ya) + c6 * (x3 - za)) * (c1 * (x1 - xb) + c2 * (x2 - yb) + c3 * (x3 - zb))),
            term2 = (
            (c1 * (x1 - xa) + c2 * (x2 - ya) + c3 * (x3 - za)) * (c7 * (x1 - xb) + c8 * (x2 - yb) + c9 * (x3 - zb))
            - (c7 * (x1 - xa) + c8 * (x2 - ya) + c9 * (x3 - za)) * (c1 * (x1 - xb) + c2 * (x2 - yb) + c3 * (x3 - zb))),
            term3 = (
            (c4 * (x1 - xa) + c5 * (x2 - ya) + c6 * (x3 - za)) * (c7 * (x1 - xb) + c8 * (x2 - yb) + c9 * (x3 - zb))
            - (c7 * (x1 - xa) + c8 * (x2 - ya) + c9 * (x3 - za)) * (c4 * (x1 - xb) + c5 * (x2 - yb) + c6 * (x3 - zb))),
            term4 = (c1 * (x1 - xa) - c1 * (x1 - xb) + c2 * (x2 - ya) - c2 * (x2 - yb) + c3 * (x3 - za) -
                     c3 * (x3 - zb)),
            term5 = (c4 * (x1 - xa) - c4 * (x1 - xb) + c5 * (x2 - ya) - c5 * (x2 - yb) + c6 * (x3 - za) -
                     c6 * (x3 - zb)),
            term6 = (c7 * (x1 - xa) - c7 * (x1 - xb) + c8 * (x2 - ya) - c8 * (x2 - yb) + c9 * (x3 - za) -
                     c9 * (x3 - zb));
    out = sqrt((term1 * term1 + term2 * term2 + term3 * term3) / (term4 * term4 + term5 * term5 + term6 * term6));
    return out;
}