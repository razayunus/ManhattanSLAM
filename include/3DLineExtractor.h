/**
* This file is part of Structure-SLAM.
* Copyright (C) 2020 Yanyan Li <yanyan.li at tum.de> (Technical University of Munich)
*
*/

#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H


#include <stdio.h>
#include <fstream>
#include<iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cxcore.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define EPS    (1e-10)

typedef struct RandomPoint3ds {
    cv::Point3d pos;
    double W_sqrt[3]; // used for mah-dist from pt to ln
    cv::Mat U, W; // cov = U*D*U.t, D = diag(W); W is vector
    double DU[9];

    RandomPoint3ds() {}

} RandomPoint3d;

typedef struct RandomLine3ds {

    std::vector<RandomPoint3d> pts;  //supporting collinear points
    cv::Point3d A, B;
    cv::Point3d u, d; // following the representation of Zhang's paper 'determining motion from...'
    RandomLine3ds() {}

} RandomLine3d;

template<class bidiiter>
//Fisher-Yates shuffle
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        bidiiter r = begin;
        std::advance(r, rand() % left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

bool verify3dLine(const std::vector<RandomPoint3d> &pts, const cv::Point3d &A, const cv::Point3d &B);

void computeLine3d_svd(const std::vector<RandomPoint3d> &pts, const std::vector<int> &idx, cv::Point3d &mean,
                       cv::Point3d &drct);

cv::Point3d projectPt3d2Ln3d(const cv::Point3d &P, const cv::Point3d &mid, const cv::Point3d &drct);

cv::Mat array2mat(double a[], int n);

cv::Point3d mat2cvpt3d(cv::Mat m);

RandomLine3d extract3dline_mahdist(const std::vector<RandomPoint3d> &pts);

double mah_dist3d_pt_line(const RandomPoint3d &pt, const cv::Point3d &q1, const cv::Point3d &q2);

void computeLine3d_svd(const std::vector<RandomPoint3d> &pts, const std::vector<int> &idx, cv::Point3d &mean,
                       cv::Point3d &drct);

RandomPoint3d compPt3dCov(cv::Point3d pt, cv::Mat K, double);

double depthStdDev(double d);

#endif LINEEXTRACTOR_H
