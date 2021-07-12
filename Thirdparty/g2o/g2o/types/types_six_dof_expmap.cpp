// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "types_six_dof_expmap.h"

#include "../core/factory.h"
#include "../stuff/macros.h"

namespace g2o {

    using namespace std;


    Vector2d project2d(const Vector3d &v) {
        Vector2d res;
        res(0) = v(0) / v(2);
        res(1) = v(1) / v(2);
        return res;
    }

    Vector3d unproject2d(const Vector2d &v) {
        Vector3d res;
        res(0) = v(0);
        res(1) = v(1);
        res(2) = 1;
        return res;
    }

    VertexSE3Expmap::VertexSE3Expmap() : BaseVertex<6, SE3Quat>() {
    }

    bool VertexSE3Expmap::read(std::istream &is) {
        Vector7d est;
        for (int i = 0; i < 7; i++)
            is >> est[i];
        SE3Quat cam2world;
        cam2world.fromVector(est);
        setEstimate(cam2world.inverse());
        return true;
    }

    bool VertexSE3Expmap::write(std::ostream &os) const {
        SE3Quat cam2world(estimate().inverse());
        for (int i = 0; i < 7; i++)
            os << cam2world[i] << " ";
        return os.good();
    }

// Only Pose

    bool EdgeSE3ProjectXYZOnlyPose::read(std::istream &is) {
        for (int i = 0; i < 2; i++) {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream &os) const {

        for (int i = 0; i < 2; i++) {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++) {
                os << " " << information()(i, j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
        VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
        Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
        _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
        _jacobianOplusXi(0, 2) = y * invz * fx;
        _jacobianOplusXi(0, 3) = -invz * fx;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = x * invz_2 * fx;

        _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
        _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
        _jacobianOplusXi(1, 2) = -x * invz * fy;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -invz * fy;
        _jacobianOplusXi(1, 5) = y * invz_2 * fy;
    }

    Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d &trans_xyz) const {
        Vector2d proj = project2d(trans_xyz);
        Vector2d res;
        res[0] = proj[0] * fx + cx;
        res[1] = proj[1] * fy + cy;
        return res;
    }


    Vector3d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Vector3d &trans_xyz) const {
        const float invz = 1.0f / trans_xyz[2];
        Vector3d res;
        res[0] = trans_xyz[0] * invz * fx + cx;
        res[1] = trans_xyz[1] * invz * fy + cy;
        res[2] = res[0] - bf * invz;
        return res;
    }


    bool EdgeStereoSE3ProjectXYZOnlyPose::read(std::istream &is) {
        for (int i = 0; i <= 3; i++) {
            is >> _measurement[i];
        }
        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeStereoSE3ProjectXYZOnlyPose::write(std::ostream &os) const {

        for (int i = 0; i <= 3; i++) {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++) {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
        VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
        Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
        _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
        _jacobianOplusXi(0, 2) = y * invz * fx;
        _jacobianOplusXi(0, 3) = -invz * fx;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = x * invz_2 * fx;

        _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
        _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
        _jacobianOplusXi(1, 2) = -x * invz * fy;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -invz * fy;
        _jacobianOplusXi(1, 5) = y * invz_2 * fy;

        _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * y * invz_2;
        _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) + bf * x * invz_2;
        _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2);
        _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - bf * invz_2;
    }

// Translation only

    Vector3d EdgeStereoSE3ProjectXYZOnlyTranslation::cam_project(const Vector3d &trans_xyz) const {
        const float invz = 1.0f / trans_xyz[2];
        Vector3d res;
        res[0] = trans_xyz[0] * invz * fx + cx;
        res[1] = trans_xyz[1] * invz * fy + cy;
        res[2] = res[0] - bf * invz;
        return res;
    }


    bool EdgeStereoSE3ProjectXYZOnlyTranslation::read(std::istream &is) {
        for (int i = 0; i <= 3; i++) {
            is >> _measurement[i];
        }
        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeStereoSE3ProjectXYZOnlyTranslation::write(std::ostream &os) const {

        for (int i = 0; i <= 3; i++) {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i <= 2; i++)
            for (int j = i; j <= 2; j++) {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeStereoSE3ProjectXYZOnlyTranslation::linearizeOplus() {
        VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
        Vector3d xyz_trans = vi->estimate().mapTrans(Xc);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = 0;
        _jacobianOplusXi(0, 2) = 0;
        _jacobianOplusXi(0, 3) = -invz * fx;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = x * invz_2 * fx;

        _jacobianOplusXi(1, 0) = 0;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(1, 2) = 0;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -invz * fy;
        _jacobianOplusXi(1, 5) = y * invz_2 * fy;

        _jacobianOplusXi(2, 0) = 0;
        _jacobianOplusXi(2, 1) = 0;
        _jacobianOplusXi(2, 2) = 0;
        _jacobianOplusXi(2, 3) = _jacobianOplusXi(0, 3);
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = _jacobianOplusXi(0, 5) - bf * invz_2;
    }

    bool EdgeSE3ProjectXYZOnlyTranslation::read(std::istream &is) {
        for (int i = 0; i < 2; i++) {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyTranslation::write(std::ostream &os) const {

        for (int i = 0; i < 2; i++) {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++) {
                os << " " << information()(i, j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZOnlyTranslation::linearizeOplus() {
        VertexSE3Expmap *vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
        Vector3d xyz_trans = vi->estimate().mapTrans(Xc);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = 0;
        _jacobianOplusXi(0, 2) = 0;
        _jacobianOplusXi(0, 3) = -invz * fx;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = x * invz_2 * fx;

        _jacobianOplusXi(1, 0) = 0;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(1, 2) = 0;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -invz * fy;
        _jacobianOplusXi(1, 5) = y * invz_2 * fy;
    }

    Vector2d EdgeSE3ProjectXYZOnlyTranslation::cam_project(const Vector3d &trans_xyz) const {
        Vector2d proj = project2d(trans_xyz);
        Vector2d res;
        res[0] = proj[0] * fx + cx;
        res[1] = proj[1] * fy + cy;
        return res;
    }


} // end namespace
