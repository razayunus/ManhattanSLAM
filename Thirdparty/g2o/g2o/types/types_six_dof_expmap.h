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

// Modified by Raúl Mur Artal (2014)
// Added EdgeSE3ProjectXYZ (project using focal_length in x,y directions)
// Modified by Raúl Mur Artal (2016)
// Added EdgeStereoSE3ProjectXYZ (project using focal_length in x,y directions)
// Added EdgeSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)
// Added EdgeStereoSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)

#ifndef G2O_SIX_DOF_TYPES_EXPMAP
#define G2O_SIX_DOF_TYPES_EXPMAP

#include "../core/base_vertex.h"
#include "../core/base_binary_edge.h"
#include "../core/base_unary_edge.h"
#include "se3_ops.h"
#include "se3quat.h"
#include "types_sba.h"
#include "plane_3d.h"
#include <Eigen/Geometry>

namespace g2o {
    namespace types_six_dof_expmap {
        void init();
    }

    using namespace Eigen;

/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
    class VertexSE3Expmap : public BaseVertex<6, SE3Quat> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexSE3Expmap();

        bool read(std::istream &is);

        bool write(std::ostream &os) const;

        virtual void setToOriginImpl() {
            _estimate = SE3Quat();
        }

        virtual void oplusImpl(const double *update_) {
            Eigen::Map<const Vector6d> update(update_);
            setEstimate(SE3Quat::exp(update) * estimate());
        }
    };

    class EdgeSE3ProjectXYZOnlyPose : public BaseUnaryEdge<2, Vector2d, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeSE3ProjectXYZOnlyPose() {}

        bool read(std::istream &is);

        bool write(std::ostream &os) const;

        void computeError() {
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Vector2d obs(_measurement);
            _error = obs - cam_project(v1->estimate().map(Xw));
        }

        virtual void linearizeOplus();

        Vector2d cam_project(const Vector3d &trans_xyz) const;

        Vector3d Xw;
        double fx, fy, cx, cy;
    };

//Manhattan world, compute translation only
    class EdgeSE3ProjectXYZOnlyTranslation : public BaseUnaryEdge<2, Vector2d, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeSE3ProjectXYZOnlyTranslation() {}

        bool read(std::istream &is);

        bool write(std::ostream &os) const;

        void computeError() {
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Vector2d obs(_measurement);
            _error = obs - cam_project(v1->estimate().mapTrans(Xc));
        }

        virtual void linearizeOplus();

        Vector2d cam_project(const Vector3d &trans_xyz) const;

        Vector3d Xc;
        double fx, fy, cx, cy;
    };

    class EdgeStereoSE3ProjectXYZOnlyPose : public BaseUnaryEdge<3, Vector3d, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeStereoSE3ProjectXYZOnlyPose() {}

        bool read(std::istream &is);

        bool write(std::ostream &os) const;

        void computeError() {
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Vector3d obs(_measurement);
            _error = obs - cam_project(v1->estimate().map(Xw));
        }

        virtual void linearizeOplus();

        Vector3d cam_project(const Vector3d &trans_xyz) const;

        Vector3d Xw;
        double fx, fy, cx, cy, bf;
    };

    class EdgeStereoSE3ProjectXYZOnlyTranslation : public BaseUnaryEdge<3, Vector3d, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeStereoSE3ProjectXYZOnlyTranslation() {}

        bool read(std::istream &is);

        bool write(std::ostream &os) const;

        void computeError() {
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Vector3d obs(_measurement);
            _error = obs - cam_project(v1->estimate().mapTrans(Xc));
        }

        virtual void linearizeOplus();

        Vector3d cam_project(const Vector3d &trans_xyz) const;

        Vector3d Xc;
        double fx, fy, cx, cy, bf;
    };

    class EdgeLineProjectXYZOnlyPose : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeLineProjectXYZOnlyPose() {}

        virtual void computeError() {
            const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
            Eigen::Vector3d obs = _measurement;
            Eigen::Vector2d proj = cam_project(v1->estimate().map(Xw));
            _error(0) = obs(0) * proj(0) + obs(1) * proj(1) + obs(2);
            _error(1) = 0;
            _error(2) = 0;
        }

        double chiline() {
            return _error(0) * _error(0);
        }

        virtual void linearizeOplus() {
            g2o::VertexSE3Expmap *vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
            Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

            double x = xyz_trans[0];
            double y = xyz_trans[1];
            double invz = 1.0 / xyz_trans[2];
            double invz_2 = invz * invz;

            double lx = _measurement(0);
            double ly = _measurement(1);

            _jacobianOplusXi(0, 0) = -fy * ly - fx * lx * x * y * invz_2 - fy * ly * y * y * invz_2;
            _jacobianOplusXi(0, 1) = fx * lx + fx * lx * x * x * invz_2 + fy * ly * x * y * invz_2;
            _jacobianOplusXi(0, 2) = -fx * lx * y * invz + fy * ly * x * invz;
            _jacobianOplusXi(0, 3) = fx * lx * invz;
            _jacobianOplusXi(0, 4) = fy * ly * invz;
            _jacobianOplusXi(0, 5) = -(fx * lx * x + fy * ly * y) * invz_2;
            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
            _jacobianOplusXi(1, 3) = 0;
            _jacobianOplusXi(1, 4) = 0;
            _jacobianOplusXi(1, 5) = 0;
            _jacobianOplusXi(2, 0) = 0;
            _jacobianOplusXi(2, 1) = 0;
            _jacobianOplusXi(2, 2) = 0;
            _jacobianOplusXi(2, 3) = 0;
            _jacobianOplusXi(2, 4) = 0;
            _jacobianOplusXi(2, 5) = 0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        Eigen::Vector2d cam_project(const Eigen::Vector3d &trans_xyz) {
            Eigen::Vector2d proj = g2o::project(trans_xyz);
            Eigen::Vector2d res;
            res[0] = proj[0] * fx + cx;
            res[1] = proj[1] * fy + cy;
            return res;
        }

        Eigen::Vector3d Xw;
        double fx, fy, cx, cy;
    };

    class EdgeLineProjectXYZOnlyTranslation : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeLineProjectXYZOnlyTranslation() {}

        virtual void computeError() {
            const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
            Eigen::Vector3d obs = _measurement;
            Eigen::Vector2d proj = cam_project(v1->estimate().mapTrans(Xc));
            _error(0) = obs(0) * proj(0) + obs(1) * proj(1) + obs(2);
            _error(1) = 0;
            _error(2) = 0;
        }

        double chiline() {
            return _error(0) * _error(0);
        }

        virtual void linearizeOplus() {
            const g2o::VertexSE3Expmap *vi = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
            Eigen::Vector3d xyz_trans = vi->estimate().mapTrans(Xc);

            double x = xyz_trans[0];
            double y = xyz_trans[1];
            double invz = 1.0 / xyz_trans[2];
            double invz_2 = invz * invz;

            double lx = _measurement(0);
            double ly = _measurement(1);

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;
            _jacobianOplusXi(0, 3) = fx * lx * invz;
            _jacobianOplusXi(0, 4) = fy * ly * invz;
            _jacobianOplusXi(0, 5) = -(fx * lx * x + fy * ly * y) * invz_2;
            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
            _jacobianOplusXi(1, 3) = 0;
            _jacobianOplusXi(1, 4) = 0;
            _jacobianOplusXi(1, 5) = 0;
            _jacobianOplusXi(2, 0) = 0;
            _jacobianOplusXi(2, 1) = 0;
            _jacobianOplusXi(2, 2) = 0;
            _jacobianOplusXi(2, 3) = 0;
            _jacobianOplusXi(2, 4) = 0;
            _jacobianOplusXi(2, 5) = 0;
        }

        bool read(std::istream &is) {
            for (int i = 0; i < 3; i++) {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const {
            for (int i = 0; i < 3; i++) {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i; j < 3; ++j) {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        Eigen::Vector2d cam_project(const Eigen::Vector3d &trans_xyz) {
            Eigen::Vector2d proj = g2o::project(trans_xyz);
            Eigen::Vector2d res;
            res[0] = proj[0] * fx + cx;
            res[1] = proj[1] * fy + cy;
            return res;
        }

        Eigen::Vector3d Xc;
        double fx, fy, cx, cy;
    };

    class EdgePlaneOnlyPose : public BaseUnaryEdge<3, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePlaneOnlyPose() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

            _error = localPlane.ominus(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        Plane3D Xw;
    };

    class EdgePlaneOnlyTranslation : public BaseUnaryEdge<3, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePlaneOnlyTranslation() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            _error = localPlane.ominus(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;

            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;

            _jacobianOplusXi(2, 0) = 0;
            _jacobianOplusXi(2, 1) = 0;
            _jacobianOplusXi(2, 2) = 0;
        }

        Plane3D Xc;
    };

    class EdgeParallelPlaneOnlyPose : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeParallelPlaneOnlyPose() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        Plane3D Xw;
    };

    class EdgeParallelPlaneOnlyTranslation : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeParallelPlaneOnlyTranslation() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            // measurement function: remap the plane in global coordinates
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;

            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
        }

        Plane3D Xc;
    };

    class EdgeVerticalPlaneOnlyPose : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeVerticalPlaneOnlyPose() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

            _error = localPlane.ominus_ver(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        Plane3D Xw;
    };

    class EdgeVerticalPlaneOnlyTranslation : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeVerticalPlaneOnlyTranslation() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            // measurement function: remap the plane in global coordinates
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            _error = localPlane.ominus_ver(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;

            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
        }

        Plane3D Xc;
    };

} // end namespace

#endif
