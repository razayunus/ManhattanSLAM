## List of Known Dependencies

In this document we list all the pieces of code included by ManhattanSLAM and linked libraries which are not property of
the authors of ManhattanSLAM. The code of this project is based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

### For code adopted from ORB-SLAM2

##### Code in **src** and **include** folders

* *ORBextractor.cc*. This is a modified version of orb.cpp of OpenCV library. The original code is BSD licensed.

* *PnPsolver.h, PnPsolver.cc*. This is a modified version of the epnp.h and epnp.cc of Vincent Lepetit. This code can be
  found in popular BSD licensed computer vision libraries
  as [OpenCV](https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/epnp.cpp)
  and [OpenGV](https://github.com/laurentkneip/opengv/blob/master/src/absolute_pose/modules/Epnp.cpp). The original code
  is FreeBSD.

* Function *ORBmatcher::DescriptorDistance* in *ORBmatcher.cc*. The code is
  from: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel. The code is in the public domain.

##### Code in Thirdparty folder

* All code in **DBoW2** folder. This is a modified version of [DBoW2](https://github.com/dorian3d/DBoW2)
  and [DLib](https://github.com/dorian3d/DLib) library. All files included are BSD licensed.

* All code in **g2o** folder. This is a modified version of [g2o](https://github.com/RainerKuemmerle/g2o). All files
  included are BSD licensed.

##### Library dependencies

* **Pangolin (visualization and user interface)**.
  [MIT license](https://en.wikipedia.org/wiki/MIT_License).

* **OpenCV**. BSD license.

* **Eigen3**. For versions greater than 3.1.1 is MPL2, earlier versions are LGPLv3.

### For code specific to ManhattanSLAM

##### Code in **src** and **include** folders

* *Surfel.h, SurfelFusion.h, SurfelMapping.h, SurfelFusion.cc, SurfelMapping.cc*. This code is adopted and modified
  from https://github.com/HKUST-Aerial-Robotics/DenseSurfelMapping, which is open-source.

* *include/peac/**. This code is adopted from https://github.com/ai4ce/peac, which allows the use of code for research
  purposes, given the copy of thier license appears in all their files.

##### Library dependencies

* **PCL**. BSD license.

* **tinyply**. The software is in the public domain.
