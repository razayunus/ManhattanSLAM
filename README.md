# ManhattanSLAM

**Authors:** Raza Yunus, Yanyan Li and Federico Tombari

ManhattanSLAM is a real-time SLAM library for **RGB-D** cameras that computes the camera pose trajectory, a sparse 3D
reconstruction (containing point, line and plane features) and a dense surfel-based 3D reconstruction. Further details
can be found in the related publication. The code is based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

<a href="https://www.youtube.com/embed/UE8A6mUOPLE" target="_blank"><img
src="https://img.youtube.com/vi/UE8A6mUOPLE/0.jpg"
alt="ManhattanSLAM" width="240" height="180" border="10" /></a>

### Related Publication:

Raza Yunus, Yanyan Li and Federico Tombari, **ManhattanSLAM: Robust Planar Tracking and Mapping Leveraging Mixture of
Manhattan Frames**, *in 2021 IEEE International Conference on Robotics and Automation (ICRA)
.* **[PDF](https://arxiv.org/pdf/2103.15068.pdf)**.

# 1. License

ManhattanSLAM is released under
a [GPLv3 license](https://github.com/razayunus/ManhattanSLAM/blob/master/License-gpl.txt). For a list of all
code/library dependencies (and associated licenses), please
see [Dependencies.md](https://github.com/razayunus/ManhattanSLAM/blob/master/Dependencies.md).

If you use ManhattanSLAM in an academic work, please cite:

```
@inproceedings{yunus2021manhattanslam,
    author = {R. Yunus, Y. Li and F. Tombari},
    title = {ManhattanSLAM: Robust Planar Tracking and Mapping Leveraging Mixture of Manhattan Frames},
    year = {2021},
    booktitle = {2021 IEEE international conference on Robotics and automation (ICRA)},
}
```

# 2. Prerequisites

We have tested the library in **Ubuntu 16.04** and **Ubuntu 20.04**, but it should be easy to compile on other platforms. A powerful
computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results. Following is the list
of dependecies for ManhattanSLAM and their versions tested by us:

- **OpenCV:** 3.3.0, 3.4.3
- **PCL:** 1.7.2, 1.10
- **Eigen3:** 3.3
- **DBoW2:** Included in Thirdparty folder
- **g2o:** Included in Thirdparty folder
- **Pangolin**
- **tinyply:** 2.3.2

# 3. Building and testing

Clone the repository:

```
git clone https://github.com/razayunus/ManhattanSLAM
```

There is a script `build.sh` to build the *Thirdparty* libraries and *ManhattanSLAM*. Please make sure you have
installed all required dependencies (see section 2). Execute:

```
cd ManhattanSLAM
chmod +x build.sh
./build.sh
```

This will create **libManhattanSLAM.so** in *lib* folder and the executable **manhattan_slam** in *Example* folder.

To test the system:

1. Download a sequence for one of the following datasets and uncompress it:
    - **TUM RGB-D: https://vision.in.tum.de/data/datasets/rgbd-dataset**
    - **ICL-NUIM: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html**
    - **TAMU RGB-D: http://telerobot.cs.tamu.edu/MFG/rgbd/livo/data.html**

2. Associate RGB images and depth images using the python
   script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). You can generate an associations
   file by executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

**Note:** For ICL-NUIM sequences, the association files are already given but the association is defined as ``depth > rgb`` rather than ``rgb > depth``. This can be changed by transforming ``associations.txt`` as:
```
cat associations.txt | sed 's/depth/temp/g;s/rgb/depth/g;s/temp/rgb/g' | tee associations.txt > /dev/null
```

3. Execute the following command. Change `Config.yaml` to ICL.yaml for ICL-NUIM sequences, TAMU.yaml for TAMU RGB-D
   sequences or TUM1.yaml, TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences of TUM RGB-D
   respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the
   path to the corresponding associations file.

  ```
  ./Example/manhattan_slam Vocabulary/ORBvoc.txt Example/Config.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
  ```
