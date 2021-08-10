##List of Known Dependencies 已知依賴項列表

###ORB-SLAM2 version 1.0

In this document we list all the pieces of code included  by ORB-SLAM2 and linked libraries which are not property of the authors of ORB-SLAM2.在本文檔中，我們列出了 ORB-SLAM2 和鏈接庫中包含的所有代碼段，它們不屬於 ORB-SLAM2 作者的財產。


#####Code in **src** and **include** folders

* *ORBextractor.cc*.
This is a modified version of orb.cpp of OpenCV library. The original code is BSD licensed. 這是 OpenCV 庫的 orb.cpp 的修改版本。原始代碼是 BSD 許可的。

* *PnPsolver.h, PnPsolver.cc*.
This is a modified version of the epnp.h and epnp.cc of Vincent Lepetit. 
This code can be found in popular BSD licensed computer vision libraries as [OpenCV](https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/epnp.cpp) and [OpenGV](https://github.com/laurentkneip/opengv/blob/master/src/absolute_pose/modules/Epnp.cpp). The original code is FreeBSD. 這是 Vincent Lepetit 的 epnp.h 和 epnp.cc 的修改版本。此代碼可以在流行的 BSD 許可計算機視覺庫中找到，如 OpenCV 和 OpenGV。原始代碼是 FreeBSD。

* Function *ORBmatcher::DescriptorDistance* in *ORBmatcher.cc*.
The code is from: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel.
The code is in the public domain. 代碼在公共領域。

#####Code in Thirdparty folder

* All code in **DBoW2** folder.
This is a modified version of [DBoW2](https://github.com/dorian3d/DBoW2) and [DLib](https://github.com/dorian3d/DLib) library. All files included are BSD licensed. DBoW2 文件夾中的所有代碼。這是 DBoW2 和 DLib 庫的修改版本。包含的所有文件都是 BSD 許可的。

* All code in **g2o** folder.
This is a modified version of [g2o](https://github.com/RainerKuemmerle/g2o). All files included are BSD licensed. g2o 文件夾中的所有代碼。這是 g2o 的修改版本。包含的所有文件都是 BSD 許可的。

#####Library dependencies 

* **Pangolin (visualization and user interface 可視化和用戶界面)**.
[MIT license](https://en.wikipedia.org/wiki/MIT_License).

* **OpenCV**.
BSD license.

* **Eigen3**.
For versions greater than 3.1.1 is MPL2, earlier versions are LGPLv3.

* **ROS (Optional, only if you build Examples/ROS)**.
BSD license. In the manifest.xml the only declared package dependencies are roscpp, tf, sensor_msgs, image_transport, cv_bridge, which are all BSD licensed. ROS（可選，僅當您構建示例/ROS 時）。 BSD 許可證。在 manifest.xml 中，唯一聲明的包依賴項是 roscpp、tf、sensor_msgs、image_transport、cv_bridge，它們都是 BSD 許可的。




