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


#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <thread>
#include <opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

private:

    // Input sensor
    // System內部嵌套定義的枚舉類型，MONOCULAR=0, STEREO=1, RGBD=2
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    // 用於特征匹配和場景識別的詞匯表
    ORBVocabulary* orb_vocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    // 用於重定位和閉環檢測的關鍵幀數據庫
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    // ORB-SLAM中的地圖對象，用於保存所有關鍵幀和地圖點。
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    // 軌跡跟蹤器。接收一幀特征點計算相機位置，判定是否需要新增關鍵幀，創建一些新的地圖點，如果軌跡跟丟了將進行重定位。
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    // 局部地圖管理器。管理局部地圖，並進行局部的BA(local bundle adustment)。
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    // 閉環探測器。為每個新的關鍵幀搜索閉環。
    // 如果檢測到閉環就觸发一個位姿圖優化並在一個新的線程中進行完整的BA(full bundle adjustment)
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    // 基於 Pangolin 的可視化對象，用於繪制地圖和當前的相機位姿。
    Viewer* mpViewer;

    // 看字面意思應該是關鍵幀渲染器，用於通過 mpViewer 繪制關鍵幀(或者說是相機位姿)。
    FrameDrawer* mpFrameDrawer;

    // 看字面意思應該是地圖渲染器，用於通過 mpViewer 繪制地圖(或者說是地圖點)。
    MapDrawer* mpMapDrawer;

    // ==================================================
    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.

    // 用於進行 LOCAL MAPPING 的線程。
    std::thread* mptLocalMapping;

    // 用於進行 LOOP CLOSING 的線程。
    std::thread* mptLoopClosing;

    // 用於可視化的線程。
    std::thread* mptViewer;
    // ==================================================

    // Reset flag
    // 保護成員變量 mbReset 的信號量。
    std::mutex mMutexReset;

    // 重置標志。
    bool mbReset;

    // Change mode flags
    // 保護工作模式標志的信號量。
    std::mutex mMutexMode;

    // 是否使用定位模式
    bool mbActivateLocalizationMode;

    // 未激活定位模式。(不是很理解為什麽要弄兩個這東西，是不是有什麽安全上的考慮？留待以後分析具體源碼時在探討吧)
    bool mbDeactivateLocalizationMode;

    // Tracking state
    // 跟蹤狀態。
    int mTrackingState;

    // 地圖點。
    std::vector<MapPoint*> mTrackedMapPoints;

    // 關鍵點。
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;

    // 保護跟蹤狀態的信號量。
    std::mutex mMutexState;

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, 
           const bool bUseViewer = true);

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);

    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear map)
    void Reset();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
