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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "ORBmatcher.h"
#include "PnPsolver.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    static const int start_idx;
    static const int end_idx;

    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, bool bReuseMap=false);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp, const int idx=0);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);


public:
    // Tracking states
    enum eTrackingState{
        // 系統尚未準備好
        SYSTEM_NOT_READY = -1,

        // 還沒有收到圖片
        NO_IMAGES_YET = 0,

        // 尚未初始化
        NOT_INITIALIZED = 1,

        // 一切正常
        OK = 2,

        // 跟丟了
        LOST = 3
    };

    // 軌跡跟蹤器的狀態，是一個枚舉類型    
    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    // 傳感器類型，單目(monocular) 0，雙目(stereo) 1，深度相機(RGBD) 2
    int mSensor;

    // Current Frame
    // 當前幀
    Frame mCurrentFrame;

    // 灰度圖，對於深度相機就是RGB圖像轉換而來的，對於雙目則是左目的灰度圖像。
    cv::Mat gray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;

    /* vector<int> 用於單目初始化，記錄了初始幀中各個特征點與當前幀匹配的特征點索引。
       關鍵點匹配中，和『影像 F1 第 i1 個關鍵點』配對成功的是『影像 F2 第 bestIdx2 個關鍵點』*/ 
    std::vector<int> mvIniMatches;

    // 紀錄前一幀匹配成功的關鍵點的位置，由於假設兩幀之間運動極小，因此關鍵點在第二幀的位置，應和前一幀差不多
    // F2.GetFeaturesInArea 在尋找關鍵點時，會根據 mvbPrevMatched 在他的附近尋找關鍵點
    std::vector<cv::Point2f> mvbPrevMatched;

    // 用於單目初始化，記錄了初始化過程中，成功三角化的特征點的3D坐標。
    std::vector<cv::Point3f> mvIniP3D;

    // 用於單目初始化，記錄了用於初始化的參考幀。
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    // 用於結束程序的時候恢覆完整的相機軌跡。為每一幀記錄和其參考關鍵幀的相對變換（位姿轉換），不是和前一幀。
    list<cv::Mat> mlRelativeFramePoses;

    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    // 只用於定位的標志，此時局部建圖(local mapping)功能處於未激活(deactivated)狀態。
    // 是否僅追蹤不建圖
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track(const int idx=0);

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization(const int idx=0);
    void CreateInitialMapMonocular(const int idx=0);

    void CheckReplacedInLastFrame(const int idx=0);
    bool TrackReferenceKeyFrame(const int idx=0);
    void UpdateLastFrame();
    bool TrackWithMotionModel(const int idx=0);

    bool Relocalization(const int idx=0);

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap(const int idx=0);
    void SearchLocalPoints();

    bool NeedNewKeyFrame(const int idx=0);
    void CreateNewKeyFrame(const int idx=0);

    // *****

    inline void recordTrackingResult();
    inline bool update(bool bOK, const int idx=0);
    inline void createRelocatePnPsolver(vector<KeyFrame *> &vpCandidateKFs, int &nCandidates,
                                           vector<bool> &vbDiscarded, ORBmatcher &matcher, 
                                           vector<vector<MapPoint *>> &vvpMapPointMatches, 
                                           vector<PnPsolver *> &vpPnPsolvers);
    inline string checkRelocalization(cv::Mat &Tcw, vector<bool> &bInliers, int &nGood, 
                                      const int i, ORBmatcher &matcher2,
                                         vector<vector<MapPoint *>> &vvpMapPointMatches,
                                         vector<KeyFrame *> &vpCandidateKFs, bool &bMatch);

    inline bool relocate(int &nCandidates, vector<bool> &vbDiscarded, 
                            vector<KeyFrame *> &vpCandidateKFs, const vector<PnPsolver *> &vpPnPsolvers, 
                            vector<vector<MapPoint *>> &vvpMapPointMatches);

    // *****

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    // 對於純定位模式，該標志表示沒有在地圖中找到合適的匹配，系統將嘗試重定位來確定零偏(zero drift)。
    bool mbVO;

    // Other Thread Pointers
    // 局部地圖管理器指針
    LocalMapping* mpLocalMapper;

    // 閉環探測指針
    LoopClosing* mpLoopClosing;

    // ORB
    // ORB特征點提取器，一般都使用 mpORBextractorLeft 來提取特征點，
    // 對於雙目相機需要 mpORBextractorRight 來提取右目的特征點， 
    // mpIniORBextractor用於單目相機初始化。
    ORBextractor* mpIniORBextractor;
    ORBextractor* mpORBextractorLeft;
    ORBextractor* mpORBextractorRight;
    
    // BoW
    // ORB 字典
    ORBVocabulary* mpORBVocabulary;

    // 關鍵幀數據庫
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    // 用於單目的初始化器
    Initializer* mpInitializer;

    // Local Map
    // 當前參考關鍵幀
    KeyFrame* mpReferenceKF;

    // Local 關鍵幀列表
    std::vector<KeyFrame*> mvpLocalKeyFrames;

    // Local 地圖點列表
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    // ORB-SLAM2 系統對象指針
    System* mpSystem;
    
    // Drawers
    // 可視化對象
    Viewer* mpViewer;

    // 幀繪圖器
    FrameDrawer* mpFrameDrawer;

    // 地圖繪圖器
    MapDrawer* mpMapDrawer;

    // Map
    Map* mpMap;

    // Calibration matrix
    // 相機內參矩陣
    cv::Mat K;

    // 鏡頭畸變系數
    cv::Mat mDistCoef;

    // 基線，對於雙目就是兩個相機之間的距離。對於 RGBD 相機，ORB SLAM2 給定了一個假想的基線，模擬雙目的數據。
    float mbf;

    // New KeyFrame rules (according to fps)
    // 創建關鍵幀所需經歷的最小幀數量。
    int mMinFrames;

    // 創建關鍵幀所需經歷的最大幀數量。
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    // 遠近點判定閾值。對於雙目或者深度相機，近點的深度是比較可信的，可以直接獲取定位；
    // 而遠處的點則需要通過兩個關鍵幀的匹配三角化得到。
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    // 只試用深度相機。因為有些數據集(比如tum)其深度圖數據是經過縮放的。
    float mDepthMapFactor;

    // Current matches in frame
    int mnMatchesInliers;

    // Last Frame, KeyFrame and Relocalisation Info
    // 上一個關鍵幀。
    KeyFrame* mpLastKeyFrame;

    // 上一個幀。
    Frame mLastFrame;

    // 上一個關鍵幀 ID。
    unsigned int mnLastKeyFrameId;

    // 上一個重定位幀ID。
    unsigned int mnLastRelocFrameId;

    // Motion Model
    cv::Mat mVelocity;

    // Color order (true RGB, false BGR, ignored if grayscale)
    // 顏色通道，true —— RGB，false —— BGR。
    bool mbRGB;

    // 一個地圖點列表。
    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
