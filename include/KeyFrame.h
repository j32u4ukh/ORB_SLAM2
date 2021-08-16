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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>

namespace ORB_SLAM2
{

    class Map;
    class MapPoint;
    class Frame;
    class KeyFrameDatabase;

    class KeyFrame
    {
    public:
        KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);

        // Pose functions
        void SetPose(const cv::Mat &Tcw);
        cv::Mat GetPose();
        cv::Mat GetPoseInverse();
        cv::Mat GetCameraCenter();
        cv::Mat GetStereoCenter();
        cv::Mat GetRotation();
        cv::Mat GetTranslation();

        // Bag of Words Representation
        void ComputeBoW();

        // Covisibility graph functions
        void AddConnection(KeyFrame *pKF, const int &weight);
        void EraseConnection(KeyFrame *pKF);
        void UpdateConnections();
        void UpdateBestCovisibles();
        std::set<KeyFrame *> GetConnectedKeyFrames();
        std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();
        std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N);

        // 取得『關鍵幀』的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序，且觀察到的地圖點數量「大於」 w）』
        std::vector<KeyFrame *> GetCovisiblesByWeight(const int &w);

        int GetWeight(KeyFrame *pKF);

        // Spanning tree functions
        void AddChild(KeyFrame *pKF);
        void EraseChild(KeyFrame *pKF);
        void ChangeParent(KeyFrame *pKF);
        std::set<KeyFrame *> GetChilds();
        KeyFrame *GetParent();
        bool hasChild(KeyFrame *pKF);

        // Loop Edges
        void AddLoopEdge(KeyFrame *pKF);
        std::set<KeyFrame *> GetLoopEdges();

        // MapPoint observation functions
        void AddMapPoint(MapPoint *pMP, const size_t &idx);
        void EraseMapPointMatch(const size_t &idx);
        void EraseMapPointMatch(MapPoint *pMP);
        void ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP);
        std::set<MapPoint *> GetMapPoints();
        std::vector<MapPoint *> GetMapPointMatches();
        int getTrackedMapPointNumber(const int &minObs);
        MapPoint *GetMapPoint(const size_t &idx);

        // KeyPoint functions
        std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
        cv::Mat UnprojectStereo(int i);

        // Image
        bool IsInImage(const float &x, const float &y) const;

        // Enable/Disable bad flag changes
        void SetNotErase();
        void SetErase();

        // Set/check bad flag
        void SetBadFlag();
        bool isBad();

        // Compute Scene Depth (q=2 median). Used in monocular.
        float ComputeSceneMedianDepth(const int q);

        static bool weightComp(int a, int b)
        {
            return a > b;
        }

        static bool lId(KeyFrame *pKF1, KeyFrame *pKF2)
        {
            return pKF1->mnId < pKF2->mnId;
        }

        // The following variables are accesed from only 1 thread or never change (no mutex needed).
    public:
        static long unsigned int nNextId;
        long unsigned int mnId;
        const long unsigned int mnFrameId;

        const double mTimeStamp;

        // Grid (to speed up feature matching)
        const int mnGridCols;
        const int mnGridRows;
        const float mfGridElementWidthInv;
        const float mfGridElementHeightInv;

        // Variables used by the tracking
        // 紀錄『提供哪一幀作為參考幀』
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnFuseTargetForKF;

        // Variables used by the local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnBAFixedForKF;

        // Variables used by the keyframe database
        long unsigned int mnLoopQuery;
        int mnLoopWords;
        float mLoopScore;
        long unsigned int mnRelocQuery;
        int mnRelocWords;
        float mRelocScore;

        // Variables used by loop closing
        // BundleAdjustment 優化後的位姿估計 
        cv::Mat mTcwGBA;

        // 保存上一次 mTcwGBA 的數值     
        cv::Mat mTcwBefGBA;

        long unsigned int mnBAGlobalForKF;

        // Calibration parameters
        const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

        // Number of KeyPoints
        const int N;

        // KeyPoints, stereo coordinate and descriptors (all associated by an index)
        const std::vector<cv::KeyPoint> mvKeys;

        // 已校正關鍵點（推測是像素坐標）
        const std::vector<cv::KeyPoint> mvKeysUn;

        // negative value for monocular points
        const std::vector<float> mvuRight; 

        // negative value for monocular points
        const std::vector<float> mvDepth;  

        // 描述子
        const cv::Mat mDescriptors;

        // BowVector == std::map<WordId, WordValue>
        // WordValue: tf * idf
        DBoW2::BowVector mBowVec;

        // FeatureVector == std::map<NodeId, std::vector<unsigned int> >
        // 以一張圖片的每個特徵點在詞典某一層節點下爲條件進行分組，用來加速圖形特徵匹配——
        // 兩兩圖像特徵匹配只需要對相同 NodeId 下的特徵點進行匹配就好。
        // std::vector<unsigned int>：觀察到該特徵的 地圖點/關鍵點 的索引值
        DBoW2::FeatureVector mFeatVec;

        // Pose relative to parent (this is computed when bad flag is activated)
        cv::Mat mTcp;

        // Scale
        const int mnScaleLevels;
        const float mfScaleFactor;
        const float mfLogScaleFactor;
        const std::vector<float> mvScaleFactors;
        const std::vector<float> mvLevelSigma2;
        const std::vector<float> mvInvLevelSigma2;

        // Image bounds and calibration
        const int mnMinX;
        const int mnMinY;
        const int mnMaxX;
        const int mnMaxY;
        const cv::Mat mK;

        // The following variables need to be accessed trough a mutex to be thread safe.
    protected:
        // SE3 Pose and camera center
        // 世界座標到相機座標的轉換矩陣
        cv::Mat Tcw;

        // 相機座標到世界座標的轉換矩陣
        cv::Mat Twc;

        // 相機中心點位置
        cv::Mat Ow;

        // Stereo middel point. Only for visualization
        cv::Mat Cw; 

        // MapPoints associated to keypoints
        // 關鍵點觀察到的地圖點
        std::vector<MapPoint *> mvpMapPoints;

        // BoW
        KeyFrameDatabase *mpKeyFrameDB;
        ORBVocabulary *mpORBvocabulary;

        // Grid over the image to speed up feature matching
        // 紀錄網格 mGrid[nGridPosX][nGridPosY] 所包含的關鍵點的索引值
        std::vector<std::vector<std::vector<size_t>>> mGrid;

        // 『已連結關鍵幀』與『觀察到的地圖點個數』之對應（根據觀察到的地圖點數量由大到小排序）
        std::map<KeyFrame *, int> mConnectedKeyFrameWeights;

        // 『已連結關鍵幀』（根據觀察到的地圖點數量由大到小排序）
        std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;

        // 『已連結關鍵幀』觀察到的地圖點數量（已由大到小排序）
        std::vector<int> mvOrderedWeights;

        // Spanning Tree and Loop Edges
        bool mbFirstConnection;
        KeyFrame *mpParent;
        std::set<KeyFrame *> mspChildrens;
        std::set<KeyFrame *> mspLoopEdges;

        // Bad flags
    
        // 不要移除當前關鍵幀
        bool mbNotErase;

        bool mbToBeErased;
        bool mbBad;

        float mHalfBaseline; // Only for visualization

        Map *mpMap;

        std::mutex mMutexPose;
        std::mutex mMutexConnections;
        std::mutex mMutexFeatures;
    };

} //namespace ORB_SLAM

#endif // KEYFRAME_H
