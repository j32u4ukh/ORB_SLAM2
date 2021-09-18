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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int beObservedNumber();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();
    
    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void increaseVisibleEstimateNumber(int n=1);
    void increaseFoundNumber(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int predictScale(const float &distance, const float scale_factor, const int scale_level);
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

    /* 模仿 octomap 對空間點的佔用機率進行更新      

     * 佔據三維體元被擊中（Hit）的概率值為 0.7 對應的 log-odd 為 0.85
     * 空閒體元被穿越（traverse）的概率值為 0.4 對應的 log-odd 為 -0.4
     * 可以呼叫 getProHit/getProHitLog、getProMiss/getProMissLog 檢視預設的引數設定。
     * 更新時，佔用機率低於 0.4 則將地圖點移除，大於 0.7
     
     * setClampingThresMax/setClampingThresMin 這兩個函式決定了一個體元執行 log-odd 更新的閾值範圍。
     * 也就是說某一個佔據體元的概率值爬升到 0.97（對應的 log-odd 為 3.5）
     * 或者空閒體元的概率值下降到 0.12（對應的 log-odd 為 -2）便不再進行 log-odd 更新計算。
     p: 空間點的佔用機率

     log_odd = log( p / (1 - p) )

     p = 1 / ( 1 + exp(-log_odd) )
     */
    inline void hit(double delta = 0.0);
    inline void miss(double delta = 0.0);
    inline double getHitProb();
    inline double getHitLog();

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;

    // 這個地圖點被幾個關鍵幀觀察到
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     // 觀察到這個地圖點的『關鍵幀』，以及其『關鍵點』的索引值
     std::map<KeyFrame*, size_t> mObservations;

     // Mean viewing direction
     // 平均『相機指向地圖點』之正規化向量（地圖點之法向量），由觀察到這個地圖點的所有關鍵幀所共同構成的向量
     cv::Mat mNormalVector;

     // Best descriptor to fast matching
     // 地圖點的描述子：『所有描述這個地圖點的描述子的集合』的中心描述子
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;

     // 估計能夠看到地圖點的關鍵幀數量
     int mnVisible;

     // 實際觀測到地圖點的關鍵幀數量
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     // Scale invariance distances
     // 『地圖點 pMP』最小可能深度
     float mfMinDistance;

     // 『地圖點 pMP』最大可能深度
     float mfMaxDistance;

     Map* mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;

     /* 模仿 octomap 對空間點的佔用機率進行更新      

     * 佔據三維體元被擊中（Hit）的概率值為 0.7 對應的 log-odd 為 0.85
     * 空閒體元被穿越（traverse）的概率值為 0.4 對應的 log-odd 為 -0.4
     * 可以呼叫 getProHit/getProHitLog、getProMiss/getProMissLog 檢視預設的引數設定。
     
     * setClampingThresMax/setClampingThresMin 這兩個函式決定了一個體元執行 log-odd 更新的閾值範圍。
     * 也就是說某一個佔據體元的概率值爬升到 0.97（對應的 log-odd 為 3.5）
     * 或者空閒體元的概率值下降到 0.12（對應的 log-odd 為 -2）便不再進行 log-odd 更新計算。
     p: 空間點的佔用機率

     log_odd = log( p / (1 - p) )

     p = 1 / ( 1 + exp(-log_odd) )
     */
     double max_odd = 3.5;
     double min_odd = -2.0;
     static const double delta_odd;

     // 佔據三維體元被擊中（Hit）的概率值為 0.7 對應的 log-odd 為 0.85
     double log_odd = 0.85;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
