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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool hasNewKeyFrames();
    void ProcessNewKeyFrame();
    void createNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;

    // 『執行續 LocalMapping』是否已結束
    bool mbFinished;
    
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    // 新關鍵幀容器
    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    // 是否中止『執行續 LocalMapping』
    bool mbStopped;

    // 請求中止『執行續 LocalMapping』（不會光 mbStopRequested 被改為 true 就中止）
    bool mbStopRequested;

    // 請求不要中止『執行續 LocalMapping』
    bool mbNotStop;

    std::mutex mMutexStop;

    // 設置『是否接受關鍵幀（此時 LOCAL MAPPING 線程是否處於空閑的狀態）』
    bool mbAcceptKeyFrames;
    
    std::mutex mMutexAccept;

    inline void createMonoMapPointsByKeyFrame(const int nn);
    inline void createStereoMapPointsByKeyFrame(const int nn);

    inline void createMonoMapPointsByKeyPoints(vector<pair<size_t, size_t>> &vMatchedIndices,
                                               KeyFrame *pKF2, int &nnew,
                                               const float fx1, const float fy1,
                                               const float cx1, const float cy1,
                                               const float invfx1, const float invfy1,
                                               const float fx2, const float fy2,
                                               const float cx2, const float cy2,
                                               const float invfx2, const float invfy2,
                                               const cv::Mat Rcw1, const cv::Mat Rwc1,
                                               const cv::Mat Rcw2, const cv::Mat Rwc2,
                                               const cv::Mat tcw1, const cv::Mat tcw2,
                                               const cv::Mat Tcw1, const cv::Mat Tcw2,
                                               const cv::Mat Ow1, const cv::Mat Ow2,
                                               const float ratioFactor);
    inline void createStereoMapPointsByKeyPoints(vector<pair<size_t, size_t>> &vMatchedIndices,
                                                 KeyFrame *pKF2, int &nnew,
                                                 const float fx1, const float fy1,
                                                 const float cx1, const float cy1,
                                                 const float invfx1, const float invfy1,
                                                 const float fx2, const float fy2,
                                                 const float cx2, const float cy2,
                                                 const float invfx2, const float invfy2,
                                                 const cv::Mat Rcw1, const cv::Mat Rwc1,
                                                 const cv::Mat Rcw2, const cv::Mat Rwc2,
                                                 const cv::Mat tcw1, const cv::Mat tcw2,
                                                 const cv::Mat Tcw1, const cv::Mat Tcw2,
                                                 const cv::Mat Ow1, const cv::Mat Ow2,
                                                 const float ratioFactor);

    inline float reprojectError(const cv::Mat pos, const float kp_x, const float kp_y,
                                const float fx, const float fy, const float cx, const float cy);

    inline float reprojectError(const cv::Mat pos, const float kp_x, const float kp_y,
                                const float fx, const float fy,
                                const float cx, const float cy,
                                const float mbf, const float kp_r);
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
