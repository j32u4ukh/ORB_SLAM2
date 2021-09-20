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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include <mutex>

namespace ORB_SLAM2
{

    long unsigned int MapPoint::nNextId = 0;
    const double MapPoint::delta_odd = 0.05;
    mutex MapPoint::mGlobalMutex;

    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap, cv::Vec3b color) : 
                       mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), 
                       mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), 
                       mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0), 
                       mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), 
                       mnFound(1), mbBad(false), mpReplaced(static_cast<MapPoint *>(NULL)), 
                       mfMinDistance(0), mfMaxDistance(0), mpMap(pMap), mp_color(color)
    {
        Pos.copyTo(mWorldPos);
        mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    /// NOTE: 20210829
    // 若觀察到這個地圖點的關鍵幀太少，移除當前地圖點(若可以就直接將不好的點排除，不要一直判斷它好不好)
    bool MapPoint::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }
    
    /// NOTE: 20210829
    // 檢查是否已添加過『關鍵幀 pKF』到當前地圖點
    bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        // mObservations：觀察到這個地圖點的『關鍵幀』，以及其『關鍵點』的索引值
        return (mObservations.count(pKF));
    }

    /// NOTE: 20210829
    // 地圖點被關鍵幀的第 idx 個關鍵點觀察到
    void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        // 若已添加，則無須再次重複
        if (mObservations.count(pKF)){
            return;
        }

        // 『關鍵幀 pKF』的第 idx 個『關鍵點』觀察到這個地圖點的
        mObservations[pKF] = idx;

        if (pKF->mvuRight[idx] >= 0)
        {
            nObs += 2;
        }
        else{
            nObs++;
        }
    }

    /// NOTE: 20210829
    // 利用所有觀察到這個地圖點的關鍵幀來估計關鍵幀們平均指向的方向，以及該地圖點可能的深度範圍(最近與最遠)
    void MapPoint::UpdateNormalAndDepth()
    {
        map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);

            if (mbBad)
            {
                return;
            }

            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos.clone();
        }

        if (observations.empty()){
            return;
        }

        // 所有觀察到這個地圖點的關鍵幀的『相機中心 指向 地圖點 的向量』的總和
        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);

        int n = 0;
        KeyFrame *pKF;
        cv::Mat Owi, normal_i;

        for(pair<KeyFrame *, size_t> obs : observations)
        {
            pKF = obs.first;
            Owi = pKF->GetCameraCenter();

            // 相機中心 指向 地圖點 的向量
            /// TODO: normal_i = Pos - Owi; 避免和其他執行續同時修改 mWorldPos
            normal_i = mWorldPos - Owi;

            // normal 為正歸化後的 normal_i 的累加
            normal = normal + normal_i / cv::norm(normal_i);

            n++;
        }
        
        // 相機中心 指向 地圖點 的向量
        cv::Mat PC = Pos - pRefKF->GetCameraCenter();

        // 相機到地圖點的距離
        const float dist = cv::norm(PC);
        
        const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels;

        {
            unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance = dist * levelScaleFactor;
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
            
            // 平均『相機指向地圖點』之正規化向量
            mNormalVector = normal / n;
        }
    }

    /// NOTE: 20210829
    // 以『所有描述這個地圖點的描述子的集合』的中心描述子，作為地圖點的描述子
    void MapPoint::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        map<KeyFrame *, size_t> observations;

        {
            unique_lock<mutex> lock1(mMutexFeatures);

            if (mbBad)
            {
                return;
            }

            observations = mObservations;
        }

        if (observations.empty())
        {
            return;
        }

        vector<cv::Mat> vDescriptors;        
        vDescriptors.reserve(observations.size());

        KeyFrame *pKF;

        for(pair<KeyFrame *, size_t> obs : observations){

            pKF = obs.first;

            if (!pKF->isBad()){

                // 這個地圖點是被關鍵幀 pKF 的第 mit->second 個關鍵點觀察到，
                // 因此取出第 mit->second 個關鍵點的描述子
                vDescriptors.push_back(pKF->mDescriptors.row(obs.second));
            }
        }

        if (vDescriptors.empty()){
            return;
        }

        // Compute distances between them
        const size_t N = vDescriptors.size();
        float Distances[N][N];
        size_t i, j;
        int distij;

        for (i = 0; i < N; i++)
        {
            Distances[i][i] = 0;

            for (j = i + 1; j < N; j++)
            {
                // 計算『觀察到同一地圖點的各關鍵幀的關鍵點的描述子』彼此之間的距離
                distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        // 找出最小的距離
        int BestMedian = INT_MAX;
        int BestIdx = 0, median;

        // 遍歷 N 個描述子
        for (i = 0; i < N; i++)
        {
            // 取出第 i 個描述子和其他描述子之間的距離（包含和自己的，距離為 0）
            vector<int> vDists(Distances[i], Distances[i] + N);
            sort(vDists.begin(), vDists.end());

            // 取得中位數
            median = vDists[0.5 * (N - 1)];

            // 篩選最小的中位數
            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);

            // 以『和其他描述子之間距離最短』的描述子作為這個地圖點的描述子
            // 類似『所有描述這個地圖點的描述子的集合』的中心點
            mDescriptor = vDescriptors[BestIdx].clone();
        }
    }

    // 觀察到這個地圖點的『關鍵幀』，以及其『關鍵點』的索引值
    map<KeyFrame *, size_t> MapPoint::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    // 獲取其查找率(在TRACKING線程中判定匹配到地圖點的關鍵幀數量與預測可以看到地圖點的關鍵幀數量之比)
    float MapPoint::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);

        /* mnFound 記錄了實際觀測到地圖點的關鍵幀數量，而 mnVisible 則是估計能夠看到該點的關鍵幀數量。
        mnFound 和 mnVisible 都是在 TRACKING 線程中得到更新的。在根據局部地圖優化位姿的時候，
        通過 Tracking 對象的成員函數 SearchLocalPoints 對局部地圖中的地圖點進行粗略篩選，會根據觀測到地圖點的
        視角余弦值來估計當前幀能否觀測到響應的地圖點，並通過地圖點的接口 increaseVisibleEstimateNumber 增加 mnVisible 的計數。*/
        return static_cast<float>(mnFound) / mnVisible;
    }

    // 清空這個地圖點、觀察到這個地圖點的所有關鍵幀，以及它自己對應的關鍵點索引值
    void MapPoint::SetBadFlag()
    {
        map<KeyFrame *, size_t> obs;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;

            // 暫存關鍵幀資訊，函式結束後便會釋放記憶體空間
            obs = mObservations;

            // 清空觀察到這個地圖點的所有關鍵幀，以及它自己對應的關鍵點索引值
            mObservations.clear();

            /// TODO: nObs = 0
        }

        map<KeyFrame *, size_t>::iterator mit, mend = obs.end();

        for (mit = obs.begin(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;

            // 『關鍵幀 pKF』第 mit->second 個關鍵點觀察到的地圖點，設為 NULL
            pKF->EraseMapPointMatch(mit->second);
        }

        // 清除『當前地圖點』
        mpMap->EraseMapPoint(this);
    }

    // 這個地圖點被幾個關鍵幀觀察到
    int MapPoint::beObservedNumber()
    {
        unique_lock<mutex> lock(mMutexFeatures);

        // 這個地圖點被幾個關鍵幀觀察到
        return nObs;
    }

    // 取得地圖點的世界座標
    cv::Mat MapPoint::GetWorldPos()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    // 考慮金字塔層級的『地圖點 pMP』最大可能深度
    float MapPoint::GetMaxDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    // 考慮金字塔層級的『地圖點 pMP』最小可能深度
    float MapPoint::GetMinDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    // 地圖點之法向量
    cv::Mat MapPoint::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);

        // 平均『相機指向地圖點』之正規化向量
        return mNormalVector.clone();
    }

    // 『關鍵幀 pKF』根據當前『地圖點 pMP』的深度，估計場景規模
    int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF)
    {
        return predictScale(currentDist, pKF->mfLogScaleFactor, pKF->mnScaleLevels);
    }

    // 根據當前距離與最遠可能距離，換算出當前尺度
    int MapPoint::PredictScale(const float &currentDist, Frame *pF)
    {
        return predictScale(currentDist, pF->mfLogScaleFactor, pF->mnScaleLevels);
    }
    
    int MapPoint::predictScale(const float &distance, const float scale_factor, const int scale_level)
    {
        float ratio;

        {
            unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / distance;
        }

        int scale = ceil(log(ratio) / scale_factor);

        if (scale < 0)
        {
            scale = 0;
        }
        else if (scale >= scale_level)
        {
            scale = scale_level - 1;
        }

        return scale;
    }
    
    // 取得地圖點描述子
    cv::Mat MapPoint::GetDescriptor()
    {
        unique_lock<mutex> lock(mMutexFeatures);

        // 地圖點的描述子：『所有描述這個地圖點的描述子的集合』的中心描述子
        return mDescriptor.clone();
    }

    // 將被『較多』關鍵幀觀察到的地圖點，取代被『較少』關鍵幀觀察到的地圖點
    void MapPoint::Replace(MapPoint *pMP)
    {
        if (pMP->mnId == this->mnId){
            return;
        }

        int nvisible, nfound;
        map<KeyFrame *, size_t> obs;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);

            // 觀察到這個地圖點的『關鍵幀』，以及其『關鍵點』的索引值
            obs = mObservations;

            mObservations.clear();
            mbBad = true;

            // 估計能夠看到地圖點的關鍵幀數量
            nvisible = mnVisible;

            nfound = mnFound;
            mpReplaced = pMP;
        }

        for(pair<KeyFrame *, size_t> kf_kpidx : obs)
        {
            // Replace measurement in keyframe
            KeyFrame *pKF = kf_kpidx.first;
            size_t kp_idx = kf_kpidx.second;

            // 檢查是否還沒被添加過『關鍵幀 pKF』到當前地圖中
            if (!pMP->IsInKeyFrame(pKF))
            {
                // 關鍵幀的第 idx 個關鍵幀觀察到的地圖點汰換成『地圖點 pMP』
                pKF->ReplaceMapPointMatch(kp_idx, pMP);

                // 『地圖點 pMP』被『關鍵幀 pKF』的第 (mit->second) 個關鍵點觀察到
                pMP->AddObservation(pKF, kp_idx);
            }
            else
            {
                // 『關鍵幀 pKF』第 (mit->second) 個關鍵點觀察到的『地圖點 pMP』，設為 NULL
                pKF->EraseMapPointMatch(kp_idx);
            }
        }

        // 增加實際觀測到地圖點的關鍵幀數量
        pMP->increaseFoundNumber(nfound);

        // 增加對『能夠看到地圖點的關鍵幀數量』的估計
        pMP->increaseVisibleEstimateNumber(nvisible);

        // 以『所有描述這個地圖點的描述子的集合』的中心描述子，作為地圖點的描述子
        pMP->ComputeDistinctiveDescriptors();

        // 清除『地圖點 pMP』
        mpMap->EraseMapPoint(this);
    }

    // 增加實際觀測到地圖點的關鍵幀數量
    void MapPoint::increaseFoundNumber(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        // mnFound：實際觀測到地圖點的關鍵幀數量
        mnFound += n;
    }

    // 增加對『能夠看到地圖點的關鍵幀數量』的估計
    void MapPoint::increaseVisibleEstimateNumber(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    // 取得『是關鍵幀 pKF 的第幾個 關鍵點 觀察到這個地圖點的』
    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        if (mObservations.count(pKF))
        {
            // 『關鍵幀 pKF』的第幾個『關鍵點』觀察到這個地圖點的
            return mObservations[pKF];
        }
        else
        {
            return -1;
        }
    }

    // 移除『關鍵幀 pKF』，更新關鍵幀的計數，若『觀察到這個地圖點的關鍵幀』太少（少於 3 個），則將地圖點與關鍵幀等全部移除
    void MapPoint::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;

        {
            unique_lock<mutex> lock(mMutexFeatures);

            if (mObservations.count(pKF))
            {
                // 取得觀察到『關鍵幀 pKF』的關鍵點的索引值
                int idx = mObservations[pKF];

                if (pKF->mvuRight[idx] >= 0){
                    nObs -= 2;
                }

                // 單目的 mvuRight 會是負的
                else{
                    // 減少 1 個關鍵幀的計數
                    nObs--;
                }

                // 從 mObservations 移除『關鍵幀 pKF』
                mObservations.erase(pKF);

                // 若為當前地圖點的參考關鍵幀
                if (mpRefKF == pKF)
                {
                    // 以更新後的 mObservations 的第一個關鍵幀作為參考關鍵幀
                    mpRefKF = mObservations.begin()->first;
                }

                // If only 2 observations or less, discard point
                // 若觀察到這個地圖點的關鍵幀太少（少於 3 個）
                if (nObs <= 2)
                {
                    bBad = true;
                }
            }
        }

        if (bBad){
            // 清空這個地圖點、觀察到這個地圖點的所有關鍵幀，以及它自己對應的關鍵點索引值
            SetBadFlag();
        }
    }

    void MapPoint::SetWorldPos(const cv::Mat &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);
    }

    // 取得參考關鍵幀
    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);

        return mpRefKF;
    }

    // 返回要更新的地圖點
    MapPoint *MapPoint::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);

        return mpReplaced;
    }

    // ==================================================
    // 模仿 octomap 對空間點的佔用機率進行更新  
    // ==================================================
    void MapPoint::hit(double delta)
    {
        unique_lock<mutex> lock1(odd_mutex);

        if(delta == 0.0)
        {
            delta = delta_odd;
        }
        
        log_odd = min(log_odd + delta, max_odd);
    }

    void MapPoint::miss(double delta)
    {
        unique_lock<mutex> lock1(odd_mutex);

        if(delta == 0.0)
        {
            delta = delta_odd;
        }

        log_odd = max(log_odd - delta, min_odd);
    }

    double MapPoint::getHitProb()
    {
        unique_lock<mutex> lock1(odd_mutex);
        return 1.0/(1.0 + exp(-log_odd));
    }

    double MapPoint::getHitLog()
    {
        unique_lock<mutex> lock1(odd_mutex);
        return log_odd;
    }

    cv::Vec3b MapPoint::getColor()
    {
        return mp_color;
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    MapPoint::MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF) : 
                       mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), 
                       mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), 
                       mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0), 
                       mnCorrectedReference(0), mnBAGlobalForKF(0), 
                       mpRefKF(static_cast<KeyFrame *>(NULL)), mnVisible(1), mnFound(1), mbBad(false), 
                       mpReplaced(NULL), mpMap(pMap)
    {
        Pos.copyTo(mWorldPos);
        cv::Mat Ow = pFrame->GetCameraCenter();
        mNormalVector = mWorldPos - Ow;
        mNormalVector = mNormalVector / cv::norm(mNormalVector);

        cv::Mat PC = Pos - Ow;
        const float dist = cv::norm(PC);
        const int level = pFrame->mvKeysUn[idxF].octave;
        const float levelScaleFactor = pFrame->mvScaleFactors[level];
        const int nLevels = pFrame->mnScaleLevels;

        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

        pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    
    
} //namespace ORB_SLAM
