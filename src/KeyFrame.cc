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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <mutex>

namespace ORB_SLAM2
{

    long unsigned int KeyFrame::nNextId = 0;

    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB) : 
                       mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), 
                       mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(F.mfGridElementWidthInv), 
                       mfGridElementHeightInv(F.mfGridElementHeightInv), mnTrackReferenceForFrame(0), 
                       mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnLoopQuery(0), 
                       mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), 
                       fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy), 
                       mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), 
                       mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mvuRight(F.mvuRight), mvDepth(F.mvDepth), 
                       mDescriptors(F.mDescriptors.clone()), mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), 
                       mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor), 
                       mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), 
                       mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), 
                       mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), 
                       mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB), mpMap(pMap), 
                       mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), 
                       mbNotErase(false), mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb / 2)
    {
        mnId = nNextId++;
        mGrid.resize(mnGridCols);

        for (int i = 0; i < mnGridCols; i++)
        {
            mGrid[i].resize(mnGridRows);

            for (int j = 0; j < mnGridRows; j++)
            {
                mGrid[i][j] = F.mGrid[i][j];
            }
        }

        SetPose(F.mTcw);
    }

    void KeyFrame::ComputeBoW()
    {
        if (mBowVec.empty() || mFeatVec.empty())
        {
            // 將描述子轉換為 vector<cv::Mat>
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);

            // Feature vector associate features with nodes in the 4th level (from leaves up)
            // We assume the vocabulary tree has 6 levels, change the 4 otherwise
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void KeyFrame::SetPose(const cv::Mat &Tcw_)
    {
        unique_lock<mutex> lock(mMutexPose);
        Tcw_.copyTo(Tcw);
        cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
        cv::Mat Rwc = Rcw.t();
        Ow = -Rwc * tcw;

        Twc = cv::Mat::eye(4, 4, Tcw.type());
        Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
        Ow.copyTo(Twc.rowRange(0, 3).col(3));
        cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
        Cw = Twc * center;
    }

    cv::Mat KeyFrame::GetPose()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.clone();
    }

    // 相機座標到世界座標的轉換矩陣
    cv::Mat KeyFrame::GetPoseInverse()
    {
        unique_lock<mutex> lock(mMutexPose);

        // 相機座標到世界座標的轉換矩陣
        return Twc.clone();
    }

    // 取得相機中心點位置
    cv::Mat KeyFrame::GetCameraCenter()
    {
        unique_lock<mutex> lock(mMutexPose);

        return Ow.clone();
    }

    cv::Mat KeyFrame::GetStereoCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Cw.clone();
    }

    cv::Mat KeyFrame::GetRotation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0, 3).colRange(0, 3).clone();
    }

    cv::Mat KeyFrame::GetTranslation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0, 3).col(3).clone();
    }

    void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
    {
        // weight：關鍵幀觀察到的地圖點數

        {
            unique_lock<mutex> lock(mMutexConnections);

            // 若未曾添加關鍵幀 pKF
            // mConnectedKeyFrameWeights：『關鍵幀』與『觀察到的地圖點個數』之對應關係
            if (!mConnectedKeyFrameWeights.count(pKF))
            {
                // 加入字典管理
                mConnectedKeyFrameWeights[pKF] = weight;
            }

            // 關鍵幀 pKF 已在字典中，但數值不同
            else if (mConnectedKeyFrameWeights[pKF] != weight)
            {
                // 更新字典數值（關鍵幀觀察到的地圖點數量）
                mConnectedKeyFrameWeights[pKF] = weight;
            }

            // 關鍵幀 pKF 已在字典中，且數值相同 -> 重複添加，直接返回
            else
            {
                return;
            }
        }

        // 將共視關鍵幀根據觀察到的地圖點數量排序，再存入 mvpOrderedConnectedKeyFrames
        UpdateBestCovisibles();
    }

    // 將共視關鍵幀根據觀察到的地圖點數量由大到小排序，再存入 mvpOrderedConnectedKeyFrames
    void KeyFrame::UpdateBestCovisibles()
    {
        unique_lock<mutex> lock(mMutexConnections);
        vector<pair<int, KeyFrame *>> vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());

        for(pair<KeyFrame *, int> kf_weight : mConnectedKeyFrameWeights)
        {
            vPairs.push_back(make_pair(kf_weight.second, kf_weight.first));
        }

        // 首先對『關鍵幀觀察到的地圖點數量（mit->second）』升序排序，若相同再對『KeyFrame *』升序排序
        sort(vPairs.begin(), vPairs.end());
        list<KeyFrame *> lKFs;
        list<int> lWs;

        for(pair<int, KeyFrame *> weight_kf : vPairs)
        {
            // 將『關鍵幀觀察到的地圖點數量』加入 lWs 進行管理
            lWs.push_front(weight_kf.first);

            // 將『KeyFrame *』加入 lKFs 進行管理
            lKFs.push_front(weight_kf.second);
        }

        /// NOTE: 將關鍵幀加到 lKFs 當中，再賦值給 mvpOrderedConnectedKeyFrames 的理由大概是，
        /// 可以各自在自己的執行續繼續操作，降低需要暫停執行續的需求
        // 根據觀察到的地圖點數量由大到小排序的共視關鍵幀
        mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
        
        // 『已連結關鍵幀』觀察到的地圖點數量（已由大到小排序）
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
    }

    // 取得『已連結關鍵幀』
    set<KeyFrame *> KeyFrame::GetConnectedKeyFrames()
    {
        unique_lock<mutex> lock(mMutexConnections);
        set<KeyFrame *> s;

        // mConnectedKeyFrameWeights：『關鍵幀』與『觀察到的地圖點個數』之對應關係
        for(pair<KeyFrame *, int> kf_weight : mConnectedKeyFrameWeights)
        {
            s.insert(kf_weight.first);
        }

        return s;
    }

    // 取得『共視關鍵幀』（根據觀察到的地圖點數量排序）
    vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
    {
        unique_lock<mutex> lock(mMutexConnections);

        // 根據觀察到的地圖點數量由大到小排序的共視關鍵幀
        return mvpOrderedConnectedKeyFrames;
    }

    // 自『根據觀察到的地圖點數量排序的共視關鍵幀』當中返回至多 N 個共視關鍵幀
    vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
    {
        unique_lock<mutex> lock(mMutexConnections);

        int size = min((int)mvpOrderedConnectedKeyFrames.size(), N);

        // 根據觀察到的地圖點數量由大到小排序的共視關鍵幀
        return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), 
                                  mvpOrderedConnectedKeyFrames.begin() + size);

        // if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        // {
        //     return mvpOrderedConnectedKeyFrames;
        // }
        // else
        // {
        //     return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), 
        //                               mvpOrderedConnectedKeyFrames.begin() + N);
        // }
    }

    // 取得『關鍵幀』的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序，且觀察到的地圖點數量「大於」 w）』
    vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w)
    {
        unique_lock<mutex> lock(mMutexConnections);

        if (mvpOrderedConnectedKeyFrames.empty()){
            return vector<KeyFrame *>();
        }

        /* 
        lower_bound & upper_bound 【用途】針對「已經排序」的資料進行binary search。
        vector <int> v;
        sort(v.begin(), v.end());

        lower_bound：找出vector中「大於或等於」val的「最小值」的位置：
            auto it = lower_bound(v.begin(), v.end(), val);

        upper_bound：找出vector中「大於」val的「最小值」的位置：
            auto it = upper_bound(v.begin(), v.end(), val);

        自定義了 __comp 的 upper_bound： 查找[first, last)區域中第一個不符合 __comp 規則的元素的 iterator
        */
       
        // 取得『已連結關鍵幀觀察到的地圖點數量（已由大到小排序）』，地圖點數量「大於」 w 的「最小值」的位置
        vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), 
                                               mvOrderedWeights.end(), 
                                               w, 
                                               KeyFrame::weightComp);

        // iterator 的個數
        int n = it - mvOrderedWeights.begin();

        // 返回地圖點數量「大於」 w 的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序）』
        return vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), 
                                  mvpOrderedConnectedKeyFrames.begin() + n);
    }

    int KeyFrame::GetWeight(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexConnections);
        
        if (mConnectedKeyFrameWeights.count(pKF))
        {
            return mConnectedKeyFrameWeights[pKF];
        }
        else
        {
            return 0;
        }
    }

    // 關鍵幀的第 idx 個關鍵點觀察到了地圖點 pMP
    void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
    {
        /* 可能新增地圖點的情況：
        1. 三角測量後
        */
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMP;
    }

    // 關鍵幀第 idx 個關鍵點觀察到的地圖點，設為 NULL
    void KeyFrame::EraseMapPointMatch(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
    }

    // 從當前關鍵幀觀察到的地圖點當中移除『地圖點 pMP』，表示其實沒有觀察到
    void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
    {
        // 『關鍵幀 this』的第幾個『關鍵點』觀察到『地圖點 pMP』的
        int idx = pMP->GetIndexInKeyFrame(this);

        if (idx >= 0)
        {
            // 移除當前關鍵幀觀察到的第 idx 個地圖點，表示其實沒有觀察到
            mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
        }
    }

    // 關鍵幀的第 idx 個關鍵幀觀察到的地圖點汰換成『地圖點 pMP』
    void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP)
    {
        mvpMapPoints[idx] = pMP;
    }

    // 取得關鍵幀觀察到的地圖點
    set<MapPoint *> KeyFrame::GetMapPoints()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        set<MapPoint *> s;

        for(MapPoint *pMP : mvpMapPoints)
        {
            if(pMP)
            {
                if (!pMP->isBad())
                {
                    s.insert(pMP);
                }
            }
        }
        
        return s;
    }

    // 有多少個地圖點，是被足夠多的關鍵幀所觀察到的
    int KeyFrame::TrackedMapPoints(const int &minObs)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        const bool bCheckObs = minObs > 0;
        int nPoints = 0;
        
        for(MapPoint *pMP : mvpMapPoints)
        {
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    if (bCheckObs)
                    {
                        // 這個地圖點被足過多的關鍵幀觀察到
                        if (pMP->Observations() >= minObs)
                        {
                            nPoints++;
                        }
                    }
                    else{
                        nPoints++;
                    }
                }
            }
        }

        return nPoints;
    }

    // 取得關鍵幀觀察到的地圖點
    vector<MapPoint *> KeyFrame::GetMapPointMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);

        // 關鍵幀觀察到的地圖點
        return mvpMapPoints;
    }

    MapPoint *KeyFrame::GetMapPoint(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }

    // 其他關鍵幀和當前關鍵幀觀察到相同的地圖點，且各自都觀察到足夠多的地圖點，則會和當前幀產生鏈結
    void KeyFrame::UpdateConnections()
    {
        // 
        map<KeyFrame *, int> KFcounter;
        vector<MapPoint *> vpMP;

        {
            unique_lock<mutex> lockMPs(mMutexFeatures);
            vpMP = mvpMapPoints;
        }

        //For all map points in keyframe check in which other keyframes are they seen
        //Increase counter for those keyframes
        for(MapPoint *pMP : vpMP)
        {
            if (!pMP)
            {
                continue;
            }

            if (pMP->isBad())
            {
                continue;
            }

            // 地圖點被關鍵幀的第 idx 個關鍵點觀察到
            map<KeyFrame *, size_t> observations = pMP->GetObservations();

            for(pair<KeyFrame *, size_t> obs : observations){

                KeyFrame * kf = obs.first;

                // 若 mnId 與當前關鍵幀相同則跳過
                if (kf->mnId == mnId)
                {
                    continue;
                }

                // 累計關鍵幀 kf 觀察到地圖點的個數
                KFcounter[kf]++;
            }            
        }
        
        // This should not happen
        if (KFcounter.empty())
        {
            return;
        }

        //If the counter is greater than threshold add connection
        //In case no keyframe counter is over threshold add the one with maximum counter
        int nmax = 0;
        KeyFrame *pKFmax = NULL;
        int th = 15;

        vector<pair<int, KeyFrame *>> vPairs;
        vPairs.reserve(KFcounter.size());

        for(pair<KeyFrame *, int> kf_count : KFcounter){

            KeyFrame * kf = kf_count.first;
            int count = kf_count.second;

            if (count > nmax)
            {
                nmax = count;
                pKFmax = kf;
            }

            // 若關鍵幀觀察到的地圖點數量足夠多（大於所設門檻數量），應該表示『這個關鍵幀很重要』
            if (count >= th)
            {
                vPairs.push_back(make_pair(count, kf));

                // 和當前關鍵幀添加連結
                kf->AddConnection(this, count);
            }
        }

        // 若沒有任一關鍵幀觀察到的地圖點數量足夠多，則將最多的那一關鍵幀和當前關鍵幀添加連結
        if (vPairs.empty())
        {
            vPairs.push_back(make_pair(nmax, pKFmax));
            pKFmax->AddConnection(this, nmax);
        }

        sort(vPairs.begin(), vPairs.end());
        list<KeyFrame *> lKFs;
        list<int> lWs;

        for(pair<int, KeyFrame *> count_kf : vPairs)
        {
            // 將『關鍵幀觀察到的地圖點數量』加入 lWs 進行管理
            lWs.push_front(count_kf.first);

            // 將『KeyFrame *』加入 lKFs 進行管理
            lKFs.push_front(count_kf.second);
        }

        {
            unique_lock<mutex> lockCon(mMutexConnections);

            // mspConnectedKeyFrames = spConnectedKeyFrames;
            // mConnectedKeyFrameWeights：『關鍵幀』與『觀察到的地圖點個數』之對應關係（由大到小排序）
            mConnectedKeyFrameWeights = KFcounter;
            mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
            mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

            if (mbFirstConnection && mnId != 0)
            {
                // 將『共視程度最高的關鍵幀』作為當前關鍵幀的父關鍵幀
                mpParent = mvpOrderedConnectedKeyFrames.front();

                // 當前關鍵幀為『父關鍵幀 mpParent』的子關鍵幀
                mpParent->AddChild(this);

                mbFirstConnection = false;
            }
        }
    }

    void KeyFrame::AddChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.insert(pKF);
    }

    void KeyFrame::EraseChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.erase(pKF);
    }

    void KeyFrame::ChangeParent(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mpParent = pKF;
        pKF->AddChild(this);
    }

    set<KeyFrame *> KeyFrame::GetChilds()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens;
    }

    KeyFrame *KeyFrame::GetParent()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mpParent;
    }

    bool KeyFrame::hasChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens.count(pKF);
    }

    void KeyFrame::AddLoopEdge(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mbNotErase = true;
        mspLoopEdges.insert(pKF);
    }

    set<KeyFrame *> KeyFrame::GetLoopEdges()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspLoopEdges;
    }

    // 請求不要移除當前關鍵幀，避免在『執行續 LoopClosing』處理關鍵幀時被移除
    void KeyFrame::SetNotErase()
    {
        unique_lock<mutex> lock(mMutexConnections);

        // 請求不要移除當前關鍵幀
        mbNotErase = true;
    }

    // 取消『不要移除當前關鍵幀』的請求，若有『移除當前關鍵幀』的請求，則設為移除當前關鍵幀，和觀察者不足的地圖點及其關鍵幀
    void KeyFrame::SetErase()
    {
        {
            unique_lock<mutex> lock(mMutexConnections);

            if (mspLoopEdges.empty())
            {
                // 取消『不要移除當前關鍵幀』的請求
                mbNotErase = false;
            }
        }

        if (mbToBeErased)
        {
            // 移除當前關鍵幀，和當前關鍵幀觀察到的地圖點，但觀察者不足的地圖點及其關鍵幀
            SetBadFlag();
        }
    }

    // 提昇『子關鍵幀』層級到與自己相同，和『父關鍵幀』高度共視者，成為『父關鍵幀-候選』，最後移除當前幀
    // 當前關鍵幀觀察到的地圖點，若『觀察到這個地圖點的關鍵幀』太少（少於 3 個），則將地圖點與關鍵幀等全部移除
    void KeyFrame::SetBadFlag()
    {
        {
            unique_lock<mutex> lock(mMutexConnections);

            if (mnId == 0)
            {
                return;
            }
            else if (mbNotErase)
            {
                mbToBeErased = true;
                return;
            }
        }

        // mConnectedKeyFrameWeights：『關鍵幀』與『觀察到的地圖點個數』之對應關係
        for(pair<KeyFrame *, int> kf_weight : mConnectedKeyFrameWeights)
        {
            // 從『關鍵幀』與『觀察到的地圖點個數』之對應關係移除『關鍵幀 pKF』，
            // 並將共視關鍵幀根據觀察到的地圖點數量重新排序
            kf_weight.first->EraseConnection(this);
        }
        
        for(MapPoint* mp : mvpMapPoints)
        {
            if(mp)
            {
                // 移除『關鍵幀 pKF』，更新關鍵幀的計數，若『觀察到這個地圖點的關鍵幀』太少（少於 3 個），
                // 則將地圖點與關鍵幀等全部移除
                mp->EraseObservation(this);
            }
        }

        {
            unique_lock<mutex> lock(mMutexConnections);
            unique_lock<mutex> lock1(mMutexFeatures);

            mConnectedKeyFrameWeights.clear();
            mvpOrderedConnectedKeyFrames.clear();

            // Update Spanning Tree
            set<KeyFrame *> sParentCandidates;

            // 加入『當前關鍵幀』的『父關鍵幀』
            sParentCandidates.insert(mpParent);

            // Assign at each iteration one children with a parent (the pair with highest covisibility 
            // weight) Include that children as new parent candidate for the rest
            while (!mspChildrens.empty())
            {
                bool bContinue = false;

                int max = -1;
                KeyFrame *pC;
                KeyFrame *pP;

                for(KeyFrame *pKF : mspChildrens)
                {
                    if (pKF->isBad())
                    {
                        continue;
                    }

                    // Check if a parent candidate is connected to the keyframe
                    // 根據和『子關鍵幀 pKF』共同觀察到的地圖點數量排序的共視關鍵幀
                    vector<KeyFrame *> vpConnected = pKF->GetVectorCovisibleKeyFrames();

                    // 遍歷『子關鍵幀 pKF』的共視關鍵幀
                    for(KeyFrame * connect_kf : vpConnected)
                    {
                        for(KeyFrame * pc_kf : sParentCandidates)
                        {
                            // 『子關鍵幀 pKF』的共視關鍵幀 ＝＝ sParentCandidates 的關鍵幀
                            // 此種情況可能多次發生，
                            // 而共同觀察到的地圖點數量最多的才會被設為『子關鍵幀 pKF』的『父關鍵幀』
                            if (connect_kf->mnId == pc_kf->mnId)
                            {
                                // 取得『子關鍵幀 pKF』和『關鍵幀 vpConnected[i]』共同觀察到的地圖點數量
                                int w = pKF->GetWeight(connect_kf);

                                // 過濾最大值
                                if (w > max)
                                {
                                    pC = pKF;
                                    pP = connect_kf;

                                    // 最大的共同觀察到的地圖點數量
                                    max = w;

                                    bContinue = true;
                                }
                            }
                        }
                    }
                }

                if (bContinue)
                {
                    // pC：『子關鍵幀 pKF』; pP：共同觀察到的地圖點數量最多的『子關鍵幀 pKF』的共視關鍵幀
                    pC->ChangeParent(pP);

                    // 將『子關鍵幀 pKF』加入 sParentCandidates
                    sParentCandidates.insert(pC);

                    // 從 mspChildrens 移除『子關鍵幀 pKF』 -> 
                    mspChildrens.erase(pC);
                }
                else
                {
                    break;
                }
            }

            // If a children has no covisibility links with any parent candidate, 
            // assign to the original parent of this KF
            /// NOTE: 此時仍在 mspChildrens 當中的關鍵幀必非和 sParentCandidates 當中的關鍵幀沒有共視關係，
            /// 而是共同觀察到的地圖點數量最多的才會被設為『子關鍵幀 pKF』的『父關鍵幀』
            if (!mspChildrens.empty())
            {
                for(KeyFrame * child_kf : mspChildrens)
                {
                    // 由於當前幀表現不佳，即將被移除，因此提昇『子關鍵幀』的層級，
                    // 將當前幀的『子關鍵幀』的『父關鍵幀』設為自己的『父關鍵幀』
                    child_kf->ChangeParent(mpParent);
                }                
            }

            // 將自己從『父關鍵幀』的『子關鍵幀』當中除名
            mpParent->EraseChild(this);

            mTcp = Tcw * mpParent->GetPoseInverse();
            mbBad = true;
        }

        mpMap->EraseKeyFrame(this);
        mpKeyFrameDB->erase(this);
    }

    /* 關鍵幀被認定不佳的可能情形有以下數種：
    1. 地圖中的關鍵幀太少
    2. 關鍵幀中沒有用於迴路檢測的候選點
    3. 當前關鍵幀沒有形成共視圖
    4. 位姿優化後，內點數量過少（少於 20 點）
    5. 迴路檢測時，這一批關鍵幀配對到的點數量過少（少於 40 點）; 或大於 40 點，但該幀沒有配對到

    (若可以就直接將不好的排除，不要一直判斷它好不好)
    */
    bool KeyFrame::isBad()
    {
        unique_lock<mutex> lock(mMutexConnections);
        return mbBad;
    }

    // 從『關鍵幀』與『觀察到的地圖點個數』之對應關係移除『關鍵幀 pKF』，並將共視關鍵幀根據觀察到的地圖點數量重新排序
    void KeyFrame::EraseConnection(KeyFrame *pKF)
    {
        bool bUpdate = false;

        {
            unique_lock<mutex> lock(mMutexConnections);

            if (mConnectedKeyFrameWeights.count(pKF))
            {
                // 從『關鍵幀』與『觀察到的地圖點個數』之對應關係移除『關鍵幀 pKF』
                mConnectedKeyFrameWeights.erase(pKF);

                bUpdate = true;
            }
        }

        if (bUpdate)
        {
            // 將共視關鍵幀根據觀察到的地圖點數量排序，再存入 mvpOrderedConnectedKeyFrames
            UpdateBestCovisibles();
        }
    }

    // 取得區域內的候選關鍵點的索引值
    vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
    {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));

        if (nMinCellX >= mnGridCols){
            return vIndices;
        }

        const int nMaxCellX = min((int)mnGridCols - 1, 
                                  (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));

        if (nMaxCellX < 0){
            return vIndices;
        }

        const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));

        if (nMinCellY >= mnGridRows){
            return vIndices;
        }

        const int nMaxCellY = min((int)mnGridRows - 1, 
                                  (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));

        if (nMaxCellY < 0){
            return vIndices;
        }

        int ix, iy;
        size_t j, jend;

        for (ix = nMinCellX; ix <= nMaxCellX; ix++)
        {
            for (iy = nMinCellY; iy <= nMaxCellY; iy++)
            {
                // 取得網格 mGrid[ix][iy] 所包含的關鍵點的索引值
                const vector<size_t> vCell = mGrid[ix][iy];

                for (j = 0, jend = vCell.size(); j < jend; j++)
                {
                    // vCell[j]：關鍵點的索引值
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    // 判斷 kpUn.pt 是否在指定區域內
                    if (fabs(distx) < r && fabs(disty) < r){

                        // 紀錄區域內的候選關鍵點的索引值
                        vIndices.push_back(vCell[j]);
                    }
                }
            }
        }

        return vIndices;
    }

    // 傳入座標點是否在關鍵幀的成像範圍內
    bool KeyFrame::IsInImage(const float &x, const float &y) const
    {
        return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
    }

    cv::Mat KeyFrame::UnprojectStereo(int i)
    {
        const float z = mvDepth[i];

        if (z > 0)
        {
            const float u = mvKeys[i].pt.x;
            const float v = mvKeys[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;

            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

            unique_lock<mutex> lock(mMutexPose);

            return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + Twc.rowRange(0, 3).col(3);
        }
        else
        {
            return cv::Mat();
        }
    }

    // 取得當前關鍵幀的座標系之下，關鍵幀觀察到的所有地圖點的深度中位數（當 q = 2）
    float KeyFrame::ComputeSceneMedianDepth(const int q)
    {
        vector<MapPoint *> vpMapPoints;
        cv::Mat Tcw_;

        {
            unique_lock<mutex> lock(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPose);
            vpMapPoints = mvpMapPoints;

            // 取出當前關鍵幀的位姿
            Tcw_ = Tcw.clone();
        }

        vector<float> vDepths;
        vDepths.reserve(N);

        // 計算當前幀的旋轉
        cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
        Rcw2 = Rcw2.t();

        // 計算當前幀的平移
        float zcw = Tcw_.at<float>(2, 3);

        for(MapPoint *pMP : mvpMapPoints)
        {
            if (pMP)
            {
                // 取出地圖點的位置
                cv::Mat x3Dw = pMP->GetWorldPos();

                // 將地圖點轉換到當前幀的座標系之下
                float z = Rcw2.dot(x3Dw) + zcw;

                // 取得當前幀的座標系之下，地圖點的深度
                vDepths.push_back(z);
            }
        }
        
        sort(vDepths.begin(), vDepths.end());

        // 取出地圖點深度的中位數（當 q = 2）
        return vDepths[(vDepths.size() - 1) / q];
    }

} //namespace ORB_SLAM
