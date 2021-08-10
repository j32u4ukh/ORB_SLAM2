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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <mutex>
#include <unistd.h>

namespace ORB_SLAM2
{
    // LocalMapping::Run & Tracking::NeedNewKeyFrame & Optimizer::LocalBundleAdjustment 將被暫時停止
    void LocalMapping::RequestStop()
    {
        unique_lock<mutex> lock(mMutexStop);

        // LocalMapping::Run & Tracking::NeedNewKeyFrame 將被暫時停止
        mbStopRequested = true;

        unique_lock<mutex> lock2(mMutexNewKFs);

        // Optimizer::LocalBundleAdjustment 將被暫時停止
        mbAbortBA = true;
    }

    // 是否中止『執行續 LocalMapping』
    bool LocalMapping::isStopped()
    {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    // 清空『新關鍵幀容器』
    void LocalMapping::Release()
    {
        unique_lock<mutex> lock(mMutexStop);
        unique_lock<mutex> lock2(mMutexFinish);

        if (mbFinished){
            return;
        }

        mbStopped = false;
        mbStopRequested = false;
        list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin();
        list<KeyFrame *>::iterator lend = mlNewKeyFrames.end();

        for (; lit != lend; lit++){
            delete *lit;
        }
            
        mlNewKeyFrames.clear();

        cout << "Local Mapping RELEASE" << endl;
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    LocalMapping::LocalMapping(Map *pMap, const float bMonocular) : 
                               mpMap(pMap), mbMonocular(bMonocular), mbResetRequested(false), 
                               mbFinishRequested(false), mbFinished(true), mbAbortBA(false), 
                               mbStopped(false), mbStopRequested(false), mbNotStop(false), 
                               mbAcceptKeyFrames(true)
    {
    }

    void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser)
    {
        mpLoopCloser = pLoopCloser;
    }

    void LocalMapping::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
    }

    void LocalMapping::Run()
    {
        mbFinished = false;

        while (1)
        {
            // Tracking will see that Local Mapping is busy
            // 通過接口 SetAcceptKeyFrames 將成員變量 mbAcceptKeyFrames 置為 false，
            // 目的是要告知 TRACKING 線程，當前 LocalMapping 正在忙
            // 設置『不接受關鍵幀（表示此時 LOCAL MAPPING 線程處於忙碌的狀態）』，無法接受關鍵幀的創建
            SetAcceptKeyFrames(false);

            // Check if there are keyframes in the queue
            // 檢查『新關鍵幀容器 mlNewKeyFrames』是否『不為空』（有新關鍵幀）
            if (CheckNewKeyFrames())
            {
                // BoW conversion and insertion in Map
                // 處理『關鍵幀 mpCurrentKeyFrame』和其他關鍵幀、地圖點、地圖之間的連結
                ProcessNewKeyFrame();

                // Check recent MapPoints
                // 遍歷所有新增的地圖點，通過多道篩選才繼續保留，否則標注為差並移除，存在過久也會被移除但不標注為差
                MapPointCulling();

                // Triangulate new MapPoints
                // 根據配對資訊，透過三角測量找出空間中的地圖點
                CreateNewMapPoints();

                // CheckNewKeyFrames：檢查『新關鍵幀容器 mlNewKeyFrames』是否不為空（有關鍵幀）
                // 若關鍵幀容器為空
                if (!CheckNewKeyFrames())
                {
                    // Find more matches in neighbor keyframes and fuse point duplications
                    // 將『關鍵幀 mpCurrentKeyFrame』觀察到的地圖點與現有的融合，更新關鍵幀之間的共視關係與連結
                    SearchInNeighbors();
                }

                mbAbortBA = false;

                // 『新關鍵幀容器 mlNewKeyFrames』為空（沒有有關鍵幀） 且 沒有請求中止『執行續 LocalMapping』
                if (!CheckNewKeyFrames() && !stopRequested())
                {
                    // Local BA
                    // 『地圖 mpMap』中的關鍵幀數量是否多於 2 個
                    if (mpMap->KeyFramesInMap() > 2)
                    {
                        // 根據區域的共視關係，取出關鍵幀與地圖點來進行多次優化，
                        // 優化後的誤差若仍過大的估計會被移除，並更新估計結果
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
                    }

                    // Check redundant local Keyframes
                    // 地圖點在金字塔層級相同、高 1 階或更低的層級中看到，則該關鍵幀被認為是冗餘的 -> SetBadFlag()
                    KeyFrameCulling();
                }

                // 將『關鍵幀 pKF』加入『關鍵幀的隊列』當中
                mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
            }
            else if (Stop())
            {
                // Safe area to stop
                // 檢查『執行續 LocalMapping』是否在中止狀態 且 LocalMapping 尚未結束
                while (isStopped() && !CheckFinish())
                {
                    usleep(3000);
                }

                if (CheckFinish())
                {
                    break;
                }
            }

            // 若有需要，清空『新關鍵幀容器』以及『最近新增的地圖點』
            ResetIfRequested();

            // Tracking will see that Local Mapping is busy
            // 設置『接受關鍵幀（表示此時 LOCAL MAPPING 線程處於空閑的狀態）』，可以接受關鍵幀的創建
            SetAcceptKeyFrames(true);

            if (CheckFinish())
            {
                break;
            }

            usleep(3000);
        }

        SetFinish();
    }

    void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexNewKFs);

        // 將新的關鍵幀放入容器 mlNewKeyFrames 中
        mlNewKeyFrames.push_back(pKF);

        // 將成員變量 mbAbortBA 設置為 true，表示要暫停 BA 優化
        mbAbortBA = true;
    }

    // 檢查『新關鍵幀容器 mlNewKeyFrames』是否不為空（有關鍵幀）
    bool LocalMapping::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexNewKFs);

        return (!mlNewKeyFrames.empty());
    }

    // 處理『關鍵幀 mpCurrentKeyFrame』和其他關鍵幀、地圖點、地圖之間的連結
    void LocalMapping::ProcessNewKeyFrame()
    {
        /* 由於 TRACKING 線程也會通過 InsertKeyFrame 操作隊列 mlNewKeyFrames，為了防止公共資源的沖突，
        所以在這里對信號量 mMutexNewKFs 加鎖，在程序控制流從大括號中退出的時候，就會釋放局部對象 lock，
        進而解鎖 mMutexNewKFs。*/
        {
            unique_lock<mutex> lock(mMutexNewKFs);

            // 一個局部的環境中從隊列 mlNewKeyFrames 中將對首的關鍵幀，也就是最早的那個
            mpCurrentKeyFrame = mlNewKeyFrames.front();

            // 移除第一個關鍵幀
            mlNewKeyFrames.pop_front();
        }

        // Compute Bags of Words structures
        // 關鍵幀的詞袋計算
        mpCurrentKeyFrame->ComputeBoW();

        // Associate MapPoints to the new keyframe and update normal and descriptor
        // 用 vpMapPointMatches 儲存『關鍵幀 mpCurrentKeyFrame』觀察到的地圖點
        const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        for (size_t i = 0; i < vpMapPointMatches.size(); i++)
        {
            // 由於 vpMapPointMatches中 的每個元素都與關鍵幀中的每個特征點相對應，並不是所有的特征點都
            // 成功匹配到了一個地圖點，那些沒有匹配的特征點所對應的地圖點就是 NULL
            MapPoint *pMP = vpMapPointMatches[i];

            if (pMP)
            {
                if (!pMP->isBad())
                {
                    // 檢查是否已添加過『關鍵幀 mpCurrentKeyFrame』到『地圖點 pMP』
                    // （似乎只在雙目和RGB-D相機下有意義？）
                    if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                    {
                        // 沒有重覆出現過，就相應的更新地圖點的法向量、深度信息、特征描述。

                        // 『地圖點 pMP』被『關鍵幀 mpCurrentKeyFrame』的第 idx 個關鍵點觀察到
                        pMP->AddObservation(mpCurrentKeyFrame, i);

                        // 利用所有觀察到『地圖點 pMP』的關鍵幀來估計關鍵幀們平均指向的方向，
                        // 以及『地圖點 pMP』可能的深度範圍(最近與最遠)
                        pMP->UpdateNormalAndDepth();
                        
                        // 以『所有描述這個地圖點的描述子的集合』的中心描述子，作為『地圖點 pMP』的描述子
                        pMP->ComputeDistinctiveDescriptors();
                    }

                    // this can only happen for new stereo points inserted by the Tracking
                    // 和單目無關，暫時跳過
                    else 
                    {
                        mlpRecentAddedMapPoints.push_back(pMP);
                    }
                }
            }
        }

        // Update links in the Covisibility Graph
        // 通過關鍵幀的接口完成共視圖的更新
        // 其他關鍵幀和『關鍵幀 mpCurrentKeyFrame』觀察到相同的地圖點，且各自都觀察到足夠多的地圖點，
        // 則會和當前幀產生鏈結
        mpCurrentKeyFrame->UpdateConnections();

        // Insert Keyframe in Map
        // 將關鍵幀添加到地圖中
        // 將『關鍵幀 pKF』加到地圖的『關鍵幀陣列』中
        mpMap->AddKeyFrame(mpCurrentKeyFrame);
    }

    // 遍歷所有新增的地圖點，通過多道篩選才繼續保留，否則標注為差並移除，存在過久也會被移除但不標注為差
    void LocalMapping::MapPointCulling()
    {
        /* 一幀圖像所能生成的地圖點是很多的，為了保證系統的效率，必需對地圖點進行一些篩選。
        ORB-SLAM 論文中指出，地圖點只有通過了一些嚴格的篩選之後才能將被認為是可跟蹤的(trackable)並且沒有被錯誤的三角化。
        
        原文提出了兩個篩選條件：
        1. 地圖點在 TRACKING 線程中的查找率(Found Ratio)不能低於 25%
        2. 創建了地圖點，並經過了兩幀關鍵幀之後，至少要有三個關鍵幀能夠觀測到該點。
        */

        // Check Recent Added MapPoints
        // 獲取了最近新增的地圖點集合的叠代器
        list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();

        // 『關鍵幀 mpCurrentKeyFrame』的 ID
        const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

        int nThObs;

        // 根據單目相機還是深度相機設定一個閾值，用於篩選條件 2 的判定
        if (mbMonocular){
            nThObs = 2;
        }
        else{
            nThObs = 3;
        }
            
        const int cnThObs = nThObs;

        // 遍歷所有新增的地圖點，通過多道篩選才繼續保留，否則標注為差並移除，存在過久也會被移除但不標注為差
        while (lit != mlpRecentAddedMapPoints.end())
        {
            MapPoint *pMP = *lit;

            // 如果地圖點的接口 isBad 返回 true 表示這個地圖點曾因為某種原因被認定為野點，將被拋棄掉
            if (pMP->isBad())
            {
                lit = mlpRecentAddedMapPoints.erase(lit);
            }

            // 第一個篩選條件的判定
            // 查找率 Found Ratio
            // 這個所謂的查找率，是在 TRACKING 線程中判定匹配到地圖點的關鍵幀數量與預測可以看到地圖點的關鍵幀數量之比。
            else if (pMP->GetFoundRatio() < 0.25f)
            {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }

            // 第二篩選條件
            // 『當前幀』和『首個觀察到地圖點 pMP 的關鍵幀』之間應至少間隔 3 幀
            // pMP->Observations() <= cnThObs 單目模式下，觀察到『地圖點 pMP』的關鍵幀，至少需要 3 個以上
            else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
            {
                // 『地圖點 pMP』不夠好，清空這個地圖點、觀察到這個地圖點的所有關鍵幀，以及它自己對應的關鍵點索引值
                pMP->SetBadFlag();

                // 清除當前地圖點的指標，並指向下一個地圖點
                lit = mlpRecentAddedMapPoints.erase(lit);
            }

            // 這個條件是說地圖點已經生成並存在很長一段時間了，所以並沒有將之標記為 Bad ，僅僅從容器中移除而已
            else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
            {
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else
            {
                lit++;
            }
        }
    }

    // 根據配對資訊，透過三角測量找出空間中的地圖點
    void LocalMapping::CreateNewMapPoints()
    {
        /* 每當我們有新的關鍵幀插入地圖中後，LocalMapping 都會從共視圖中提取出與新插入的關鍵幀直接關聯的關鍵幀。
        計算它們與新關鍵幀之間的基礎矩陣，並據此通過詞袋模型得到匹配的特征點。
        針對每個匹配的特征點對，計算它們的歸一化平面坐標，以及視差角余弦值。
        篩選出那些具有足夠大的視差角的匹配點對進行三角化。
        成功三角化求得特征點所對應的世界坐標之後，還將進一步把那些負深度的、重投影誤差較大的、
        違反尺度一致性的匹配點對剔除掉。
        */

        // Retrieve neighbor keyframes in covisibility graph
        // nn 是關聯幀數量，估計仍然是處於運行效率的考慮，通過它來設定了一個上限
        int nn = 10;

        if (mbMonocular){
            nn = 20;
        }

        // GetBestCovisibilityKeyFrames：獲取臨接圖中的關聯幀
        // 自『根據觀察到的地圖點數量排序的共視關鍵幀』當中返回至多 nn（對單目而言是 20） 個共視關鍵幀
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        ORBmatcher matcher(0.6, false);

        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        
        // 獲取當前幀的位姿 Tcw1
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));

        // 相機中心坐標 Ow1
        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

        // 取得相機內參 fx1, fy1 ……
        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        // ratioFactor 是一個比例因子，用於篩選尺度一致性
        // 根據當前關鍵幀的尺度因子，計算 ratioFactor ，它將用於根據尺度一致性來判定新建的地圖點是否合理
        const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

        // 計數器 nnew 用於對新建的地圖點計數
        int nnew = 0;

        // Search matches with epipolar restriction and triangulate
        // 遍歷『根據觀察到的地圖點數量排序的共視關鍵幀』（對單目而言有 20 幀）
        for (size_t i = 0; i < vpNeighKFs.size(); i++)
        {
            // 1. 至少與一個關鍵幀配合，三角化能夠與之匹配的 ORB 特征點，構建新的地圖點。
            // 2. 如果有新的關鍵幀插入了，為了計算效率，就不再繼續與其它關鍵幀進行匹配三角化了。
            // ＝ CheckNewKeyFrames：檢查『新關鍵幀容器 mlNewKeyFrames』是否不為空
            if (i > 0 && CheckNewKeyFrames()){
                return;
            }

            // 第 i 個共視關鍵幀
            KeyFrame *pKF2 = vpNeighKFs[i];

            // Check first that baseline is not too short
            // 獲取臨接關鍵幀的相機中心坐標
            cv::Mat Ow2 = pKF2->GetCameraCenter();

            // 兩相機之間的距離，為計算視差時的基線
            cv::Mat vBaseline = Ow2 - Ow1;

            // 計算基線長度
            const float baseline = cv::norm(vBaseline);

            // 非單目
            if (!mbMonocular)
            {
                // 如果基線太短，就不適合對特征點進行三角化了，誤差會比較大
                if (baseline < pKF2->mb){
                    continue;
                }
            }

            // 單目
            else
            {
                // 取得當前關鍵幀的座標系之下，關鍵幀觀察到的所有地圖點的深度中位數（當 q = 2）
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);

                // 計算『基線：深度』比例值
                const float ratioBaselineDepth = baseline / medianDepthKF2;

                // 若視差相對於深度太短，則跳過
                if (ratioBaselineDepth < 0.01){
                    continue;
                }
            }

            // Compute Fundamental Matrix
            // 計算『基礎矩陣(Fundamental Matrix)』
            cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

            // Search matches that fullfil epipolar constraint
            vector<pair<size_t, size_t>> vMatchedIndices;

            /* 進行特征點匹配。匹配的特征點對在兩個關鍵幀中的索引被保存在容器 vMatchedIndices 中，
            匹配器的接口 SearchForTriangulation 還會對匹配的特征點進行篩選，只保留那些滿足對極約束的點對。*/
            // 兩幀之間形成配對的各自地圖點索引值，（『關鍵幀 pKF1』的地圖點索引值，『關鍵幀 pKF2』的地圖點索引值），
            // 存入 vMatchedIndices
            matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

            // 獲取『共視關鍵幀 pKF2』的位姿(Tcw2)和相機內參(fx2,fy2 ……)
            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));

            // 取出『共視關鍵幀 pKF2』的相機內參
            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // 遍歷所有的匹配點對，並分別進行三角化構建地圖點
            for(pair<size_t, size_t> matched_indice : vMatchedIndices){

                // 獲取匹配點對在兩幀中的索引，保存在 idx1 和 idx2 中。
                const int &idx1 = matched_indice.first;
                const int &idx2 = matched_indice.second;
                
                // 根據這兩個索引值分別獲取『關鍵幀 mpCurrentKeyFrame』和『共視關鍵幀 pKF2』上的特徵點 kp1, kp2
                const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
                const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
                bool bStereo1 = kp1_ur >= 0;

                const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                const float kp2_ur = pKF2->mvuRight[idx2];
                bool bStereo2 = kp2_ur >= 0;

                // ==================================================
                // 根據針孔相機模型將『像素坐標』轉換為『歸一化平面坐標』，並計算視差角的餘弦值。
                // ==================================================
                // Check parallax between rays                
                cv::Mat xn1 = (cv::Mat_<float>(3, 1) << 
                                        (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
                cv::Mat xn2 = (cv::Mat_<float>(3, 1) << 
                                        (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

                // Rwc 將相機座標轉換到世界座標下，ray 為世界座標原點指向特徵點的向量（同時也是它們在世界座標下的座標）
                cv::Mat ray1 = Rwc1 * xn1;
                cv::Mat ray2 = Rwc2 * xn2;

                // 計算兩向量餘弦值
                const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));
                // 如果餘弦值為負數，意味著視差角超過了 90 度。在短時間內出現這種情況的可能性很小，一般都是算錯了，
                // 出現了誤匹配才會发生的。餘弦值接近 1，意味著視差角太小，這樣的數據進行三角化容易產生較大的計算誤差。
                // 所以 ORB-SLAM2 只在有足夠大的視差角的情況下對匹配特征點進行三角化。
                // ==================================================

                float cosParallaxStereo = cosParallaxRays + 1;
                float cosParallaxStereo1 = cosParallaxStereo;
                float cosParallaxStereo2 = cosParallaxStereo;

                // 非單目，暫時跳過
                if (bStereo1){
                    cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2, 
                                                                    mpCurrentKeyFrame->mvDepth[idx1]));
                }
                else if (bStereo2){
                    cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));
                }
                    
                cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

                cv::Mat x3D;

                // 0 < cosParallaxRays < 0.9998 表示『角度沒超過 90 度（不是誤比對）』，
                // 也『沒有因視差過小而導致餘弦值很接近 1』
                if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && 
                                                    (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
                {
                    // Linear Triangulation Method
                    cv::Mat A(4, 4, CV_32F);
                    A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                    A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                    A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                    A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                    cv::Mat w, u, vt;
                    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                    x3D = vt.row(3).t();

                    if (x3D.at<float>(3) == 0){
                        continue;
                    }

                    // Euclidean coordinates
                    // 得到特征點在世界坐標下的估計之後，三角化的工作就完成了
                    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
                }

                // 非單目，暫時跳過
                else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
                {
                    x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
                }

                // 非單目，暫時跳過
                else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
                {
                    x3D = pKF2->UnprojectStereo(idx2);
                }

                else{
                    // No stereo and very low parallax
                    continue; 
                }

                // ==================================================
                // ORB-SLAM2 還是對三角化後的地圖點進一步的篩選了一下。首先三角化的點一定在兩幀相機的前方
                // ==================================================
                cv::Mat x3Dt = x3D.t();

                // Check triangulation in front of cameras
                float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);

                if (z1 <= 0){
                    continue;
                }

                float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);

                if (z2 <= 0){
                    continue;
                }
                // ==================================================

                // ==================================================
                // 計算三角化後的地圖點在兩幀圖像中的重投影誤差，剔除那些誤差較大的點
                // ==================================================
                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];

                // 計算特徵點在空間中的位置
                const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);

                const float invz1 = 1.0 / z1;

                // 若為單目
                if (!bStereo1)
                {
                    /* computeReprojectionError(const float fx, const float fy, const float cx, 
                    const float cy,  const float x, const float y, const float inv_z, 
                    const cv::KeyPoint keypoint) 

                    // 計算像素座標
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float v1 = fy1 * y1 * invz1 + cy1;

                    // 計算重投影誤差
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;

                    // 重投影誤差過大則跳過
                    return errX1 * errX1 + errY1 * errY1;                    
                    */

                    // 計算像素座標
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float v1 = fy1 * y1 * invz1 + cy1;

                    // 計算重投影誤差
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;

                    // 重投影誤差過大則跳過
                    if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1){
                        continue;
                    }                        
                }

                // 非單目，暫時跳過
                else
                {
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                    float v1 = fy1 * y1 * invz1 + cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    float errX1_r = u1_r - kp1_ur;

                    if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1){
                        continue;
                    }
                }
                // ==================================================

                //Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

                // 計算特徵點在空間中的位置
                const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
                const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);

                const float invz2 = 1.0 / z2;

                // 若為單目
                if (!bStereo2)
                {
                    // 計算像素座標
                    float u2 = fx2 * x2 * invz2 + cx2;
                    float v2 = fy2 * y2 * invz2 + cy2;

                    // 計算重投影誤差
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;

                    // 重投影誤差過大則跳過
                    if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2){
                        continue;
                    }
                }

                // 非單目，暫時跳過
                else
                {
                    float u2 = fx2 * x2 * invz2 + cx2;
                    float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                    float v2 = fy2 * y2 * invz2 + cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    float errX2_r = u2_r - kp2_ur;

                    if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2){
                        continue;
                    }
                }

                // ==================================================
                // 計算三角化後地圖點在兩幀圖像中的深度比例，以及兩幀圖像的尺度因子的比例關系，剔除那些差異較大的點。
                // ==================================================
                //Check scale consistency
                cv::Mat normal1 = x3D - Ow1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = x3D - Ow2;
                float dist2 = cv::norm(normal2);

                if (dist1 == 0 || dist2 == 0){
                    continue;
                }

                const float ratioDist = dist2 / dist1;
                const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / 
                                                                    pKF2->mvScaleFactors[kp2.octave];

                /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
                if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor){
                    continue;
                }
                // ==================================================

                // Triangulation is succesfull
                // 如果成功進行了三角化，就會新建一個地圖點，並相應的更新關鍵幀與該地圖點之間的可視關系。
                MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

                // 『地圖點 pMP』被『關鍵幀 mpCurrentKeyFrame』的第 idx1 個關鍵點所觀察到
                pMP->AddObservation(mpCurrentKeyFrame, idx1);

                // 『地圖點 pMP』同時也被『關鍵幀 pKF2』的第 idx2 個關鍵點所觀察到
                pMP->AddObservation(pKF2, idx2);

                // 『關鍵幀 mpCurrentKeyFrame』的第 idx1 個關鍵點觀察到了『地圖點 pMP』
                mpCurrentKeyFrame->AddMapPoint(pMP, idx1);

                // 『關鍵幀 mpCurrentKeyFrame』的第 idx1 個關鍵點觀察到了『地圖點 pMP』
                pKF2->AddMapPoint(pMP, idx2);

                // 以『所有描述地圖點 pMP 的描述子的集合』的中心描述子，作為『地圖點 pMP』的描述子
                pMP->ComputeDistinctiveDescriptors();

                // 利用所有觀察到『地圖點 pMP』的關鍵幀來估計關鍵幀們平均指向的方向，
                // 以及該地圖點可能的深度範圍(最近與最遠)
                pMP->UpdateNormalAndDepth();

                // 將『地圖點 pMP』加入地圖進行管理
                mpMap->AddMapPoint(pMP);

                // 將『地圖點 pMP』列為近期加入的地圖點
                mlpRecentAddedMapPoints.push_back(pMP);

                // 通過一個計數器 nnew 來累計新建的地圖點數量。
                nnew++;
            }
        }
    }

    // 將『關鍵幀 mpCurrentKeyFrame』觀察到的地圖點與現有的融合，更新關鍵幀之間的共視關係與連結
    void LocalMapping::SearchInNeighbors()
    {
        // Retrieve neighbor keyframes
        int nn = 10;

        if (mbMonocular){
            nn = 20;
        }

        // 返回至多 nn(單目為 20) 個『關鍵幀 mpCurrentKeyFrame』的共視關鍵幀（根據觀察到的地圖點數量排序）
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        // 『關鍵幀 mpCurrentKeyFrame』的共視關鍵幀和『共視關鍵幀的共視關鍵幀』
        vector<KeyFrame *> vpTargetKFs;

        // 遍歷『關鍵幀 mpCurrentKeyFrame』的共視關鍵幀
        for(KeyFrame * pKFi : vpNeighKFs){

            /// NOTE: mnFuseTargetForKF 似乎是在避免重複當前環節用的變數
            if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId){
                continue;
            }

            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

            // Extend to some second neighbors
            // 返回至多 5 個『關鍵幀 pKFi』的共視關鍵幀（根據觀察到的地圖點數量排序）
            // 對『關鍵幀 mpCurrentKeyFrame』而言就是共視的共視（自己和部份共視幀也被包含在此當中）
            const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);

            for(KeyFrame *pKFi2 : vpSecondNeighKFs){

                if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || 
                    pKFi2->mnId == mpCurrentKeyFrame->mnId){
                    continue;
                }

                vpTargetKFs.push_back(pKFi2);
            }
        }
        
        // Search matches by projection from current KF in target KFs
        ORBmatcher matcher;

        // 『關鍵幀 mpCurrentKeyFrame』觀察到的地圖點
        vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        for(KeyFrame *pKFi : vpTargetKFs){
            // 『關鍵幀 pKFi』觀察到的地圖點和『現有地圖點』兩者的描述子距離很近，
            // 保留被更多關鍵幀觀察到的一點取代另一點
            /// TODO: 保留較新的一點，除非觀察到舊點的關鍵幀數量顯著多於新點
            matcher.Fuse(pKFi, vpMapPointMatches);
        }

        // Search matches by projection from target KFs in current KF
        // 共視關鍵幀所觀察到的地圖點
        vector<MapPoint *> vpFuseCandidates;
        vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

        /// TODO: 檢視是否可以和上方的迴圈合併？都是 for(KeyFrame *pKFi : vpTargetKFs)
        // 遍歷『關鍵幀 mpCurrentKeyFrame』的共視關鍵幀和『共視關鍵幀的共視關鍵幀』
        for(KeyFrame *pKFi : vpTargetKFs)
        {
            // 取得『關鍵幀 pKFi』觀察到的地圖點
            vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

            for(MapPoint *pMP : vpMapPointsKFi)
            {
                if (!pMP){
                    continue;
                }

                if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId){
                    continue;
                }

                pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
                vpFuseCandidates.push_back(pMP);
            }
        }
        
        // 『關鍵幀 mpCurrentKeyFrame』觀察到的地圖點和『現有地圖點』兩者的描述子距離很近，
        // 保留被更多關鍵幀觀察到的一點取代另一點
        matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

        // Update points
        // 更新後的『關鍵幀 mpCurrentKeyFrame』觀察到的地圖點
        vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        for(MapPoint *pMP : vpMapPointMatches){

            if (pMP)
            {
                if (!pMP->isBad())
                {
                    // 以『所有描述這個地圖點的描述子的集合』的中心描述子，作為地圖點的描述子
                    pMP->ComputeDistinctiveDescriptors();

                    // 利用所有觀察到這個地圖點的關鍵幀來估計關鍵幀們平均指向的方向，
                    // 以及該地圖點可能的深度範圍(最近與最遠)
                    pMP->UpdateNormalAndDepth();
                }
            }
        }

        // Update connections in covisibility graph
        // 其他關鍵幀和『關鍵幀 mpCurrentKeyFrame』觀察到相同的地圖點，且各自都觀察到足夠多的地圖點，則會與之產生鏈結
        mpCurrentKeyFrame->UpdateConnections();
    }

    // 計算『基礎矩陣(Fundamental Matrix)』
    cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
    {
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();

        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        // 由 2 到 1 的旋轉
        cv::Mat R12 = R1w * R2w.t();

        // 由 2 到 1 的平移
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

        // 向量 t12 的『反對稱矩陣』 t^
        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        // E = (t^)R = t12x * R12
        // F = K^-T * E * K^-1
        return K1.t().inv() * t12x * R12 * K2.inv();
    }

    // 是否中止『執行續 LocalMapping』
    bool LocalMapping::Stop()
    {
        unique_lock<mutex> lock(mMutexStop);

        // 是否請求中止 且 沒有『請求不要中止』
        if (mbStopRequested && !mbNotStop)
        {
            mbStopped = true;
            cout << "Local Mapping STOP" << endl;
            return true;
        }

        return false;
    }

    // 請求中止『執行續 LocalMapping』
    bool LocalMapping::stopRequested()
    {
        unique_lock<mutex> lock(mMutexStop);

        // 請求中止『執行續 LocalMapping』（不會光 mbStopRequested 被改為 true 就中止）
        return mbStopRequested;
    }

    // 是否接受關鍵幀（此時 LOCAL MAPPING 線程是否處於空閑的狀態）
    bool LocalMapping::AcceptKeyFrames()
    {
        unique_lock<mutex> lock(mMutexAccept);
        return mbAcceptKeyFrames;
    }

    // 設置『是否接受關鍵幀（此時 LOCAL MAPPING 線程是否處於空閑的狀態）』
    void LocalMapping::SetAcceptKeyFrames(bool flag)
    {
        unique_lock<mutex> lock(mMutexAccept);
        
        mbAcceptKeyFrames = flag;
    }

    // 防止 LocalMapping 暫停用
    bool LocalMapping::SetNotStop(bool flag)
    {
        unique_lock<mutex> lock(mMutexStop);

        if (flag && mbStopped)
        {
            return false;
        }

        // mbNotStop 決定了 Stop 函數是否能夠將成員變量 mbStopped 設置為 true
        // 請求不要中止『執行續 LocalMapping』
        mbNotStop = flag;

        return true;
    }

    // 將 mbAbortBA 設為 true
    void LocalMapping::InterruptBA()
    {
        mbAbortBA = true;
    }

    // 地圖點在相對小關鍵幀（相同、高 1 階或更精細的比例）中看到，則該關鍵幀被認為是冗餘的 -> SetBadFlag()
    void LocalMapping::KeyFrameCulling()
    {        
        /*
        Check redundant keyframes (only local keyframes)
        A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
        in at least other 3 keyframes (in the same or finer scale)
        We only consider close stereo points

        檢查冗餘關鍵幀（僅局部關鍵幀）如果關鍵幀看到的 90% 的 MapPoints 至少在其他 3 個關鍵幀（相同或更精細的比例）中
        看到，則該關鍵幀被認為是冗餘的 我們只考慮接近的立體點

        在更細緻的層級就可看到，無須在當前幀也看到，該關鍵幀被認為是冗餘的
        */

        // 『關鍵幀 mpCurrentKeyFrame』的共視關鍵幀（根據觀察到的地圖點數量排序）
        vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

        for(KeyFrame *pKF : vpLocalKeyFrames){

            if (pKF->mnId == 0){
                continue;
            }

            // 『關鍵幀 pKF』的關鍵點觀察到的地圖點
            const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

            int nObs = 3;
            const int thObs = nObs;
            int nRedundantObservations = 0;

            // 『關鍵幀 pKF』的關鍵點觀察到的『有效地圖點』個數
            int nMPs = 0;

            for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
            {
                MapPoint *pMP = vpMapPoints[i];

                if (pMP)
                {
                    // 觀察到這個地圖點的關鍵幀『不會太少』
                    if (!pMP->isBad())
                    {
                        // 不是單目
                        if (!mbMonocular)
                        {
                            if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0){
                                continue;
                            }
                        }

                        // 『關鍵幀 pKF』的關鍵點觀察到的『有效地圖點』個數
                        nMPs++;

                        // 這個地圖點被足夠多的關鍵幀觀察到
                        if (pMP->Observations() > thObs)
                        {
                            // 根據『關鍵點索引值 i』取得關鍵點，再取得其所在的金字塔層級
                            const int &scaleLevel = pKF->mvKeysUn[i].octave;

                            // 觀察到『共視地圖點 pMP』的『關鍵幀』，以及其『關鍵點』的索引值
                            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                            
                            // 『地圖點 pMP』在相對小關鍵幀（相同、高 1 階或更精細的比例）中看到，
                            // 則該關鍵幀被認為是冗餘的
                            int nObs = 0;

                            for(pair<KeyFrame *, size_t> obs : observations){

                                KeyFrame *pKFi = obs.first;
                                size_t kp_idx = obs.second;

                                if (pKFi == pKF){
                                    continue;
                                }

                                // 根據『關鍵點索引值 kp_idx』取得關鍵點，再取得其所在的金字塔層級
                                const int &scaleLeveli = pKFi->mvKeysUn[kp_idx].octave;

                                // 相對小關鍵幀（相同、高 1 階或更精細的比例）中看到
                                if (scaleLeveli <= scaleLevel + 1)
                                {
                                    nObs++;

                                    if (nObs >= thObs){
                                        break;
                                    }
                                }
                            }

                            if (nObs >= thObs)
                            {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }

            // 『地圖點 pMP』在相對小關鍵幀（相同、高 1 階或更精細的比例）中看到，
            // 則該關鍵幀被認為是冗餘的
            if (nRedundantObservations > 0.9 * nMPs){
                pKF->SetBadFlag();
            }
        }

    }

    // 取得傳入向量的『反對稱矩陣』
    cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
    {
        return (cv::Mat_<float>(3, 3) << 
                              0, -v.at<float>(2),  v.at<float>(1),
                 v.at<float>(2),               0, -v.at<float>(0),
                -v.at<float>(1),  v.at<float>(0),              0);
    }

    // 請求清空『新關鍵幀容器』以及『最近新增的地圖點』
    void LocalMapping::RequestReset()
    {
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        while (1)
        {
            {
                unique_lock<mutex> lock2(mMutexReset);

                if (!mbResetRequested){
                    break;
                }
            }

            usleep(3000);
        }
    }

    // 若有需要，清空『新關鍵幀容器』以及『最近新增的地圖點』
    void LocalMapping::ResetIfRequested()
    {
        unique_lock<mutex> lock(mMutexReset);

        if (mbResetRequested)
        {
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested = false;
        }
    }

    void LocalMapping::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool LocalMapping::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void LocalMapping::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
        unique_lock<mutex> lock2(mMutexStop);
        mbStopped = true;
    }

    // 『執行續 LocalMapping』是否已結束
    bool LocalMapping::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);

        return mbFinished;
    }

} //namespace ORB_SLAM
