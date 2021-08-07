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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include <mutex>
#include <thread>
#include <unistd.h>

namespace ORB_SLAM2
{

    LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale) : 
                             mpMap(pMap), mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mbFixScale(bFixScale),
                             mbResetRequested(false), mbFinishRequested(false), mbFinished(true),
                             mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
                             mbStopGBA(false), mpThreadGBA(NULL), mnFullBAIdx(0)
    {
        mnCovisibilityConsistencyTh = 3;
    }

    void LoopClosing::SetTracker(Tracking *pTracker)
    {
        mpTracker = pTracker;
    }

    void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    void LoopClosing::Run()
    {
        mbFinished = false;

        while (1)
        {
            // Check if there are keyframes in the queue
            // 檢查『關鍵幀的隊列』是否有關鍵幀
            if (CheckNewKeyFrames())
            {
                // Detect loop candidates and check covisibility consistency
                // 篩選和當前關鍵幀有相同單字的關鍵幀，若是第一次出現則加入容器，若之前出現過，則更新出現次數
                // 若容器內的關鍵幀和之前的容器至少共享一個關鍵幀的次數足夠多，則可用於檢測迴路，
                // 檢查是否有可用於檢測迴路的關鍵幀
                if (DetectLoop())
                {
                    // Compute similarity transformation [sR|t]
                    // In the stereo/RGBD case s=1
                    // 「返回迴路檢測是否成功（是否觀測到迴路的形成）」
                    // 利用『當前關鍵幀 mpCurrentKF』和『候選關鍵幀』找出『相似轉換矩陣 Sim3』，
                    // 不斷更新當前關鍵幀的特征點和地圖點的匹配，內點足夠多的『候選關鍵幀』可作為
                    // 『選定的閉環關鍵幀 mpMatchedKF』協助後續校正，並根據匹配的個數決定閉環是否成功
                    if (ComputeSim3())
                    {
                        // Perform loop fusion and pose graph optimization
                        // 『當前關鍵幀』轉換到『共視關鍵幀』的『相似轉換矩陣』作為頂點;
                        // 『共視關鍵幀』之間的轉換的『相似轉換矩陣』作為『邊』
                        // 優化後，重新估計各個『相似轉換矩陣』以及地圖點的位置
                        CorrectLoop();
                    }
                }
            }

            // 若有重置請求，重置『關鍵幀隊列』、『最新一筆關鍵幀的 id』以及『重置請求』
            ResetIfRequested();

            // 檢查結束 LoopClosing 執行續的請求
            if (CheckFinish())
            {
                break;
            }

            usleep(5000);
        }

        // 標注 LoopClosing 執行續已停止
        SetFinish();
    }

    // 將『關鍵幀 pKF』加入『關鍵幀的隊列』當中
    void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexLoopQueue);

        if (pKF->mnId != 0){
            mlpLoopKeyFrameQueue.push_back(pKF);
        }
    }

    // 檢查『關鍵幀的隊列』是否有關鍵幀
    bool LoopClosing::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexLoopQueue);

        // 『關鍵幀的隊列』是否不為空（有關鍵幀）
        return (!mlpLoopKeyFrameQueue.empty());
    }

    // 篩選和當前關鍵幀有相同單字的關鍵幀，若是第一次出現則加入容器，若之前出現過，則更新出現次數
    // 若容器內的關鍵幀和之前的容器至少共享一個關鍵幀的次數足夠多，則可用於檢測迴路，返回是否有可用於檢測迴路的關鍵幀
    bool LoopClosing::DetectLoop()
    {
        {
            unique_lock<mutex> lock(mMutexLoopQueue);

            // 從『關鍵幀的隊列』中取得第一個關鍵幀
            mpCurrentKF = mlpLoopKeyFrameQueue.front();

            // 從『關鍵幀的隊列』中取得第一個關鍵幀
            mlpLoopKeyFrameQueue.pop_front();

            // Avoid that a keyframe can be erased while it is being process by this thread
            // 請求不要移除當前關鍵幀，避免在『執行續 LoopClosing』處理關鍵幀時被移除
            mpCurrentKF->SetNotErase();
        }

        // If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
        // 如果『地圖包含的關鍵幀少於 10 個』或『已通過上次循環檢測的關鍵幀少於 10 個』
        if (mpCurrentKF->mnId < mLastLoopKFid + 10)
        {
            mpKeyFrameDB->add(mpCurrentKF);

            // 取消『不要移除當前關鍵幀』的請求
            // 若有『移除當前關鍵幀』的請求，則設為移除當前關鍵幀，和觀察者不足的地圖點及其關鍵幀
            mpCurrentKF->SetErase();

            return false;
        }

        // Compute reference BoW similarity score
        // This is the lowest score to a connected keyframe in the covisibility graph
        // We will impose loop candidates to have a higher similarity than this

        // 取得『關鍵幀 mpCurrentKF』的共視關鍵幀（根據觀察到的地圖點數量排序），用於協助篩選相似分數最小
        const vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();

        // 取得『關鍵幀 mpCurrentKF』的單字權重
        const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;

        // 相似分數最小
        float minScore = 1;

        // 篩選相似分數最小
        for(KeyFrame *pKF : vpConnectedKeyFrames){

            if (pKF->isBad()){
                continue;
            }
            
            // 取得『共視關鍵幀 pKF』的單字權重
            const DBoW2::BowVector &BowVec = pKF->mBowVec;

            // 比較『關鍵幀 mpCurrentKF』和『共視關鍵幀 pKF』的相似程度，值越大代表相似度越高[0, 1]
            float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

            // 篩選相似分數最小
            if (score < minScore){
                minScore = score;
            }
        }

        // for (size_t i = 0; i < vpConnectedKeyFrames.size(); i++)
        // {
        //     // 第 i 個『共視關鍵幀 pKF』
        //     KeyFrame *pKF = vpConnectedKeyFrames[i];
        //     if (pKF->isBad()){
        //         continue;
        //     }            
        //     // 取得『共視關鍵幀 pKF』的單字權重
        //     const DBoW2::BowVector &BowVec = pKF->mBowVec;
        //     // 比較『關鍵幀 mpCurrentKF』和『共視關鍵幀 pKF』的相似程度，值越大代表相似度越高[0, 1]
        //     float score = mpORBVocabulary->score(CurrentBowVec, BowVec);
        //     // 篩選相似分數最小
        //     if (score < minScore){
        //         minScore = score;
        //     }
        // }

        // Query the database imposing the minimum score
        // 計算和『關鍵幀 pKF』有相同單字的『關鍵幀及其共視關鍵幀』和『關鍵幀 pKF』的相似程度，將相似程度高的關鍵幀返回
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

        // If there are no loop candidates, just add new keyframe and return false
        if (vpCandidateKFs.empty())
        {
            mpKeyFrameDB->add(mpCurrentKF);
            mvConsistentGroups.clear();
            mpCurrentKF->SetErase();
            return false;
        }

        // For each loop candidate check consistency with previous loop candidates
        // Each candidate expands a covisibility group (keyframes connected to the loop candidate
        // in the covisibility graph)
        // A group is consistent with a previous group if they share at least a keyframe
        // We must detect a consistent loop in several consecutive keyframes to accept it
        /* 
        * 對於每個循環候選檢查與先前循環候選的一致性
        * 每個候選擴展一個共視群（連接到 共視圖 中循環候選的關鍵幀）
        * 如果他們至少共享一個關鍵幀，則該組與前一組一致
        * 我們必須在幾個連續的關鍵幀中檢測到一致的循環才能接受它
        */
        mvpEnoughConsistentCandidates.clear();

        vector<ConsistentGroup> vCurrentConsistentGroups;

        //  std::vector<pair<set<KeyFrame*>,int>> mvConsistentGroups
        vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);

        // 遍歷候選關鍵幀
        for(KeyFrame *pCandidateKF : vpCandidateKFs){
            
            // 『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』
            set<KeyFrame *> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();

            // spCandidateGroup：『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』和『候選關鍵幀 pCandidateKF』
            spCandidateGroup.insert(pCandidateKF);

            bool bEnoughConsistent = false;
            bool bConsistentForSomeGroup = false;
            
            for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
            {
                // 第 iG 個 ConsistentGroup 的關鍵幀們
                set<KeyFrame *> sPreviousGroup = mvConsistentGroups[iG].first;

                bool bConsistent = false;

                for(KeyFrame * sit : spCandidateGroup){

                    // 若『關鍵幀 (*sit)』已存在 sPreviousGroup 當中
                    if (sPreviousGroup.count(sit))
                    {
                        // 第 iG 個 ConsistentGroup 和 sPreviousGroup 具有『一致性（至少共享一個關鍵幀）』
                        bConsistent = true;

                        // spCandidateGroup 當中至少有一個『關鍵幀 (*sit)』已存在 sPreviousGroup 當中
                        bConsistentForSomeGroup = true;

                        break;
                    }
                }

                // 若 mvConsistentGroups 當中的關鍵幀有一個已存在於 sPreviousGroup 當中
                if (bConsistent)
                {
                    int nPreviousConsistency = mvConsistentGroups[iG].second;

                    // 和 sPreviousGroup 至少共享一個關鍵幀的次數
                    int nCurrentConsistency = nPreviousConsistency + 1;

                    // 若第 iG 個共視群之前不存在
                    if (!vbConsistentGroup[iG])
                    {
                        // spCandidateGroup：『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』和
                        // 『候選關鍵幀 pCandidateKF』
                        ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                        vCurrentConsistentGroups.push_back(cg);

                        // this avoid to include the same group more than once
                        // 標注第 iG 個共視群已存在
                        vbConsistentGroup[iG] = true; 
                    }

                    // 『一致性分數 nCurrentConsistency』大於門檻值 且 是第一次超過門檻
                    if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                    {
                        mvpEnoughConsistentCandidates.push_back(pCandidateKF);

                        // this avoid to insert the same candidate more than once
                        bEnoughConsistent = true; 
                    }
                }
            }

            // If the group is not consistent with any previous group insert 
            // with consistency counter set to zero
            // spCandidateGroup 當中沒有任一個『關鍵幀 (*sit)』存在 sPreviousGroup 當中
            // 當 sPreviousGroup 還沒有內容時，也會先進到這個區塊
            if (!bConsistentForSomeGroup)
            {
                // spCandidateGroup：『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』，形成 ConsistentGroup
                ConsistentGroup cg = make_pair(spCandidateGroup, 0);
                vCurrentConsistentGroups.push_back(cg);
            }
        }
        
        // for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++)
        // {
        //     // 『候選關鍵幀 pCandidateKF』
        //     KeyFrame *pCandidateKF = vpCandidateKFs[i];
        //     // 『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』
        //     set<KeyFrame *> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        //     // spCandidateGroup：『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』和『候選關鍵幀 pCandidateKF』
        //     spCandidateGroup.insert(pCandidateKF);
        //     bool bEnoughConsistent = false;
        //     bool bConsistentForSomeGroup = false;
        //     for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
        //     {
        //         // 第 iG 個 ConsistentGroup 的關鍵幀們
        //         set<KeyFrame *> sPreviousGroup = mvConsistentGroups[iG].first;
        //         bool bConsistent = false;
        //         set<KeyFrame *>::iterator sit = spCandidateGroup.begin();
        //         set<KeyFrame *>::iterator send = spCandidateGroup.end();
        //         for (; sit != send; sit++)
        //         {
        //             // 若『關鍵幀 (*sit)』已存在 sPreviousGroup 當中
        //             if (sPreviousGroup.count(*sit))
        //             {
        //                 // 第 iG 個 ConsistentGroup 和 sPreviousGroup 具有『一致性（至少共享一個關鍵幀）』
        //                 bConsistent = true;
        //                 // spCandidateGroup 當中至少有一個『關鍵幀 (*sit)』已存在 sPreviousGroup 當中
        //                 bConsistentForSomeGroup = true;
        //                 break;
        //             }
        //         }
        //         // 若 mvConsistentGroups 當中的關鍵幀有一個已存在於 sPreviousGroup 當中
        //         if (bConsistent)
        //         {
        //             int nPreviousConsistency = mvConsistentGroups[iG].second;
        //             // 和 sPreviousGroup 至少共享一個關鍵幀的次數
        //             int nCurrentConsistency = nPreviousConsistency + 1;
        //             // 若第 iG 個共視群之前不存在
        //             if (!vbConsistentGroup[iG])
        //             {
        //                 // spCandidateGroup：『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』和
        //                 // 『候選關鍵幀 pCandidateKF』
        //                 ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
        //                 vCurrentConsistentGroups.push_back(cg);
        //                 // this avoid to include the same group more than once
        //                 // 標注第 iG 個共視群已存在
        //                 vbConsistentGroup[iG] = true; 
        //             }
        //             // 『一致性分數 nCurrentConsistency』大於門檻值 且 是第一次超過門檻
        //             if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
        //             {
        //                 mvpEnoughConsistentCandidates.push_back(pCandidateKF);
        //                 // this avoid to insert the same candidate more than once
        //                 bEnoughConsistent = true; 
        //             }
        //         }
        //     }
        //     // If the group is not consistent with any previous group insert 
        //     // with consistency counter set to zero
        //     // spCandidateGroup 當中沒有任一個『關鍵幀 (*sit)』存在 sPreviousGroup 當中
        //     // 當 sPreviousGroup 還沒有內容時，也會先進到這個區塊
        //     if (!bConsistentForSomeGroup)
        //     {
        //         // spCandidateGroup：『候選關鍵幀 pCandidateKF』的『已連結關鍵幀』，形成 ConsistentGroup
        //         ConsistentGroup cg = make_pair(spCandidateGroup, 0);
        //         vCurrentConsistentGroups.push_back(cg);
        //     }
        // }

        // Update Covisibility Consistent Groups
        // 更新 mvConsistentGroups 為 vCurrentConsistentGroups
        mvConsistentGroups = vCurrentConsistentGroups;

        // Add Current Keyframe to database
        mpKeyFrameDB->add(mpCurrentKF);

        if (mvpEnoughConsistentCandidates.empty())
        {
            mpCurrentKF->SetErase();
            return false;
        }
        else
        {
            return true;
        }

        // /// NOTE: 這裡應該不會被執行到吧
        // mpCurrentKF->SetErase();
        // return false;
    }

    // 「返回迴路檢測是否成功（是否觀測到迴路的形成）」
    // 利用『當前關鍵幀 mpCurrentKF』和『候選關鍵幀』找出『相似轉換矩陣 Sim3』，不斷更新當前關鍵幀的特征點和地圖點的匹配，
    // 內點足夠多的『候選關鍵幀』可作為『選定的閉環關鍵幀 mpMatchedKF』協助後續校正，並根據匹配的個數決定閉環是否成功
    bool LoopClosing::ComputeSim3()
    {
        // For each consistent loop candidate we try to compute a Sim3

        const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

        // We compute first ORB matches for each candidate
        // If enough matches are found, we setup a Sim3Solver
        ORBmatcher matcher(0.75, true);

        vector<Sim3Solver *> vpSim3Solvers;
        vpSim3Solvers.resize(nInitialCandidates);

        vector<vector<MapPoint *>> vvpMapPointMatches;
        vvpMapPointMatches.resize(nInitialCandidates);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nInitialCandidates);

        // candidates with enough matches
        int nCandidates = 0;

        for (int i = 0; i < nInitialCandidates; i++)
        {
            // pKF：mvpEnoughConsistentCandidates[i] detectloop 給出的候選關鍵幀
            KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

            // avoid that local mapping erase it while it is being processed in this thread
            // 請求不要移除當前關鍵幀，避免在『執行續 LoopClosing』處理關鍵幀時被移除
            pKF->SetNotErase();

            // 若『關鍵幀 pKF』不佳
            if (pKF->isBad())
            {
                // 標注為廢棄
                vbDiscarded[i] = true;
                continue;
            }

            // mpCurrentKF：當前關鍵幀
            // vvpMapPointMatches[i]：mpCurrentKF 和 mvpEnoughConsistentCandidates[i]（pKF）匹配的地圖點
            // 『關鍵幀 pKF1』和『關鍵幀 pKF2』上的關鍵點距離足夠小的關鍵點個數
            int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

            // 『關鍵幀 pKF1』和『關鍵幀 pKF2』配對的點數少於 20 
            if (nmatches < 20)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // vvpMapPointMatches[i]：mpCurrentKF和mvpEnoughConsistentCandidates[i] （pKF）匹配的地圖點
                // 將『關鍵幀 pKF1』和『關鍵幀 pKF2』的對結果用於優化
                Sim3Solver *pSolver = 
                                new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
                pSolver->SetRansacParameters(0.99, 20, 300);
                vpSim3Solvers[i] = pSolver;
            }

            // 候選個數加一
            nCandidates++;
        }

        bool bMatch = false;

        // Perform alternatively RANSAC iterations for each candidate
        // until one is succesful or all fail
        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nInitialCandidates; i++)
            {
                if (vbDiscarded[i]){
                    continue;
                }

                KeyFrame *pKF = mvpEnoughConsistentCandidates[i];

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                Sim3Solver *pSolver = vpSim3Solvers[i];

                // Sim3Solver 執行 5 次
                // 返回最佳規模尺度下的『相似轉換矩陣』
                cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If RANSAC returns a Sim3, perform a guided matching and optimize with 
                // all correspondences
                if (!Scm.empty())
                {
                    // vvpMapPointMatches[i]：mpCurrentKF 和 mvpEnoughConsistentCandidates[i]（pKF）
                    // 匹配的地圖點
                    vector<MapPoint *> vpMapPointMatches(vvpMapPointMatches[i].size(), 
                                                         static_cast<MapPoint *>(NULL));

                    for (size_t j = 0, jend = vbInliers.size(); j < jend; j++)
                    {
                        // 若為內點（已在 pSolver->iterate 作過判斷）
                        if (vbInliers[j]){
                            vpMapPointMatches[j] = vvpMapPointMatches[i][j];
                        }
                    }

                    // 最佳旋轉矩陣、平移向量、規模尺度，皆為在 pSolver->iterate 中求出
                    cv::Mat R = pSolver->GetEstimatedRotation();
                    cv::Mat t = pSolver->GetEstimatedTranslation();                    
                    const float s = pSolver->GetEstimatedScale();

                    // 利用『相似轉換矩陣』將 mpCurrentKF 和 pKF 各自觀察到的地圖點投影到彼此上，
                    // 若雙方都找到同樣的匹配關係，則替換 vpMapPointMatches 的地圖點
                    matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

                    g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);

                    // 將『地圖點 pMP1、pMP2』的位置轉換到相機座標系下，作為『頂點』加入優化，
                    // 相對應的特徵點位置作為『邊』加入，優化並排除誤差過大的估計後，重新估計『相似轉換矩陣』
                    const int nInliers = Optimizer::OptimizeSim3(
                                            mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                    // If optimization is succesful stop ransacs and continue
                    if (nInliers >= 20)
                    {
                        bMatch = true;

                        // mpMatchedKF：選定的閉環關鍵幀
                        mpMatchedKF = pKF;

                        g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), 
                                       Converter::toVector3d(pKF->GetTranslation()), 1.0);
                        mg2oScw = gScm * gSmw;
                        mScw = Converter::toCvMat(mg2oScw);

                        // vpMapPointMatches ＝ vvpMapPointMatches[i]
                        // mpCurrentKF 和 mvpEnoughConsistentCandidates[i] 匹配的地圖點
                        mvpCurrentMatchedPoints = vpMapPointMatches;
                        break;
                    }
                }
            }
        }

        if (!bMatch)
        {
            for (int i = 0; i < nInitialCandidates; i++)
            {
                mvpEnoughConsistentCandidates[i]->SetErase();
            }

            mpCurrentKF->SetErase();

            return false;
        }

        // Retrieve MapPoints seen in Loop Keyframe and neighbors
        // 取得『選定的閉環關鍵幀 mpMatchedKF』的『共視關鍵幀』（根據觀察到的地圖點數量排序）
        vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();

        // 『選定的閉環關鍵幀 mpMatchedKF』及其『共視關鍵幀』
        vpLoopConnectedKFs.push_back(mpMatchedKF);

        mvpLoopMapPoints.clear();

        // 遍歷『關鍵幀 mpMatchedKF』及其『共視關鍵幀』
        for(KeyFrame *pKF : vpLoopConnectedKFs){

            // 取得『關鍵幀 pKF』觀察到的地圖點
            vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

            for(MapPoint *pMP : vpMapPoints){

                if (pMP)
                {
                    if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
                    {
                        // 『關鍵幀 mpMatchedKF』及其『共視關鍵幀』所觀察到的地圖點
                        mvpLoopMapPoints.push_back(pMP);
                        pMP->mnLoopPointForKF = mpCurrentKF->mnId;
                    }
                }
            }
        }

        // vector<KeyFrame *>::iterator vit = vpLoopConnectedKFs.begin();
        // // 遍歷『關鍵幀 mpMatchedKF』及其『共視關鍵幀』
        // for (; vit != vpLoopConnectedKFs.end(); vit++)
        // {
        //     KeyFrame *pKF = *vit;
        //     取得『關鍵幀 pKF』觀察到的地圖點
        //     vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();       
        //     for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        //     {
        //         MapPoint *pMP = vpMapPoints[i];
        //         if (pMP)
        //         {
        //             if (!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
        //             {
        //                 // 『關鍵幀 mpMatchedKF』及其『共視關鍵幀』所觀察到的地圖點
        //                 mvpLoopMapPoints.push_back(pMP);
        //                 pMP->mnLoopPointForKF = mpCurrentKF->mnId;
        //             }
        //         }
        //     }
        // }

        // Find more matches projecting with the computed Sim3
        // vvpMapPointMatches[i]：mpCurrentKF 和 mvpEnoughConsistentCandidates[i]（pKF）
        // mvpLoopMapPoints：和『關鍵幀 mpCurrentKF』已配對的『關鍵幀及其共視關鍵幀觀察到的地圖點』
        // 已透過 Sim3 尋找了匹配關係，這裡在『已配對的關鍵幀及其共視關鍵幀觀察到的地圖點』當中進一步尋找匹配關係
        // 『地圖點們 vpPoints』投影到『關鍵幀 pKF』上，vpMatched 為匹配結果，第 idx 個特徵點對應『地圖點 pMP』
        // mvpLoopMapPoints 的地圖點若再次匹配成功，也會被保存在 mvpCurrentMatchedPoints
        matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

        // If enough matches accept Loop
        int nTotalMatches = 0;
        
        // mvpCurrentMatchedPoints：根據已配對的地圖點與關鍵幀，再次匹配成功後找到的『地圖點』
        // mvpCurrentMatchedPoints[idx] = pMP -> 『關鍵幀 pKF』的第 idx 個特徵點對應『地圖點 pMP』                
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
        {
            if (mvpCurrentMatchedPoints[i]){
                nTotalMatches++;
            }
        }

        // 當成功匹配的地圖點足夠多（大於等於 40 點）
        if (nTotalMatches >= 40)
        {
            for (int i = 0; i < nInitialCandidates; i++){

                // 移除『選定的閉環關鍵幀 mpMatchedKF』以外的關鍵幀
                if (mvpEnoughConsistentCandidates[i] != mpMatchedKF){
                    mvpEnoughConsistentCandidates[i]->SetErase();
                }
            
            }

            // 觀測到迴路的形成
            return true;
        }
        else
        {
            // 移除所有候選幀
            for (int i = 0; i < nInitialCandidates; i++){
                mvpEnoughConsistentCandidates[i]->SetErase();
            }

            // 移除當前幀
            mpCurrentKF->SetErase();

            // 沒有觀測到迴路的形成
            return false;
        }
    }

    // 『當前關鍵幀』轉換到『共視關鍵幀』的『相似轉換矩陣』作為頂點;『共視關鍵幀』之間的轉換的『相似轉換矩陣』作為『邊』
    // 優化後，重新估計各個『相似轉換矩陣』以及地圖點的位置
    void LoopClosing::CorrectLoop()
    {
        cout << "Loop detected!" << endl;

        // Send a stop signal to Local Mapping
        // Avoid new keyframes are inserted while correcting the loop
        // LocalMapping::Run & Tracking::NeedNewKeyFrame & Optimizer::LocalBundleAdjustment 將被暫時停止
        mpLocalMapper->RequestStop();

        // If a Global Bundle Adjustment is running, abort it
        // 若『執行續 RunGlobalBundleAdjustment』正在執行，將其停止
        if (isRunningGBA())
        {
            unique_lock<mutex> lock(mMutexGBA);

            // 停止『執行續 RunGlobalBundleAdjustment』當中的 BundleAdjustment
            mbStopGBA = true;

            mnFullBAIdx++;

            if (mpThreadGBA)
            {
                // detach 不等待 thread 執行結束
                mpThreadGBA->detach();
                delete mpThreadGBA;
            }
        }

        // Wait until Local Mapping has effectively stopped
        // 『執行續 LocalMapping』中止前持續等待
        while (!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }

        // Ensure current keyframe is updated
        // 其他關鍵幀和『關鍵幀 mpCurrentKF』觀察到相同的地圖點，且各自都觀察到足夠多的地圖點，則會和當前幀產生鏈結
        mpCurrentKF->UpdateConnections();

        // Retrive keyframes connected to the current keyframe and compute corrected 
        // Sim3 pose by propagation
        // 檢索連接到當前關鍵幀的關鍵幀並通過傳播計算校正後的 Sim3 位姿

        // 取得『共視關鍵幀』（根據觀察到的地圖點數量排序）
        mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();

        // 『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
        mvpCurrentConnectedKFs.push_back(mpCurrentKF);

        /*
        typedef map<KeyFrame*,
                    g2o::Sim3,
                    std::less<KeyFrame*>,
                    Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3>>> KeyFrameAndPose;
        */
        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;

        // 紀錄當前關鍵幀座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣 mg2oScw』
        CorrectedSim3[mpCurrentKF] = mg2oScw;

        // 相機座標（『關鍵幀 mpCurrentKF』座標系）到世界座標的轉換矩陣
        cv::Mat Twc = mpCurrentKF->GetPoseInverse();

        {
            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            for(KeyFrame *pKFi : mvpCurrentConnectedKFs){

                // 『關鍵幀 pKFi』的位姿
                cv::Mat Tiw = pKFi->GetPose();

                if (pKFi != mpCurrentKF)
                {
                    // 『關鍵幀 mpCurrentKF』座標系到『關鍵幀 pKFi』座標系
                    cv::Mat Tic = Tiw * Twc;

                    cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
                    cv::Mat tic = Tic.rowRange(0, 3).col(3);

                    // 構成『相似轉換矩陣』
                    g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
                    g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;

                    //Pose corrected with the Sim3 of the loop closure
                    // 紀錄當『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系
                    // 對應的『相似轉換矩陣 g2oCorrectedSiw』
                    CorrectedSim3[pKFi] = g2oCorrectedSiw;
                }

                cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
                cv::Mat tiw = Tiw.rowRange(0, 3).col(3);

                // 由『關鍵幀 pKFi』的位姿構成的『相似轉換矩陣 g2oSiw』
                g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);

                // Pose without correction
                // 紀錄『關鍵幀 pKFi』對應的『相似轉換矩陣 g2oSiw』
                NonCorrectedSim3[pKFi] = g2oSiw;
            }

            // vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin();
            // vector<KeyFrame *>::iterator vend = mvpCurrentConnectedKFs.end();
            // // 遍歷『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
            // for (; vit != vend; vit++)
            // {
            //     KeyFrame *pKFi = *vit;
            //     // 『關鍵幀 pKFi』的位姿
            //     cv::Mat Tiw = pKFi->GetPose();
            //     if (pKFi != mpCurrentKF)
            //     {
            //         // 『關鍵幀 mpCurrentKF』座標系到『關鍵幀 pKFi』座標系
            //         cv::Mat Tic = Tiw * Twc;
            //         cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
            //         cv::Mat tic = Tic.rowRange(0, 3).col(3);
            //         // 構成『相似轉換矩陣』
            //         g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
            //         g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
            //         //Pose corrected with the Sim3 of the loop closure
            //         // 紀錄當『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系
            //         // 對應的『相似轉換矩陣 g2oCorrectedSiw』
            //         CorrectedSim3[pKFi] = g2oCorrectedSiw;
            //     }
            //     cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
            //     cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
            //     // 由『關鍵幀 pKFi』的位姿構成的『相似轉換矩陣 g2oSiw』
            //     g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
            //     // Pose without correction
            //     // 紀錄『關鍵幀 pKFi』對應的『相似轉換矩陣 g2oSiw』
            //     NonCorrectedSim3[pKFi] = g2oSiw;
            // }

        /*
        typedef map<KeyFrame*,
                    g2o::Sim3,
                    std::less<KeyFrame*>,
                    Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3>>> KeyFrameAndPose;
        */
            // Correct all MapPoints obsrved by current keyframe and neighbors, 
            // so that they align with the other side of the loop
            KeyFrameAndPose::iterator mit = CorrectedSim3.begin();
            KeyFrameAndPose::iterator mend = CorrectedSim3.end();

            // 遍歷 CorrectedSim3（紀錄當『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系
            // 對應的『相似轉換矩陣 g2oCorrectedSiw』）
            for (; mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                // 『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系
                // 對應的『相似轉換矩陣 g2oCorrectedSiw』
                g2o::Sim3 g2oCorrectedSiw = mit->second;

                // 『關鍵幀 pKFi』座標系轉換到座標系『關鍵幀 mpCurrentKF』
                // 對應的『相似轉換矩陣 g2oCorrectedSiw』
                g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

                // 『關鍵幀 pKFi』對應的『相似轉換矩陣 g2oSiw』
                g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];
                
                // 『關鍵幀 pKFi』觀察到的地圖點
                vector<MapPoint *> vpMPsi = pKFi->GetMapPointMatches();

                for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
                {
                    MapPoint *pMPi = vpMPsi[iMP];

                    if (!pMPi){
                        continue;
                    }

                    if (pMPi->isBad()){
                        continue;
                    }

                    if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId){
                        continue;
                    }

                    // Project with non-corrected pose and project back with corrected pose
                    // 取得『關鍵幀 pKFi』觀察到的『地圖點 pMPi』的世界座標
                    cv::Mat P3Dw = pMPi->GetWorldPos();

                    Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);

                    // 將『空間點 eigP3Dw』從世界座標系轉換到座標系『關鍵幀 mpCurrentKF』
                    // 轉換過程中，規模尺度也轉換為『關鍵幀 mpCurrentKF』的尺度
                    Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = 
                                                            g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                    // 轉換為『關鍵幀 mpCurrentKF』的尺度後，再重新轉換為世界座標
                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);

                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;

                    // 利用所有觀察到『地圖點 pMPi』的關鍵幀來估計關鍵幀們平均指向的方向，
                    // 以及該地圖點可能的深度範圍(最近與最遠)
                    pMPi->UpdateNormalAndDepth();
                }

                // Update keyframe pose with corrected Sim3. 
                // First transform Sim3 to SE3 (scale translation)
                // 『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系
                // 對應的『相似轉換矩陣 g2oCorrectedSiw』，拆分為旋轉、平移、規模
                Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                double s = g2oCorrectedSiw.scale();

                // [R t/s; 0 1] 縮放規模
                eigt *= (1. / s); 

                // 規模校正後的『轉換矩陣』
                cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

                // 更新『關鍵幀 pKFi』的位姿
                pKFi->SetPose(correctedTiw);

                // Make sure connections are updated
                // 其他關鍵幀和『關鍵幀 pKFi』觀察到相同的地圖點，且各自都觀察到足夠多的地圖點，則會和當前幀產生鏈結
                pKFi->UpdateConnections();
            }

            // Start Loop Fusion
            // Update matched map points and replace if duplicated
            // mvpCurrentMatchedPoints：根據已配對的地圖點與關鍵幀，再次匹配成功後找到的『地圖點』
            for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
            {
                // mvpCurrentMatchedPoints[idx] = pMP -> 『關鍵幀 pKF』的第 idx 個特徵點對應『地圖點 pMP』
                if (mvpCurrentMatchedPoints[i])
                {
                    // 『關鍵幀 mpCurrentKF』的第 idx 個特徵點對應『地圖點 pLoopMP』
                    MapPoint *pLoopMP = mvpCurrentMatchedPoints[i];

                    // 『關鍵幀 mpCurrentKF』的第 idx 個特徵點對應『地圖點 pCurMP』
                    MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);

                    // 若『關鍵幀 mpCurrentKF』的第 idx 個特徵點對應的特徵點已存在
                    if (pCurMP){
                        // 『地圖點 pLoopMP』取代『地圖點 pCurMP』
                        // 將被『較多』關鍵幀觀察到的地圖點，取代被『較少』關鍵幀觀察到的地圖點
                        pCurMP->Replace(pLoopMP);
                    }
                    else
                    {
                        // 『關鍵幀 mpCurrentKF』的第 idx 個關鍵點觀察到了『地圖點 pLoopMP』
                        mpCurrentKF->AddMapPoint(pLoopMP, i);

                        // 『地圖點 pLoopMP』被『關鍵幀 mpCurrentKF』的第 idx 個關鍵點觀察到
                        pLoopMP->AddObservation(mpCurrentKF, i);

                        // 以『所有描述這個地圖點的描述子的集合』的中心描述子，作為地圖點的描述子
                        pLoopMP->ComputeDistinctiveDescriptors();
                    }
                }
            }
        }

        // Project MapPoints observed in the neighborhood of the loop keyframe
        // into the current keyframe and neighbors using corrected poses.
        // Fuse duplications.
        // CorrectedSim3：紀錄當『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系
        // 對應的『相似轉換矩陣 g2oCorrectedSiw』
        // 原有地圖點投影到『關鍵幀 pKF』進行匹配，匹配成功則加入，
        // 若認定和原有地圖點為同一點，將被『較多』關鍵幀觀察到的地圖點，取代被『較少』關鍵幀觀察到的地圖點
        SearchAndFuse(CorrectedSim3);

        // After the MapPoint fusion, new links in the covisibility graph will appear attaching 
        // both sides of the loop
        // LoopConnections[pKFi]：『關鍵幀 pKFi』的『已連結關鍵幀』
        map<KeyFrame *, set<KeyFrame *>> LoopConnections;

        // mvpCurrentConnectedKFs：『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
        for(KeyFrame *pKFi : mvpCurrentConnectedKFs){

            // 取得『關鍵幀 pKFi』的『共視關鍵幀』（根據觀察到的地圖點數量排序）
            vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

            // Update connections. Detect new links.
            // 其他關鍵幀和『關鍵幀 pKFi』觀察到相同的地圖點，且各自都觀察到足夠多的地圖點，則會和當前幀產生鏈結
            pKFi->UpdateConnections();

            // 取得『關鍵幀 pKFi』的『已連結關鍵幀』
            LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();

            // 從『關鍵幀 pKFi』的『已連結關鍵幀』當中移除『關鍵幀 pKFi』的『共視關鍵幀』
            for(KeyFrame * pre_kf : vpPreviousNeighbors){
                LoopConnections[pKFi].erase(pre_kf);
            }

            // 從『關鍵幀 pKFi』的『已連結關鍵幀』當中移除『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
            for(KeyFrame * curr_conn_kf : mvpCurrentConnectedKFs){
                LoopConnections[pKFi].erase(curr_conn_kf);
            }
        }

        // vector<KeyFrame *>::iterator vit = mvpCurrentConnectedKFs.begin();
        // vector<KeyFrame *>::iterator vend = mvpCurrentConnectedKFs.end();
        // for (; vit != vend; vit++)
        // {
        //     KeyFrame *pKFi = *vit;
        //     // 取得『關鍵幀 pKFi』的『共視關鍵幀』（根據觀察到的地圖點數量排序）
        //     vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();
        //     // Update connections. Detect new links.
        //     // 其他關鍵幀和『關鍵幀 pKFi』觀察到相同的地圖點，且各自都觀察到足夠多的地圖點，則會和當前幀產生鏈結
        //     pKFi->UpdateConnections();
        //     // 取得『關鍵幀 pKFi』的『已連結關鍵幀』
        //     LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        //     vector<KeyFrame *>::iterator vit_prev = vpPreviousNeighbors.begin();
        //     vector<KeyFrame *>::iterator vend_prev = vpPreviousNeighbors.end();
        //     // 從『關鍵幀 pKFi』的『已連結關鍵幀』當中移除『關鍵幀 pKFi』的『共視關鍵幀』
        //     for (; vit_prev != vend_prev; vit_prev++)
        //     {
        //         LoopConnections[pKFi].erase(*vit_prev);
        //     }
        //     // mvpCurrentConnectedKFs：『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
        //     vector<KeyFrame *>::iterator vit2 = mvpCurrentConnectedKFs.begin();
        //     vector<KeyFrame *>::iterator vend2 = mvpCurrentConnectedKFs.end();
        //     // 從『關鍵幀 pKFi』的『已連結關鍵幀』當中移除『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
        //     for (; vit2 != vend2; vit2++)
        //     {
        //         LoopConnections[pKFi].erase(*vit2);
        //     }
        // }

        // Optimize graph
        // LoopConnections：『關鍵幀 pKFi』的『已連結關鍵幀』，移除『關鍵幀 pKFi』的『共視關鍵幀』和
        // 『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
        // 『當前關鍵幀』轉換到『共視關鍵幀』的『相似轉換矩陣』作為頂點;
        // 『共視關鍵幀』之間的轉換的『相似轉換矩陣』作為『邊』
        // 優化後，重新估計各個『相似轉換矩陣』以及地圖點的位置
        Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, 
                                          NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

        // 增加『重要變革的索引值』
        mpMap->InformNewBigChange();

        // Add loop edge
        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);

        // Launch a new thread to perform Global Bundle Adjustment
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

        // Loop closed. Release Local Mapping.
        // 清空『新關鍵幀容器』
        mpLocalMapper->Release();

        mLastLoopKFid = mpCurrentKF->mnId;
    }

    // 原有地圖點投影到『關鍵幀 pKF』進行匹配，匹配成功則加入，
    // 若認定和原有地圖點為同一點，將被『較多』關鍵幀觀察到的地圖點，取代被『較少』關鍵幀觀察到的地圖點
    void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
    {
        ORBmatcher matcher(0.8);
        /*
        typedef map<KeyFrame*,
                    g2o::Sim3,
                    std::less<KeyFrame*>,
                    Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3>>> KeyFrameAndPose;
        */

        for(pair<KeyFrame *const, g2o::Sim3> kf_pose : CorrectedPosesMap){

            KeyFrame *pKF = kf_pose.first;

            // 『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKF』座標系
            // 對應的『相似轉換矩陣 g2oScw』
            g2o::Sim3 g2oScw = kf_pose.second;

            cv::Mat cvScw = Converter::toCvMat(g2oScw);

            // mvpLoopMapPoints：和『關鍵幀 mpCurrentKF』已配對的『關鍵幀及其共視關鍵幀』觀察到的地圖點
            vector<MapPoint *> vpReplacePoints(mvpLoopMapPoints.size(), 
                                               static_cast<MapPoint *>(NULL));

            // 『關鍵幀 pKF』觀察到的地圖點和『現有地圖點』兩者的描述子距離很近
            // 被認定為相同的地圖點存在 vpReplacePoints 當中
            matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();

            for (int i = 0; i < nLP; i++)
            {
                MapPoint *pRep = vpReplacePoints[i];

                if (pRep)
                {
                    // 將被『較多』關鍵幀觀察到的地圖點，取代被『較少』關鍵幀觀察到的地圖點
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }
        }

        // KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin();
        // KeyFrameAndPose::const_iterator mend = CorrectedPosesMap.end();
        // for (; mit != mend; mit++)
        // {
        //     KeyFrame *pKF = mit->first;
        //     // 『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKF』座標系
        //     // 對應的『相似轉換矩陣 g2oScw』
        //     g2o::Sim3 g2oScw = mit->second;
        //     cv::Mat cvScw = Converter::toCvMat(g2oScw);
        //     // mvpLoopMapPoints：和『關鍵幀 mpCurrentKF』已配對的『關鍵幀及其共視關鍵幀』觀察到的地圖點
        //     vector<MapPoint *> vpReplacePoints(mvpLoopMapPoints.size(), static_cast<MapPoint *>(NULL));
        //     // 『關鍵幀 pKF』觀察到的地圖點和『現有地圖點』兩者的描述子距離很近
        //     // 被認定為相同的地圖點存在 vpReplacePoints 當中
        //     matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);
        //     // Get Map Mutex
        //     unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        //     const int nLP = mvpLoopMapPoints.size();
        //     for (int i = 0; i < nLP; i++)
        //     {
        //         MapPoint *pRep = vpReplacePoints[i];
        //         if (pRep)
        //         {
        //             // 將被『較多』關鍵幀觀察到的地圖點，取代被『較少』關鍵幀觀察到的地圖點
        //             pRep->Replace(mvpLoopMapPoints[i]);
        //         }
        //     }
        // }
    }

    // 請求重置狀態
    void LoopClosing::RequestReset()
    {
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        while (1)
        {
            {
                unique_lock<mutex> lock2(mMutexReset);

                // 完成重置後，mbResetRequested 會被改成 false，因此這個迴圈的目的是在等待確實被重置
                if (!mbResetRequested){
                    break;
                }
            }

            // 確實被重置之前，暫停 LoopClosing
            usleep(5000);
        }
    }

    // 若有重置請求，重置『關鍵幀隊列』、『最新一筆關鍵幀的 id』以及『重置請求』
    void LoopClosing::ResetIfRequested()
    {
        unique_lock<mutex> lock(mMutexReset);

        if (mbResetRequested)
        {
            mlpLoopKeyFrameQueue.clear();
            mLastLoopKFid = 0;
            mbResetRequested = false;
        }
    }

    // 呼叫 Optimizer::GlobalBundleAdjustemnt 後，『子關鍵幀』以『父關鍵幀』優化後的位姿估計為基礎，
    // 再進行兩幀間的相對運動，更新『子關鍵幀』的位姿估計，而『父關鍵幀』的位姿估計也改為優化後的位姿
    void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
    {
        cout << "Starting Global Bundle Adjustment" << endl;

        int idx = mnFullBAIdx;

        // 從地圖中取出所有『關鍵幀』和『地圖點』，進行 BundleAdjustment
        // nLoopKF：關鍵幀 Id
        Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, nLoopKF, false);

        // Update all MapPoints and KeyFrames
        // Local Mapping was active during BA, that means that there might be new keyframes
        // not included in the Global BA and they are not consistent with the updated map.
        // We need to propagate the correction through the spanning tree
        {
            unique_lock<mutex> lock(mMutexGBA);

            if (idx != mnFullBAIdx){
                return;
            }

            // 是否繼續 LoopClosing::RunGlobalBundleAdjustment
            if (!mbStopGBA)
            {
                cout << "Global Bundle Adjustment finished" << endl;
                cout << "Updating map ..." << endl;

                // LocalMapping::Run & Tracking::NeedNewKeyFrame & Optimizer::LocalBundleAdjustment 
                // 將被暫時停止
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                // 持續等待，直到『執行續 LocalMapping』確實停下來
                while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
                {
                    usleep(1000);
                }

                // Get Map Mutex
                unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

                // Correct keyframes starting at map first keyframe
                // 取得原始的所有關鍵幀
                list<KeyFrame *> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), 
                                             mpMap->mvpKeyFrameOrigins.end());

                // 設置各個『關鍵幀』用於 GBA 從世界座標系到相機座標系的轉換矩陣 
                while (!lpKFtoCheck.empty())
                {
                    KeyFrame *pKF = lpKFtoCheck.front();

                    // 取得『關鍵幀 pKF』的『子關鍵幀』
                    const set<KeyFrame *> sChilds = pKF->GetChilds();

                    // 『關鍵幀 pKF』的相機座標到世界座標的轉換矩陣
                    cv::Mat Twc = pKF->GetPoseInverse();

                    // 遍歷『關鍵幀 pKF』的『子關鍵幀』
                    for(KeyFrame * pChild : sChilds){

                        if (pChild->mnBAGlobalForKF != nLoopKF)
                        {
                            // 『關鍵幀 pKF』的相機座標到『子關鍵幀』的相機座標的轉換矩陣
                            cv::Mat Tchildc = pChild->GetPose() * Twc;

                            // *Tcorc*pKF->mTcwGBA;
                            // pKF->mTcwGBA：『關鍵幀 pKF』在 BundleAdjustment 優化後的位姿估計 
                            // Tchildc：相當於『關鍵幀 pKF』往『子關鍵幀 pChild』的相對運動
                            // Tchildc * pKF->mTcwGBA：
                            // 以『關鍵幀 pKF』優化後的位姿估計為基礎，進行相對運動到『子關鍵幀 pChild』
                            pChild->mTcwGBA = Tchildc * pKF->mTcwGBA; 

                            pChild->mnBAGlobalForKF = nLoopKF;
                        }

                        lpKFtoCheck.push_back(pChild);
                    }

                    // set<KeyFrame *>::const_iterator sit = sChilds.begin();
                    // // 遍歷『關鍵幀 pKF』的『子關鍵幀』
                    // for (; sit != sChilds.end(); sit++)
                    // {
                    //     KeyFrame *pChild = *sit;
                    //     if (pChild->mnBAGlobalForKF != nLoopKF)
                    //     {
                    //         // 『關鍵幀 pKF』的相機座標到『子關鍵幀』的相機座標的轉換矩陣
                    //         cv::Mat Tchildc = pChild->GetPose() * Twc;
                    //         // *Tcorc*pKF->mTcwGBA;
                    //         // pKF->mTcwGBA：『關鍵幀 pKF』在 BundleAdjustment 優化後的位姿估計 
                    //         // Tchildc：相當於『關鍵幀 pKF』往『子關鍵幀 pChild』的相對運動
                    //         // Tchildc * pKF->mTcwGBA：
                    //         // 以『關鍵幀 pKF』優化後的位姿估計為基礎，進行相對運動到『子關鍵幀 pChild』
                    //         pChild->mTcwGBA = Tchildc * pKF->mTcwGBA; 
                    //         pChild->mnBAGlobalForKF = nLoopKF;
                    //     }
                    //     lpKFtoCheck.push_back(pChild);
                    // }

                    // 保存上一次 mTcwGBA 的數值    
                    pKF->mTcwBefGBA = pKF->GetPose();

                    // 利用 mTcwGBA 更新『關鍵幀 pKF』的位姿估計
                    pKF->SetPose(pKF->mTcwGBA);

                    // 移除 lpKFtoCheck 第一個元素
                    lpKFtoCheck.pop_front();
                }

                // Correct MapPoints
                // 取得所有『地圖點』
                const vector<MapPoint *> vpMPs = mpMap->GetAllMapPoints();

                for(MapPoint * pMP : vpMPs){

                    if (pMP->isBad()){
                        continue;
                    }

                    if (pMP->mnBAGlobalForKF == nLoopKF)
                    {
                        // If optimized by Global BA, just update
                        pMP->SetWorldPos(pMP->mPosGBA);
                    }
                    else
                    {
                        // Update according to the correction of its reference keyframe
                        // 取得『地圖點 pMP』的參考關鍵幀
                        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();

                        if (pRefKF->mnBAGlobalForKF != nLoopKF){
                            continue;
                        }

                        // Map to non-corrected camera
                        cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                        cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);

                        // 將『地圖點 pMP』從世界座標轉換到相機座標下
                        cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

                        // Backproject using corrected camera
                        cv::Mat Twc = pRefKF->GetPoseInverse();
                        cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                        cv::Mat twc = Twc.rowRange(0, 3).col(3);

                        // 利用 mTcwBefGBA 轉換到相機座標系，再利用 GetPoseInverse 轉回世界座標系
                        // 即撤銷以原本的轉換矩陣而得的世界座標，再以優化後的位姿估計來估計空間中的世界座標系
                        pMP->SetWorldPos(Rwc * Xc + twc);
                    }
                }

                // for (size_t i = 0; i < vpMPs.size(); i++)
                // {
                //     MapPoint *pMP = vpMPs[i];
                //     if (pMP->isBad()){
                //         continue;
                //     }
                //     if (pMP->mnBAGlobalForKF == nLoopKF)
                //     {
                //         // If optimized by Global BA, just update
                //         pMP->SetWorldPos(pMP->mPosGBA);
                //     }
                //     else
                //     {
                //         // Update according to the correction of its reference keyframe
                //         // 取得『地圖點 pMP』的參考關鍵幀
                //         KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                //         if (pRefKF->mnBAGlobalForKF != nLoopKF){
                //             continue;
                //         }
                //         // Map to non-corrected camera
                //         cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
                //         cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
                //         // 將『地圖點 pMP』從世界座標轉換到相機座標下
                //         cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;
                //         // Backproject using corrected camera
                //         cv::Mat Twc = pRefKF->GetPoseInverse();
                //         cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                //         cv::Mat twc = Twc.rowRange(0, 3).col(3);
                //         // 利用 mTcwBefGBA 轉換到相機座標系，再利用 GetPoseInverse 轉回世界座標系
                //         // 即撤銷以原本的轉換矩陣而得的世界座標，再以優化後的位姿估計來估計空間中的世界座標系
                //         pMP->SetWorldPos(Rwc * Xc + twc);
                //     }
                // }

                // 增加『重要變革的索引值』
                mpMap->InformNewBigChange();

                // 清空『新關鍵幀容器』
                mpLocalMapper->Release();

                cout << "Map updated!" << endl;
            }

            mbFinishedGBA = true;
            mbRunningGBA = false;
        }
    }

    // 請求結束 LoopClosing 執行續
    void LoopClosing::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    // 檢查結束 LoopClosing 執行續的請求
    bool LoopClosing::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    // 標注 LoopClosing 執行續已停止
    void LoopClosing::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    // LoopClosing 執行續是否已有效停止
    bool LoopClosing::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

} //namespace ORB_SLAM
