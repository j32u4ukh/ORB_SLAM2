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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include <mutex>

using namespace std;

namespace ORB_SLAM2
{

    KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary &voc) : mpVoc(&voc)
    {
        mvInvertedFile.resize(voc.size());
    }

    void KeyFrameDatabase::add(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutex);

        DBoW2::BowVector bow_vector = pKF->mBowVec;

        for(pair<DBoW2::WordId, DBoW2::WordValue> bow : bow_vector)
        {
            mvInvertedFile[bow.first].push_back(pKF);
        }

        // DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin();
        // DBoW2::BowVector::const_iterator vend = pKF->mBowVec.end();
        // for (; vit != vend; vit++){
        //     mvInvertedFile[vit->first].push_back(pKF);
        // }
    }

    void KeyFrameDatabase::erase(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutex);

        // Erase elements in the Inverse File for the entry
        DBoW2::BowVector bow_vector = pKF->mBowVec;

        for(pair<DBoW2::WordId, DBoW2::WordValue> bow : bow_vector)
        {
            // List of keyframes that share the word
            list<KeyFrame *> &lKFs = mvInvertedFile[bow.first];

            list<KeyFrame *>::iterator lit = lKFs.begin();
            list<KeyFrame *>::iterator lend = lKFs.end();

            for (; lit != lend; lit++)
            {
                if (pKF == *lit)
                {
                    lKFs.erase(lit);
                    break;
                }
            }
        }

        // DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin();
        // DBoW2::BowVector::const_iterator vend = pKF->mBowVec.end();
        // for (; vit != vend; vit++)
        // {
        //     // List of keyframes that share the word
        //     list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];
        //     list<KeyFrame *>::iterator lit = lKFs.begin();
        //     list<KeyFrame *>::iterator lend = lKFs.end();
        //     for (; lit != lend; lit++)
        //     {
        //         if (pKF == *lit)
        //         {
        //             lKFs.erase(lit);
        //             break;
        //         }
        //     }
        // }
    }

    void KeyFrameDatabase::clear()
    {
        mvInvertedFile.clear();
        mvInvertedFile.resize(mpVoc->size());
    }

    // 計算和『關鍵幀 pKF』有相同單字的『關鍵幀及其共視關鍵幀』和『關鍵幀 pKF』的相似程度，將相似程度高的關鍵幀返回
    vector<KeyFrame *> KeyFrameDatabase::DetectLoopCandidates(KeyFrame *pKF, float minScore)
    {
        // minScore：『關鍵幀 pKF』和其『共視關鍵幀』計算相似程度，當中的最小值（相似程度最低）

        // 『關鍵幀 pKF』的『已連結關鍵幀』
        set<KeyFrame *> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();

        // 和『關鍵幀 pKF』有一到多個相同單字的『關鍵幀』（pKFi->mnLoopWords 紀錄有多少相同單字）
        // 從『具有相同單字的關鍵幀』當中來篩選
        list<KeyFrame *> lKFsSharingWords;

        // Search all keyframes that share a word with current keyframes
        // Discard keyframes connected to the query keyframe
        {
            unique_lock<mutex> lock(mMutex);

            // BowVector == std::map<WordId, WordValue>
            // WordValue: tf * idf
            DBoW2::BowVector bow_vector = pKF->mBowVec;

            for(pair<DBoW2::WordId, DBoW2::WordValue> bow : bow_vector)
            {
                // mvInvertedFile[vit->first]：含有『單字 vit->first』的關鍵幀陣列
                list<KeyFrame *> &lKFs = mvInvertedFile[bow.first];

                for(KeyFrame *pKFi : lKFs)
                {
                    if (pKFi->mnLoopQuery != pKF->mnId)
                    {
                        pKFi->mnLoopWords = 0;

                        // 是否已在『關鍵幀 pKF』的『已連結關鍵幀』當中
                        if (!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->mnLoopQuery = pKF->mnId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }

                    // 紀錄『關鍵幀 pKFi』和『關鍵幀 pKF』有多少相同單字
                    pKFi->mnLoopWords++;
                }
            }

            // DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin();
            // DBoW2::BowVector::const_iterator vend = pKF->mBowVec.end();
            // for (; vit != vend; vit++)
            // {
            //     // mvInvertedFile[vit->first]：含有『單字 vit->first』的關鍵幀陣列
            //     list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];
            //     list<KeyFrame *>::iterator lit = lKFs.begin();
            //     list<KeyFrame *>::iterator lend = lKFs.end();
            //     for (; lit != lend; lit++)
            //     {
            //         KeyFrame *pKFi = *lit;
            //         if (pKFi->mnLoopQuery != pKF->mnId)
            //         {
            //             pKFi->mnLoopWords = 0;
            //             // 是否已在『關鍵幀 pKF』的『已連結關鍵幀』當中
            //             if (!spConnectedKeyFrames.count(pKFi))
            //             {
            //                 pKFi->mnLoopQuery = pKF->mnId;
            //                 lKFsSharingWords.push_back(pKFi);
            //             }
            //         }
            //         // 紀錄『關鍵幀 pKFi』和『關鍵幀 pKF』有多少相同單字
            //         pKFi->mnLoopWords++;
            //     }
            // }
        }

        // 若 lKFsSharingWords 為空，則返回空陣列
        if (lKFsSharingWords.empty()){
            return vector<KeyFrame *>();
        }

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;

        for(KeyFrame * kf : lKFsSharingWords)
        {
            // kf->mnLoopWords：紀錄『關鍵幀 kf』和『關鍵幀 pKF』有多少相同單字
            if (kf->mnLoopWords > maxCommonWords){

                // 取得最多相同單字個數
                maxCommonWords = kf->mnLoopWords;
            }
        }

        // list<KeyFrame *>::iterator lit = lKFsSharingWords.begin();
        // list<KeyFrame *>::iterator lend = lKFsSharingWords.end();
        // for (; lit != lend; lit++)
        // {
        //     // (*lit)->mnLoopWords：紀錄『關鍵幀 (*lit)』和『關鍵幀 pKF』有多少相同單字
        //     if ((*lit)->mnLoopWords > maxCommonWords){
        //         // 取得最多相同單字個數
        //         maxCommonWords = (*lit)->mnLoopWords;
        //     }
        // }

        // key: 和『關鍵幀 pKF』的相似程度分數, value: 關鍵幀
        list<pair<float, KeyFrame *>> lScoreAndMatch;

        // 相同單字個數下限
        int minCommonWords = maxCommonWords * 0.8f;

        // 『關鍵幀 pKFi』和『關鍵幀 pKF』相同單字的個數大於下限的『關鍵幀 pKFi』個數
        int nscores = 0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for(KeyFrame *pKFi : lKFsSharingWords)
        {
            // 若『關鍵幀 pKFi』和『關鍵幀 pKF』相同單字的個數大於下限
            if (pKFi->mnLoopWords > minCommonWords)
            {
                nscores++;

                // 計算『關鍵幀 pKFi』和『關鍵幀 pKF』的相似程度
                float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

                // 紀錄『關鍵幀 pKFi』和『關鍵幀 pKF』的相似程度
                pKFi->mLoopScore = si;

                // 相似程度 大於 『關鍵幀 pKF』和其『共視關鍵幀』的相似程度最小值
                if (si >= minScore){
                    // 將『相似程度』與『關鍵幀 pKFi』作為一組，放入 lScoreAndMatch
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
                }
            }
        }

        // lit = lKFsSharingWords.begin();
        // lend = lKFsSharingWords.end();
        // for (; lit != lend; lit++)
        // {
        //     KeyFrame *pKFi = *lit;
        //     // 若『關鍵幀 pKFi』和『關鍵幀 pKF』相同單字的個數大於下限
        //     if (pKFi->mnLoopWords > minCommonWords)
        //     {
        //         nscores++;
        //         // 計算『關鍵幀 pKFi』和『關鍵幀 pKF』的相似程度
        //         float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
        //         // 紀錄『關鍵幀 pKFi』和『關鍵幀 pKF』的相似程度
        //         pKFi->mLoopScore = si;
        //         // 相似程度 大於 『關鍵幀 pKF』和其『共視關鍵幀』的相似程度最小值
        //         if (si >= minScore){
        //             // 將『相似程度』與『關鍵幀 pKFi』作為一組，放入 lScoreAndMatch
        //             lScoreAndMatch.push_back(make_pair(si, pKFi));
        //         }
        //     }
        // }

        if (lScoreAndMatch.empty()){
            return vector<KeyFrame *>();
        }

        list<pair<float, KeyFrame *>> lAccScoreAndMatch;
        float bestAccScore = minScore;

        // Lets now accumulate score by covisibility
        // lScoreAndMatch key: 和『關鍵幀 pKF』的相似程度分數, value: 關鍵幀
        for(pair<float, KeyFrame *> score_match : lScoreAndMatch)
        {
            // it->first：『關鍵幀 pKFi』和『關鍵幀 pKF』的相似程度分數
            float bestScore = score_match.first;
            float accScore = score_match.first;

            KeyFrame *pKFi = score_match.second;
            KeyFrame *pBestKF = pKFi;

            // 自『根據觀察到的地圖點數量排序的共視關鍵幀』當中返回至多 10 個共視關鍵幀
            vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            // 『關鍵幀 pKFi』的『共視關鍵幀』
            for(KeyFrame *pKF2 : vpNeighs)
            {
                // 若『關鍵幀 pKF2』曾協助迴路檢測 且 和『關鍵幀 pKF』的相同單字的個數大於下限
                if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords)
                {
                    // accumulation 累積
                    // 『關鍵幀 pKF2』和『關鍵幀 pKF』的相似分數加入『累積相似分數 accScore』
                    accScore += pKF2->mLoopScore;

                    // 篩選和『關鍵幀 pKF』最相似的關鍵幀，及其相似分數
                    if (pKF2->mLoopScore > bestScore)
                    {
                        pBestKF = pKF2;
                        bestScore = pKF2->mLoopScore;
                    }
                }
            }

            // 『關鍵幀 pKFi』的『共視關鍵幀』當中，和『關鍵幀 pKF』最相似的關鍵幀，及其相似分數
            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));

            if (accScore > bestAccScore){
                bestAccScore = accScore;
            }
        }

        // list<pair<float, KeyFrame *>>::iterator it = lScoreAndMatch.begin();
        // list<pair<float, KeyFrame *>>::iterator itend = lScoreAndMatch.end();
        // for (; it != itend; it++)
        // {
        //     KeyFrame *pKFi = it->second;
        //     // 自『根據觀察到的地圖點數量排序的共視關鍵幀』當中返回至多 10 個共視關鍵幀
        //     vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
        //     // it->first：『關鍵幀 pKFi』和『關鍵幀 pKF』的相似程度分數
        //     float bestScore = it->first;
        //     float accScore = it->first;
        //     KeyFrame *pBestKF = pKFi;
        //     vector<KeyFrame *>::iterator vit = vpNeighs.begin();
        //     vector<KeyFrame *>::iterator vend = vpNeighs.end();
        //     for (; vit != vend; vit++)
        //     {
        //         // 『關鍵幀 pKFi』的『共視關鍵幀』
        //         KeyFrame *pKF2 = *vit;
        //         // 若『關鍵幀 pKF2』曾協助迴路檢測 且 和『關鍵幀 pKF』的相同單字的個數大於下限
        //         if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords)
        //         {
        //             // accumulation 累積
        //             // 『關鍵幀 pKF2』和『關鍵幀 pKF』的相似分數加入『累積相似分數 accScore』
        //             accScore += pKF2->mLoopScore;
        //             // 篩選和『關鍵幀 pKF』最相似的關鍵幀，及其相似分數
        //             if (pKF2->mLoopScore > bestScore)
        //             {
        //                 pBestKF = pKF2;
        //                 bestScore = pKF2->mLoopScore;
        //             }
        //         }
        //     }
        //     // 『關鍵幀 pKFi』的『共視關鍵幀』當中，和『關鍵幀 pKF』最相似的關鍵幀，及其相似分數
        //     lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        //     if (accScore > bestAccScore){
        //         bestAccScore = accScore;
        //     }
        // }

        // Return all those keyframes with a score higher than 0.75*bestScore
        // 相似分數下限
        float minScoreToRetain = 0.75f * bestAccScore;

        set<KeyFrame *> spAlreadyAddedKF;
        vector<KeyFrame *> vpLoopCandidates;
        vpLoopCandidates.reserve(lAccScoreAndMatch.size());

        for(pair<float, KeyFrame *> score_match : lAccScoreAndMatch)
        {
            // 『關鍵幀』的『共視關鍵幀 it->second』的相似分數 大於 相似分數下限
            if (score_match.first > minScoreToRetain)
            {
                KeyFrame *pKFi = score_match.second;

                // 避免重複添加『關鍵幀 pKFi』
                if (!spAlreadyAddedKF.count(pKFi))
                {
                    vpLoopCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        // it = lAccScoreAndMatch.begin();
        // itend = lAccScoreAndMatch.end();
        // for (; it != itend; it++)
        // {
        //     // 『關鍵幀』的『共視關鍵幀 it->second』的相似分數 大於 相似分數下限
        //     if (it->first > minScoreToRetain)
        //     {
        //         KeyFrame *pKFi = it->second;
        //         // 避免重複添加『關鍵幀 pKFi』
        //         if (!spAlreadyAddedKF.count(pKFi))
        //         {
        //             vpLoopCandidates.push_back(pKFi);
        //             spAlreadyAddedKF.insert(pKFi);
        //         }
        //     }
        // }

        return vpLoopCandidates;
    }

    // 當前幀的詞袋模型包含的所有關鍵幀，篩選出『重定位詞較多』、『BoW 相似性得分較高』的關鍵幀
    // 這裡的分數同時考慮了其他觀察到相同地圖點的關鍵幀的 BoW 相似性得分
    vector<KeyFrame *> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
    {
        // 當前幀的詞袋模型包含的所有關鍵幀
        list<KeyFrame *> lKFsSharingWords;

        // Search all keyframes that share a word with current frame
        // 將當前幀的詞袋模型包含的關鍵幀取出，並紀錄各幀『包含多少用於重定位的 Word』
        {
            unique_lock<mutex> lock(mMutex);

            // 分類樹中 leaf 的數值與權重(葉) Bag of Words Vector structures.
            // BowVector == std::map<WordId: int, WordValue: double>
            for(pair<DBoW2::WordId, DBoW2::WordValue> id_value : F->mBowVec){

                list<KeyFrame *> &lKFs = mvInvertedFile[id_value.first];

                for(KeyFrame *pKFi : lKFs){

                    // mnRelocQuery：推測為『紀錄提出重定位請求的 Frame 的 Id』
                    if (pKFi->mnRelocQuery != F->mnId)
                    {
                        // mnRelocWords
                        pKFi->mnRelocWords = 0;

                        pKFi->mnRelocQuery = F->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }

                    // mnRelocWords：推測為『包含多少用於重定位的 Word』
                    pKFi->mnRelocWords++;
                }
            }

            // DBoW2::BowVector::const_iterator vit = F->mBowVec.begin();
            // DBoW2::BowVector::const_iterator vend = F->mBowVec.end();
            // for (; vit != vend; vit++)
            // {
            //     list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];
            //     list<KeyFrame *>::iterator lit = lKFs.begin();
            //     list<KeyFrame *>::iterator lend = lKFs.end();
            //     for (; lit != lend; lit++)
            //     {
            //         KeyFrame *pKFi = *lit;
            //         // mnRelocQuery：推測為『紀錄提出重定位請求的 Frame 的 Id』
            //         if (pKFi->mnRelocQuery != F->mnId)
            //         {
            //             // mnRelocWords
            //             pKFi->mnRelocWords = 0;
            //             pKFi->mnRelocQuery = F->mnId;
            //             lKFsSharingWords.push_back(pKFi);
            //         }
            //         // mnRelocWords：推測為『包含多少用於重定位的 Word』
            //         pKFi->mnRelocWords++;
            //     }
            // }
        }

        if (lKFsSharingWords.empty())
        {
            return vector<KeyFrame *>();
        }

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;

        // 尋找包含最多『重定位詞』的 KeyFrame
        for(KeyFrame * kf : lKFsSharingWords){

            if (kf->mnRelocWords > maxCommonWords)
            {
                maxCommonWords = kf->mnRelocWords;
            }
        }

        // list<KeyFrame *>::iterator lit = lKFsSharingWords.begin();
        // list<KeyFrame *>::iterator lend = lKFsSharingWords.end();
        // for (; lit != lend; lit++)
        // {
        //     if ((*lit)->mnRelocWords > maxCommonWords)
        //     {
        //         maxCommonWords = (*lit)->mnRelocWords;
        //     }
        // }

        // 『重定位詞』個數下限，定為最大值的 8 成
        int minCommonWords = maxCommonWords * 0.8f;

        // 紀錄各關鍵幀和『當前幀 F』的 BoW 相似性得分
        list<pair<float, KeyFrame *>> lScoreAndMatch;

        // nscores 沒用處
        int nscores = 0;

        // Compute similarity score.
        for(KeyFrame *pKFi : lKFsSharingWords){

            // 若關鍵幀 pKFi 的『重定位詞』個數大於下限
            if (pKFi->mnRelocWords > minCommonWords)
            {
                // nscores 沒用處
                nscores++;

                // 重定位分數
                float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
                pKFi->mRelocScore = si;

                lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }

        // lit = lKFsSharingWords.begin();
        // lend = lKFsSharingWords.end();        
        // for (; lit != lend; lit++)
        // {
        //     KeyFrame *pKFi = *lit;
        //     // 若關鍵幀 pKFi 的『重定位詞』個數大於下限
        //     if (pKFi->mnRelocWords > minCommonWords)
        //     {
        //         // nscores 沒用處
        //         nscores++;
        //         // 重定位分數
        //         float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
        //         pKFi->mRelocScore = si;
        //         lScoreAndMatch.push_back(make_pair(si, pKFi));
        //     }
        // }

        if (lScoreAndMatch.empty()){
            return vector<KeyFrame *>();
        }

        // lScoreAndMatch 的進階版，紀錄各關鍵幀和『當前幀 F』的 BoW 相似性得分
        // 此處分數同時考慮其他觀察到相同地圖點的關鍵幀的 BoW 相似性得分，關鍵幀則更新為 BoW 相似性得分最高者
        list<pair<float, KeyFrame *>> lAccScoreAndMatch;
        float bestAccScore = 0;

        // Lets now accumulate score by covisibility
        for(pair<float, KeyFrame *> score_match : lScoreAndMatch){

            // 關鍵幀和『當前幀 F』的 BoW 相似性得分
            float bestScore = score_match.first;

            KeyFrame *pKFi = score_match.second;

            // 返回至多 10 個已連結的有序關鍵幀（已連結：彼此觀察到相同地圖點的關鍵幀）
            vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float accScore = bestScore;
            KeyFrame *pBestKF = pKFi;

            for(KeyFrame *pKF2 : vpNeighs){
                
                // 若當前幀沒有對 pKF2 提出過重定位請求，則跳過
                if (pKF2->mnRelocQuery != F->mnId){
                    continue;
                }

                // 將其他觀察到相同地圖點的關鍵幀的 BoW 相似性得分都考慮進來
                accScore += pKF2->mRelocScore;

                // 更新 BoW 相似性得分最高值，以及其對應的關鍵幀
                if (pKF2->mRelocScore > bestScore)
                {
                    pBestKF = pKF2;
                    bestScore = pKF2->mRelocScore;
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));

            // 更新最高的 BoW 相似性得分
            if (accScore > bestAccScore){
                bestAccScore = accScore;
            }
        }

        // list<pair<float, KeyFrame *>>::iterator it = lScoreAndMatch.begin();
        // list<pair<float, KeyFrame *>>::iterator itend = lScoreAndMatch.end();
        // for (; it != itend; it++)
        // {
        //     KeyFrame *pKFi = it->second;
        //     // 返回至多 10 個已連結的有序關鍵幀（已連結：彼此觀察到相同地圖點的關鍵幀）
        //     vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
        //     // 關鍵幀和『當前幀 F』的 BoW 相似性得分
        //     float bestScore = it->first;
        //     float accScore = bestScore;
        //     KeyFrame *pBestKF = pKFi;
        //     vector<KeyFrame *>::iterator vit = vpNeighs.begin();
        //     vector<KeyFrame *>::iterator vend = vpNeighs.end();
        //     for (; vit != vend; vit++)
        //     {
        //         KeyFrame *pKF2 = *vit;
        //         // 若當前幀沒有對 pKF2 提出過重定位請求，則跳過
        //         if (pKF2->mnRelocQuery != F->mnId){
        //             continue;
        //         }
        //         // 將其他觀察到相同地圖點的關鍵幀的 BoW 相似性得分都考慮進來
        //         accScore += pKF2->mRelocScore;
        //         // 更新 BoW 相似性得分最高值，以及其對應的關鍵幀
        //         if (pKF2->mRelocScore > bestScore)
        //         {
        //             pBestKF = pKF2;
        //             bestScore = pKF2->mRelocScore;
        //         }
        //     }
        //     lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        //     // 更新最高的 BoW 相似性得分
        //     if (accScore > bestAccScore){
        //         bestAccScore = accScore;
        //     }
        // }

        // Return all those keyframes with a score higher than 0.75 * bestScore
        // BoW 相似性得分至少為最高分的 0.75 倍
        float minScoreToRetain = 0.75f * bestAccScore;

        // 篩選重定位候補的關鍵幀
        vector<KeyFrame *> vpRelocCandidates;
        vpRelocCandidates.reserve(lAccScoreAndMatch.size());

        // 協助 vpRelocCandidates 不要重複添加
        set<KeyFrame *> spAlreadyAddedKF;

        for(pair<float, KeyFrame *> score_match : lAccScoreAndMatch){

            const float &si = score_match.first;

            // 若 BoW 相似性得分大於最低要求
            if (si > minScoreToRetain)
            {
                KeyFrame *pKFi = score_match.second;

                if (!spAlreadyAddedKF.count(pKFi))
                {
                    // 將關鍵幀加入 vpRelocCandidates，將用於協助重定位
                    vpRelocCandidates.push_back(pKFi);

                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        // it = lAccScoreAndMatch.begin();
        // itend = lAccScoreAndMatch.end();
        // for (; it != itend; it++)
        // {
        //     const float &si = it->first;
        //     // 若 BoW 相似性得分大於最低要求
        //     if (si > minScoreToRetain)
        //     {
        //         KeyFrame *pKFi = it->second;
        //         if (!spAlreadyAddedKF.count(pKFi))
        //         {
        //             // 將關鍵幀加入 vpRelocCandidates，將用於協助重定位
        //             vpRelocCandidates.push_back(pKFi);
        //             spAlreadyAddedKF.insert(pKFi);
        //         }
        //     }
        // }

        return vpRelocCandidates;
    }

} //namespace ORB_SLAM
