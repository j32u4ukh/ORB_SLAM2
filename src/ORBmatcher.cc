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

#include "ORBmatcher.h"

#include <limits.h>
#include <tuple>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include <stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    // 第一個參數是一個接受最佳匹配的系數，只有當最佳匹配點的漢明距離小於次加匹配點距離的 nnratio 倍時才接收匹配點，
    // 第二個參數表示匹配特征點時是否考慮方向。
    ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : 
                           mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    // 計算描述子之間的距離（相似程度）
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;
        unsigned int v;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            v = *pa ^ *pb;
            
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    // 兩幀之間形成配對的各自地圖點索引值，（『關鍵幀 pKF1』的地圖點索引值，『關鍵幀 pKF2』的地圖點索引值）
    int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                           vector<pair<size_t, size_t>> &vMatchedPairs, 
                                           const bool bOnlyStereo)
    {
        //Compute epipole in second image
        cv::Mat Cw = pKF1->GetCameraCenter();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        // 從相機 1 轉換到相機 2
        cv::Mat C2 = R2w * Cw + t2w;

        // 逆深度
        const float invz = 1.0f / C2.at<float>(2);

        // 投影在『關鍵幀 pKF2』上的像素座標（利用逆深度控制規模）
        const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
        const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches = 0;
        vector<bool> vbMatched2(pKF2->N, false);
        vector<int> vMatches12(pKF1->N, -1);

        // rotHist 為 HISTO_LENGTH 個 vector<int> 的組合
        vector<int> rotHist[HISTO_LENGTH];

        for (int i = 0; i < HISTO_LENGTH; i++){
            // rotHist[i] 為 vector<int>，預先劃分 500 個空間
            rotHist[i].reserve(500);
        }

        const float factor = 1.0f / HISTO_LENGTH;

        // FeatureVector == std::map<NodeId, std::vector<unsigned int> >
        // 以一張圖片的每個特徵點在詞典某一層節點下爲條件進行分組，用來加速圖形特徵匹配——
        // 兩兩圖像特徵匹配只需要對相同 NodeId 下的特徵點進行匹配就好。
        // std::vector<unsigned int>：觀察到該特徵的 地圖點/關鍵點 的索引值

        // 『關鍵幀 pKF1』的特徵向量
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();

        // 『關鍵幀 pKF2』的特徵向量
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while (f1it != f1end && f2it != f2end)
        {
            // 兩個特徵向量的 NodeId 相同
            if (f1it->first == f2it->first)
            {
                // 『關鍵幀 pKF1』觀察到『特徵 f1it->first』的地圖點的索引值 idx1
                for(const size_t idx1 : f1it->second){

                    // 『關鍵幀 pKF1』的第 idx1 個地圖點
                    MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

                    // If there is already a MapPoint skip
                    // 若地圖點已存在，則跳過
                    if (pMP1){
                        continue;
                    }

                    // 是否為雙目（單目的 mvuRight 應為負值）
                    const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;

                    if (bOnlyStereo)
                    {
                        if (!bStereo1)
                        {
                            continue;
                        }
                    }

                    // 『關鍵幀 pKF1』的第 idx1 個已校正關鍵點
                    const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

                    // 『關鍵幀 pKF1』的第 idx1 個關鍵點的描述子
                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    // 『關鍵幀 pKF2』觀察到『特徵 f2it->first』的地圖點的索引值 idx2
                    for(const size_t idx2 : f2it->second){

                        // 『關鍵幀 pKF2』的第 idx2 個地圖點
                        MapPoint *pMP2 = pKF2->GetMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        if (vbMatched2[idx2] || pMP2){
                            continue;
                        }

                        // 是否為雙目（單目的 mvuRight 應為負值）
                        const bool bStereo2 = pKF2->mvuRight[idx2] >= 0;

                        if (bOnlyStereo){
                            if (!bStereo2){
                                continue;
                            }
                        }

                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                        const int dist = DescriptorDistance(d1, d2);

                        if (dist > TH_LOW || dist > bestDist){
                            continue;
                        }

                        // 『關鍵幀 pKF2』的第 idx2 個關鍵點的描述子
                        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                        // 是否為單目
                        if (!bStereo1 && !bStereo2)
                        {
                            // 計算重投影誤差
                            const float distex = ex - kp2.pt.x;
                            const float distey = ey - kp2.pt.y;
                            const float diste = distex * distex + distey * distey;

                            // 若誤差足夠小
                            if (diste < 100 * pKF2->mvScaleFactors[kp2.octave]){
                                continue;
                            }
                        }

                        // 檢查極線長度是否足夠小（越長越難找到正確的投影點）
                        /// NOTE: 這裡只要符合條件就會替換數值，是否『關鍵幀 pKF2』的特徵向量有作一定的排序？
                        /// 使得後面的極線長度會越來越小？還是忘記過濾最佳數值？
                        if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2))
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    // 若『有找到』極線長度是否足夠小的組合，bestIdx2 便會替換成該組合的索引值，進而大於等於 0
                    if (bestIdx2 >= 0)
                    {
                        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];

                        // 『關鍵幀 pKF1』第 idx1 個地圖點和『關鍵幀 pKF2』第 bestIdx2 個地圖點形成配對
                        vMatches12[idx1] = bestIdx2;

                        nmatches++;

                        // 檢查方向
                        if (mbCheckOrientation)
                        {
                            updateRotHist(kp1, kp2, factor, idx1, rotHist);

                            // // 計算角度差
                            // float rot = kp1.angle - kp2.angle;

                            // if (rot < 0.0){
                            //     rot += 360.0f;
                            // }

                            // // 角度差換算成直方圖的格子索引值
                            // int bin = round(rot * factor);

                            // if (bin == HISTO_LENGTH){
                            //     bin = 0;
                            // }

                            // assert(bin >= 0 && bin < HISTO_LENGTH);
                            // rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }

            // 若『關鍵幀 pKF1』的 NodeId 較『關鍵幀 pKF2』小，f1it 指向相同的 NodeId
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }

            // 若『關鍵幀 pKF2』的 NodeId 較『關鍵幀 pKF1』小，f2it 指向相同的 NodeId
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            nmatches = convergenceMatched(nmatches, rotHist, vMatches12, -1);
        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        size_t i, iend = vMatches12.size();

        for (i = 0; i < iend; i++)
        {
            if (vMatches12[i] < 0){
                continue;
            }

            // vMatches12：『關鍵幀 pKF1』第 idx1 個地圖點和『關鍵幀 pKF2』第 bestIdx2 個地圖點形成配對
            // 兩幀之間形成配對的各自地圖點索引值，（『關鍵幀 pKF1』的地圖點索引值，『關鍵幀 pKF2』的地圖點索引值）
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }

        return nmatches;
    }

    // 檢查極線長度是否足夠小（越長越難找到正確的投影點）
    bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, 
                                           const cv::Mat &F12, const KeyFrame *pKF2)
    {
        // Epipolar line in second image l = x1'F12 = [a b c]
        const float a = kp1.pt.x * F12.at<float>(0, 0) + 
                        kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);

        const float b = kp1.pt.x * F12.at<float>(0, 1) + 
                        kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);

        const float c = kp1.pt.x * F12.at<float>(0, 2) + 
                        kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

        const float num = a * kp2.pt.x + b * kp2.pt.y + c;

        const float den = a * a + b * b;

        if (den == 0){
            return false;
        }

        // 計算對極的長度
        const float dsqr = num * num / den;

        return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
    }

    // 篩選前三多直方格的索引值
    void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, 
                                        int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++)
        {
            const int s = histo[i].size();

            if (s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3 < 0.1f * (float)max1)
        {
            ind3 = -1;
        }
    }

    // 『關鍵幀 pKF』觀察到的地圖點和『現有地圖點』兩者的描述子距離很近，保留被更多關鍵幀觀察到的一點取代另一點
    int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
    {
        // vpMapPoints：不同於『關鍵幀 pKF』的其他關鍵幀所觀察到的地圖點

        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        // 相機中心點位置
        cv::Mat Ow = pKF->GetCameraCenter();

        int nFused = 0, bestDist, bestIdx;

        const int nMPs = vpMapPoints.size();

        MapPoint *pMP;
        MapPoint *pMPinKF;
        std::tuple<bool, int, int> continue_dist_idx;
        // const cv::Mat sim3 = cv::Mat::eye(3, 3, CV_32F);
        // const cv::Mat t21 = cv::Mat::zeros(3, 1, CV_32F);

        std::tuple<float, float, float> u_v_ur;
        std::tuple<bool, float> valid_dist;
        float dist3D, u, v, ur;

        for (int i = 0; i < nMPs; i++)
        {
            pMP = vpMapPoints[i];             

            if (!pMP){
                continue;
            }

            if (pMP->isBad() || pMP->IsInKeyFrame(pKF)){
                continue;
            }

            /* std::tuple<bool, int, int> 
            ORBmatcher::selectFuseTarget(KeyFrame *pKF, MapPoint *pMP, float th, 
                                         cv::Mat sim3, cv::Mat Rcw, cv::Mat t1w, cv::Mat t21, cv::Mat Ow, 
                                         const float fx, const float fy, const float cx, const float cy, 
                                         const float bf, bool consider_error, bool consider_included_angle)
            */
            // continue_dist_idx = selectFuseTarget(pKF, pMP, th, 
            //                                      sim3, Rcw, tcw, t21, Ow, 
            //                                      fx, fy, cx, cy, 
            //                                      bf, true, true);
            // if(std::get<0>(continue_dist_idx))
            // {
            //     continue;
            // }
            // bestDist = std::get<1>(continue_dist_idx);
            // bestIdx = std::get<2>(continue_dist_idx);

            // 『地圖點 pMP』的世界座標
            cv::Mat p3Dw = pMP->GetWorldPos();

            // 『地圖點 pMP』由世紀座標系 轉換到 相機座標系
            cv::Mat p3Dc = Rcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc.at<float>(2) < 0.0f)
            {
                continue;
            }
            
            u_v_ur = getPixelCoordinatesStereo(p3Dc, bf, fx, fy, cx, cy);

            u = std::get<0>(u_v_ur);
            v = std::get<1>(u_v_ur);
            
            // Point must be inside the image
            // 傳入座標點是否超出關鍵幀的成像範圍
            if (!pKF->IsInImage(u, v))
            {
                continue;
            }

            ur = std::get<2>(u_v_ur);

            // // Depth must be positive
            // if (p3Dc.at<float>(2) < 0.0f){
            //     continue;
            // }
            // const float invz = 1 / p3Dc.at<float>(2);
            // // 重投影之歸一化平面座標
            // const float x = p3Dc.at<float>(0) * invz;
            // const float y = p3Dc.at<float>(1) * invz;
            // // 重投影之像素座標
            // const float u = fx * x + cx;
            // const float v = fy * y + cy;
            // // Point must be inside the image
            // // 傳入座標點是否超出關鍵幀的成像範圍
            // if (!pKF->IsInImage(u, v)){
            //     continue;
            // }

            valid_dist = isValidDistance(pMP, p3Dw, Ow);

            // 若非有效深度估計
            if(!std::get<0>(valid_dist)){
                continue;
            }

            dist3D = std::get<1>(valid_dist);

            // // 考慮金字塔層級的『地圖點 pMP』最大可能深度
            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // // 考慮金字塔層級的『地圖點 pMP』最小可能深度
            // const float minDistance = pMP->GetMinDistanceInvariance();
            // // 『相機中心 Ow』指向『地圖點 pMP』之向量
            // cv::Mat PO = p3Dw - Ow;
            // // 『地圖點 pMP』深度（『相機中心 Ow』到『地圖點 pMP』之距離）
            // const float dist3D = cv::norm(PO);
            // // Depth must be inside the scale pyramid of the image
            // // 『相機中心 Ow』到『地圖點 pMP』之距離應在可能深度的區間內
            // if (dist3D < minDistance || dist3D > maxDistance){
            //     continue;
            // }
            // // Viewing angle must be less than 60 deg
            // // 地圖點之法向量
            // cv::Mat Pn = pMP->GetNormal();
            // // 計算 PO 和 Pn 的夾角是否超過 60 度（餘弦值超過 0.5）
            // if (PO.dot(Pn) < 0.5 * dist3D){
            //     continue;
            // }

            // 『關鍵幀 pKF』根據當前『地圖點 pMP』的深度，估計場景規模
            int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

            // 指定區域內的候選關鍵點的索引值
            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty()){
                continue;
            }

            // Match to the most similar keypoint in the radius
            // 取得『地圖點 pMP』描述子
            const cv::Mat dMP = pMP->GetDescriptor();

            bestDist = INT_MAX;
            bestIdx = -1;

            // std::tuple<bool, cv::KeyPoint, int>{continue, kp, kpLevel};
            std::tuple<bool, cv::KeyPoint, int> continue_kp_level;
            cv::KeyPoint kp;
            int kpLevel;
            
            for(const size_t idx : vIndices)
            {
                continue_kp_level = checkFuseTarget(pKF, idx, nPredictedLevel);

                if(std::get<0>(continue_kp_level))
                {
                    continue;
                }

                kp = std::get<1>(continue_kp_level);
                kpLevel = std::get<2>(continue_kp_level);

                // // 指定區域內的候選關鍵點
                // const cv::KeyPoint &kp = pKF->mvKeysUn[idx];
                // // 『關鍵點 kp』的金字塔層級
                // const int &kpLevel = kp.octave;
                // // kpLevel 可以是：(nPredictedLevel - 1) 或 nPredictedLevel
                // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel){
                //     continue;
                // }

                // 非單目，暫時跳過
                if (pKF->mvuRight[idx] >= 0)
                {
                    // Check reprojection error in stereo
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float &kpr = pKF->mvuRight[idx];
                    const float ex = u - kpx;
                    const float ey = v - kpy;
                    const float er = ur - kpr;
                    const float e2 = ex * ex + ey * ey + er * er;

                    if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8){
                        continue;
                    }
                }

                // 單目
                else
                {
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;

                    // 計算重投影誤差
                    const float ex = u - kpx;
                    const float ey = v - kpy;
                    const float e2 = ex * ex + ey * ey;

                    // 若重投影誤差過大則跳過
                    if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99){
                        continue;
                    }
                }

                updateFuseTarget(pKF, idx, dMP, bestDist, bestIdx);

                // // 取得『關鍵幀 pKF』的第 idx 個特徵點的描述子
                // const cv::Mat &dKF = pKF->mDescriptors.row(idx);
                // // 計算『地圖點 pMP』描述子和『關鍵幀 pKF』的第 idx 個特徵點的描述子之間的距離
                // const int dist = DescriptorDistance(dMP, dKF);
                // // 篩選距離最近的『距離 bestDist』和『關鍵幀索引值 bestIdx』
                // if (dist < bestDist)
                // {
                //     bestDist = dist;
                //     bestIdx = idx;
                // }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            // 若『地圖點 pMP』描述子和『關鍵幀 pKF』的第 bestIdx 個特徵點的描述子之間的距離足夠小
            // 可視為同一地圖點
            if (bestDist <= TH_LOW)
            {
                // 取出『關鍵幀 pKF』的第 bestIdx 個特徵點所對應的地圖點
                pMPinKF = pKF->GetMapPoint(bestIdx);

                // 若『地圖點 pMPinKF』已存在
                if (pMPinKF)
                {
                    if (!pMPinKF->isBad())
                    {
                        // 被較多關鍵幀觀察到的地圖點取代被較少關鍵幀觀察到的地圖點
                        if (pMPinKF->getObservationNumber() > pMP->getObservationNumber())
                        {
                            pMP->Replace(pMPinKF);
                        }
                        else{
                            pMPinKF->Replace(pMP);
                        }
                    }
                }

                // 若『地圖點 pMPinKF』不存在，則紀錄『關鍵幀 pKF』觀察到『地圖點 pMP』這一訊息
                else
                {
                    pMP->AddObservation(pKF, bestIdx);
                    pKF->AddMapPoint(pMP, bestIdx);
                }

                nFused++;
            }
        }
        
        return nFused;
    }

    // 『關鍵幀 pKF』觀察到的地圖點和『現有地圖點』兩者的描述子距離很近，保留『關鍵幀 pKF』觀察到的地圖點
    int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, 
                         vector<MapPoint *> &vpReplacePoint)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));

        cv::Mat Rcw = sRcw / scw;
        cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
        cv::Mat Ow = -Rcw.t() * tcw;

        // Set of MapPoints already found in the KeyFrame
        // 取得『關鍵幀 pKF』觀察到的地圖點
        const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

        int nFused = 0, bestDist, bestIdx;

        // vpPoints：和『關鍵幀 mpCurrentKF』已配對的『關鍵幀及其共視關鍵幀』觀察到的地圖點
        const int nPoints = vpPoints.size();
        MapPoint *pMP, *pMPinKF;
        // std::tuple<bool, int, int> continue_dist_idx;
        // const cv::Mat sim3 = cv::Mat::eye(3, 3, CV_32F);
        // const cv::Mat t21 = cv::Mat::zeros(3, 1, CV_32F);

        std::tuple<float, float, float> u_v_invz;
        std::tuple<bool, cv::KeyPoint, int> continue_kp_level;
        std::tuple<bool, float> valid_dist;
        float dist3D, u, v;

        // Get 3D Coords.
        cv::Mat p3Dw, p3Dc;
    
        // For each candidate MapPoint project and match
        for (int iMP = 0; iMP < nPoints; iMP++)
        {
            pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if (pMP->isBad() || spAlreadyFound.count(pMP)){
                continue;
            }

            /* std::tuple<bool, int, int> 
            ORBmatcher::selectFuseTarget(KeyFrame *pKF, MapPoint *pMP, float th, 
                                         cv::Mat sim3, cv::Mat Rcw, cv::Mat t1w, cv::Mat t21, cv::Mat Ow, 
                                         const float fx, const float fy, const float cx, const float cy, 
                                         const float bf, bool consider_error, bool consider_included_angle)

            bf 在需要考慮誤差時才會用到，因此這裡無須 bf
            */
            // continue_dist_idx = selectFuseTarget(pKF, pMP, th, 
            //                                      sim3, Rcw, tcw, t21, Ow, 
            //                                      fx, fy, cx, cy, 
            //                                      0, false, true);

            // if(std::get<0>(continue_dist_idx))
            // {
            //     continue;
            // }

            // bestDist = std::get<1>(continue_dist_idx);
            // bestIdx = std::get<2>(continue_dist_idx);

            // Get 3D Coords.
            p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            // 『地圖點 pMP』由世界座標轉換到『關鍵幀 pKF』座標系
            p3Dc = Rcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc.at<float>(2) < 0.0f)
            {
                continue;
            }
            
            u_v_invz = getPixelCoordinates(p3Dc, fx, fy, cx, cy);
            u = std::get<0>(u_v_invz);
            v = std::get<1>(u_v_invz);

            // Point must be inside the image
            // 傳入座標點是否超出關鍵幀的成像範圍
            if (!pKF->IsInImage(u, v))
            {
                continue;
            }

            // // Depth must be positive
            // if (p3Dc.at<float>(2) < 0.0f){
            //     continue;
            // }

            // // Project into Image
            // const float invz = 1.0 / p3Dc.at<float>(2);

            // // 重投影之歸一化平面座標
            // const float x = p3Dc.at<float>(0) * invz;
            // const float y = p3Dc.at<float>(1) * invz;

            // // 重投影之像素座標
            // const float u = fx * x + cx;
            // const float v = fy * y + cy;

            valid_dist = isValidDistance(pMP, p3Dw, Ow);

            // 若非有效的深度估計
            if(!std::get<0>(valid_dist)){
                continue;
            }

            dist3D = std::get<1>(valid_dist);

            // // Depth must be inside the scale pyramid of the image
            // // 考慮金字塔層級的『地圖點 pMP』最大可能深度
            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            
            // // 考慮金字塔層級的『地圖點 pMP』最小可能深度
            // const float minDistance = pMP->GetMinDistanceInvariance();
            
            // // 『相機中心 Ow』指向『地圖點 pMP』之向量
            // cv::Mat PO = p3Dw - Ow;

            // // 『地圖點 pMP』深度（『相機中心 Ow』到『地圖點 pMP』之距離）
            // const float dist3D = cv::norm(PO);

            // if (dist3D < minDistance || dist3D > maxDistance){
            //     continue;
            // }

            // // Viewing angle must be less than 60 deg
            // cv::Mat Pn = pMP->GetNormal();

            // if (PO.dot(Pn) < 0.5 * dist3D){
            //     continue;
            // }

            // Compute predicted scale level
            const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

            // 取得區域內的候選關鍵點的索引值
            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty()){
                continue;
            }

            // Match to the most similar keypoint in the radius
            // 取得地圖點描述子
            const cv::Mat dMP = pMP->GetDescriptor();

            bestDist = INT_MAX;
            bestIdx = -1;

            for(const size_t idx : vIndices)
            {
                continue_kp_level = checkFuseTarget(pKF, idx, nPredictedLevel);

                if(std::get<0>(continue_kp_level))
                {
                    continue;
                }

                // // 『關鍵點 kp』的金字塔層級
                // const int &kpLevel = pKF->mvKeysUn[idx].octave;

                // // kpLevel 可以是：(nPredictedLevel - 1) 或 nPredictedLevel
                // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel){
                //     continue;
                // }

                updateFuseTarget(pKF, idx, dMP, bestDist, bestIdx);

                // // 取得『關鍵幀 pKF』的第 idx 個特徵點的描述子
                // const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                // // 『關鍵幀 pKF』的第 idx 個特徵點的描述子 和 『地圖點 pMP』的描述子 之間的距離
                // int dist = DescriptorDistance(dMP, dKF);

                // // 篩選距離最近的『距離 bestDist』和『關鍵幀索引值 bestIdx』
                // if (dist < bestDist)
                // {
                //     bestDist = dist;
                //     bestIdx = idx;
                // }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            // 若『地圖點 pMP』描述子和『關鍵幀 pKF』的第 bestIdx 個特徵點的描述子之間的距離足夠小
            // 可視為同一地圖點
            if (bestDist <= TH_LOW)
            {
                // 取出『關鍵幀 pKF』的第 bestIdx 個特徵點所對應的地圖點
                pMPinKF = pKF->GetMapPoint(bestIdx);

                if (pMPinKF)
                {
                    if (!pMPinKF->isBad()){
                        vpReplacePoint[iMP] = pMPinKF;
                    }
                }
                else
                {
                    // 『地圖點 pMP』被『關鍵幀 pKF』的第 bestIdx 個關鍵點觀察到
                    pMP->AddObservation(pKF, bestIdx);

                    // 『關鍵幀 pKF』的第 bestIdx 個關鍵點觀察到『地圖點 pMP』
                    pKF->AddMapPoint(pMP, bestIdx);
                }

                nFused++;
            }
        }
        
        return nFused;
    }

    // 『關鍵幀 pKF1』和『關鍵幀 pKF2』上的關鍵點距離足夠小的關鍵點個數
    int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
        // 取得『關鍵幀 pKF1』的已校正關鍵點
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;

        // 取得『關鍵幀 pKF1』的特徵向量
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;

        // 取得『關鍵幀 pKF1』的地圖點
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();

        // 取得『關鍵幀 pKF1』的描述子
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        vpMatches12 = vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(), false);

        vector<int> rotHist[HISTO_LENGTH];

        for (int i = 0; i < HISTO_LENGTH; i++){
            rotHist[i].reserve(500);
        }

        const float factor = 1.0f / HISTO_LENGTH;

        int nmatches = 0;

        // FeatureVector == std::map<NodeId, std::vector<unsigned int> >
        // 以一張圖片的每個特徵點在詞典某一層節點下爲條件進行分組，用來加速圖形特徵匹配——
        // 兩兩圖像特徵匹配只需要對相同 NodeId 下的特徵點進行匹配就好。
        // std::vector<unsigned int>：觀察到該特徵的 地圖點/關鍵點 的索引值
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        vector<unsigned int> mp_indexs1, mp_indexs2;
        MapPoint *pMP1;

        while (f1it != f1end && f2it != f2end)
        {
            if (f1it->first == f2it->first)
            {
                mp_indexs1 = f1it->second;
                mp_indexs2 = f2it->second;

                for(const size_t idx1 : mp_indexs1)
                {
                    pMP1 = vpMapPoints1[idx1];

                    if (!pMP1){
                        continue;
                    }

                    if (pMP1->isBad()){
                        continue;
                    }

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    int bestDist1 = 256;
                    int bestIdx2 = -1;
                    int bestDist2 = 256;

                    for(const size_t idx2 : mp_indexs2)
                    {
                        MapPoint *pMP2 = vpMapPoints2[idx2];

                        if (vbMatched2[idx2] || !pMP2){
                            continue;
                        }

                        if (pMP2->isBad()){
                            continue;
                        }

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        // 『關鍵幀 pKF1』和『關鍵幀 pKF2』的關鍵點觀察到相似的特徵，計算兩關鍵點之間的距離
                        // 計算描述子之間的距離（相似程度）
                        int dist = DescriptorDistance(d1, d2);

                        if (dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdx2 = idx2;
                        }
                        else if (dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }
                    }

                    // 若『關鍵幀 pKF1』和『關鍵幀 pKF2』上的關鍵點距離足夠小
                    if (bestDist1 < TH_LOW)
                    {
                        // 最小距離比第二小的距離要小的多
                        if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                        {
                            // 『關鍵幀 pKF1』的第 idx1 個關鍵點，對應著『關鍵幀 pKF2』的第 bestIdx2 個地圖點
                            vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2] = true;
                            nmatches++;

                            if (mbCheckOrientation)
                            {
                                updateRotHist(vKeysUn1[idx1], vKeysUn2[bestIdx2], 
                                              factor, idx1, rotHist);
                            }                            
                        }
                    }
                }

                f1it++;
                f2it++;
            }

            // NodeId 相同才進行比較，這裡將兩者的指標指向相同的 NodeId
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            nmatches = convergenceMatched(nmatches, rotHist, 
                                          vpMatches12, static_cast<MapPoint *>(NULL));
        }

        return nmatches;
    }

    // 利用詞袋模型，快速匹配兩幀同時觀察到的地圖點 vpMapPointMatches
    int ORBmatcher::SearchByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches)
    {
        // 從『參考關鍵幀 pKF』取出已匹配成功的地圖點
        const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

        // 初始化匹配地圖點，長度初始化為『當前幀 F』的地圖點個數
        vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

        // 取得『參考關鍵幀 pKF』的特徵向量
        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

        int nmatches = 0;

        // rotHist[i], rotHist[j] 都是 vector<int>
        vector<int> rotHist[HISTO_LENGTH];

        for (int i = 0; i < HISTO_LENGTH; i++){
            rotHist[i].reserve(500);
        }

        const float factor = 1.0f / HISTO_LENGTH;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();

        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

        MapPoint *pMP;
        int bestDist1, bestIdxF, bestDist2;

        // FeatureVector == std::map<NodeId, std::vector<unsigned int> >
        while (KFit != KFend && Fit != Fend)
        {
            // first: NodeId
            // 相近的特徵點利用節點區分在一起，以加速運算，因此兩幀之間要比較時，只要比較節點(NodeId)相同的即可
            if (KFit->first == Fit->first)
            {
                // 特徵索引值列表 second: std::vector<unsigned int>
                // 包含了所有屬於『參考關鍵幀 pKF』詞袋模型的 KFit->first 節點的地圖點的索引值
                const vector<unsigned int> vIndicesKF = KFit->second;

                // 包含了所有屬於『當前幀 F』詞袋模型的 Fit->first 節點的地圖點的索引值
                const vector<unsigned int> vIndicesF = Fit->second;

                for(const unsigned int realIdxKF : vIndicesKF)
                {
                    // 『參考關鍵幀 pKF』的第 realIdxKF 個地圖點在詞袋模型中，屬於 KFit->first 節點
                    pMP = vpMapPointsKF[realIdxKF];

                    if (!pMP){
                        continue;
                    }

                    if (pMP->isBad()){
                        continue;
                    }

                    // 同樣利用 realIdxKF 取得相對應的特徵點的描述子
                    const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

                    bestDist1 = 256;
                    bestIdxF = -1;
                    bestDist2 = 256;

                    // 遍歷所有屬於『當前幀 F』詞袋模型的 Fit->first 節點的地圖點的索引值
                    for(const unsigned int realIdxF : vIndicesF){

                        if (vpMapPointMatches[realIdxF]){
                            continue;
                        }

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        // 計算兩特徵點之間的距離
                        const int dist = DescriptorDistance(dKF, dF);

                        // 篩選兩特徵點之間的最短距離（bestDist1），以及其特徵點索引值（bestIdxF）
                        if (dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF = realIdxF;
                        }
                        else if (dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }
                    }

                    // 若兩特徵點之間的距離足夠小
                    if (bestDist1 <= TH_LOW)
                    {
                        // 且比第二近的距離小的多
                        if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                        {
                            // 第 bestIdxF 個匹配地圖點設為『參考關鍵幀 pKF』的第 realIdxKF 個地圖點 pMP
                            vpMapPointMatches[bestIdxF] = pMP;

                            // 取得第 realIdxKF 個地圖點相對應的（已校正）關鍵點
                            const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                            // 成功配對數
                            nmatches++;

                            if (mbCheckOrientation)
                            {
                                updateRotHist(kp, F.mvKeys[bestIdxF], factor, bestIdxF, rotHist);
                            }                            
                        }
                    }
                }

                // 更新 const_iterator
                KFit++;
                Fit++;
            }

            // KFit 起始節點較 Fit 小
            else if (KFit->first < Fit->first)
            {
                // 將 KFit 起始節點設為 Fit->first
                KFit = vFeatVecKF.lower_bound(Fit->first);
            }

            // KFit 起始節點較 Fit 大
            else
            {
                // 將 Fit 起始節點設為 KFit->first
                Fit = F.mFeatVec.lower_bound(KFit->first);
            }
        }

        if (mbCheckOrientation)
        {
            nmatches = convergenceMatched(nmatches, rotHist, 
                                          vpMapPointMatches, static_cast<MapPoint *>(NULL));
        }

        return nmatches;
    }

    // 利用『相似轉換矩陣』將 pKF1 和 pKF2 各自觀察到的地圖點投影到彼此上，
    // 若雙方都找到同樣的匹配關係，則替換 vpMatches12 的地圖點
    int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,
                                 const float &s12, const cv::Mat &R12, const cv::Mat &t12, 
                                 const float th)
    {
        const float &fx = pKF1->fx;
        const float &fy = pKF1->fy;
        const float &cx = pKF1->cx;
        const float &cy = pKF1->cy;

        // Camera 1 from world
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();

        //Camera 2 from world
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        //Transformation between cameras
        cv::Mat sR12 = s12 * R12;
        cv::Mat sR21 = (1.0 / s12) * R12.t();
        cv::Mat t21 = -sR21 * t12;

        // 『關鍵幀 pKF1』觀察到的地圖點
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
        
        const int N1 = vpMapPoints1.size();

        // 『關鍵幀 pKF2』觀察到的地圖點
        const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();

        const int N2 = vpMapPoints2.size();

        vector<bool> vbAlreadyMatched1(N1, false);
        vector<bool> vbAlreadyMatched2(N2, false);
        MapPoint *pMP;
        int idx, bestDist, bestIdx;

        for (int i = 0; i < N1; i++)
        {
            pMP = vpMatches12[i];

            if (pMP)
            {
                vbAlreadyMatched1[i] = true;

                // 『關鍵幀 pKF2』的第 idx2 個特徵點觀察到『地圖點 pMP』               
                idx = pMP->GetIndexInKeyFrame(pKF2);

                if (idx >= 0 && idx < N2){
                    vbAlreadyMatched2[idx] = true;
                }
            }
        }
        
        // vnMatch1 匹配關係：『關鍵幀 pKF1』的第 i1 個地圖點＆『關鍵幀 pKF2』的第 bestIdx 個特徵點
        vector<int> vnMatch1(N1, -1);

        std::tuple<bool, cv::KeyPoint, int> continue_kp_level;
        std::tuple<float, float, float> u_v_invz;
        std::tuple<bool, float> valid_dist;
        float dist3D, u, v;

        cv::Mat p3Dw, p3Dc1, p3Dc2;

        // Transform from KF1 to KF2 and search
        // 『關鍵幀 pKF1』投影到『關鍵幀 pKF2』上尋找匹配點
        for (int i1 = 0; i1 < N1; i1++)
        {
            // 『地圖點 pMP』：『關鍵幀 pKF1』的第 i1 個地圖點
            pMP = vpMapPoints1[i1];

            if (!pMP || vbAlreadyMatched1[i1]){
                continue;
            }

            if (pMP->isBad()){
                continue;
            }

            p3Dw = pMP->GetWorldPos();
            p3Dc1 = R1w * p3Dw + t1w;

            // 利用『相似轉換矩陣』將『關鍵幀 pKF1』上的特徵點轉換到『關鍵幀 pKF2』的座標系下
            p3Dc2 = sR21 * p3Dc1 + t21;

            // Depth must be positive
            if (p3Dc2.at<float>(2) < 0.0){
                continue;
            }

            u_v_invz = getPixelCoordinates(p3Dc2, fx, fy, cx, cy);
            u = std::get<0>(u_v_invz);
            v = std::get<1>(u_v_invz);

            // const float invz = 1.0 / p3Dc2.at<float>(2);
            // const float x = p3Dc2.at<float>(0) * invz;
            // const float y = p3Dc2.at<float>(1) * invz;

            // // 像素座標
            // const float u = fx * x + cx;
            // const float v = fy * y + cy;

            // Point must be inside the image
            // 傳入座標點是否在關鍵幀的成像範圍內
            if (!pKF2->IsInImage(u, v)){
                continue;
            }

            valid_dist = isValidDistanceSim3(pMP, p3Dw);

            // 若非有效的深度估計
            if(!std::get<0>(valid_dist)){
                continue;
            }

            dist3D = std::get<1>(valid_dist);

            // // 考慮金字塔層級的『地圖點 pMP』最大可能深度
            // const float maxDistance = pMP->GetMaxDistanceInvariance();

            // // 考慮金字塔層級的『地圖點 pMP』最小可能深度
            // const float minDistance = pMP->GetMinDistanceInvariance();

            // // 『關鍵幀 pKF2』相機中心到空間點的距離，即『關鍵幀 pKF2』座標系下的深度
            // const float dist3D = cv::norm(p3Dc2);

            // // Depth must be inside the scale invariance region
            // if (dist3D < minDistance || dist3D > maxDistance){
            //     continue;
            // }

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

            // Search in a radius
            const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

            // 返回以 (u, v) 為圓心，在搜索半徑內，在指定金字塔層級找到的關鍵點的索引值
            const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty()){
                continue;
            }

            // Match to the most similar keypoint in the radius
            // 『地圖點 pMP』（『關鍵幀 pKF1』的第 i1 個地圖點）的描述子
            const cv::Mat dMP = pMP->GetDescriptor();

            bestDist = INT_MAX;
            bestIdx = -1;

            // 遍歷『關鍵幀 pKF2』當中可能和『地圖點 pMP』（『關鍵幀 pKF1』的第 i1 個地圖點）匹配的特徵點
            for(const size_t idx : vIndices)
            {
                continue_kp_level = checkFuseTarget(pKF2, idx, nPredictedLevel);

                if(std::get<0>(continue_kp_level))
                {
                    continue;
                }

                // const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

                // if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel){
                //     continue;
                // }

                updateFuseTarget(pKF2, idx, dMP, bestDist, bestIdx);

                // // 『關鍵幀 pKF2』的第 idx 個特徵點的描述子
                // const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

                // // 『關鍵幀 pKF2』的第 idx 個特徵點的描述子 和 『地圖點 pMP』的描述子 之間的距離
                // const int dist = DescriptorDistance(dMP, dKF);

                // // 篩選描述子距離最短的情況
                // if (dist < bestDist)
                // {
                //     bestDist = dist;

                //     // 『關鍵幀 pKF2』的第 bestIdx 個特徵點和『地圖點 pMP』最相似
                //     bestIdx = idx;
                // }
            }

            // 若描述子距離足夠近
            if (bestDist <= TH_HIGH)
            {
                // vnMatch1 匹配關係：『關鍵幀 pKF1』的第 i1 個地圖點＆『關鍵幀 pKF2』的第 bestIdx 個特徵點
                vnMatch1[i1] = bestIdx;
            }
        }

        // vnMatch2 匹配關係：『關鍵幀 pKF2』的第 i2 個地圖點＆『關鍵幀 pKF1』的第 bestIdx 個特徵點
        vector<int> vnMatch2(N2, -1);

        // Transform from KF2 to KF1 and search
        for (int i2 = 0; i2 < N2; i2++)
        {
            // 『地圖點 pMP』：『關鍵幀 pKF2』的第 i2 個地圖點
            pMP = vpMapPoints2[i2];

            if (!pMP || vbAlreadyMatched2[i2]){
                continue;
            }

            if (pMP->isBad()){
                continue;
            }

            p3Dw = pMP->GetWorldPos();
            p3Dc2 = R2w * p3Dw + t2w;

            // 利用『相似轉換矩陣』將『關鍵幀 pKF2』上的特徵點轉換到『關鍵幀 pKF1』的座標系下
            p3Dc1 = sR12 * p3Dc2 + t12;

            // Depth must be positive
            if (p3Dc1.at<float>(2) < 0.0){
                continue;
            }

            u_v_invz = getPixelCoordinates(p3Dc1, fx, fy, cx, cy);
            u = std::get<0>(u_v_invz);
            v = std::get<1>(u_v_invz);

            // const float invz = 1.0 / p3Dc1.at<float>(2);

            // // 重投影之歸一化平面座標
            // const float x = p3Dc1.at<float>(0) * invz;
            // const float y = p3Dc1.at<float>(1) * invz;

            // const float u = fx * x + cx;
            // const float v = fy * y + cy;

            // Point must be inside the image
            if (!pKF1->IsInImage(u, v)){
                continue;
            }

            valid_dist = isValidDistanceSim3(pMP, p3Dw);

            // 若非有效的深度估計
            if(!std::get<0>(valid_dist)){
                continue;
            }

            dist3D = std::get<1>(valid_dist);

            // // 考慮金字塔層級的『地圖點 pMP』最大可能深度
            // const float maxDistance = pMP->GetMaxDistanceInvariance();

            // // 考慮金字塔層級的『地圖點 pMP』最小可能深度
            // const float minDistance = pMP->GetMinDistanceInvariance();

            // // 『關鍵幀 pKF1』相機中心到空間點的距離，即『關鍵幀 pKF1』座標系下的深度
            // const float dist3D = cv::norm(p3Dc1);

            // // Depth must be inside the scale pyramid of the image
            // if (dist3D < minDistance || dist3D > maxDistance){
            //     continue;
            // }

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

            // Search in a radius of 2.5*sigma(ScaleLevel)
            const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

            // 返回以 (u, v) 為圓心，在搜索半徑內，在指定金字塔層級找到的關鍵點的索引值
            const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty()){
                continue;
            }

            // Match to the most similar keypoint in the radius
            // 『地圖點 pMP』（『關鍵幀 pKF2』的第 i2 個地圖點）的描述子
            const cv::Mat dMP = pMP->GetDescriptor();

            bestDist = INT_MAX;
            bestIdx = -1;

            // 遍歷『關鍵幀 pKF1』當中可能和『地圖點 pMP』（『關鍵幀 pKF2』的第 i2 個地圖點）匹配的特徵點
            for(const size_t idx : vIndices)
            {
                continue_kp_level = checkFuseTarget(pKF1, idx, nPredictedLevel);

                if(std::get<0>(continue_kp_level))
                {
                    continue;
                }

                // const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

                // if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel){
                //     continue;
                // }

                updateFuseTarget(pKF1, idx, dMP, bestDist, bestIdx);

                // // 『關鍵幀 pKF1』的第 idx 個特徵點的描述子
                // const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

                // // 『關鍵幀 pKF1』的第 idx 個特徵點的描述子 和 『地圖點 pMP』的描述子 之間的距離
                // const int dist = DescriptorDistance(dMP, dKF);

                // // 篩選描述子距離最短的情況
                // if (dist < bestDist)
                // {
                //     bestDist = dist;

                //     // 『關鍵幀 pKF1』的第 bestIdx 個特徵點和『地圖點 pMP』最相似
                //     bestIdx = idx;
                // }
            }

            // 若描述子距離足夠近
            if (bestDist <= TH_HIGH)
            {
                // vnMatch2 匹配關係：『關鍵幀 pKF2』的第 i2 個地圖點＆『關鍵幀 pKF1』的第 bestIdx 個特徵點
                vnMatch2[i2] = bestIdx;
            }
        }

        // Check agreement
        int nFound = 0, i1, idx2, idx1;

        // 『關鍵幀 pKF1』索引值
        for (i1 = 0; i1 < N1; i1++)
        {
            // vnMatch1 匹配關係：『關鍵幀 pKF1』的第 i1 個地圖點＆『關鍵幀 pKF2』的第 idx2 個特徵點
            idx2 = vnMatch1[i1];

            // 初始值為 -1，>= 0 表示有配對成功
            if (idx2 >= 0)
            {
                // vnMatch2 匹配關係：『關鍵幀 pKF2』的第 idx2 個地圖點＆『關鍵幀 pKF1』的第 idx1 個特徵點
                idx1 = vnMatch2[idx2];

                if (idx1 == i1)
                {
                    // 原本的地圖點，替換成『關鍵幀 pKF2』觀察到的地圖點
                    vpMatches12[i1] = vpMapPoints2[idx2];
                    nFound++;
                }
            }
        }

        return nFound;
    }

    // 『地圖點們 vpPoints』投影到『關鍵幀 pKF』上，vpMatched 為匹配結果，第 idx 個特徵點對應『地圖點 pMP』
    int ORBmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, 
                                       vector<MapPoint *> &vpMatched, int th)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw / scw;
        cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
        cv::Mat Ow = -Rcw.t() * tcw;

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());

        // 將 NULL 的地圖點移除
        spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

        int nmatches = 0, bestDist, bestIdx;

        // Get 3D Coords.
        cv::Mat p3Dw, p3Dc;

        std::tuple<float, float, float> u_v_invz;
        std::tuple<bool, cv::KeyPoint, int> continue_kp_level;
        std::tuple<bool, float> valid_dist;
        float u, v, dist;

        // For each Candidate MapPoint Project and Match
        for(MapPoint *pMP : vpPoints){

            // Discard Bad MapPoints and already found
            // spAlreadyFound.count(pMP)：排除已存在的地圖點
            if (pMP->isBad() || spAlreadyFound.count(pMP)){
                continue;
            }

            // Get 3D Coords.
            cv::Mat p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            cv::Mat p3Dc = Rcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc.at<float>(2) < 0.0f)
            {
                continue;
            }
            
            u_v_invz = getPixelCoordinates(p3Dc, fx, fy, cx, cy);
            u = std::get<0>(u_v_invz);
            v = std::get<1>(u_v_invz);

            // Point must be inside the image
            // 傳入座標點是否超出關鍵幀的成像範圍
            if (!pKF->IsInImage(u, v))
            {
                continue;
            }
            
            // // Depth must be positive
            // if (p3Dc.at<float>(2) < 0.0){
            //     continue;
            // }

            // // Project into Image
            // const float invz = 1 / p3Dc.at<float>(2);

            // // 重投影之歸一化平面座標
            // const float x = p3Dc.at<float>(0) * invz;
            // const float y = p3Dc.at<float>(1) * invz;

            // // 重投影之像素座標
            // const float u = fx * x + cx;
            // const float v = fy * y + cy;

            // // Point must be inside the image
            // if (!pKF->IsInImage(u, v)){
            //     continue;
            // }

            valid_dist = isValidDistance(pMP, p3Dw, Ow);

            // 若非有效的深度估計
            if(!std::get<0>(valid_dist)){
                continue;
            }

            dist = std::get<1>(valid_dist);

            // // Depth must be inside the scale invariance region of the point
            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();

            // // 相機中心指向空間點
            // cv::Mat PO = p3Dw - Ow;

            // // 深度
            // const float dist = cv::norm(PO);

            // if (dist < minDistance || dist > maxDistance){
            //     continue;
            // }

            // // Viewing angle must be less than 60 deg
            // // 取得『地圖點 pMP』的法向量
            // cv::Mat Pn = pMP->GetNormal();

            // if (PO.dot(Pn) < 0.5 * dist){
            //     continue;
            // }

            // 『關鍵幀 pKF』根據當前『地圖點 pMP』的深度，估計場景規模
            int nPredictedLevel = pMP->PredictScale(dist, pKF);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

            // 取得區域內的候選關鍵點的索引值
            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty()){
                continue;
            }

            // Match to the most similar keypoint in the radius
            // 取得『地圖點 pMP』描述子
            const cv::Mat dMP = pMP->GetDescriptor();

            bestDist = INT_MAX;
            bestIdx = -1;

            for(const size_t idx : vIndices){

                if (vpMatched[idx]){
                    continue;
                }

                continue_kp_level = checkFuseTarget(pKF, idx, nPredictedLevel);

                // // 『關鍵點 kp』的金字塔層級
                // const int &kpLevel = pKF->mvKeysUn[idx].octave;

                // // kpLevel 可以是：(nPredictedLevel - 1) 或 nPredictedLevel
                // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel){
                //     continue;
                // }

                updateFuseTarget(pKF, idx, dMP, bestDist, bestIdx);

                // // 取得『關鍵幀 pKF』的第 idx 個特徵點的描述子
                // const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                // // 計算『地圖點 pMP』描述子和『關鍵幀 pKF』的第 idx 個特徵點的描述子之間的距離
                // const int dist = DescriptorDistance(dMP, dKF);

                // // 篩選距離最近的『距離 bestDist』和『關鍵幀索引值 bestIdx』
                // if (dist < bestDist)
                // {
                //     bestDist = dist;
                //     bestIdx = idx;
                // }
            }

            if (bestDist <= TH_LOW)
            {
                // 『關鍵幀 pKF』的第 idx 個特徵點對應『地圖點 pMP』
                vpMatched[bestIdx] = pMP;
                nmatches++;
            }
        }

        return nmatches;
    }

    // 尋找 CurrentFrame 當中和『LastFrame』特徵點對應的位置，形成 CurrentFrame 的地圖點，並返回匹配成功的個數
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, 
                                       const bool bMono)
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];

        for (int i = 0; i < HISTO_LENGTH; i++){
            rotHist[i].reserve(500);
        }

        const float factor = 1.0f / HISTO_LENGTH;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        /// NOTE: 旋轉矩陣為正交矩陣，因此轉置和逆相同，但加負號是什麼意思？
        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat tlc = Rlw * twc + tlw;

        // Mono 模式下，這兩個都是 false
        const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;
        const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;

        std::tuple<float, float, float> u_v_invz;
        int nLastOctave, bestDist, bestIdx;
        vector<size_t> vIndices2;
        float radius, u, v, invz;
        MapPoint *pMP;
        
        for (int i = 0; i < LastFrame.N; i++)
        {
            // 依序取出前一幀觀察到的地圖點
            // LastFrame 的第 i 個地圖點
            pMP = LastFrame.mvpMapPoints[i];

            if (pMP)
            {
                // 若該地圖點不是 Outlier
                if (!LastFrame.mvbOutlier[i])
                {
                    // Project
                    // 取出地圖點的世界座標
                    cv::Mat x3Dw = pMP->GetWorldPos();

                    // 將地圖點轉換到相機座標下
                    cv::Mat x3Dc = Rcw * x3Dw + tcw;

                    // const float xc = x3Dc.at<float>(0);
                    // const float yc = x3Dc.at<float>(1);

                    // // 取得逆深度
                    // const float invzc = 1.0 / x3Dc.at<float>(2);

                    // // 深度必定為正，不會有負數
                    // if (invzc < 0){
                    //     continue;
                    // }

                    // // 相機座標 轉換到 像素座標
                    // float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
                    // float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

                    // Depth must be positive
                    if (x3Dc.at<float>(2) < 0.0f)
                    {
                        continue;
                    }
            
                    u_v_invz = getPixelCoordinates(x3Dc,
                                                   CurrentFrame.fx, CurrentFrame.fy, 
                                                   CurrentFrame.cx, CurrentFrame.cy);

                    u = std::get<0>(u_v_invz);
                    v = std::get<1>(u_v_invz);
                    invz = std::get<2>(u_v_invz);

                    // 檢查像素點位置是否超出成像範圍
                    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX){
                        continue;
                    }

                    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY){
                        continue;
                    }

                    // 取得前一幀影像金字塔的層級
                    nLastOctave = LastFrame.mvKeys[i].octave;

                    // Search in a window. Size depends on scale
                    // 計算金字塔層級對應的搜索半徑
                    radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                    if (bForward){
                        vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
                    }
                    else if (bBackward){
                        vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
                    }

                    // Mono 模式下，前兩項都是 false，直接進入此區塊
                    else{
                        // 返回以 (u, v) 為圓心，在搜索半徑內，在指定金字塔層級找到的關鍵點的索引值
                        vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 
                                                                   nLastOctave - 1, nLastOctave + 1);
                    }

                    // 搜索半徑內沒有找到特徵點
                    if (vIndices2.empty()){
                        continue;
                    }

                    // 『LastFrame 的第 i 個地圖點』的描述子
                    const cv::Mat dMP = pMP->GetDescriptor();

                    bestDist = INT_MAX;
                    bestIdx = -1;

                    // 遍歷搜索半徑內找到的特徵點的索引值
                    for(const size_t i2 : vIndices2)
                    {
                        // 取得特徵點相對應的地圖點
                        // LastFrame 的特徵點反覆在 CurrentFrame 上尋找對應的點
                        // 因此前面的流程中已形成 CurrentFrame.mvpMapPoints[i2] 是有可能的
                        if (CurrentFrame.mvpMapPoints[i2]){

                            // 若該地圖點被至少 1 個關鍵幀觀察到，則無須再進行後續匹配（因為已經匹配成功）
                            if (CurrentFrame.mvpMapPoints[i2]->getObservationNumber() > 0){
                                continue;
                            }
                        }

                        // 單目的 mvuRight 會是負值，因此暫時跳過
                        if (CurrentFrame.mvuRight[i2] > 0)
                        {
                            const float ur = u - CurrentFrame.mbf * invz;
                            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);

                            if (er > radius){
                                continue;
                            }
                        }

                        // updateFuseTarget(&CurrentFrame, i2, dMP, bestDist, bestIdx);

                        // 取得 CurrentFrame 的第 i2 個特徵點的描述子
                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        // 計算『LastFrame 的第 i 個地圖點的描述子』和『CurrentFrame 的第 i2 個特徵點的描述子』
                        // 之間的距離，距離足夠小則表示匹配成功
                        const int dist = DescriptorDistance(dMP, d);

                        // 過濾兩者距離最小的距離和索引值
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx = i2;
                        }
                    }

                    // 若描述子之間的最小距離足夠小
                    if (bestDist <= TH_HIGH)
                    {
                        // 將 CurrentFrame 第 bestIdx2 個地圖點替換成 pMP
                        CurrentFrame.mvpMapPoints[bestIdx] = pMP;

                        // 配對數 +1
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            updateRotHist(LastFrame.mvKeysUn[i], CurrentFrame.mvKeysUn[bestIdx], 
                                          factor, bestIdx, rotHist);
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        if (mbCheckOrientation)
        {
            nmatches = convergenceMatched(nmatches, rotHist, 
                                          CurrentFrame.mvpMapPoints, static_cast<MapPoint *>(NULL));
        }

        return nmatches;
    }

    // 尋找 CurrentFrame 當中和『關鍵幀 pKF』的特徵點對應的位置，形成 CurrentFrame 的地圖點，並返回匹配成功的個數
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, 
                                       const set<MapPoint *> &sAlreadyFound, const float th, 
                                       const int ORBdist)
    {
        /*
        th：搜索半徑的尺度參數
        */
        int nmatches = 0;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);
        const cv::Mat Ow = -Rcw.t() * tcw;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];

        for (int i = 0; i < HISTO_LENGTH; i++){
            rotHist[i].reserve(500);
        }

        const float factor = 1.0f / HISTO_LENGTH;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        
        std::tuple<float, float, float> u_v_invz;
        std::tuple<bool, float> valid_dist;
        cv::Mat x3Dw, x3Dc;
        float u, v, dist3D;
        MapPoint *pMP;

        // 遍歷『關鍵幀 pKF』的已配對地圖點
        for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
        {
            pMP = vpMPs[i];

            if (pMP)
            {
                if (!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    x3Dw = pMP->GetWorldPos();
                    x3Dc = Rcw * x3Dw + tcw;

                    u_v_invz = getPixelCoordinates(x3Dc,
                                                   CurrentFrame.fx, CurrentFrame.fy, 
                                                   CurrentFrame.cx, CurrentFrame.cy);

                    u = std::get<0>(u_v_invz);
                    v = std::get<1>(u_v_invz);

                    // const float xc = x3Dc.at<float>(0);
                    // const float yc = x3Dc.at<float>(1);
                    // const float invzc = 1.0 / x3Dc.at<float>(2);

                    // const float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
                    // const float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

                    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX){
                        continue;
                    }

                    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY){
                        continue;
                    }

                    valid_dist = isValidDistance(pMP, x3Dw, Ow);

                    // 若非有效深度估計
                    if(!std::get<0>(valid_dist)){
                        continue;
                    }

                    dist3D = std::get<1>(valid_dist);

                    // // Compute predicted scale level
                    // cv::Mat PO = x3Dw - Ow;
                    // float dist3D = cv::norm(PO);

                    // const float maxDistance = pMP->GetMaxDistanceInvariance();
                    // const float minDistance = pMP->GetMinDistanceInvariance();

                    // // Depth must be inside the scale pyramid of the image
                    // if (dist3D < minDistance || dist3D > maxDistance){
                    //     continue;
                    // }

                    // 根據當前距離與最遠可能距離，換算出當前尺度
                    int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);

                    // Search in a window
                    const float radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel];

                    // 返回以 (u, v) 為圓心，在搜索半徑內，在指定金字塔層級找到的關鍵點的索引值
                    const vector<size_t> vIndices2 = 
                                    CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, 
                                                                nPredictedLevel + 1);

                    if (vIndices2.empty()){
                        continue;
                    }

                    // 取得『關鍵幀 pKF』的已配對地圖點的描述子
                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    // 遍歷關鍵點的索引值
                    for(const size_t i2 : vIndices2){

                        if (CurrentFrame.mvpMapPoints[i2]){
                            continue;
                        }

                        updateFuseTarget(&CurrentFrame, i2, dMP, bestDist, bestIdx2);

                        // // 取得當前幀的第 i2 個特徵點的描述子
                        // const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        // // 計算描述子之間的距離（相似程度）
                        // const int dist = DescriptorDistance(dMP, d);

                        // if (dist < bestDist)
                        // {
                        //     bestDist = dist;
                        //     bestIdx2 = i2;
                        // }
                    }

                    if (bestDist <= ORBdist)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            updateRotHist(pKF->mvKeysUn[i], CurrentFrame.mvKeysUn[bestIdx2], 
                                          factor, bestIdx2, rotHist);
                        }
                    }
                }
            }
        }

        if (mbCheckOrientation)
        {
            nmatches = convergenceMatched(nmatches, rotHist, 
                                          CurrentFrame.mvpMapPoints, static_cast<MapPoint *>(NULL));
        }

        return nmatches;
    }

    // Used in Tracking::SearchLocalPoints
    int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th)
    {
        int nmatches = 0, bestDist, bestLevel, bestDist2, bestLevel2, bestIdx;

        // const bool bFactor = th != 1.0;
        MapPoint *pMP;
        float r;

        for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
        {
            pMP = vpMapPoints[iMP];

            if (!pMP->mbTrackInView){
                continue;
            }

            if (pMP->isBad()){
                continue;
            }

            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            r = th * RadiusByViewingCos(pMP->mTrackViewCos);

            // if (bFactor){
            //     r *= th;
            // }

            const vector<size_t> vIndices = 
                                            F.GetFeaturesInArea(pMP->mTrackProjX,
                                                                pMP->mTrackProjY, 
                                                                r * F.mvScaleFactors[nPredictedLevel], 
                                                                nPredictedLevel - 1, 
                                                                nPredictedLevel);

            if (vIndices.empty()){
                continue;
            }

            const cv::Mat MPdescriptor = pMP->GetDescriptor();

            bestDist = 256;
            bestLevel = -1;
            bestDist2 = 256;
            bestLevel2 = -1;
            bestIdx = -1;

            // Get best and second matches with near keypoints
            for(const size_t idx : vIndices){

                if (F.mvpMapPoints[idx])
                {
                    if (F.mvpMapPoints[idx]->getObservationNumber() > 0)
                    {
                        continue;
                    }
                }

                if (F.mvuRight[idx] > 0)
                {
                    const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);

                    if (er > r * F.mvScaleFactors[nPredictedLevel]){
                        continue;
                    }
                }

                const cv::Mat &d = F.mDescriptors.row(idx);

                const int dist = DescriptorDistance(MPdescriptor, d);

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeysUn[idx].octave;
                    bestIdx = idx;
                }
                else if (dist < bestDist2)
                {
                    bestLevel2 = F.mvKeysUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= TH_HIGH)
            {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2){
                    continue;
                }

                F.mvpMapPoints[bestIdx] = pMP;
                nmatches++;
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, 
                                            vector<int> &vnMatches12, int windowSize)
    {
        int nmatches = 0;
        vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

        // 直方圖，長度為 HISTO_LENGTH，用於紀錄關鍵點分別落在哪些角度區間
        vector<int> rotHist[HISTO_LENGTH];

        // 『角度換算到直方圖中的序號』的尺度變量
        const float factor = 1.0f / HISTO_LENGTH;

        for (int i = 0; i < HISTO_LENGTH; i++){
            rotHist[i].reserve(500);
        }
        
        // F2.mvKeysUn.size()： F2 當中已校正之關鍵點的數量
        vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
        vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

        // 遍歷 F1 的（已校正關鍵點）索引值
        for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++)
        {
            // 取得 F1 中的關鍵點
            cv::KeyPoint kp1 = F1.mvKeysUn[i1];

            // 取得當前關鍵點是在影像金字塔的哪個層級找到的
            int level1 = kp1.octave;

            // 只處理原圖？ level1 0 為原圖
            if (level1 > 0){
                continue;
            }

            // 從 F2 取出指定區域內，由『指定金字塔層級(level1: 0)』找到的關鍵點的索引值
            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, 
                                                            vbPrevMatched[i1].y, 
                                                            windowSize, level1, level1);

            if (vIndices2.empty()){
                continue;
            }

            // 取出影像 F1 當中的第 i1 個關鍵點的描述子
            cv::Mat d1 = F1.mDescriptors.row(i1);

            // 最小距離
            int bestDist = INT_MAX;

            // 第二小距離
            int bestDist2 = INT_MAX;

            // 影像 F2 當中和 d1 距離最近的關鍵點的索引值
            int bestIdx2 = -1;

            // 遍歷 F2 的（已校正關鍵點）索引值
            for(size_t i2 : vIndices2){

                // 取出影像 F2 當中的第 i2 個關鍵點的描述子
                cv::Mat d2 = F2.mDescriptors.row(i2);

                // d1：影像 F1 當中的第 i1 個關鍵點的描述子
                // d2：影像 F2 當中的第 i2 個關鍵點的描述子
                // 計算兩個關鍵點的距離
                int dist = DescriptorDistance(d1, d2);

                // 若關鍵點之間的距離，比已匹配距離更遠，則直接計算下一點
                if (vMatchedDistance[i2] <= dist){
                    continue;
                }

                if (dist < bestDist)
                {
                    // 最小距離
                    bestDist2 = bestDist;

                    // 第二小距離
                    bestDist = dist;

                    // 影像 F2 當中和 d1 距離最近的關鍵點的索引值
                    bestIdx2 = i2;
                }
                else if (dist < bestDist2)
                {
                    bestDist2 = dist;
                }
            }

            // 若最小距離足夠小
            if (bestDist <= TH_LOW)
            {
                // 且『最小距離』比『第二小距離 的 mfNNratio 倍』還小
                if (bestDist < (float)bestDist2 * mfNNratio)
                {
                    // 若未曾匹配過，數值會是 -1， >= 0 表示曾經匹配過
                    if (vnMatches21[bestIdx2] >= 0)
                    {
                        // 將之前的數據還原
                        vnMatches12[vnMatches21[bestIdx2]] = -1;
                        nmatches--;
                    }

                    // 關鍵點匹配中，和『影像 F1 第 i1 個關鍵點』配對成功的是『影像 F2 第 bestIdx2 個關鍵點』
                    vnMatches12[i1] = bestIdx2;

                    // bestIdx2：影像 F2 當中和 d1 距離最近的關鍵點的索引值
                    // 關鍵點匹配中，和『影像 F2 第 bestIdx2 個關鍵點』配對成功的是『影像 F1 第 i1 個關鍵點』
                    vnMatches21[bestIdx2] = i1;

                    // 第 bestIdx2 組匹配成功的距離為 bestDist
                    vMatchedDistance[bestIdx2] = bestDist;

                    // 更新成功匹配個數
                    nmatches++;

                    // 檢查方向
                    if (mbCheckOrientation)
                    {
                        updateRotHist(F1.mvKeysUn[i1], F2.mvKeysUn[bestIdx2], factor, i1, rotHist);

                        // // 計算兩關鍵點之間的角度
                        // float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;

                        // if (rot < 0.0){
                        //     rot += 360.0f;
                        // }

                        // // 角度換算到直方圖中的序號
                        // int bin = round(rot * factor);

                        // if (bin == HISTO_LENGTH){
                        //     bin = 0;
                        // }

                        // assert(bin >= 0 && bin < HISTO_LENGTH);

                        // // 直方圖，長度為 HISTO_LENGTH，用於紀錄關鍵點分別落在哪些角度區間
                        // // i1 和其匹配到的關鍵點之夾角，落在第 bin 個區間
                        // rotHist[bin].push_back(i1);
                    }
                }
            }
        }

        // 檢查方向
        if (mbCheckOrientation)
        {
            nmatches = convergenceMatched(nmatches, rotHist, vnMatches12, -1);
        }

        // Update prev matched
        for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++){

            if (vnMatches12[i1] >= 0){

                // 更新為當前各個匹配成功的關鍵點的位置，協助尋找下一幀的關鍵點
                vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;
            }
        }

        return nmatches;
    }

    float ORBmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if (viewCos > 0.998)
        {
            return 2.5;
        }
        else{
            return 4.0;
        }
    }
    
    // T: vector<int> / vector<MapPoint *>
    // default_value: -1 / static_cast<MapPoint *>(NULL)
    template<class T, class D>
    int ORBmatcher::convergenceMatched(int n_match, vector<int> *rot_hist, 
                                       vector<T> &v_matched, D default_value){
        int i, ind1 = -1, ind2 = -1, ind3 = -1;
        size_t j, jend;

        // 篩選前三多直方格的索引值
        ComputeThreeMaxima(rot_hist, HISTO_LENGTH, ind1, ind2, ind3);

        for (i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
            {
                continue;
            }

            jend = rot_hist[i].size();

            for (j = 0; j < jend; j++)
            {
                /// NOTE: rot_hist 當中就是儲存配對到的資訊，因此不檢查也可以
                v_matched[rot_hist[i][j]] = default_value;
                n_match--;
            }
        }

        return n_match;
    }

    void ORBmatcher::updateRotHist(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, 
                                   const float factor, const int idx, std::vector<int> *rot_hist){
        // 計算角度差
        float rot = kp1.angle - kp2.angle;

        if (rot < 0.0){
            rot += 360.0f;
        }

        // 角度差換算成直方圖的格子索引值
        int bin = round(rot * factor);

        // HISTO_LENGTH >= bin >= 0
        bin = min(HISTO_LENGTH, max(bin, 0));

        if (bin == HISTO_LENGTH){
            bin = 0;
        }
        
        // assert(bin >= 0 && bin < HISTO_LENGTH);
        rot_hist[bin].push_back(idx);
    }

    // std::tuple<float, float, float>{u, v, invz}
    std::tuple<float, float, float> ORBmatcher::getPixelCoordinates(cv::Mat sp3Dc,
                                                                    const float fx,
                                                                    const float fy,
                                                                    const float cx,
                                                                    const float cy)
    {
        const float invz = 1 / sp3Dc.at<float>(2);
        
        // 重投影之歸一化平面座標
        const float x = sp3Dc.at<float>(0) * invz;
        const float y = sp3Dc.at<float>(1) * invz;

        // 重投影之像素座標
        const float u = fx * x + cx;
        const float v = fy * y + cy;
        
        return std::tuple<float, float, float>{u, v, invz};
    }

    // std::tuple<bool, float, float, float>{need_continue, u, v, ur}
    std::tuple<float, float, float> ORBmatcher::getPixelCoordinatesStereo(cv::Mat sp3Dc, 
                                                                                const float bf, 
                                                                                const float fx, 
                                                                                const float fy,
                                                                                const float cx, 
                                                                                const float cy)
    {
        // std::tuple<float, float, float>{u, v, invz}
        std::tuple<float, float, float> u_v_invz;
        u_v_invz = getPixelCoordinates(sp3Dc, fx, fy, cx, cy);

        float u = std::get<0>(u_v_invz);
        float invz = std::get<2>(u_v_invz);
        float ur = u - bf * invz;
        std::get<2>(u_v_invz) = ur;

        return u_v_invz;
    }
    
    std::tuple<bool, float> ORBmatcher::isValidDistance(MapPoint *pMP, cv::Mat p3Dw, cv::Mat Ow)
    {
        // 考慮金字塔層級的『地圖點 pMP』最大可能深度
        const float maxDistance = pMP->GetMaxDistanceInvariance();

        // 考慮金字塔層級的『地圖點 pMP』最小可能深度
        const float minDistance = pMP->GetMinDistanceInvariance();

        // 『相機中心 Ow』指向『地圖點 pMP』之向量
        cv::Mat PO = p3Dw - Ow;

        // 『地圖點 pMP』深度（『相機中心 Ow』到『地圖點 pMP』之距離）
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        // 『相機中心 Ow』到『地圖點 pMP』之距離應在可能深度的區間內
        if (dist3D < minDistance || dist3D > maxDistance)
        {
            return std::tuple<bool, float>{false, -1.0f};
        }
        
        // Viewing angle must be less than 60 deg
        // 地圖點之法向量
        // SerachBySim3 沒有這段
        cv::Mat Pn = pMP->GetNormal();

        // SerachBySim3 沒有這段
        // 計算 PO 和 Pn 的夾角是否超過 60 度（餘弦值超過 0.5）
        if (PO.dot(Pn) < 0.5 * dist3D)
        {
            return std::tuple<bool, float>{false, -1.0f};
        }

        return std::tuple<bool, float>{true, dist3D};
    }

    std::tuple<bool, float> ORBmatcher::isValidDistanceSim3(MapPoint *pMP, cv::Mat sp3Dc)
    {
        // 考慮金字塔層級的『地圖點 pMP』最大可能深度
        const float maxDistance = pMP->GetMaxDistanceInvariance();

        // 考慮金字塔層級的『地圖點 pMP』最小可能深度
        const float minDistance = pMP->GetMinDistanceInvariance();

        // 『地圖點 pMP』深度（『相機中心 Ow』到『地圖點 pMP』之距離）
        const float dist3D = cv::norm(sp3Dc);

        // Depth must be inside the scale pyramid of the image
        // 『相機中心 Ow』到『地圖點 pMP』之距離應在可能深度的區間內
        if (dist3D < minDistance || dist3D > maxDistance)
        {
            return std::tuple<bool, float>{false, -1.0f};
        }

        return std::tuple<bool, float>{true, dist3D};
    }

    // std::tuple<bool, cv::KeyPoint, int>{continue, kp, kpLevel};
    std::tuple<bool, cv::KeyPoint, int> ORBmatcher::checkFuseTarget(KeyFrame *pKF, const size_t idx,
                                                                    int nPredictedLevel)
    {
        // 指定區域內的候選關鍵點
        const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

        // 『關鍵點 kp』的金字塔層級
        const int &kpLevel = kp.octave;

        bool need_continue = false;

        // kpLevel 可以是：(nPredictedLevel - 1) 或 nPredictedLevel
        if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
        {
            need_continue = true;
        }
        
        return std::tuple<bool, cv::KeyPoint, int>{need_continue, kp, kpLevel};
    }

    void ORBmatcher::updateFuseTarget(Frame *frame, const size_t idx, const cv::Mat dMP, 
                                      int &bestDist, int &bestIdx)
    {
        // 取得『關鍵幀 pKF』的第 idx 個特徵點的描述子
        const cv::Mat dKF = frame->mDescriptors.row(idx);

        updateFuseTarget(dKF, idx, dMP, bestDist, bestIdx);
    }

    void ORBmatcher::updateFuseTarget(KeyFrame *pKF, const size_t idx, const cv::Mat dMP, 
                                      int &bestDist, int &bestIdx)
    {
        // 取得『關鍵幀 pKF』的第 idx 個特徵點的描述子
        const cv::Mat dKF = pKF->mDescriptors.row(idx);

        updateFuseTarget(dKF, idx, dMP, bestDist, bestIdx);
    } 

    void ORBmatcher::updateFuseTarget(const cv::Mat dKF, const size_t idx, const cv::Mat dMP, 
                                      int &bestDist, int &bestIdx)
    {
        // 計算『地圖點 pMP』描述子和『關鍵幀 pKF』的第 idx 個特徵點的描述子之間的距離
        int dist = DescriptorDistance(dMP, dKF);

        // 篩選距離最近的『距離 bestDist』和『關鍵幀索引值 bestIdx』
        if (dist < bestDist)
        {
            bestDist = dist;
            bestIdx = idx;
        }
    }   
    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

} //namespace ORB_SLAM
