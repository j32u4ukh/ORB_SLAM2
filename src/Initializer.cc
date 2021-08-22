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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include <thread>

namespace ORB_SLAM2
{
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    // ReferenceFrame：用於初始化的參考幀
    Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
    {
        // 相機的內參矩陣
        K = ReferenceFrame.K.clone();

        // 參考幀中提取的特征點（已校正）
        mvKeys1 = ReferenceFrame.mvKeysUn;

        mSigma = sigma;
        mSigma2 = sigma * sigma;
        mMaxIterations = iterations;
    }

    // 同時利用『單應性矩陣』和『基礎矩陣』估計空間點 vP3D（過程中包含估計旋轉和平移），挑選較佳的估計結果
    // 並確保『兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小』，返回是否順利估計
    // 估計 旋轉 Rcw, 平移 tcw, 空間點位置 mvIniP3D
    bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21,
                                 cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
    {
        // Fill structures with current keypoints and matches with reference frame
        // Reference Frame: 1, Current Frame: 2
        // 使用成員變量 mvKeys2 記錄下當前幀中的特征點
        mvKeys2 = CurrentFrame.mvKeysUn;

        // vMatches12 中記錄了匹配的特征點。
        mvMatches12.clear();
        mvMatches12.reserve(mvKeys2.size());
        mvbMatched1.resize(mvKeys1.size());

        // 成員容器 mvMatches12 中記錄下兩幀圖像匹配的特征點對， mvbMatched1 則標記了參考幀中的特征點是否有匹配對象。
        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            // 索引值 >= 0 表示有匹配到
            if (vMatches12[i] >= 0)
            {
                mvMatches12.push_back(make_pair(i, vMatches12[i]));
                mvbMatched1[i] = true;
            }
            else
            {
                mvbMatched1[i] = false;
            }
        }

        // 所有匹配點的數量
        const int N = mvMatches12.size();

        // 協助重置 vAvailableIndices
        vector<size_t> vAllIndices;
        vAllIndices.reserve(N);

        // 記錄了用於 RANSAC 叠代采樣時的備選匹配點索引。
        vector<size_t> vAvailableIndices;

        for (int i = 0; i < N; i++)
        {
            vAllIndices.push_back(i);
        }

        // Generate sets of 8 points for each RANSAC iteration
        // 記錄了所有 RANSAC 樣本，每個樣本都有 8 個特征點，共 mMaxIterations 組，確保可解出基礎矩陣和單應矩陣。
        mvSets = vector<vector<size_t>>(mMaxIterations, vector<size_t>(8, 0));

        DUtils::Random::SeedRandOnce(0);
        int randi, idx;
        size_t j;

        for (int it = 0; it < mMaxIterations; it++)
        {
            vAvailableIndices = vAllIndices;

            // Select a minimum set
            // 取出 8 個特徵點索引值
            for (j = 0; j < 8; j++)
            {
                // 隨機取得索引值
                randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
                idx = vAvailableIndices[randi];

                mvSets[it][j] = idx;

                // 將取出的第 randi 個的數值，以最後一個數值取代
                vAvailableIndices[randi] = vAvailableIndices.back();

                // 將最後一個數值移除，形成抽出不放回的狀態
                vAvailableIndices.pop_back();
            }
        }

        // Launch threads to compute in parallel a fundamental matrix and a homography
        vector<bool> vbMatchesInliersH, vbMatchesInliersF;

        // 局部變量 SH 和 SF 分別給出了兩個矩陣的評分，H 和 F 則是解出的單應和基礎矩陣。
        float SH, SF;
        cv::Mat H, F;

        // 兩個線程中，完成單應矩陣和基礎矩陣的計算。
        // FindHomography： 尋找單應性矩陣以及重投影得分，並區分關鍵點為內點還是外點
        thread threadH(&Initializer::FindHomography, this, ref(vbMatchesInliersH), ref(SH), ref(H));
        thread threadF(&Initializer::FindFundamental, this, ref(vbMatchesInliersF), ref(SF), ref(F));

        // Wait until both threads have finished
        threadH.join();
        threadF.join();

        // Compute ratio of scores
        float RH = SH / (SH + SF);

        // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
        /* 函數 ReconstructH 和 ReconstructF 分別完成了單應矩陣和基礎矩陣的奇異值分解。
        我們知道，分解出來的旋轉矩陣和平移向量並不是唯一的。
        單應矩陣可以分解出來 8 種組合，基礎矩陣則有 4 種，而相機的位姿只有一種可能。
        ORB-SLAM 針對每一種組合，對匹配到的特征點進行三角化估計對應點的空間三維坐標。 
        然後選擇三角化後的空間點都在兩幀相機的前方並且重投影誤差最小的那個組合，作為相機的位姿估計。

        函數 ReconstructH 和 ReconstructF 的參數列表十分相似，
        前三個參數分別記錄了 1.特征點是否匹配的標記、2.待分解的矩陣、3.相機的內參矩陣。
        R21 和 t21 是分解之後的變換，描述了第二幀相機相對於第一幀相機的旋轉矩陣和平移向量。
        vP3D 記錄了三角化後的空間點坐標，vbTriangulated 則標記了各個特征點是否被成功三角化。
        最後兩個標量參數是兩個閾值，分別限定了視差和三角化特征點的數量，如果不能超過閾值則拒絕之。*/
        if (RH > 0.40)
        {
            // 『兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小』條件下，
            // 估計空間點 vP3D（過程中包含估計旋轉和平移）時，單應矩陣會有 8 種可能，比較各種可能的表現，挑選最佳的可能點
            return ReconstructH(vbMatchesInliersH, H, K, R21, t21, vP3D, vbTriangulated, 1.0, 50);
        }

        // if(pF_HF > 0.6)
        else
        {
            // 利用『基礎矩陣』估計空間點 vP3D（過程中包含估計旋轉和平移）時會有 4 種可能，比較各種可能的表現，
            // 挑選最佳的可能點
            return ReconstructF(vbMatchesInliersF, F, K, R21, t21, vP3D, vbTriangulated, 1.0, 50);
        }

        return false;
    }

    // 尋找單應性矩陣以及重投影得分，並區分關鍵點為內點還是外點
    void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
    {
        // Number of putative matches
        const int N = mvMatches12.size();

        // Normalize coordinates
        vector<cv::Point2f> vPn1, vPn2;

        // 包含尺度變量和偏移變量的 3 X 3 矩陣
        cv::Mat T1, T2;

        // mvKeys1：前一幀中提取的特征點（已校正）
        // 計算關鍵點的中心後，轉換為相對位置，並以距離中心點的平均距離作為規模控制的尺度變量
        Normalize(mvKeys1, vPn1, T1);

        // mvKeys2：當前幀中提取的特征點（已校正）
        // 計算關鍵點的中心後，轉換為相對位置，並以距離中心點的平均距離作為規模控制的尺度變量
        Normalize(mvKeys2, vPn2, T2);

        //
        cv::Mat T2inv = T2.inv();

        // Best Results variables
        score = 0.0;
        vbMatchesInliers = vector<bool>(N, false);

        // Iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat H21i, H12i, Hn;
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;
        int it, idx;
        size_t j;

        // Perform all RANSAC iterations and save the solution with highest score
        for (it = 0; it < mMaxIterations; it++)
        {
            // Select a minimum set
            for (j = 0; j < 8; j++)
            {
                idx = mvSets[it][j];

                // mvMatches12[idx]： 一組匹配成功的點對，關鍵點的各自的索引值
                // .first：前一幀的關鍵點的索引值; .second：當前幀的關鍵點的索引值
                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            // 利用八組隨機挑選的點對，計算單應性矩陣的中間矩陣
            Hn = ComputeH21(vPn1i, vPn2i);

            // 單應性矩陣
            H21i = T2inv * Hn * T1;

            H12i = H21i.inv();

            // currentScore 為投影誤差小於門檻值的程度，currentScore 越大表示誤差越小
            // 透過重投影誤差，區分是內點還是外點，並根據誤差程度衡量單應性矩陣（誤差越小，分數越高）
            currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

            if (currentScore > score)
            {
                H21 = H21i.clone();
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }

    // 計算關鍵點的中心後，轉換為相對位置，並以距離中心點的平均距離作為規模控制的尺度變量
    void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys,
                                vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
    {
        // 所有關鍵點的中心位置
        float meanX = 0;
        float meanY = 0;

        const int N = vKeys.size();

        vNormalizedPoints.resize(N);

        for (int i = 0; i < N; i++)
        {
            meanX += vKeys[i].pt.x;
            meanY += vKeys[i].pt.y;
        }

        meanX = meanX / N;
        meanY = meanY / N;

        // 所有關鍵點與中心位置的平均距離
        float meanDevX = 0;
        float meanDevY = 0;

        for (int i = 0; i < N; i++)
        {
            // vNormalizedPoints： 『相對於中心』的座標點
            vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
            vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

            meanDevX += fabs(vNormalizedPoints[i].x);
            meanDevY += fabs(vNormalizedPoints[i].y);
        }

        meanDevX = meanDevX / N;
        meanDevY = meanDevY / N;

        float sX = 1.0 / meanDevX;
        float sY = 1.0 / meanDevY;

        for (int i = 0; i < N; i++)
        {
            // 根據平均距離縮放，達到控制規模尺度的效果
            vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
            vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
        }

        T = cv::Mat::eye(3, 3, CV_32F);

        // 尺度變數
        T.at<float>(0, 0) = sX;
        T.at<float>(1, 1) = sY;

        // 偏移變數
        T.at<float>(0, 2) = -meanX * sX;
        T.at<float>(1, 2) = -meanY * sY;
    }

    // 計算單應性矩陣
    cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();

        cv::Mat A(2 * N, 9, CV_32F);

        for (int i = 0; i < N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;

            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(2 * i, 0) = 0.0;
            A.at<float>(2 * i, 1) = 0.0;
            A.at<float>(2 * i, 2) = 0.0;
            A.at<float>(2 * i, 3) = -u1;
            A.at<float>(2 * i, 4) = -v1;
            A.at<float>(2 * i, 5) = -1;
            A.at<float>(2 * i, 6) = v2 * u1;
            A.at<float>(2 * i, 7) = v2 * v1;
            A.at<float>(2 * i, 8) = v2;

            A.at<float>(2 * i + 1, 0) = u1;
            A.at<float>(2 * i + 1, 1) = v1;
            A.at<float>(2 * i + 1, 2) = 1;
            A.at<float>(2 * i + 1, 3) = 0.0;
            A.at<float>(2 * i + 1, 4) = 0.0;
            A.at<float>(2 * i + 1, 5) = 0.0;
            A.at<float>(2 * i + 1, 6) = -u2 * u1;
            A.at<float>(2 * i + 1, 7) = -u2 * v1;
            A.at<float>(2 * i + 1, 8) = -u2;
        }

        cv::Mat u, w, vt;

        // 將 A 利用 SVD 拆解成 w, u, vt 三個子矩陣
        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        return vt.row(8).reshape(0, 3);
    }

    // 透過重投影誤差，區分是內點還是外點，並根據誤差程度衡量『單應性矩陣』（誤差越小，分數越高）
    float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12,
                                       vector<bool> &vbMatchesInliers, float sigma)
    {
        const int N = mvMatches12.size();

        // H21： 單應性矩陣
        const float h11 = H21.at<float>(0, 0);
        const float h12 = H21.at<float>(0, 1);
        const float h13 = H21.at<float>(0, 2);
        const float h21 = H21.at<float>(1, 0);
        const float h22 = H21.at<float>(1, 1);
        const float h23 = H21.at<float>(1, 2);
        const float h31 = H21.at<float>(2, 0);
        const float h32 = H21.at<float>(2, 1);
        const float h33 = H21.at<float>(2, 2);

        // H12：單應性矩陣的逆
        const float h11inv = H12.at<float>(0, 0);
        const float h12inv = H12.at<float>(0, 1);
        const float h13inv = H12.at<float>(0, 2);
        const float h21inv = H12.at<float>(1, 0);
        const float h22inv = H12.at<float>(1, 1);
        const float h23inv = H12.at<float>(1, 2);
        const float h31inv = H12.at<float>(2, 0);
        const float h32inv = H12.at<float>(2, 1);
        const float h33inv = H12.at<float>(2, 2);

        vbMatchesInliers.resize(N);

        float score = 0;

        const float th = 5.991;

        const float invSigmaSquare = 1.0 / (sigma * sigma);

        for (int i = 0; i < N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;

            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // Reprojection error in first image
            // x2in1 = H12*x2

            const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
            const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
            const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

            const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

            const float chiSquare1 = squareDist1 * invSigmaSquare;

            if (chiSquare1 > th)
            {
                bIn = false;
            }
            else
            {
                score += th - chiSquare1;
            }

            // Reprojection error in second image
            // x1in2 = H21*x1

            const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
            const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
            const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

            const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

            const float chiSquare2 = squareDist2 * invSigmaSquare;

            if (chiSquare2 > th)
            {
                bIn = false;
            }
            else
            {
                score += th - chiSquare2;
            }

            // 誤差足夠小，則認定為內點
            if (bIn)
            {
                vbMatchesInliers[i] = true;
            }

            // 誤差過大，認定為 outlier
            else
            {
                vbMatchesInliers[i] = false;
            }
        }

        return score;
    }

    void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
    {
        // Number of putative matches
        const int N = vbMatchesInliers.size();

        // Normalize coordinates
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mvKeys1, vPn1, T1);
        Normalize(mvKeys2, vPn2, T2);

        cv::Mat T2t = T2.t();

        // Best Results variables
        score = 0.0;
        vbMatchesInliers = vector<bool>(N, false);

        // Iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat F21i, Fn;
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;
        int it, idx, j;

        // Perform all RANSAC iterations and save the solution with highest score
        for (it = 0; it < mMaxIterations; it++)
        {
            // Select a minimum set
            for (j = 0; j < 8; j++)
            {
                idx = mvSets[it][j];

                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            Fn = ComputeF21(vPn1i, vPn2i);

            F21i = T2t * Fn * T1;

            currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

            if (currentScore > score)
            {
                F21 = F21i.clone();
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }

    // 計算基礎矩陣
    cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();

        cv::Mat A(N, 9, CV_32F);

        for (int i = 0; i < N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(i, 0) = u2 * u1;
            A.at<float>(i, 1) = u2 * v1;
            A.at<float>(i, 2) = u2;
            A.at<float>(i, 3) = v2 * u1;
            A.at<float>(i, 4) = v2 * v1;
            A.at<float>(i, 5) = v2;
            A.at<float>(i, 6) = u1;
            A.at<float>(i, 7) = v1;
            A.at<float>(i, 8) = 1;
        }

        cv::Mat u, w, vt;

        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        cv::Mat Fpre = vt.row(8).reshape(0, 3);

        cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        w.at<float>(2) = 0;

        return u * cv::Mat::diag(w) * vt;
    }

    // 透過重投影誤差，區分是內點還是外點，並根據誤差程度衡量『基礎矩陣』（誤差越小，分數越高）
    float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
    {
        const int N = mvMatches12.size();

        const float f11 = F21.at<float>(0, 0);
        const float f12 = F21.at<float>(0, 1);
        const float f13 = F21.at<float>(0, 2);
        const float f21 = F21.at<float>(1, 0);
        const float f22 = F21.at<float>(1, 1);
        const float f23 = F21.at<float>(1, 2);
        const float f31 = F21.at<float>(2, 0);
        const float f32 = F21.at<float>(2, 1);
        const float f33 = F21.at<float>(2, 2);

        vbMatchesInliers.resize(N);

        float score = 0;

        const float th = 3.841;
        const float thScore = 5.991;

        const float invSigmaSquare = 1.0 / (sigma * sigma);

        for (int i = 0; i < N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // Reprojection error in second image
            // l2=F21x1=(a2,b2,c2)

            const float a2 = f11 * u1 + f12 * v1 + f13;
            const float b2 = f21 * u1 + f22 * v1 + f23;
            const float c2 = f31 * u1 + f32 * v1 + f33;

            const float num2 = a2 * u2 + b2 * v2 + c2;

            const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

            const float chiSquare1 = squareDist1 * invSigmaSquare;

            if (chiSquare1 > th)
            {
                bIn = false;
            }
            else
            {
                score += thScore - chiSquare1;
            }

            // Reprojection error in second image
            // l1 =x2tF21=(a1,b1,c1)

            const float a1 = f11 * u2 + f21 * v2 + f31;
            const float b1 = f12 * u2 + f22 * v2 + f32;
            const float c1 = f13 * u2 + f23 * v2 + f33;

            const float num1 = a1 * u1 + b1 * v1 + c1;

            const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

            const float chiSquare2 = squareDist2 * invSigmaSquare;

            if (chiSquare2 > th)
            {
                bIn = false;
            }
            else
            {
                score += thScore - chiSquare2;
            }

            if (bIn)
            {
                vbMatchesInliers[i] = true;
            }
            else
            {
                vbMatchesInliers[i] = false;
            }
        }

        return score;
    }

    // 『兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小』條件下，估計空間點 vP3D（過程中包含估計旋轉和平移）時，
    // 單應矩陣會有 8 種可能，比較各種可能的表現，挑選最佳的可能點
    bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                                   cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                                   vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        /*
        vbMatchesInliers： 特征點是否匹配的標記
        H21： 待分解的矩陣
        K： 相機的內參矩陣
        R21 和 t21 是分解之後的變換，描述了第二幀相機相對於第一幀相機的旋轉矩陣和平移向量。
        vP3D 記錄了三角化後的空間點坐標，vbTriangulated 則標記了各個特征點是否被成功三角化。
        最後兩個標量參數是兩個閾值，分別限定了視差和三角化特征點的數量，如果不能超過閾值則拒絕之。
        */

        // 內點的數量
        int N = 0;

        for(bool is_inlier : vbMatchesInliers){

            if (is_inlier)
            {
                N++;
            }
        }

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewise planar environment.
        // International Journal of Pattern Recognition and Artificial Intelligence, 1988

        cv::Mat invK = K.inv();
        cv::Mat A = invK * H21 * K;

        cv::Mat U, w, Vt, V;
        cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
        V = Vt.t();

        // 計算行列式相乘值
        float s = cv::determinant(U) * cv::determinant(Vt);

        float d1 = w.at<float>(0);
        float d2 = w.at<float>(1);
        float d3 = w.at<float>(2);

        if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
        {
            return false;
        }

        vector<cv::Mat> vR, vt, vn;
        vR.reserve(8);
        vt.reserve(8);
        vn.reserve(8);

        //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
        float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
        float x1[] = {aux1, aux1, -aux1, -aux1};
        float x3[] = {aux3, -aux3, aux3, -aux3};

        //case d'=d2
        float aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

        float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

        for (int i = 0; i < 4; i++)
        {
            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = ctheta;
            Rp.at<float>(0, 2) = -stheta[i];
            Rp.at<float>(2, 0) = stheta[i];
            Rp.at<float>(2, 2) = ctheta;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = -x3[i];
            tp *= d1 - d3;

            cv::Mat t = U * tp;
            vt.push_back(t / cv::norm(t));

            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;

            if (n.at<float>(2) < 0)
            {
                n = -n;
            }

            vn.push_back(n);
        }

        //case d'=-d2
        float aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

        float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

        for (int i = 0; i < 4; i++)
        {
            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = cphi;
            Rp.at<float>(0, 2) = sphi[i];
            Rp.at<float>(1, 1) = -1;
            Rp.at<float>(2, 0) = sphi[i];
            Rp.at<float>(2, 2) = -cphi;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = x3[i];
            tp *= d1 + d3;

            cv::Mat t = U * tp;
            vt.push_back(t / cv::norm(t));

            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;

            if (n.at<float>(2) < 0){
                n = -n;
            }

            vn.push_back(n);
        }

        int bestGood = 0;
        int secondBestGood = 0;
        int bestSolutionIdx = -1;
        float bestParallax = -1;
        vector<cv::Point3f> bestP3D;
        vector<bool> bestTriangulated;

        // Instead of applying the visibility constraints proposed in the Faugeras' paper
        // (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax
        // 挑選出 8 組種表現最好的那組
        for (size_t i = 0; i < 8; i++)
        {
            float parallaxi;
            vector<cv::Point3f> vP3Di;
            vector<bool> vbTriangulatedi;

            // 利用三角測量,計算空間點。確保兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小
            // nGood： 『兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小』的點的個數
            int nGood = CheckRT(vR[i], vt[i], mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K,
                                vP3Di, 4.0 * mSigma2, vbTriangulatedi, parallaxi);

            // 更新 bestGood 和 secondBestGood
            if (nGood > bestGood)
            {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
                bestParallax = parallaxi;

                // 更新空間點的估計
                bestP3D = vP3Di;

                bestTriangulated = vbTriangulatedi;
            }
            else if (nGood > secondBestGood)
            {
                secondBestGood = nGood;
            }
        }

        // 若兩相機間有足夠的夾角，且符合要求的點數足夠多
        if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax &&
            bestGood > minTriangulated && bestGood > 0.9 * N)
        {
            vR[bestSolutionIdx].copyTo(R21);
            vt[bestSolutionIdx].copyTo(t21);
            vP3D = bestP3D;
            vbTriangulated = bestTriangulated;

            return true;
        }

        return false;
    }

    // 利用三角測量,計算空間點。確保兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小
    int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1,
                             const vector<cv::KeyPoint> &vKeys2, const vector<Match> &vMatches12,
                             vector<bool> &vbMatchesInliers, const cv::Mat &K,
                             vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood,
                             float &parallax)
    {
        // Calibration parameters
        const float fx = K.at<float>(0, 0);
        const float fy = K.at<float>(1, 1);
        const float cx = K.at<float>(0, 2);
        const float cy = K.at<float>(1, 2);

        vbGood = vector<bool>(vKeys1.size(), false);
        vP3D.resize(vKeys1.size());

        vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // Camera 1 Projection Matrix K[I|0]
        cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
        K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

        // Camera 1 相機原點
        cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

        // Camera 2 Projection Matrix K[R|t]
        cv::Mat P2(3, 4, CV_32F);
        R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        t.copyTo(P2.rowRange(0, 3).col(3));
        P2 = K * P2;

        // Camera 2 相機原點
        cv::Mat O2 = -R.t() * t, p3dC1, p3dC2, normal1, normal2;
        float dist1, dist2, cosParallax, im1x, im1y, invZ1, squareError1, 
              im2x, im2y, invZ2, squareError2;

        int nGood = 0;

        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            // 若為 outlier
            if (!vbMatchesInliers[i])
            {
                continue;
            }

            const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
            const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
            
            // 進行三角測量，獲得空間點 p3dC1 (x, y, z)
            Triangulate(kp1, kp2, P1, P2, p3dC1);

            // 若任一值為無限（無窮遠），則 vbGood 標注為 false
            if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) ||
                !isfinite(p3dC1.at<float>(2)))
            {
                vbGood[vMatches12[i].first] = false;
                continue;
            }

            // Check parallax
            normal1 = p3dC1 - O1;
            dist1 = cv::norm(normal1);

            normal2 = p3dC1 - O2;
            dist2 = cv::norm(normal2);

            // 利用餘弦定理，計算夾角
            cosParallax = normal1.dot(normal2) / (dist1 * dist2);

            // Check depth in front of first camera (only if enough parallax,
            // as "infinite" points can easily go to negative depth)
            // cosParallax 接近 1 表示幾乎無夾角
            if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
            {
                continue;
            }

            // Check depth in front of second camera (only if enough parallax,
            // as "infinite" points can easily go to negative depth)
            // Camera 1 下的 p3dC1 轉換到 Camera 2 下的 p3dC2
            p3dC2 = R * p3dC1 + t;

            if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
            {
                continue;
            }

            // Check reprojection error in first image
            invZ1 = 1.0 / p3dC1.at<float>(2);
            im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
            im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

            // 參考幀上的重投影誤差
            squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) +
                           (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

            if (squareError1 > th2)
            {
                continue;
            }

            // Check reprojection error in second image
            im2x, im2y;
            invZ2 = 1.0 / p3dC2.at<float>(2);
            im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
            im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

            // 當前幀上的重投影誤差
            squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) +
                           (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

            if (squareError2 > th2)
            {
                continue;
            }

            vCosParallax.push_back(cosParallax);
            vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1),
                                                    p3dC1.at<float>(2));

            // 兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小
            nGood++;

            if (cosParallax < 0.99998)
            {
                vbGood[vMatches12[i].first] = true;
            }
        }

        if (nGood > 0)
        {
            sort(vCosParallax.begin(), vCosParallax.end());

            size_t idx = min(50, int(vCosParallax.size() - 1));

            // 弧度 轉 角度 => * 180 / CV_PI
            parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
        }
        else
        {
            parallax = 0;
        }

        return nGood;
    }

    // 進行三角測量，獲得空間點 x3D (x, y, z)
    void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1,
                                  const cv::Mat &P2, cv::Mat &x3D)
    {
        cv::Mat A(4, 4, CV_32F);

        A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
        A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
        A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
        A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        x3D = vt.row(3).t();

        // 三角測量所得空間點
        // 四元數 轉 (x, y, z)
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
    }

    // 利用『基礎矩陣』估計空間點 vP3D（過程中包含估計旋轉和平移）時會有 4 種可能，比較各種可能的表現，挑選最佳的可能點
    bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                                   cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D,
                                   vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        /*
        1.vbMatchesInliers： 特征點是否匹配的標記
        2.F21： 待分解的矩陣
        3.K： 相機的內參矩陣
        R21 和 t21 是分解之後的變換，描述了第二幀相機相對於第一幀相機的旋轉矩陣和平移向量。
        vP3D 記錄了三角化後的空間點坐標，vbTriangulated 則標記了各個特征點是否被成功三角化。
        最後兩個標量參數是兩個閾值，分別限定了視差和三角化特征點的數量，如果不能超過閾值則拒絕之。
        */

        // 內點的數量
        int N = 0;

        for(bool is_inlier : vbMatchesInliers){

            if (is_inlier)
            {
                N++;
            }
        }

        // Compute Essential Matrix from Fundamental Matrix
        // 利用『基礎矩陣Fundamental Matrix』計算『本質矩陣 Essential Matrix』
        cv::Mat E21 = K.t() * F21 * K;

        cv::Mat R1, R2, t;

        // Recover the 4 motion hypotheses
        // 將『本質矩陣』拆解成兩種方向的旋轉(R1, R2)和一種平移(t)
        DecomposeE(E21, R1, R2, t);

        cv::Mat t1 = t;
        cv::Mat t2 = -t;

        // Reconstruct with the 4 hypotheses and check
        vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
        vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
        float parallax1, parallax2, parallax3, parallax4;

        // CheckRT：利用三角測量,計算空間點。確保兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小
        int nGood1 = CheckRT(R1, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, 
                             vP3D1, 4.0 * mSigma2, vbTriangulated1, parallax1);
        int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, 
                             vP3D2, 4.0 * mSigma2, vbTriangulated2, parallax2);
        int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, 
                             vP3D3, 4.0 * mSigma2, vbTriangulated3, parallax3);
        int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, 
                             vP3D4, 4.0 * mSigma2, vbTriangulated4, parallax4);

        int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

        R21 = cv::Mat();
        t21 = cv::Mat();

        int nMinGood = max(static_cast<int>(0.9 * N), minTriangulated);

        int nsimilar = 0;

        if (nGood1 > 0.7 * maxGood){
            nsimilar++;
        }

        if (nGood2 > 0.7 * maxGood){
            nsimilar++;
        }

        if (nGood3 > 0.7 * maxGood){
            nsimilar++;
        }
        
        if (nGood4 > 0.7 * maxGood){
            nsimilar++;
        }

        // If there is not a clear winner or not enough triangulated points reject initialization
        // 若 nGoodX > 0.7 * maxGood 表示最多的那組沒有明顯多於第二名，拒絕該初始化
        if (maxGood < nMinGood || nsimilar > 1)
        {
            return false;
        }

        // If best reconstruction has enough parallax initialize
        if (maxGood == nGood1)
        {
            if (parallax1 > minParallax)
            {
                vP3D = vP3D1;
                vbTriangulated = vbTriangulated1;

                R1.copyTo(R21);
                t1.copyTo(t21);

                return true;
            }
        }
        else if (maxGood == nGood2)
        {
            if (parallax2 > minParallax)
            {
                vP3D = vP3D2;
                vbTriangulated = vbTriangulated2;

                R2.copyTo(R21);
                t1.copyTo(t21);

                return true;
            }
        }
        else if (maxGood == nGood3)
        {
            if (parallax3 > minParallax)
            {
                vP3D = vP3D3;
                vbTriangulated = vbTriangulated3;

                R1.copyTo(R21);
                t2.copyTo(t21);

                return true;
            }
        }
        else if (maxGood == nGood4)
        {
            if (parallax4 > minParallax)
            {
                vP3D = vP3D4;
                vbTriangulated = vbTriangulated4;

                R2.copyTo(R21);
                t2.copyTo(t21);
                
                return true;
            }
        }

        return false;
    }

    // 將『本質矩陣』拆解成兩種方向的旋轉(R1, R2)和一種平移(t)
    void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
    {
        cv::Mat u, w, vt;
        cv::SVD::compute(E, w, u, vt);

        u.col(2).copyTo(t);
        t = t / cv::norm(t);

        cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
        W.at<float>(0, 1) = -1;
        W.at<float>(1, 0) = 1;
        W.at<float>(2, 2) = 1;

        R1 = u * W * vt;

        // 若行列式小於 0 則將旋轉矩陣乘以 -1
        if (cv::determinant(R1) < 0){
            R1 = -R1;
        }

        R2 = u * W.t() * vt;

        if (cv::determinant(R2) < 0){
            R2 = -R2;
        }
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    
} //namespace ORB_SLAM
