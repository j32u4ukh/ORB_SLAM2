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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/StdVector>

#include "Converter.h"

#include <mutex>

namespace ORB_SLAM2
{
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    // 根據區域的共視關係，取出關鍵幀與地圖點來進行多次優化，優化後的誤差若仍過大的估計會被移除，並更新估計結果
    // 『共視關鍵幀』、『共視關鍵幀的共視關鍵幀』、『共視地圖點』作為『頂點』
    // 『觀察到共視地圖點的特徵點的位置』作為『邊』
    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
    {
        // Local KeyFrames: First Breath Search from Current Keyframe
        list<KeyFrame *> lLocalKeyFrames;

        lLocalKeyFrames.push_back(pKF);

        // 標注 pKF 已參與 pKF 的 LocalBundleAdjustment
        pKF->mnBALocalForKF = pKF->mnId;

        // 『關鍵幀 pKF』的共視關鍵幀（根據觀察到的地圖點數量排序）
        const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();

        // 篩選好的『關鍵幀 pKF』的共視關鍵幀

        for(KeyFrame *pKFi : vNeighKFs)
        {
            // 標注『關鍵幀 pKFi』已參與『關鍵幀 pKF』的 LocalBundleAdjustment，避免重複添加到 lLocalKeyFrames
            pKFi->mnBALocalForKF = pKF->mnId;

            if (!pKFi->isBad()){
                lLocalKeyFrames.push_back(pKFi);
            }
        }

        // Local MapPoints seen in Local KeyFrames
        // 共視地圖點：『共視關鍵幀 list<KeyFrame *> lLocalKeyFrames』所觀察到的地圖點
        list<MapPoint *> lLocalMapPoints;

        for(KeyFrame * local_kf : lLocalKeyFrames)
        {
            // 『關鍵幀 (*lit)』的關鍵點觀察到的地圖點
            vector<MapPoint *> vpMPs = local_kf->GetMapPointMatches();

            for(MapPoint *pMP : vpMPs)
            {
                if (pMP)
                {
                    if (!pMP->isBad())
                    {
                        if (pMP->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            // 標注『地圖點 pMP』已參與『關鍵幀 pKF』的 LocalBundleAdjustment
                            // 避免重複添加到 lLocalMapPoints
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        // 哪些關鍵幀同時也觀察到 lLocalMapPoints 當中的地圖點
        list<KeyFrame *> lFixedCameras;

        for(MapPoint* local_mp : lLocalMapPoints)
        {
            // 觀察到『共視地圖點 (*lit)』的關鍵幀，及其對應的特徵點的索引值
            map<KeyFrame *, size_t> observations = local_mp->GetObservations();

            for(pair<KeyFrame *, size_t> obs : observations)
            {
                KeyFrame *pKFi = obs.first;

                // Local && Fixed
                // 檢查『關鍵幀 pKFi』是否已參與『關鍵幀 pKF』的 LocalBundleAdjustment
                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;

                    if (!pKFi->isBad()){
                        lFixedCameras.push_back(pKFi);
                    }
                }
            }
        }

        // g2o 的優化會分別使用到 lLocalKeyFrames, lLocalMapPoints, lFixedCameras

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver;
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if (pbStopFlag)
        {
            optimizer.setForceStopFlag(pbStopFlag);
        }

        unsigned long maxKFid = 0;

        // Set Local KeyFrame vertices

        for(KeyFrame *pKFi : lLocalKeyFrames)
        {
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);

            // Local 共視關鍵幀只有第 0 幀設為 Fixed，Fixed 共視關鍵幀全部都設為 Fixed
            vSE3->setFixed(pKFi->mnId == 0);

            // Local 共視關鍵幀作為頂點加入優化        
            optimizer.addVertex(vSE3);

            if (pKFi->mnId > maxKFid)
            {
                maxKFid = pKFi->mnId;
            }
        }
        
        list<KeyFrame *>::iterator lit, lend;

        // Set Fixed KeyFrame vertices
        // Fixed 共視關鍵幀作為頂點加入優化

        for(KeyFrame *pKFi : lFixedCameras)
        {
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);

            // Fixed 共視關鍵幀全部都設為 Fixed，Local 共視關鍵幀只有第 0 幀設為 Fixed
            vSE3->setFixed(true);

            optimizer.addVertex(vSE3);

            if (pKFi->mnId > maxKFid)
            {
                maxKFid = pKFi->mnId;
            }
        }

        // Set MapPoint vertices
        // （『Local 關鍵幀個數』 + 『Fixed 關鍵幀個數』） * 『Local 關鍵幀所觀察到的地圖點個數』
        const int nExpectedSize = (lLocalKeyFrames.size() +  lFixedCameras.size()) * 
                                   lLocalMapPoints.size();

        vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        // 『共視地圖點』作為『頂點』，而『觀察到共視地圖點的特徵點的位置』作為『邊』
        for(MapPoint *pMP : lLocalMapPoints)
        {
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);

            // 『Local 共視關鍵幀所觀察到的地圖點』作為頂點加入優化
            optimizer.addVertex(vPoint);

            // 和 list<KeyFrame *> lFixedCameras 區塊很相似，但前面區塊只取出關鍵幀而已
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            // Set edges
            for(pair<KeyFrame *, size_t> obs : observations)
            {
                KeyFrame *pKFi = obs.first;

                if (!pKFi->isBad())
                {
                    size_t kp_idx = obs.second;

                    // 根據索引值，取得已校正的關鍵點
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[kp_idx];

                    // 單目的這個數值會是負的
                    const float kp_ur = pKFi->mvuRight[kp_idx];

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    // Monocular observation
                    if (kp_ur < 0)
                    {
                        // 已校正的關鍵點的位置，作為『邊』加入優化
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();
                        e->setVertex(0,
                            dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, 
                            dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                                                        optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }

                    // Stereo observation（非單目，暫時跳過）
                    else 
                    {
                        Eigen::Matrix<double, 3, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();
                        e->setVertex(0, 
                            dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, 
                            dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                                                        optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);
                    }
                }            
            }
        }
        
        if (pbStopFlag)
        {
            if (*pbStopFlag)
            {
                return;
            }
        }

        optimizer.initializeOptimization();

        // 優化 5 次
        optimizer.optimize(5);

        bool bDoMore = true;

        if (pbStopFlag)
        {
            if (*pbStopFlag)
            {
                bDoMore = false;
            }
        }

        if (bDoMore)
        {
            // Check inlier observations
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
                MapPoint *pMP = vpMapPointEdgeMono[i];

                if (pMP->isBad()){
                    continue;
                }

                // 『誤差較大』或『深度不為正』的邊
                if (e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    // 再次納入優化
                    e->setLevel(1);
                }

                // 不使用 RobustKernel
                e->setRobustKernel(0);
            }

            // 和單目無關，暫時跳過
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
                MapPoint *pMP = vpMapPointEdgeStereo[i];

                if (pMP->isBad()){
                    continue;
                }

                if (e->chi2() > 7.815 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            // Optimize again without the outliers

            optimizer.initializeOptimization(0);

            // 再優化 10 次
            optimizer.optimize(10);
        }

        vector<pair<KeyFrame *, MapPoint *>> vToErase;
        vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad()){
                continue;
            }

            // 『誤差較大』或『深度不為正』的邊
            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFMono[i];

                // 標注為要移除的 (KeyFrame, MapPoint)
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        // 和單目無關，暫時跳過
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad()){
                continue;
            }

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // 將前一步驟標注要移除的內容給實際移除
        if (!vToErase.empty())
        {
            for(pair<KeyFrame *, MapPoint *> to_earse : vToErase)
            {                
                KeyFrame *pKFi = to_earse.first;
                MapPoint *pMPi = to_earse.second;

                // 從當前關鍵幀觀察到的地圖點當中移除『地圖點 pMPi』，表示其實沒有觀察到
                pKFi->EraseMapPointMatch(pMPi);  

                // 移除『關鍵幀 pKFi』，更新關鍵幀的計數，若『觀察到這個地圖點的關鍵幀』太少（少於 3 個），
                // 則將地圖點與關鍵幀等全部移除
                pMPi->EraseObservation(pKFi);
            }
        }

        // ==================================================
        // ==================================================
        // 更新估計值 Recover optimized data

        // Keyframes
        // 利用 Local 關鍵幀的 mnId 取出相對應的頂點，並更新自身的位姿估計
        for(KeyFrame *pKF : lLocalKeyFrames)
        {
            g2o::VertexSE3Expmap *vSE3 = 
                                    static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }

        // Points
        // 依序取出地圖點之頂點，並更新地圖點的位置
        for(MapPoint *pMP : lLocalMapPoints)
        {
            // 為何地圖點的頂點 ID 會是 pMP->mnId + maxKFid + 1？
            g2o::VertexSBAPointXYZ *vPoint = 
                    static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            
            // 更新地圖點的位置
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            
            // 利用所有觀察到這個地圖點的關鍵幀來估計關鍵幀們平均指向的方向，以及該地圖點可能的深度範圍(最近與最遠)
            pMP->UpdateNormalAndDepth();
        }
    }

    // 將『地圖點 pMP1、pMP2』的位置轉換到相機座標系下，作為『頂點』加入優化，相對應的特徵點位置作為『邊』加入，
    // 優化並排除誤差過大的估計後，重新估計『相似轉換矩陣』
    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, 
                                g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver;
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Calibration 相機內參
        const cv::Mat &K1 = pKF1->K;
        const cv::Mat &K2 = pKF2->K;

        // Camera poses
        const cv::Mat R1w = pKF1->GetRotation();
        const cv::Mat t1w = pKF1->GetTranslation();

        const cv::Mat R2w = pKF2->GetRotation();
        const cv::Mat t2w = pKF2->GetTranslation();

        // Set Sim3 vertex
        g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
        vSim3->_fix_scale = bFixScale;

        // g2oS12 相似轉換矩陣
        vSim3->setEstimate(g2oS12);

        vSim3->setId(0);
        vSim3->setFixed(false);

        // 相機內參
        vSim3->_principle_point1[0] = K1.at<float>(0, 2);
        vSim3->_principle_point1[1] = K1.at<float>(1, 2);

        vSim3->_focal_length1[0] = K1.at<float>(0, 0);
        vSim3->_focal_length1[1] = K1.at<float>(1, 1);

        vSim3->_principle_point2[0] = K2.at<float>(0, 2);
        vSim3->_principle_point2[1] = K2.at<float>(1, 2);
        
        vSim3->_focal_length2[0] = K2.at<float>(0, 0);
        vSim3->_focal_length2[1] = K2.at<float>(1, 1);

        // 『相似轉換矩陣 g2oS12』封裝到 vSim3 當中，作為頂點加入優化
        optimizer.addVertex(vSim3);

        // Set MapPoint vertices
        const int N = vpMatches1.size();

        // 『關鍵幀 pKF1』觀察到的地圖點
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();

        vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
        vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
        vector<size_t> vnIndexEdge;

        vnIndexEdge.reserve(2 * N);
        vpEdges12.reserve(2 * N);
        vpEdges21.reserve(2 * N);

        const float deltaHuber = sqrt(th2);

        int nCorrespondences = 0;

        for (int i = 0; i < N; i++)
        {
            if (!vpMatches1[i]){
                continue;
            }

            // 『地圖點 pMP1』：『關鍵幀 pKF1』觀察到的第 i 個地圖點
            MapPoint *pMP1 = vpMapPoints1[i];

            // 『地圖點 pMP2』：『關鍵幀 pKF1』的第 i 個地圖點在『關鍵幀 pKF2』上對應的地圖點
            MapPoint *pMP2 = vpMatches1[i];

            // i = 0, id1 = 1, id2 = 2; i = 1, id1 = 3, id2 = 4
            // id1: 1, 3, 5, ...; id2: 2, 4, 6, ...
            const int id1 = 2 * i + 1;
            const int id2 = 2 * (i + 1);

            // 『關鍵幀 pKF2』的第 i2 個特徵點觀察到『地圖點 pMP2』
            const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

            // 將『地圖點 pMP1、pMP2』的位置轉換到相機座標系下，作為『頂點』（id1, id2）加入優化
            if (pMP1 && pMP2)
            {
                if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0)
                {
                    g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();

                    // 取得『地圖點 pMP1』的世界座標
                    cv::Mat P3D1w = pMP1->GetWorldPos();

                    // 將『地圖點 pMP1』轉換到『關鍵幀 pKF1』座標系下
                    cv::Mat P3D1c = R1w * P3D1w + t1w;

                    // 將『關鍵幀 pKF1』座標系下的『地圖點 pMP1』的位置轉換成向量表示
                    vPoint1->setEstimate(Converter::toVector3d(P3D1c));

                    vPoint1->setId(id1);
                    vPoint1->setFixed(true);

                    // 『地圖點 pMP1』的位置作為『頂點』加入優化
                    optimizer.addVertex(vPoint1);

                    g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();

                    // 取得『地圖點 pMP2』的世界座標
                    cv::Mat P3D2w = pMP2->GetWorldPos();

                    // 將『地圖點 pMP2』轉換到『關鍵幀 pKF2』座標系下
                    cv::Mat P3D2c = R2w * P3D2w + t2w;

                    // 將『關鍵幀 pKF2』座標系下的『地圖點 pMP2』的位置轉換成向量表示
                    vPoint2->setEstimate(Converter::toVector3d(P3D2c));

                    vPoint2->setId(id2);
                    vPoint2->setFixed(true);

                    // 『地圖點 pMP2』的位置作為『頂點』加入優化
                    optimizer.addVertex(vPoint2);
                }
                else{
                    continue;
                }
            }
            else{
                continue;
            }

            nCorrespondences++;

            // Set edge x1 = S12*X2
            // 『關鍵幀 pKF1』的第 i 個特徵點位置作為『邊』加入優化 
            Eigen::Matrix<double, 2, 1> obs1;
            const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
            obs1 << kpUn1.pt.x, kpUn1.pt.y;

            g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e12->setMeasurement(obs1);

            const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
            e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e12);

            // Set edge x2 = S21*X1
            // 『關鍵幀 pKF2』的第 i2 個特徵點位置作為『邊』加入優化 
            Eigen::Matrix<double, 2, 1> obs2;
            const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;

            g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();

            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e21->setMeasurement(obs2);
            float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
            e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            optimizer.addEdge(e21);

            // 『關鍵幀 pKF1』的第 i 個特徵點位置
            vpEdges12.push_back(e12);

            // 『關鍵幀 pKF2』的第 i2 個特徵點位置
            vpEdges21.push_back(e21);

            vnIndexEdge.push_back(i);
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        // Check inliers
        int nBad = 0;

        for (size_t i = 0; i < vpEdges12.size(); i++)
        {
            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];

            if (!e12 || !e21){
                continue;
            }

            // 優化後，誤差仍十分大
            if (e12->chi2() > th2 || e21->chi2() > th2)
            {
                size_t idx = vnIndexEdge[i];

                // 移除誤差過大的『邊』和『地圖點』
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
                vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);

                // 外點計數加一
                nBad++;
            }
        }

        int nMoreIterations;

        if (nBad > 0){
            nMoreIterations = 10;
        }
        else{
            nMoreIterations = 5;
        }

        if (nCorrespondences - nBad < 10){
            return 0;
        }

        // Optimize again only with inliers
        // 再次優化
        optimizer.initializeOptimization();
        optimizer.optimize(nMoreIterations);

        int nIn = 0;

        for (size_t i = 0; i < vpEdges12.size(); i++)
        {
            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];

            if (!e12 || !e21){
                continue;
            }

            // 再優化後，誤差仍十分大
            if (e12->chi2() > th2 || e21->chi2() > th2)
            {
                size_t idx = vnIndexEdge[i];

                // 移除誤差過大的『地圖點』
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            }
            else{
                nIn++;
            }
        }

        // Recover optimized Sim3
        g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));

        // 更新對『相似轉換矩陣 g2oS12』的估計
        g2oS12 = vSim3_recov->estimate();

        return nIn;
    }

    // 『當前關鍵幀』轉換到『共視關鍵幀』的『相似轉換矩陣』作為頂點;『共視關鍵幀』之間的轉換的『相似轉換矩陣』作為『邊』
    // 優化後，重新估計各個『相似轉換矩陣』以及地圖點的位置
    void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                           const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                           const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                           const map<KeyFrame *, set<KeyFrame *>> &LoopConnections, 
                                           const bool &bFixScale)
    {
        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);

        g2o::BlockSolver_7_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
        g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver;
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        solver->setUserLambdaInit(1e-16);
        optimizer.setAlgorithm(solver);

        const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

        // 取得最大 KFid（應該約等價於關鍵幀個數，關鍵幀可能被丟棄，但 KFid 會繼續計數）
        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
        vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

        const int minFeat = 100;

        // Set KeyFrame vertices
        // 遍歷所有關鍵幀

        for(KeyFrame *pKF : vpKFs)
        {
            if (pKF->isBad()){
                continue;
            }

            g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();
            const int nIDi = pKF->mnId;

            // 『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系對應的『相似轉換矩陣』
            LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

            if (it != CorrectedSim3.end())
            {
                // 根據關鍵幀的 Id 對應『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系對應的『相似轉換矩陣』
                vScw[nIDi] = it->second;
                // 估計初始值
                VSim3->setEstimate(it->second);
            }
            else
            {
                Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
                Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
                g2o::Sim3 Siw(Rcw, tcw, 1.0);
                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);
            }

            if (pKF == pLoopKF)
            {
                VSim3->setFixed(true);
            }

            VSim3->setId(nIDi);
            VSim3->setMarginalized(false);

            // 單目的規模尺度不固定
            VSim3->_fix_scale = bFixScale;

            // 關鍵幀的『相似轉換矩陣』作為『頂點』加入優化
            optimizer.addVertex(VSim3);

            vpVertices[nIDi] = VSim3;
        }
        
        set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

        const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

        // Set Loop edges
        // LoopConnections[pKFi]：『關鍵幀 pKFi』的『已連結關鍵幀』，移除『關鍵幀 pKFi』的『共視關鍵幀』和
        // 『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
        for(pair<KeyFrame *, set<KeyFrame *>> loop_onnection : LoopConnections)
        {
            KeyFrame *pKF = loop_onnection.first;

            // 『關鍵幀 pKFi』的『已連結關鍵幀』，移除『關鍵幀 pKFi』的『共視關鍵幀』和
            // 『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
            const set<KeyFrame *> &spConnections = loop_onnection.second;

            const long unsigned int nIDi = pKF->mnId;

            // 根據關鍵幀的 Id 對應『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系對應的『相似轉換矩陣』
            const g2o::Sim3 Siw = vScw[nIDi];

            // 『關鍵幀 pKFi』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
            const g2o::Sim3 Swi = Siw.inverse();

            for(KeyFrame * sit : spConnections)
            {
                const long unsigned int nIDj = sit->mnId;  
                             
                if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(sit) < minFeat)
                {
                    continue;
                }
                
                // Sjw：『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』
                const g2o::Sim3 Sjw = vScw[nIDj];

                // Swi：『關鍵幀 pKFi』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
                // Sji:『關鍵幀 pKF_i』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』
                const g2o::Sim3 Sji = Sjw * Swi;

                // 『邊』連結兩個『關鍵幀』的『相似轉換矩陣』的頂點
                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                
                // Sji:『關鍵幀 pKF_i』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』作為『邊』加入優化
                e->setMeasurement(Sji);
                e->information() = matLambda;
                optimizer.addEdge(e);

                sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
            }
        }

        // Set normal edges
        for(KeyFrame *pKF : vpKFs)
        {            
            const int nIDi = pKF->mnId;
            g2o::Sim3 Swi;

            // NonCorrectedSim3[pKF]：『關鍵幀 pKFi』對應的『相似轉換矩陣』
            LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

            if (iti != NonCorrectedSim3.end())
            {
                Swi = (iti->second).inverse();
            }

            // 應該只有最後一幀會進來這裡
            else
            {
                // vScw[nIDi]：『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKF_i』座標系對應的『相似轉換矩陣』
                Swi = vScw[nIDi].inverse();
            }

            KeyFrame *pParentKF = pKF->GetParent();

            // Spanning tree edge
            if (pParentKF)
            {
                int nIDj = pParentKF->mnId;
                g2o::Sim3 Sjw;
                LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);
                
                if (itj != NonCorrectedSim3.end())
                {
                    // 『關鍵幀 pParentKF』的『相似轉換矩陣』
                    Sjw = itj->second;
                }
                else
                {
                    // vScw[nIDj]：『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』
                    Sjw = vScw[nIDj];
                }

                // Swi：『關鍵幀 pKFi』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
                // Sji:『關鍵幀 pKF_i』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』
                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);
                e->information() = matLambda;

                optimizer.addEdge(e);
            }
            // Loop edges
            const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();

            for(KeyFrame * pLKF : sLoopEdges)
            {
                if (pLKF->mnId < pKF->mnId)
                {
                    g2o::Sim3 Slw;
                    LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                    if (itl != NonCorrectedSim3.end())
                    {
                        Slw = itl->second;
                    }
                    else{
                        Slw = vScw[pLKF->mnId];
                    }

                    g2o::Sim3 Sli = Slw * Swi;

                    g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                    el->setVertex(1, 
                        dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, 
                        dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;

                    optimizer.addEdge(el);
                }
            }

            // Covisibility graph edges
            // 取得至多 minFeat 個『共視關鍵幀』
            // 取得『關鍵幀 pKF』的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序，
            // 且觀察到的地圖點數量「大於」 minFeat）』
            const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);

            for(KeyFrame *pKFn : vpConnectedKFs)
            {
                if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
                {
                    if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                    {
                        if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), 
                                                           max(pKF->mnId, pKFn->mnId)))){
                            continue;
                        }

                        g2o::Sim3 Snw;
                        LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);
                        
                        if (itn != NonCorrectedSim3.end()){
                            Snw = itn->second;
                        }
                        else
                        {
                            Snw = vScw[pKFn->mnId];
                        }

                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                        en->setVertex(1, 
                            dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, 
                            dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;

                        optimizer.addEdge(en);
                    }
                }
            }
        }
        
        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(20);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for(KeyFrame *pKFi : vpKFs)
        {
            const int nIDi = pKFi->mnId;

            g2o::VertexSim3Expmap *VSim3 = 
                                    static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));

            // 優化後重新估計『相似轉換矩陣』
            g2o::Sim3 CorrectedSiw = VSim3->estimate();

            // 『關鍵幀 pKF_i』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();

            // 優化後重新估計『旋轉矩陣』
            Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();

            // 優化後重新估計『平移』
            Eigen::Vector3d eigt = CorrectedSiw.translation();

            // 優化後重新估計『規模尺度』
            double s = CorrectedSiw.scale();

            eigt *= (1. / s); //[R t/s;0 1]

            // 優化後重新估計『校正規模』的『轉換矩陣』
            cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(Tiw);
        }

        // Correct points. Transform to "non-optimized" reference keyframe pose and transform back 
        // with optimized pose
        for(MapPoint *pMP : vpMPs)
        {            
            if (pMP->isBad()){
                continue;
            }

            int nIDr;

            if (pMP->mnCorrectedByKF == pCurKF->mnId)
            {
                nIDr = pMP->mnCorrectedReference;
            }
            else
            {
                KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }

            g2o::Sim3 Srw = vScw[nIDr];

            // 『關鍵幀 pKF_i』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            cv::Mat P3Dw = pMP->GetWorldPos();
            Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
            Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
            
            // 優化後重新估計『地圖點』的位置
            cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);

            pMP->SetWorldPos(cvCorrectedP3Dw);
            pMP->UpdateNormalAndDepth();
        }
    }

    // 從地圖中取出所有『關鍵幀』和『地圖點』，進行 BundleAdjustment
    void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag, 
                                           const unsigned long nLoopKF, const bool bRobust)
    {
        // 取出所有『關鍵幀』
        vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

        // 取出所有『地圖點』
        vector<MapPoint *> vpMP = pMap->GetAllMapPoints();

        // 關鍵幀和地圖點作為『頂點』，而關鍵點的位置作為『邊』，優化後重新估計關鍵幀的位姿與地圖點的位置
        BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
    }

    // 關鍵幀和地圖點作為『頂點』，而關鍵點的位置作為『邊』，優化後重新估計關鍵幀的位姿與地圖點的位置
    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                     int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, 
                                     const bool bRobust)
    {
        vector<bool> vbNotIncludedMP;
        vbNotIncludedMP.resize(vpMP.size());

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver;
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);                                    
        optimizer.setAlgorithm(solver);

        if (pbStopFlag){
            optimizer.setForceStopFlag(pbStopFlag);
        }

        long unsigned int maxKFid = 0;

        // Set KeyFrame vertices
        // 將『關鍵幀』的位姿，作為『頂點』加入優化，Id 由 0 到 maxKFid 編號
        for(KeyFrame * pKF : vpKFs)
        {
            if (pKF->isBad()){
                continue;
            }

            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();

            // 取得『關鍵幀 pKF』的位姿，並轉換成 g2o 所需的類型 
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(pKF->mnId);
            vSE3->setFixed(pKF->mnId == 0);

            // 『關鍵幀 pKF』的位姿，作為『頂點』加入優化
            optimizer.addVertex(vSE3);

            if (pKF->mnId > maxKFid)
            {
                maxKFid = pKF->mnId;
            }
        }

        const float thHuber2D = sqrt(5.99);
        const float thHuber3D = sqrt(7.815);
        MapPoint *pMP;
        KeyFrame *pKF;
        size_t kp_idx;

        // Set MapPoint vertices
        // 『地圖點』的座標作為『頂點』加入優化
        for (size_t i = 0; i < vpMP.size(); i++)
        {
            pMP = vpMP[i];

            if (pMP->isBad()){
                continue;
            }

            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();

            // 『地圖點 pMP』的世界座標作為 vPoint 的初始值
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

            /// NOTE: 『地圖點 pMP』的 Id + 最大的『關鍵幀』 Id + 1， Id 由 maxKFid + 1 開始編號
            const int id = pMP->mnId + maxKFid + 1;

            vPoint->setId(id);
            vPoint->setMarginalized(true);

            // 『地圖點 pMP』的座標作為『頂點』加入優化
            optimizer.addVertex(vPoint);

            // 觀察到『地圖點 pMP』的『關鍵幀』，以及其『關鍵點』的索引值
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            int nEdges = 0;
            
            // SET EDGES
            for(pair<KeyFrame *, size_t> obs : observations)
            {
                pKF = obs.first;
                kp_idx = obs.second;

                if (pKF->isBad() || pKF->mnId > maxKFid){
                    continue;
                }

                nEdges++;

                // 『關鍵幀 pKF』的第 kp_idx 個『關鍵點 kpUn』觀察到『地圖點 pMP』
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[kp_idx];

                // 單目的 mvuRight 會是負的
                if (pKF->mvuRight[kp_idx] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    // 『邊』連接第 mnId 個關鍵幀和第 mnId 個地圖點
                    // （地圖點接續關鍵幀的 id 編號，因此是由 maxKFid + 1 開始編號）
                    e->setVertex(0, 
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, 
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                                                        optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    if (bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    // 『關鍵點 kpUn』作為『邊』加入優化
                    optimizer.addEdge(e);
                }

                // 非單目，暫時跳過
                else
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKF->mvuRight[kp_idx];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();
                    e->setVertex(0, 
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, 
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                                                        optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);
                    
                    if (bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber3D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
                    e->bf = pKF->mbf;
                    optimizer.addEdge(e);
                }
            }
            
            if (nEdges == 0)
            {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i] = true;
            }
            else
            {
                vbNotIncludedMP[i] = false;
            }
        }

        // Optimize!
        optimizer.initializeOptimization();

        // 優化 nIterations 次
        optimizer.optimize(nIterations);

        // Recover optimized data

        // Keyframes
        // 更新為優化後的位姿
        g2o::VertexSE3Expmap *vSE3;
        g2o::SE3Quat SE3quat;

        for(KeyFrame *kf : vpKFs)
        {
            if (kf->isBad()){
                continue;
            }

            vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(kf->mnId));
            
            // 估計優化後的位姿
            SE3quat = vSE3->estimate();

            // nLoopKF：關鍵幀 Id，也就是只有第一次才會直接存在『關鍵幀 pKF』的位姿中
            if (nLoopKF == 0)
            {
                kf->SetPose(Converter::toCvMat(SE3quat));
            }

            // 第二次開始會先存在 mTcwGBA 當中，之後才會在 LoopClosing::RunGlobalBundleAdjustment 用來更新位姿
            else
            {
                kf->mTcwGBA.create(4, 4, CV_32F);

                // 優化後的位姿估計存在 pKF->mTcwGBA，而非直接存在『關鍵幀 pKF』的位姿中
                Converter::toCvMat(SE3quat).copyTo(kf->mTcwGBA);

                kf->mnBAGlobalForKF = nLoopKF;
            }
        }
        
        // Points
        // 更新為優化後的估計點位置
        g2o::VertexSBAPointXYZ *vPoint;

        for (size_t i = 0; i < vpMP.size(); i++)
        {
            if (vbNotIncludedMP[i]){
                continue;
            }

            pMP = vpMP[i];

            if (pMP->isBad()){
                continue;
            }

            vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

            if (nLoopKF == 0)
            {
                // 更新『地圖點 pMP』的位置估計
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));

                // 利用所有觀察到這個地圖點的關鍵幀來估計關鍵幀們平均指向的方向，以及該地圖點可能的深度範圍(最近與最遠)
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3, 1, CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    }

    // 優化『pFrame 觀察到的地圖點』的位置，以及 pFrame 的位姿估計，並返回優化後的內點個數
    int Optimizer::PoseOptimization(Frame *pFrame)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver;
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        // 將當前幀轉換成『轉換矩陣 SE3』，並作為頂點加入 optimizer
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        // 取得 pFrame 觀察到的地圖點個數
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);

        // 取出 pFrame 觀察到的地圖點的位置，作為『邊』加入 optimizer
        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++)
            {
                // 依序取得 pFrame 觀察到的地圖點
                MapPoint *pMP = pFrame->mvpMapPoints[i];

                if (pMP)
                {
                    // 單目的這個值，會是負的
                    const float &kp_ur = pFrame->mvuRight[i];
                        
                    // 地圖點可被觀察到，因此不是 Outlier
                    pFrame->mvbOutlier[i] = false;

                    nInitialCorrespondences++;

                    // 取得與地圖點相對應的 pFrame 的特徵點
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];

                    // Monocular observation
                    if (kp_ur < 0)
                    {
                        // 特徵點的位置
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        // 一元邊：重投影誤差（僅用於優化相機位姿時）
                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, 
                                    dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        rk->setDelta(deltaMono);
                        e->setRobustKernel(rk);

                        // 相機內參
                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;

                        // 圖點的世界座標
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }

                    // Stereo observation（暫時跳過）
                    else 
                    {
                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = 
                                                            new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, 
                                    dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;

                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }
            }
        }

        // 若地圖點的數量太少（少於 3）
        if (nInitialCorrespondences < 3){
            return 0;
        }

        // We perform 4 optimizations, after each optimization we classify observation as 
        // inlier/outlier. At the next optimization, outliers are not included, but at the end 
        // they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

        // 定義每輪優化要優化幾次（每輪優化後會『將足夠好的邊設為無須再優化』，才再次進行優化）
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;

        // 優化『pFrame 觀察到的地圖點』的位置
        for (size_t it = 0; it < 4; it++)
        {
            // 將 pFrame 位姿轉型成『轉換矩陣』，作為 vSE3 的初始估計值
            /// NOTE: 似乎沒有在每次優化後更新 pFrame 的位姿，只在此函式結束前的最後更新而已
            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);

            // 優化 its[it] 次
            optimizer.optimize(its[it]);

            // 
            nBad = 0;

            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                // 取出 optimizer 的第 i 個『邊』
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                // 若為第 idx 個特徵點為 Outlier
                if (pFrame->mvbOutlier[idx])
                {
                    // 利用『邊』計算誤差
                    e->computeError();
                }

                // chi2() 基於 computeError() 所計算的誤差，取得 chi2 值
                const float chi2 = e->chi2();

                // 若 chi2 值比預設門檻高
                if (chi2 > chi2Mono[it])
                {
                    // 將第 idx 個特徵點認定為 Outlier
                    pFrame->mvbOutlier[idx] = true;

                    // 持續優化這個『邊』
                    e->setLevel(1);

                    nBad++;
                }

                // 若 chi2 值比預設門檻低
                else
                {
                    // 將第 idx 個特徵點認定為內點
                    pFrame->mvbOutlier[idx] = false;

                    // 不再優化這個『邊』
                    e->setLevel(0);
                }

                // 第 3 輪(it == 2)以後，無須使用 RobustKernel
                if (it == 2)
                {
                    e->setRobustKernel(0);
                }
            }

            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2){
                    e->setRobustKernel(0);
                }
            }

            // 若 optimizer 還需要優化的『邊』足夠少（少於 10 個），則結束優化
            if (optimizer.edges().size() < 10){
                break;
            }
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);

        // 更新 pFrame 的位姿
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    
} //namespace ORB_SLAM
