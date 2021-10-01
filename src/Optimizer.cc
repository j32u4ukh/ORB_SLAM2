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
    const float Optimizer::thHuber2D = sqrt(5.99);
    const float Optimizer::thHuber3D = sqrt(7.815);
    const float Optimizer::thHuberMono = sqrt(5.991);
    const float Optimizer::thHuberStereo = sqrt(7.815);
    const float Optimizer::deltaMono = sqrt(5.991);
    const float Optimizer::deltaStereo = sqrt(7.815);
    const Eigen::Matrix<double, 7, 7> Optimizer::matLambda = Eigen::Matrix<double, 7, 7>::Identity();
    const int Optimizer::start_idx = 1570;
    const int Optimizer::end_idx = 1600;
        
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

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
        // std::cout << "Start Optimizer::BundleAdjustment" << std::endl;

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

        unsigned long maxKFid = 0;
        
        // Set KeyFrame vertices
        // 將『關鍵幀』的位姿，作為『頂點』加入優化，Id 由 0 到 maxKFid 編號
        addKeyFramePoses(vpKFs, optimizer, maxKFid);

        addMapPoints(vpMP, optimizer, maxKFid, bRobust, vbNotIncludedMP);
        
        // Optimize!
        optimizer.initializeOptimization();

        // 優化 nIterations 次
        if(optimizer.vertices().size() == 0 || optimizer.edges().size() == 0)
        {
            std::cout << "[BundleAdjustment] #vertices: " 
                    << optimizer.vertices().size() 
                    << ", #edges: " << optimizer.edges().size() << std::endl;
        }
                  
        optimizer.optimize(nIterations);

        // Recover optimized data

        // ================================================================================
        // ================================================================================
        // 以上為『優化』、以下為『更新優化結果』
        // ================================================================================
        // ================================================================================
        updateKeyFramePoses(vpKFs, optimizer, nLoopKF);

        updateMapPoints(vpMP, vbNotIncludedMP, optimizer, maxKFid, nLoopKF);
    }

    /// NOTE: 20210906
    // 根據區域的共視關係，取出關鍵幀與地圖點來進行多次優化，優化後的誤差若仍過大的估計會被移除，並更新估計結果
    // 『共視關鍵幀』、『共視關鍵幀的共視關鍵幀』、『共視地圖點』作為『頂點』
    // 『觀察到共視地圖點的特徵點的位置』作為『邊』
    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
    {
        // std::cout << "Start Optimizer::LocalBundleAdjustment" << std::endl;

        // Local KeyFrames: First Breath Search from Current Keyframe
        list<KeyFrame *> lLocalKeyFrames;

        extractLocalKeyFrames(lLocalKeyFrames, pKF);

        // Local MapPoints seen in Local KeyFrames
        // 共視地圖點：『共視關鍵幀 list<KeyFrame *> lLocalKeyFrames』所觀察到的地圖點
        list<MapPoint *> lLocalMapPoints;

        extractLocalMapPoints(lLocalMapPoints, lLocalKeyFrames, pKF);

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        // 哪些關鍵幀同時也觀察到 lLocalMapPoints 當中的地圖點
        list<KeyFrame *> lFixedCameras;

        extractFixedCameras(lFixedCameras, lLocalMapPoints, pKF);
        
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
        unsigned long id;

        /// NOTE: [addLocalKeyFrames] Added vertices / All vertices: 7/7
        addLocalKeyFrames(lLocalKeyFrames, optimizer, maxKFid);

        // Set Fixed KeyFrame vertices
        // Fixed 共視關鍵幀作為頂點加入優化
        /// NOTE: [addFixedCameras] Added vertices / All vertices: 11/11
        /// 極少數其況發生 addFixedCameras 沒有頂點，但同時的 addLocalKeyFrames 都不為 0
        addFixedCameras(lFixedCameras, optimizer, maxKFid);
        
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
        
        /// NOTE: [addLocalMapPoints] Added edges / All edges: 5050/5050
        addLocalMapPoints(lLocalMapPoints, optimizer, maxKFid, pKF, 
                          vpEdgesMono, vpEdgeKFMono, vpMapPointEdgeMono,
                          vpEdgesStereo, vpEdgeKFStereo, vpMapPointEdgeStereo); 

        if (pbStopFlag)
        {
            if (*pbStopFlag)
            {
                return;
            }
        }

        optimizer.initializeOptimization();

        if(optimizer.vertices().size() == 0 || optimizer.edges().size() == 0)
        {
            std::cout << "[LocalBundleAdjustment - 1] #vertices: " 
                    << optimizer.vertices().size() 
                    << ", #edges: " << optimizer.edges().size() << std::endl;
        }

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
            filterMonoLocalMapPoints(vpEdgesMono, vpMapPointEdgeMono);

            // 和單目無關，暫時跳過
            // filterStereoLocalMapPoints(vpEdgesStereo, vpMapPointEdgeStereo);

            // Optimize again without the outliers
            optimizer.initializeOptimization(0);

            if(optimizer.vertices().size() == 0 || optimizer.edges().size() == 0)
            {
                std::cout << "[LocalBundleAdjustment - 2] #vertices: " 
                        << optimizer.vertices().size() 
                        << ", #edges: " << optimizer.edges().size() << std::endl;
            }

            // 再優化 10 次
            optimizer.optimize(10);
        }

        vector<pair<KeyFrame *, MapPoint *>> vToErase;
        vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());
        
        markEarseMono(vToErase, vpMapPointEdgeMono, vpEdgesMono, vpEdgeKFMono);
        
        // markEarseStereo(vToErase, vpMapPointEdgeStereo, vpEdgesStereo, vpEdgeKFStereo);
        
        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // 將前一步驟標注要移除的內容給實際移除
        if (!vToErase.empty())
        {            
            executeEarsing(vToErase);
        }

        // ================================================================================
        // ================================================================================
        // 以上為『優化』、以下為『更新優化結果』
        // ================================================================================
        // ================================================================================

        updateLocalKeyFrames(optimizer, lLocalKeyFrames);
        
        updateLocalMapPoints(optimizer, lLocalMapPoints, maxKFid);        
    }

    /// NOTE: 20210907
    // 將『地圖點 pMP1、pMP2』的位置轉換到相機座標系下，作為『頂點』加入優化，相對應的特徵點位置作為『邊』加入，
    // 優化並排除誤差過大的估計後，重新估計『相似轉換矩陣』
    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, 
                                g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
    {
        // std::cout << "Start Optimizer::OptimizeSim3" << std::endl;

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver;
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Set MapPoint vertices
        const int N = vpMatches1.size();

        /// NOTE: [OptimizeSim3] vpMatches1.size: 3641
        // std::cout << "[OptimizeSim3] vpMatches1.size: " << N << std::endl;
        
        vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
        vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
        vector<size_t> vnIndexEdge;

        vnIndexEdge.reserve(2 * N);
        vpEdges12.reserve(2 * N);
        vpEdges21.reserve(2 * N);

        int nCorrespondences = 0;
        
        addSim3MapPointsAndKeyPoints(pKF1, pKF2, bFixScale, g2oS12, optimizer, N, th2, 
                                     nCorrespondences, vpEdges12, vpEdges21, vnIndexEdge, vpMatches1);

        // Optimize!
        optimizer.initializeOptimization();
        
        if(optimizer.vertices().size() == 0 || optimizer.edges().size() == 0)
        {
            std::cout << "[OptimizeSim3 - 1] #vertices: " 
                    << optimizer.vertices().size() 
                    << ", #edges: " << optimizer.edges().size() << std::endl;
        }

        optimizer.optimize(5);

        // Check inliers
        int nBad = filterSim3Outlier(optimizer, th2, vpMatches1, vpEdges12, vpEdges21, vnIndexEdge);        
        int nMoreIterations;

        if (nBad > 0)
        {
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

        if(optimizer.vertices().size() == 0 || optimizer.edges().size() == 0)
        {
            std::cout << "[OptimizeSim3 - 2] #vertices: " 
                    << optimizer.vertices().size() 
                    << ", #edges: " << optimizer.edges().size() << std::endl;
        }

        optimizer.optimize(nMoreIterations);

        int nIn = filterSim3Inlier(optimizer, th2, vpMatches1, vpEdges12, vpEdges21, vnIndexEdge);
        // std::cout << "[OptimizeSim3] nIn: " << nIn << std::endl;

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
        // std::cout << "Start Optimizer::OptimizeEssentialGraph" << std::endl;

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

        const int minFeat = 100;

        /// NOTE: [addEssentialSim3] Added vertices / All vertices: 315/315
        addEssentialSim3(optimizer, pLoopKF, vpKFs, CorrectedSim3, vScw, nMaxKFid, bFixScale);

        set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

        /// NOTE: [addEssentialLoopConnections] Added edges / All edges: 51/165
        addEssentialLoopConnections(optimizer, sInsertedEdges, LoopConnections, vScw, minFeat, 
                                    pLoopKF, pCurKF);
        
        /// NOTE: addEssentialEdges當中：addEdgeSim3/addCovisibilityEdges 皆不為 0; addLoopEdges 全部為 0
        addEssentialEdges(vpKFs, optimizer, NonCorrectedSim3, vScw, minFeat, sInsertedEdges);
        
        // Optimize!
        optimizer.initializeOptimization();

        if(optimizer.vertices().size() == 0 || optimizer.edges().size() == 0)
        {
            std::cout << "[OptimizeEssentialGraph] #vertices: " 
                    << optimizer.vertices().size() 
                    << ", #edges: " << optimizer.edges().size() << std::endl;
        }

        optimizer.optimize(20);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        updateEssentialPoses(vpKFs, optimizer, vCorrectedSwc);

        updateEssentialMapPoints(vpMPs, pCurKF, vScw, vCorrectedSwc);
    }

    // 優化『pFrame 觀察到的地圖點』的位置，以及 pFrame 的位姿估計，並返回優化後的內點個數
    int Optimizer::PoseOptimization(Frame *pFrame, const int idx)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver;
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Set Frame vertex
        // 將當前幀轉換成『轉換矩陣 SE3』，並作為頂點加入 optimizer
        g2o::VertexSE3Expmap *vSE3 = addVertexSE3Expmap(optimizer, pFrame->mTcw, 0, false);

        // if(start_idx <= idx && idx <= end_idx)
        // {
        //     cv::Mat mat = pFrame->mTcw.colRange(0, 3).rowRange(0, 3);
        //     Eigen::Matrix<double, 3, 3> matrix = Converter::toMatrix3d(mat);
        //     Eigen::Vector3d euler = matrix.eulerAngles(2, 1, 0);

        //     cv::Mat translate = pFrame->mTcw.col(3).rowRange(0, 3);

        //     std::cout << "Optimizer::PoseOptimization, euler: " << euler.transpose() 
        //               << ", translate:" << translate.t() << std::endl;
        // }
        
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

        int nInitialCorrespondences = 0, i, n_mp, n_outlier;

        // 取出 pFrame 觀察到的地圖點的位置，作為『邊』加入 optimizer
        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            // if(start_idx <= idx && idx <= end_idx)
            // {
            //     n_mp = 0, n_outlier = 0;        

            //     for(i = 0; i < pFrame->N; i++)
            //     {
            //         if(pFrame->mvpMapPoints[i])
            //         {
            //             n_mp++;

            //             if(pFrame->mvbOutlier[i])
            //             {
            //                 n_outlier++;
            //             }
            //         }
            //     }

            //     std::cout << "[PoseOptimization] Before optimizer, n_mp: " << n_mp 
            //               << ", n_outlier: " << n_outlier << std::endl;
            // }
            
            addPoseOptimization(pFrame, nInitialCorrespondences, optimizer, vnIndexEdgeMono, 
                                vpEdgesMono, vpEdgesStereo, vnIndexEdgeStereo, idx);

            // if(start_idx <= idx && idx <= end_idx)
            // {
            //     n_mp = 0, n_outlier = 0;        

            //     for(i = 0; i < pFrame->N; i++)
            //     {
            //         if(pFrame->mvpMapPoints[i])
            //         {
            //             n_mp++;

            //             if(pFrame->mvbOutlier[i])
            //             {
            //                 n_outlier++;
            //             }
            //         }
            //     }

            //     std::cout << "[PoseOptimization] After addPoseOptimization, n_mp: " << n_mp 
            //               << ", n_outlier: " << n_outlier << std::endl;
            // }
        }

        // 若地圖點的數量太少（少於 3）
        if (nInitialCorrespondences < 3){
            return 0;
        }

        int nBad = addPoseOptimizationEdges(vSE3, pFrame, optimizer, vpEdgesMono, vnIndexEdgeMono,
                                            vpEdgesStereo, vnIndexEdgeStereo, idx);

        // if(start_idx <= idx && idx <= end_idx)
        // {
        //     n_mp = 0, n_outlier = 0;        

        //     for(i = 0; i < pFrame->N; i++)
        //     {
        //         if(pFrame->mvpMapPoints[i])
        //         {
        //             n_mp++;

        //             if(pFrame->mvbOutlier[i])
        //             {
        //                 n_outlier++;
        //             }
        //         }
        //     }

        //     std::cout << "[PoseOptimization] After addPoseOptimizationEdges, n_mp: " << n_mp 
        //               << ", n_outlier: " << n_outlier << std::endl;
        // }
        
        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);

        // 更新 pFrame 的位姿
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    // ==================================================
    // 自己封裝的函式
    // ==================================================

    // ***** Optimizer::BundleAdjustment *****

    /// NOTE: 20210906
    // 提取出 KeyFrame 的 Pose，加入優化 SparseOptimizer
    void Optimizer::addKeyFramePoses(const vector<KeyFrame *> &vpKFs, g2o::SparseOptimizer &op, 
                                     unsigned long &maxKFid)
    {
        // Set KeyFrame vertices
        // 將『關鍵幀』的位姿，作為『頂點』加入優化，Id 由 0 到 maxKFid 編號
        vector<KeyFrame *>::const_iterator i_start, i_end = vpKFs.end();
        unsigned long id;
        KeyFrame *pKF;

        for(i_start = vpKFs.begin(); i_start != i_end; i_start++)
        {
            pKF = *i_start;

            if (pKF->isBad())
            {
                continue;
            }

            id = pKF->mnId;

            addVertexSE3Expmap(op, pKF->GetPose(), id, id == 0);

            if (id > maxKFid)
            {
                maxKFid = id;
            }
        }
    }

    /// NOTE: 20210906
    void Optimizer::addMapPoints(const vector<MapPoint *> &vpMP, g2o::SparseOptimizer &op,
                                 const unsigned long maxKFid,
                                 const bool bRobust, vector<bool> &vbNotIncludedMP)
    {
        /* 會造成 0 vertices to optimize 的原因
        1. pMP->isBad()
        2. pKF->isBad()
        3. pKF->mnId > maxKFid
        */
        g2o::VertexSBAPointXYZ *vPoint;
        MapPoint *pMP;
        KeyFrame *pKF;
        size_t kp_idx;
        unsigned long id;
        cv::KeyPoint kpUn;

        // Set MapPoint vertices
        // 『地圖點』的座標作為『頂點』加入優化
        for (size_t i = 0; i < vpMP.size(); i++)
        {
            pMP = vpMP[i];

            if (pMP->isBad())
            {
                // std::cout << "pMP->isBad():" << pMP->isBad() << std::endl;
                continue;
            }

            // 觀察到『地圖點 pMP』的『關鍵幀』，以及其『關鍵點』的索引值
            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            vbNotIncludedMP[i] = true;

            // 將是否將有 Edge 被添加的判斷，移到 Vertex 建構之前，避免建構後又要移除
            for (pair<KeyFrame *, size_t> obs : observations)
            {
                pKF = obs.first;

                if (pKF->isBad() || pKF->mnId > maxKFid)
                {
                    continue;
                }

                // 只要有 Edge 將被添加，就不是 vbNotIncludedMP，就可以正式建構 Vertex 並添加 Edge
                vbNotIncludedMP[i] = false;
                break;
            }

            // 若沒有地圖點被加入，則跳過此次迴圈
            if(vbNotIncludedMP[i]){
                continue;
            }
            
            // 建構 Vertex
            id = pMP->mnId + maxKFid + 1;
            vPoint = newVertexSBAPointXYZ(pMP->GetWorldPos(), id);
            vPoint->setMarginalized(true);

            // 『地圖點 pMP』的座標作為『頂點』加入優化
            op.addVertex(vPoint);

            // SET EDGES
            for (pair<KeyFrame *, size_t> obs : observations)
            {
                pKF = obs.first;
                
                if (pKF->isBad() || pKF->mnId > maxKFid)
                {
                    // std::cout << "pMP->isBad():" << pMP->isBad()
                    //           << ", pKF->mnId: " << pKF->mnId 
                    //           << ", maxKFid: " << maxKFid
                    //           << std::endl;
                    continue;
                }

                kp_idx = obs.second;

                // 『關鍵幀 pKF』的第 kp_idx 個『關鍵點 kpUn』觀察到『地圖點 pMP』
                // const cv::KeyPoint &kpUn = pKF->mvKeysUn[kp_idx];
                kpUn = pKF->mvKeysUn[kp_idx];

                addEdgeSE3ProjectXYZ(op, kpUn, pKF, id, pKF->mnId, bRobust, thHuber2D);

                // // 單目的 mvuRight 會是負的
                // if (pKF->mvuRight[kp_idx] < 0)
                // {
                //     addEdgeSE3ProjectXYZ(op, kpUn, pKF, id, pKF->mnId, bRobust, thHuber2D);
                // }
                // // 非單目，暫時跳過
                // else
                // {
                //     addEdgeStereoSE3ProjectXYZ(op, kpUn, pKF, kp_idx, id, pKF->mnId, bRobust, thHuber3D);
                // }
            }
        }
    }

    /// NOTE: 20210906
    void Optimizer::updateKeyFramePoses(const vector<KeyFrame *> &vpKFs, g2o::SparseOptimizer &op,
                                        const unsigned long nLoopKF)
    {
        // Keyframes
        // 更新為優化後的位姿
        g2o::VertexSE3Expmap *vSE3;
        // g2o::SE3Quat SE3quat;
        cv::Mat mat;

        for(KeyFrame *kf : vpKFs)
        {
            if (kf->isBad())
            {
                continue;
            }

            vSE3 = static_cast<g2o::VertexSE3Expmap *>(op.vertex(kf->mnId));
            
            // 估計優化後的位姿
            // SE3quat = vSE3->estimate();
            mat = Converter::toCvMat(vSE3->estimate());

            // nLoopKF：關鍵幀 Id，也就是只有第一次才會直接存在『關鍵幀 pKF』的位姿中
            if (nLoopKF == 0)
            {
                // kf->SetPose(Converter::toCvMat(SE3quat));
                kf->SetPose(mat);                
            }

            // 第二次開始會先存在 mTcwGBA 當中，之後才會在 LoopClosing::RunGlobalBundleAdjustment 用來更新位姿
            else
            {
                kf->mTcwGBA.create(4, 4, CV_32F);

                // 優化後的位姿估計存在 pKF->mTcwGBA，而非直接存在『關鍵幀 pKF』的位姿中
                // Converter::toCvMat(SE3quat).copyTo(kf->mTcwGBA);
                mat.copyTo(kf->mTcwGBA);

                kf->mnBAGlobalForKF = nLoopKF;
            }
        }
    }

    /// NOTE: 20210907
    void Optimizer::updateMapPoints(const vector<MapPoint *> &vpMP, vector<bool> &vbNotIncludedMP, 
                                    g2o::SparseOptimizer &op, long unsigned int &maxKFid, 
                                    const unsigned long nLoopKF)
    {
        MapPoint *pMP;
        unsigned long id;
        g2o::VertexSBAPointXYZ *vPoint;
        size_t i, len = vpMP.size();
        cv::Mat mat;

        // Points
        // 更新為優化後的估計點位置
        for (i = 0; i < len; i++)
        {
            if (vbNotIncludedMP[i])
            {
                continue;
            }

            pMP = vpMP[i];

            if (pMP->isBad())
            {
                continue;
            }

            id = pMP->mnId + maxKFid + 1;
            vPoint = static_cast<g2o::VertexSBAPointXYZ *>(op.vertex(id));

            mat = Converter::toCvMat(vPoint->estimate());

            if (nLoopKF == 0)
            {
                // 更新『地圖點 pMP』的位置估計
                pMP->SetWorldPos(mat);

                // 利用所有觀察到這個地圖點的關鍵幀來估計關鍵幀們平均指向的方向，以及該地圖點可能的深度範圍(最近與最遠)
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3, 1, CV_32F);
                // Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                mat.copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    }


    // ***** Optimizer::LocalBundleAdjustment *****
    /// NOTE: 20210907
    void Optimizer::extractLocalKeyFrames(list<KeyFrame *> &lLocalKeyFrames, KeyFrame *pKF)
    {
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
            
            // if (!pKFi || pKFi->isBad())
            if (pKFi->isBad())
            {
                continue;
            }

            lLocalKeyFrames.push_back(pKFi);
        }
    }

    /// NOTE: 20210907
    void Optimizer::extractLocalMapPoints(list<MapPoint *> &lLocalMapPoints,
                                          const list<KeyFrame *> &lLocalKeyFrames, const KeyFrame *pKF)
    {
        vector<MapPoint *> vpMPs;

        for(KeyFrame * local_kf : lLocalKeyFrames)
        {
            // 『關鍵幀 (*lit)』的關鍵點觀察到的地圖點
            vpMPs = local_kf->GetMapPointMatches();

            // 將巢式 if 判斷式改為多次判斷，效果相同但更於閱讀
            for(MapPoint *pMP : vpMPs)
            {
                if (!pMP || pMP->isBad())
                {
                    continue;
                }

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

    /// NOTE: 20210907
    void Optimizer::extractFixedCameras(list<KeyFrame *> &lFixedCameras, 
                                        const list<MapPoint *> &lLocalMapPoints, const KeyFrame *pKF)
    {
        map<KeyFrame *, size_t> observations;
        KeyFrame *pKFi;

        for(MapPoint* local_mp : lLocalMapPoints)
        {
            // 觀察到『共視地圖點 (*lit)』的關鍵幀，及其對應的特徵點的索引值
            observations = local_mp->GetObservations();

            for(pair<KeyFrame *, size_t> obs : observations)
            {
                pKFi = obs.first;

                // Local && Fixed
                // 檢查『關鍵幀 pKFi』是否已參與『關鍵幀 pKF』的 LocalBundleAdjustment
                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;

                    if (!pKFi->isBad())
                    {
                        lFixedCameras.push_back(pKFi);
                    }
                }
            }
        }
    }

    /// NOTE: 20210907
    void Optimizer::addLocalKeyFrames(list<KeyFrame *> &lLocalKeyFrames, g2o::SparseOptimizer &op,
                                      long unsigned int &maxKFid)
    {
        int id, n_vertex = 0;

        for(KeyFrame *kf : lLocalKeyFrames)
        {
            id = kf->mnId;

            addVertexSE3Expmap(op, kf->GetPose(), id, id == 0);
            n_vertex++;

            if (id > maxKFid)
            {
                maxKFid = id;
            }
        }
    }

    /// NOTE: 20210907
    void Optimizer::addFixedCameras(list<KeyFrame *> &lFixedCameras, g2o::SparseOptimizer &op,
                                    long unsigned int &maxKFid)
    {
        int id;

        for (KeyFrame *kf : lFixedCameras)
        {
            id = kf->mnId;

            addVertexSE3Expmap(op, kf->GetPose(), id, true);

            if (id > maxKFid)
            {
                maxKFid = id;
            }
        }
    }

    /// NOTE: 20210907
    void Optimizer::addLocalMapPoints(list<MapPoint *> &lLocalMapPoints, g2o::SparseOptimizer &op, 
                                      long unsigned int &maxKFid, KeyFrame *pKF,
                                      vector<g2o::EdgeSE3ProjectXYZ *> &vpEdgesMono,
                                      vector<KeyFrame *> &vpEdgeKFMono,
                                      vector<MapPoint *> &vpMapPointEdgeMono,
                                      vector<g2o::EdgeStereoSE3ProjectXYZ *> &vpEdgesStereo,
                                      vector<KeyFrame *> &vpEdgeKFStereo,
                                      vector<MapPoint *> &vpMapPointEdgeStereo)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e_stereo;
        g2o::VertexSBAPointXYZ *vPoint;
        g2o::EdgeSE3ProjectXYZ *e;
        KeyFrame *kf;
        size_t kp_idx;
        int id, n_edge = 0, all_edge = 0;

        for(MapPoint *mp : lLocalMapPoints)
        {
            id = mp->mnId + maxKFid + 1;
            vPoint = newVertexSBAPointXYZ(mp->GetWorldPos(), id);
            vPoint->setMarginalized(true);

            // 『Local 共視關鍵幀所觀察到的地圖點』作為頂點加入優化
            op.addVertex(vPoint);

            // 和 list<KeyFrame *> lFixedCameras 區塊很相似，但前面區塊只取出關鍵幀而已
            const map<KeyFrame *, size_t> observations = mp->GetObservations();
            all_edge += observations.size();

            // Set edges
            for(pair<KeyFrame *, size_t> obs : observations)
            {
                kf = obs.first;

                if (kf->isBad())
                {
                    continue;
                }

                kp_idx = obs.second;

                // 根據索引值，取得已校正的關鍵點
                const cv::KeyPoint &kpUn = kf->mvKeysUn[kp_idx];

                // 單目的這個數值會是負的
                // const float kp_ur = kf->mvuRight[kp_idx];

                const float &invSigma2 = kf->mvInvLevelSigma2[kpUn.octave];

                // Monocular observation
                e = addEdgeSE3ProjectXYZ(op, kpUn, kf, id, kf->mnId, true, thHuberMono);
                n_edge++;

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(kf);
                vpMapPointEdgeMono.push_back(mp);

                // // Monocular observation
                // if (kp_ur < 0)
                // {
                //     e = addEdgeSE3ProjectXYZ(op, kpUn, kf, id, kf->mnId, true, thHuberMono);
                //     n_edge++;
                //     vpEdgesMono.push_back(e);
                //     vpEdgeKFMono.push_back(kf);
                //     vpMapPointEdgeMono.push_back(mp);
                // }
                // // Stereo observation（非單目，暫時跳過）
                // else 
                // {
                //     e_stereo = addEdgeStereoSE3ProjectXYZ(op, kpUn, pKF, kp_idx, 
                //                                             id, pKF->mnId, true, thHuberStereo);
                                                            
                //     vpEdgesStereo.push_back(e_stereo);
                //     vpEdgeKFStereo.push_back(kf);
                //     vpMapPointEdgeStereo.push_back(mp);
                // }         
            }
        }
    }

    /// NOTE: 20210907
    void Optimizer::filterMonoLocalMapPoints(vector<g2o::EdgeSE3ProjectXYZ *> &vpEdgesMono,
                                      vector<MapPoint *> &vpMapPointEdgeMono)
    {
        g2o::EdgeSE3ProjectXYZ *e;
        MapPoint *pMP;
        int n_edge = 0;
        size_t i, iend = vpEdgesMono.size();

        // Check inlier observations
        for (i = 0; i < iend; i++)
        {
            pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad()){
                continue;
            }

            e = vpEdgesMono[i];                

            // 『誤差較大』或『深度不為正』的邊
            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                // 再次納入優化
                e->setLevel(1);
                n_edge++;
            }

            // 不使用 RobustKernel
            e->setRobustKernel(0);
        }
    }

    /// NOTE: 20210907
    void Optimizer::markEarseMono(vector<pair<KeyFrame *, MapPoint *>> &vToErase,
                                  vector<MapPoint *> &vpMapPointEdgeMono,
                                  vector<g2o::EdgeSE3ProjectXYZ *> &vpEdgesMono,
                                  vector<KeyFrame *> &vpEdgeKFMono)
    {
        g2o::EdgeSE3ProjectXYZ *e;
        MapPoint *mp;
        KeyFrame *kf;

        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            mp = vpMapPointEdgeMono[i];

            if (mp->isBad())
            {
                continue;
            }

            e = vpEdgesMono[i];

            // 『誤差較大』或『深度不為正』的邊
            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                kf = vpEdgeKFMono[i];

                // 標注為要移除的 (KeyFrame, MapPoint)
                vToErase.push_back(make_pair(kf, mp));
            }
        }
    }

    /// NOTE: 20210907
    void Optimizer::executeEarsing(vector<pair<KeyFrame *, MapPoint *>> &vToErase)
    {
        KeyFrame *pKFi;
        MapPoint *pMP;

        for(pair<KeyFrame *, MapPoint *> to_earse : vToErase)
        {                
            pKFi = to_earse.first;
            pMP = to_earse.second;

            // 從當前關鍵幀觀察到的地圖點當中移除『地圖點 pMPi』，表示其實沒有觀察到
            pKFi->EraseMapPointMatch(pMP);  

            // 移除『關鍵幀 pKFi』，更新關鍵幀的計數，若『觀察到這個地圖點的關鍵幀』太少（少於 3 個），
            // 則將地圖點與關鍵幀等全部移除
            pMP->EraseObservation(pKFi);
        }
    }

    /// NOTE: 20210907
    void Optimizer::updateLocalKeyFrames(g2o::SparseOptimizer &op, list<KeyFrame *> &lLocalKeyFrames)
    {
        // 更新估計值 Recover optimized data
        g2o::VertexSE3Expmap *vSE3;
        g2o::SE3Quat SE3quat;
        cv::Mat mat;

        // Keyframes
        // 利用 Local 關鍵幀的 mnId 取出相對應的頂點，並更新自身的位姿估計
        for(KeyFrame *kf : lLocalKeyFrames)
        {
            vSE3 = static_cast<g2o::VertexSE3Expmap *>(op.vertex(kf->mnId));
            // SE3quat = vSE3->estimate();
            mat = Converter::toCvMat(vSE3->estimate());
            kf->SetPose(mat);
        }
    }

    /// NOTE: 20210907
    void Optimizer::updateLocalMapPoints(g2o::SparseOptimizer &op, list<MapPoint *> lLocalMapPoints,
                                         const unsigned long maxKFid)
    {
        // Points
        // 依序取出地圖點之頂點，並更新地圖點的位置
        g2o::VertexSBAPointXYZ *v_sba;

        for(MapPoint *mp : lLocalMapPoints)
        {
            // 為何地圖點的頂點 ID 會是 mp->mnId + maxKFid + 1？
            v_sba = static_cast<g2o::VertexSBAPointXYZ *>(op.vertex(mp->mnId + maxKFid + 1));
            
            // 更新地圖點的位置
            mp->SetWorldPos(Converter::toCvMat(v_sba->estimate()));
            
            // 利用所有觀察到這個地圖點的關鍵幀來估計關鍵幀們平均指向的方向，以及該地圖點可能的深度範圍(最近與最遠)
            mp->UpdateNormalAndDepth();
        }
    }

    // ***** Optimizer::OptimizeSim3 *****

    /// NOTE: 20210907
    void Optimizer::addSim3MapPointsAndKeyPoints(KeyFrame *pKF1, KeyFrame *pKF2, const bool bFixScale,
                                                 g2o::Sim3 &g2oS12, g2o::SparseOptimizer &op,
                                                 const int N, const float th2, int &nCorrespondences,
                                                 vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                                 vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                                 vector<size_t> &vnIndexEdge,
                                                 vector<MapPoint *> &vpMatches1)
    {        
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
        op.addVertex(vSim3);

        // 『關鍵幀 pKF1』觀察到的地圖點
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();

        const float deltaHuber = sqrt(th2);

        MapPoint *pMP1, *pMP2;
        g2o::VertexSBAPointXYZ *vPoint1, *vPoint2;
        cv::Mat P3D1w, P3D1c, P3D2w, P3D2c;

        // addSim3VertexAndEdge
        for (int i = 0; i < N; i++)
        {
            if (!vpMatches1[i])
            {
                continue;
            }

            // i = 0, id1 = 1, id2 = 2; i = 1, id1 = 3, id2 = 4
            // id1: 1, 3, 5, ...; id2: 2, 4, 6, ...
            const int id1 = 2 * i + 1;
            const int id2 = 2 * (i + 1);

            // 『地圖點 pMP1』：『關鍵幀 pKF1』觀察到的第 i 個地圖點
            pMP1 = vpMapPoints1[i];

            // 『地圖點 pMP2』：『關鍵幀 pKF1』的第 i 個地圖點在『關鍵幀 pKF2』上對應的地圖點
            pMP2 = vpMatches1[i];

            // 『關鍵幀 pKF2』的第 i2 個特徵點觀察到『地圖點 pMP2』
            const int i2 = pMP2->GetIndexInKeyFrame(pKF2);


            // 將『地圖點 pMP1、pMP2』的位置轉換到相機座標系下，作為『頂點』（id1, id2）加入優化
            if (pMP1 && pMP2)
            {
                if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0)
                {
                    // 取得『地圖點 pMP1』的世界座標
                    P3D1w = pMP1->GetWorldPos();

                    // 將『地圖點 pMP1』轉換到『關鍵幀 pKF1』座標系下
                    P3D1c = R1w * P3D1w + t1w;

                    vPoint1 = newVertexSBAPointXYZ(P3D1c, id1);
                    vPoint1->setFixed(true);

                    // 『地圖點 pMP1』的位置作為『頂點』加入優化
                    op.addVertex(vPoint1);

                    // 取得『地圖點 pMP2』的世界座標
                    P3D2w = pMP2->GetWorldPos();

                    // 將『地圖點 pMP2』轉換到『關鍵幀 pKF2』座標系下
                    P3D2c = R2w * P3D2w + t2w;

                    vPoint2 = newVertexSBAPointXYZ(P3D2c, id2);
                    vPoint2->setFixed(true);

                    // 『地圖點 pMP2』的位置作為『頂點』加入優化
                    op.addVertex(vPoint2);
                }
                else
                {
                    continue;
                }
            }
            else
            {
                continue;
            }

            nCorrespondences++;

            addSim3KeyPoints(i, i2, id1, id2, deltaHuber, pKF1, pKF2, op,
                             vpEdges12, vpEdges21, vnIndexEdge);
        }

        if(op.vertices().size() == 0 || op.edges().size() == 0)
        {
            std::cout << "[addSim3MapPointsAndKeyPoints] #vertices: " 
                      << op.vertices().size() 
                      << ", #edges: " << op.edges().size() << std::endl;
        }
    }

    /// NOTE: 20210907
    void Optimizer::addSim3KeyPoints(const int i, const int i2, const int id1, const int id2,
                                     const float deltaHuber, KeyFrame *pKF1, KeyFrame *pKF2,
                                     g2o::SparseOptimizer &op,
                                     vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                     vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                     vector<size_t> &vnIndexEdge)
    {
        // std::cout << "Start Optimizer::addSim3KeyPoints" << std::endl;

        // Set edge x1 = S12*X2
        // 『關鍵幀 pKF1』的第 i 個特徵點位置作為『邊』加入優化 
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        
        g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(0)));
        e12->setMeasurement(obs1);

        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        op.addEdge(e12);

        // Set edge x2 = S21*X1
        // 『關鍵幀 pKF2』的第 i2 個特徵點位置作為『邊』加入優化 
        Eigen::Matrix<double, 2, 1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];        

        g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();
        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(0)));
        e21->setMeasurement(obs2);
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        op.addEdge(e21);

        // 『關鍵幀 pKF1』的第 i 個特徵點位置
        vpEdges12.push_back(e12);

        // 『關鍵幀 pKF2』的第 i2 個特徵點位置
        vpEdges21.push_back(e21);

        vnIndexEdge.push_back(i);
    }

    /// NOTE: 20210907
    int Optimizer::filterSim3Outlier(g2o::SparseOptimizer &op, const float th2,
                                     vector<MapPoint *> &vpMatches1,
                                     vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                     vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                     vector<size_t> &vnIndexEdge)
    {
        // std::cout << "Start Optimizer::filterSim3Outlier" << std::endl;

        g2o::EdgeSim3ProjectXYZ *e12;
        g2o::EdgeInverseSim3ProjectXYZ *e21;
        size_t idx;
        int nBad = 0;

        for (size_t i = 0; i < vpEdges12.size(); i++)
        {
            e12 = vpEdges12[i];
            e21 = vpEdges21[i];

            if (!e12 || !e21)
            {
                continue;
            }

            // 優化後，誤差仍十分大
            if (e12->chi2() > th2 || e21->chi2() > th2)
            {
                idx = vnIndexEdge[i];

                // 移除誤差過大的『邊』和『地圖點』
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);

                op.removeEdge(e12);
                op.removeEdge(e21);

                vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
                vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);

                // 外點計數加一
                nBad++;
            }
        }

        // std::cout << "[filterSim3Outlier] Removed " << 2 * nBad << " edges." << std::endl;

        return nBad;
    }

    /// NOTE: 20210907
    int Optimizer::filterSim3Inlier(g2o::SparseOptimizer &op, const float th2,
                                    vector<MapPoint *> &vpMatches1,
                                    vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                    vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                    vector<size_t> &vnIndexEdge)
    {
        g2o::EdgeSim3ProjectXYZ *e12;
        g2o::EdgeInverseSim3ProjectXYZ *e21;
        size_t idx;
        int nIn = 0;

        for (size_t i = 0; i < vpEdges12.size(); i++)
        {
            e12 = vpEdges12[i];
            e21 = vpEdges21[i];

            if (!e12 || !e21)
            {
                continue;
            }

            // 再優化後，誤差仍十分大
            if (e12->chi2() > th2 || e21->chi2() > th2)
            {
                idx = vnIndexEdge[i];

                // 移除誤差過大的『地圖點』
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            }
            else
            {
                nIn++;
            }
        }

        return nIn;
    }

    // ***** Optimizer::OptimizeEssentialGraph *****

    /// NOTE: 20210907
    void Optimizer::addEssentialSim3(g2o::SparseOptimizer &op, KeyFrame *pLoopKF, 
                                     const vector<KeyFrame *> &vpKFs,
                                     const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                     vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vScw,
                                     const unsigned int nMaxKFid, const bool &bFixScale)
    {
        LoopClosing::KeyFrameAndPose::const_iterator it;
        Eigen::Matrix<double, 3, 3> Rcw;
        Eigen::Matrix<double, 3, 1> tcw;
        g2o::VertexSim3Expmap *VSim3;
        unsigned long nIDi;

        vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);
        int n_vertex = 0;
        
        for(KeyFrame *pKF : vpKFs)
        {
            if (pKF->isBad()){
                continue;
            }

            VSim3 = new g2o::VertexSim3Expmap();
            nIDi = pKF->mnId;

            // 『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系對應的『相似轉換矩陣』
            it = CorrectedSim3.find(pKF);

            if (it != CorrectedSim3.end())
            {
                // 根據關鍵幀的 Id 對應『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系對應的『相似轉換矩陣』
                vScw[nIDi] = it->second;

                // 估計初始值
                VSim3->setEstimate(it->second);
            }
            else
            {
                Rcw = Converter::toMatrix3d(pKF->GetRotation());
                tcw = Converter::toVector3d(pKF->GetTranslation());

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
            op.addVertex(VSim3);
            n_vertex++;

            vpVertices[nIDi] = VSim3;
        }

        if(op.vertices().size() == 0 || op.edges().size() == 0)
        {
            std::cout << "[addEssentialSim3] #vertices: " 
                    << op.vertices().size() 
                    << ", #edges: " << op.edges().size() << std::endl;
        }
    }

    /// NOTE: 20210907
    void Optimizer::addEssentialLoopConnections(g2o::SparseOptimizer &op, 
                                                set<pair<long unsigned int, long unsigned int>> &sInsertedEdges,
                                                const map<KeyFrame *, set<KeyFrame *>> &LoopConnections,
                                                vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vScw,
                                                const int minFeat, KeyFrame *pLoopKF, KeyFrame *pCurKF)
    {
        KeyFrame *pKF;
        unsigned long nIDi, nIDj;
        g2o::Sim3 Siw, Swi, Sjw, Sji;
        int n_loop_edge = 0, n_connecttion = 0;

        // Set Loop edges
        // LoopConnections[pKFi]：『關鍵幀 pKFi』的『已連結關鍵幀』，移除『關鍵幀 pKFi』的『共視關鍵幀』和
        // 『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
        for(pair<KeyFrame *, set<KeyFrame *>> loop_onnection : LoopConnections)
        {
            pKF = loop_onnection.first;

            // 『關鍵幀 pKFi』的『已連結關鍵幀』，移除『關鍵幀 pKFi』的『共視關鍵幀』和
            // 『關鍵幀 mpCurrentKF』和其『共視關鍵幀』
            const set<KeyFrame *> &spConnections = loop_onnection.second;
            n_connecttion += spConnections.size();

            nIDi = pKF->mnId;

            // 根據關鍵幀的 Id 對應『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKFi』座標系對應的『相似轉換矩陣』
            Siw = vScw[nIDi];

            // 『關鍵幀 pKFi』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
            Swi = Siw.inverse();

            for(KeyFrame * sit : spConnections)
            {
                nIDj = sit->mnId;  
                             
                if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(sit) < minFeat)
                {
                    continue;
                }
                
                // Sjw：『關鍵幀 mpCurrentKF』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』
                Sjw = vScw[nIDj];

                // Swi：『關鍵幀 pKFi』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
                // Sji:『關鍵幀 pKF_i』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』
                Sji = Sjw * Swi;

                addEdgeSim3(op, Sji, nIDi, nIDj);
                n_loop_edge++;

                sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
            }
        }

        if(op.vertices().size() == 0 || op.edges().size() == 0)
        {
            std::cout << "[addEssentialLoopConnections] #vertices: " 
                    << op.vertices().size() 
                    << ", #edges: " << op.edges().size() << std::endl;
        }
    }

    /// NOTE: 20210907
    void Optimizer::addEssentialEdges(const vector<KeyFrame *> vpKFs, g2o::SparseOptimizer &op, 
                                      const LoopClosing::KeyFrameAndPose &NonCorrectedSim3, 
                                      vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw, 
                                      const int minFeat, 
                                      set<pair<long unsigned int, long unsigned int>> sInsertedEdges)
    {
        LoopClosing::KeyFrameAndPose::const_iterator iti, itj;
        g2o::Sim3 Swi, Sjw, Sji;
        KeyFrame *pParentKF;
        int nIDj;
        bool need_continue;
        int nIDi;
            
        // Set normal edges
        /// NOTE: addEdgeSim3/addCovisibilityEdges 皆不為 0; addLoopEdges 全部為 0
        for(KeyFrame *pKF : vpKFs)
        {            
            nIDi = pKF->mnId;

            // NonCorrectedSim3[pKF]：『關鍵幀 pKFi』對應的『相似轉換矩陣』
            iti = NonCorrectedSim3.find(pKF);

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

            pParentKF = pKF->GetParent();

            // Spanning tree edge
            if (pParentKF)
            {
                nIDj = pParentKF->mnId;
                itj = NonCorrectedSim3.find(pParentKF);
                
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
                Sji = Sjw * Swi;

                addEdgeSim3(op, Sji, nIDi, nIDj);
            }


            // Loop edges
            addLoopEdges(pKF, op, Swi, NonCorrectedSim3, nIDi, vScw);

            /// TODO: 若這些內部函式沒有前後關係，可以將有 need_continue 的往前移，減少無效的運算
            // Covisibility graph edges
            // 取得至多 minFeat 個『共視關鍵幀』
            // 取得『關鍵幀 pKF』的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序，
            // 且觀察到的地圖點數量「大於」 minFeat）』
            addCovisibilityEdges(pKF, pParentKF, minFeat, sInsertedEdges, 
                                 NonCorrectedSim3, Swi, op, nIDi, vScw);
        }
    }

    /// NOTE: 20210907
    void Optimizer::addLoopEdges(KeyFrame *pKF, g2o::SparseOptimizer &op, const g2o::Sim3 Swi,
                                 const LoopClosing::KeyFrameAndPose &NonCorrectedSim3, const int nIDi, 
                                 vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw)
    {
        LoopClosing::KeyFrameAndPose::const_iterator itl;
        g2o::Sim3 Slw, Sli;
                
        // Loop edges
        const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();

        /// NOTE: sLoopEdges 全部為 0，導致 n_edge 也一直為 0

        for(KeyFrame * pLKF : sLoopEdges)
        {
            if (pLKF->mnId < pKF->mnId)
            {
                itl = NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                {
                    Slw = itl->second;
                }
                else{
                    Slw = vScw[pLKF->mnId];
                }

                Sli = Slw * Swi;

                addEdgeSim3(op, Sli, nIDi, pLKF->mnId);
            }
        }
    }
    
    /// NOTE: 20210907
    void Optimizer::addCovisibilityEdges(KeyFrame *pKF, KeyFrame *pParentKF, const int minFeat, 
                                         set<pair<long unsigned int, long unsigned int>> sInsertedEdges,
                                         const LoopClosing::KeyFrameAndPose &NonCorrectedSim3, 
                                         const g2o::Sim3 Swi, g2o::SparseOptimizer &op, const int nIDi, 
                                         vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw)
    {
        // Covisibility graph edges
        // 取得至多 minFeat 個『共視關鍵幀』
        // 取得『關鍵幀 pKF』的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序，
        // 且觀察到的地圖點數量「大於」 minFeat）』
        const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);

        // Loop edges
        const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();

        LoopClosing::KeyFrameAndPose::const_iterator itn;
        g2o::Sim3 Snw, Sni;

        for(KeyFrame *pKFn : vpConnectedKFs)
        {
            if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                {
                    if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), 
                                                       max(pKF->mnId, pKFn->mnId))))
                    {
                        continue;
                    }
                    
                    itn = NonCorrectedSim3.find(pKFn);
                    
                    if (itn != NonCorrectedSim3.end())
                    {
                        Snw = itn->second;
                    }
                    else
                    {
                        Snw = vScw[pKFn->mnId];
                    }

                    Sni = Snw * Swi;

                    addEdgeSim3(op, Sni, nIDi, pKFn->mnId);
                }
            }
        }
    }

    /// NOTE: 20210907
    void Optimizer::updateEssentialPoses(const vector<KeyFrame *> &vpKFs, g2o::SparseOptimizer &op,
                                        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vCorrectedSwc)
    {
        g2o::VertexSim3Expmap *VSim3;
        g2o::Sim3 CorrectedSiw;
        Eigen::Matrix3d eigR;
        Eigen::Vector3d eigt;
        cv::Mat Tiw;
        double s;
        int nIDi;

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for(KeyFrame *pKFi : vpKFs)
        {
            nIDi = pKFi->mnId;

            VSim3 = static_cast<g2o::VertexSim3Expmap *>(op.vertex(nIDi));

            // 優化後重新估計『相似轉換矩陣』
            CorrectedSiw = VSim3->estimate();

            // 『關鍵幀 pKF_i』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();

            // 優化後重新估計『旋轉矩陣』
            eigR = CorrectedSiw.rotation().toRotationMatrix();

            // 優化後重新估計『平移』
            eigt = CorrectedSiw.translation();

            // 優化後重新估計『規模尺度』
            s = CorrectedSiw.scale();

            eigt *= (1. / s); //[R t/s; 0 1]

            // 優化後重新估計『校正規模』的『轉換矩陣』
            Tiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(Tiw);
        }

        // std::cout << "[updateEssentialPoses] Update " << vpKFs.size() << " KeyFrame." << std::endl;
    }

    /// NOTE: 20210907
    void Optimizer::updateEssentialMapPoints(const vector<MapPoint *> vpMPs, KeyFrame *pCurKF, 
                                             const vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vScw, 
                                             const vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vCorrectedSwc)
    {
        Eigen::Matrix<double, 3, 1> eigP3Dw, eigCorrectedP3Dw;
        cv::Mat P3Dw, cvCorrectedP3Dw;        
        g2o::Sim3 Srw, correctedSwr;
        KeyFrame *pRefKF;
        int nIDr, n_mp = 0;

        // Correct points. Transform to "non-optimized" reference keyframe pose and transform back 
        // with optimized pose
        for(MapPoint *pMP : vpMPs)
        {            
            if (pMP->isBad()){
                continue;
            }

            n_mp++;

            if (pMP->mnCorrectedByKF == pCurKF->mnId)
            {
                nIDr = pMP->mnCorrectedReference;
            }
            else
            {
                pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }

            Srw = vScw[nIDr];

            // 『關鍵幀 pKF_i』座標系轉換到『關鍵幀 mpCurrentKF』座標系對應的『相似轉換矩陣』
            correctedSwr = vCorrectedSwc[nIDr];

            P3Dw = pMP->GetWorldPos();
            eigP3Dw = Converter::toVector3d(P3Dw);
            eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
            
            // 優化後重新估計『地圖點』的位置
            cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);

            pMP->SetWorldPos(cvCorrectedP3Dw);
            pMP->UpdateNormalAndDepth();
        }

        // std::cout << "[updateEssentialMapPoints] Update " << n_mp << " MapPoint." << std::endl;
    }

    // ***** Optimizer::PoseOptimization *****

    /// NOTE: 20210907
    void Optimizer::addPoseOptimization(Frame *pFrame, int &nInitialCorrespondences,
                                        g2o::SparseOptimizer &op, vector<size_t> &vnIndexEdgeMono,
                                        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> &vpEdgesMono,
                                        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo,
                                        vector<size_t> vnIndexEdgeStereo, const int idx)
    {
        g2o::EdgeSE3ProjectXYZOnlyPose *e;
        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e_stereo;
        cv::Mat Xw;
        MapPoint *pMP;

        const int N = pFrame->N;
        // float kp_ur;

        for (int i = 0; i < N; i++)
        {
            // 依序取得 pFrame 觀察到的地圖點
            pMP = pFrame->mvpMapPoints[i];

            if (pMP)
            {
                // 單目的這個值，會是負的
                // kp_ur = pFrame->mvuRight[i];
                
                // 地圖點可被觀察到，因此不是 Outlier
                pFrame->mvbOutlier[i] = false;

                nInitialCorrespondences++;

                // 取得與地圖點相對應的 pFrame 的特徵點
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];

                // Monocular observation
                e = newEdgeSE3ProjectXYZOnlyPose(op, pFrame, kpUn);

                // 圖點的世界座標
                /*void computeError()  {
                    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
                    Vector2d obs(_measurement);
                    _error = obs-cam_project(v1->estimate().map(Xw));
                }*/
                Xw = pMP->GetWorldPos();                    
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);
                
                op.addEdge(e);
                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);

                // // Monocular observation
                // if (kp_ur < 0)
                // {
                //     e = newEdgeSE3ProjectXYZOnlyPose(op, pFrame, kpUn);
                //     // 圖點的世界座標
                //     /*void computeError()  {
                //         const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
                //         Vector2d obs(_measurement);
                //         _error = obs-cam_project(v1->estimate().map(Xw));
                //     }*/
                //     Xw = pMP->GetWorldPos();                    
                //     e->Xw[0] = Xw.at<float>(0);
                //     e->Xw[1] = Xw.at<float>(1);
                //     e->Xw[2] = Xw.at<float>(2);                    
                //     op.addEdge(e);
                //     vpEdgesMono.push_back(e);
                //     vnIndexEdgeMono.push_back(i);
                // }
                // // Stereo observation（暫時跳過）
                // else 
                // {
                //     //SET EDGE
                //     e_stereo = newEdgeStereoSE3ProjectXYZOnlyPose(op, pFrame, kpUn, kp_ur);
                //     Xw = pMP->GetWorldPos();
                //     e_stereo->Xw[0] = Xw.at<float>(0);
                //     e_stereo->Xw[1] = Xw.at<float>(1);
                //     e_stereo->Xw[2] = Xw.at<float>(2);
                //     op.addEdge(e_stereo);
                //     vpEdgesStereo.push_back(e_stereo);
                //     vnIndexEdgeStereo.push_back(i);
                // }
            }
        }

        if(op.edges().size() == 0)
        {
            std::cout << "[addPoseOptimization] #vertices: " << op.vertices().size()  
                      << ", #edges: " << op.edges().size() << std::endl;
        }
    }

    /// NOTE: 20210907
    int Optimizer::addPoseOptimizationEdges(g2o::VertexSE3Expmap *vSE3, Frame *pFrame,
                                            g2o::SparseOptimizer &op,
                                            vector<g2o::EdgeSE3ProjectXYZOnlyPose *> &vpEdgesMono,
                                            vector<size_t> &vnIndexEdgeMono,
                                            vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> &vpEdgesStereo,
                                            vector<size_t> &vnIndexEdgeStereo, const int idx)
    {
        // We perform 4 optimizations, after each optimization we classify observation as
        // inlier/outlier. At the next optimization, outliers are not included, but at the end
        // they can be classified as inliers again.
        /// TODO: 或可轉換為靜態變數，減少重複宣告
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};

        // 定義每輪優化要優化幾次（每輪優化後會『將足夠好的 Edge 設為無須再優化』，才再次進行優化）
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0, i, n_mp, n_outlier;;
        const g2o::SE3Quat mSE3Tcw = Converter::toSE3Quat(pFrame->mTcw);

        // if(start_idx <= idx && idx <= end_idx)
        // {
        //     n_mp = 0, n_outlier = 0;        

        //     for(i = 0; i < pFrame->N; i++)
        //     {
        //         if(pFrame->mvpMapPoints[i])
        //         {
        //             n_mp++;

        //             if(pFrame->mvbOutlier[i])
        //             {
        //                 n_outlier++;
        //             }
        //         }
        //     }            

        //     std::cout << "[addPoseOptimizationEdges] Before optimize idx: " << idx
        //               << ", #vertices: " << op.vertices().size() 
        //               << ", #edges: " << op.edges().size() 
        //               << ", n_mp: " << n_mp 
        //               << ", n_outlier: " << n_outlier
        //               << "\nvSE3->estimate: " << vSE3->estimate().toVector().transpose() 
        //               << std::endl;
        // }
        
        // 優化『pFrame 觀察到的地圖點』的位置
        for (size_t it = 0; it < 4; it++)
        {
            // 將 pFrame 位姿轉型成『轉換矩陣』，作為 vSE3 的初始估計值
            /// NOTE: 似乎沒有在每次優化後更新 pFrame 的位姿，只在此函式結束前的最後更新而已
            vSE3->setEstimate(mSE3Tcw);
            op.initializeOptimization(0);

            // if(op.vertices().size() == 0 || op.edges().size() == 0)
            // {
            //     std::cout << "[addPoseOptimizationEdges] #vertices: " 
            //             << op.vertices().size() 
            //             << ", #edges: " << op.edges().size() << std::endl;
            // }

            // 優化 its[it] 次
            /// NOTE: 這裡發生了 0 vertices to optimize
            op.optimize(its[it]);

            //
            nBad = 0;

            /// NOTE: 已知進入 addPoseOptimizationMonoEdges 之前，若發生 TrackWithMotionModel failed.
            /// 會發生所有地圖點變成 outlier
            addPoseOptimizationMonoEdges(vpEdgesMono, vnIndexEdgeMono, pFrame, chi2Mono, it, nBad, idx);

            // addPoseOptimizationStereoEdges(pFrame, vpEdgesStereo, vnIndexEdgeStereo,
            //                                chi2Stereo, it, nBad);

            // 若 optimizer 還需要優化的『邊』足夠少（少於 10 個），則結束優化
            if (op.edges().size() < 10)
            {
                break;
            }
        }
        
        // if(start_idx <= idx && idx <= end_idx)
        // {
        //     n_mp = 0, n_outlier = 0;        

        //     for(i = 0; i < pFrame->N; i++)
        //     {
        //         if(pFrame->mvpMapPoints[i])
        //         {
        //             n_mp++;

        //             if(pFrame->mvbOutlier[i])
        //             {
        //                 n_outlier++;
        //             }
        //         }
        //     }

        //     std::cout << "[addPoseOptimizationEdges] Afetr optimize idx: " << idx
        //               << ", #vertices: " << op.vertices().size() 
        //               << ", #edges: " << op.edges().size() 
        //               << ", n_mp: " << n_mp 
        //               << ", n_outlier: " << n_outlier
        //               << ", vSE3->estimate: " << vSE3->estimate().toVector().transpose() 
        //               << std::endl;
        // }

        return nBad;
    }

    void Optimizer::addPoseOptimizationMonoEdges(vector<g2o::EdgeSE3ProjectXYZOnlyPose *> &vpEdgesMono, 
                                                 vector<size_t> &vnIndexEdgeMono, Frame *pFrame, 
                                                 const float chi2Mono[], const size_t it, int &nBad,
                                                 const int idx)
    {
        g2o::EdgeSE3ProjectXYZOnlyPose *e;
        size_t edge_idx;
        float chi2, cum_error = 0.0f;
        int n_outlier = 0, n_edge0 = 0, n_edge1 = 0;

        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            // 取出 optimizer 的第 i 個『邊』
            e = vpEdgesMono[i];

            // optimizer 的第 i 個『邊』(地圖點)
            edge_idx = vnIndexEdgeMono[i];

            // 若為第 edge_idx 個特徵點為 Outlier
            if (pFrame->mvbOutlier[edge_idx])
            {
                // 利用『邊』計算誤差
                e->computeError();
                n_outlier++;
            }

            // chi2() 基於 computeError() 所計算的誤差，取得 chi2 值
            chi2 = e->chi2();
            cum_error += chi2;

            // 若 chi2 值比預設門檻高
            if (chi2 > chi2Mono[it])
            {
                // 將第 edge_idx 個特徵點認定為 Outlier
                pFrame->mvbOutlier[edge_idx] = true;

                // 持續優化這個『邊』
                e->setLevel(1);
                n_edge1++;

                nBad++;
            }

            // 若 chi2 值比預設門檻低
            else
            {
                // 將第 edge_idx 個特徵點認定為內點
                pFrame->mvbOutlier[edge_idx] = false;

                // 不再優化這個『邊』
                e->setLevel(0);
                n_edge0++;
            }

            // 第 3 輪(it == 2)以後，無須使用 RobustKernel
            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        // if(start_idx <= idx && idx <= end_idx && cum_error > 1000000)
        // {
        //     std::cout << "[addPoseOptimizationMonoEdges] idx: " << idx 
        //               << ", cum_error: " << cum_error
        //               << ", n_outlier: " << n_outlier
        //               << ", n_edge0: " << n_edge0
        //               << ", n_edge1: " << n_edge1
        //               << std::endl;
        // }
    }

    void Optimizer::addPoseOptimizationStereoEdges(Frame *pFrame,
                                                   vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> &vpEdgesStereo,
                                                   vector<size_t> &vnIndexEdgeStereo,
                                                   const float chi2Stereo[], const size_t it, int &nBad)
    {
        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e;

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            e = vpEdgesStereo[i];

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
    }
    
    // ********************************************************************************
    // ********************************************************************************
    g2o::EdgeSE3ProjectXYZ *Optimizer::addEdgeSE3ProjectXYZ(g2o::SparseOptimizer &op,
                                                            const cv::KeyPoint kpUn,
                                                            const KeyFrame *pKF,
                                                            int v0, int v1, bool bRobust, double delta)
    {
        Eigen::Matrix<double, 2, 1> obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];

        g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

        // 『邊』連接第 mnId 個關鍵幀和第 mnId 個地圖點
        // （地圖點接續關鍵幀的 id 編號，因此是由 maxKFid + 1 開始編號）
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(v0)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(v1)));
        e->setMeasurement(obs);
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

        if (bRobust)
        {
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            rk->setDelta(delta);
            e->setRobustKernel(rk);
        }

        e->fx = pKF->fx;
        e->fy = pKF->fy;
        e->cx = pKF->cx;
        e->cy = pKF->cy;

        // 『關鍵點 kpUn』作為『邊』加入優化
        op.addEdge(e);

        return e;
    }

    g2o::VertexSBAPointXYZ *Optimizer::newVertexSBAPointXYZ(cv::Mat pos, const int id)
    {
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();

        // 『地圖點 pMP』的世界座標作為 vPoint 的初始值
        vPoint->setEstimate(Converter::toVector3d(pos));
        vPoint->setId(id);
        // vPoint->setMarginalized(true);

        return vPoint;
    }

    g2o::VertexSE3Expmap *Optimizer::addVertexSE3Expmap(g2o::SparseOptimizer &op, cv::Mat pose,
                                                        const int id, const bool fixed)
    {
        // Set Frame vertex
        // 將當前幀轉換成『轉換矩陣 SE3』，並作為頂點加入 optimizer
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pose));
        vSE3->setId(id);
        vSE3->setFixed(fixed);
        op.addVertex(vSE3);

        return vSE3;
    }

    /// NOTE: 20210907
    g2o::EdgeSim3 *Optimizer::addEdgeSim3(g2o::SparseOptimizer &op, g2o::Sim3 sim3,
                                          const int v0, const int v1)
    {
        // 『邊』連結兩個『關鍵幀』的『相似轉換矩陣』的頂點
        g2o::EdgeSim3 *e = new g2o::EdgeSim3();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(v0)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(v1)));

        // Sji:『關鍵幀 pKF_i』座標系轉換到『關鍵幀 pKF_j』座標系對應的『相似轉換矩陣』作為『邊』加入優化
        e->setMeasurement(sim3);
        e->information() = matLambda;
        op.addEdge(e);

        return e;
    }

    g2o::EdgeSE3ProjectXYZOnlyPose *Optimizer::newEdgeSE3ProjectXYZOnlyPose(g2o::SparseOptimizer &op,
                                                                            Frame *frame,
                                                                            const cv::KeyPoint kpUn)
    {
        // 特徵點的位置
        Eigen::Matrix<double, 2, 1> obs;
        obs << kpUn.pt.x, kpUn.pt.y;

        const float invSigma2 = frame->mvInvLevelSigma2[kpUn.octave];

        // 一元邊：重投影誤差（僅用於優化相機位姿時）
        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(0)));

        // 設置 _measurement（觀測值）
        e->setMeasurement(obs);
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(deltaMono);
        e->setRobustKernel(rk);

        // 相機內參
        e->fx = frame->fx;
        e->fy = frame->fy;
        e->cx = frame->cx;
        e->cy = frame->cy;

        return e;
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    void Optimizer::filterStereoLocalMapPoints(vector<g2o::EdgeStereoSE3ProjectXYZ *> &vpEdgesStereo,
                                      vector<MapPoint *> &vpMapPointEdgeStereo)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e_stereo;
        MapPoint *pMP;

        // 和單目無關，暫時跳過
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad()){
                continue;
            }

            e_stereo = vpEdgesStereo[i];
            
            if (e_stereo->chi2() > 7.815 || !e_stereo->isDepthPositive())
            {
                e_stereo->setLevel(1);
            }

            e_stereo->setRobustKernel(0);
        }
    }

    void Optimizer::markEarseStereo(vector<pair<KeyFrame *, MapPoint *>> &vToErase,
                                      vector<MapPoint *> &vpMapPointEdgeStereo,
                                      vector<g2o::EdgeStereoSE3ProjectXYZ *> &vpEdgesStereo,
                                      vector<KeyFrame *> &vpEdgeKFStereo)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e_stereo;
        MapPoint *mp;
        KeyFrame *kf;

        // 和單目無關，暫時跳過
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            mp = vpMapPointEdgeStereo[i];

            if (mp->isBad()){
                continue;
            }

            // vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStere
            e_stereo = vpEdgesStereo[i];            

            if (e_stereo->chi2() > 7.815 || !e_stereo->isDepthPositive())
            {
                kf = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(kf, mp));
            }
        }
    }

        

    g2o::EdgeStereoSE3ProjectXYZ* Optimizer::addEdgeStereoSE3ProjectXYZ(g2o::SparseOptimizer &op, 
                                                                        const cv::KeyPoint kpUn, 
                                                                        const KeyFrame *pKF,
                                                                        const size_t kp_idx,
                                                                        const int v0,
                                                                        const int v1,
                                                                        bool bRobust, double delta)
    {
        Eigen::Matrix<double, 3, 1> obs;
        obs << kpUn.pt.x, kpUn.pt.y, pKF->mvuRight[kp_idx];

        const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
        

        g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(v0)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(v1)));
        e->setMeasurement(obs);

        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
        e->setInformation(Info);
        
        if (bRobust)
        {
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(delta);
        }

        e->fx = pKF->fx;
        e->fy = pKF->fy;
        e->cx = pKF->cx;
        e->cy = pKF->cy;
        e->bf = pKF->mbf;

        op.addEdge(e);

        return e;
    }

    g2o::EdgeStereoSE3ProjectXYZOnlyPose* 
    Optimizer::newEdgeStereoSE3ProjectXYZOnlyPose(g2o::SparseOptimizer &op, Frame *frame, 
                                                  const cv::KeyPoint kpUn, const float kp_ur)
    {
        Eigen::Matrix<double, 3, 1> obs;
        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

        const float invSigma2 = frame->mvInvLevelSigma2[kpUn.octave];        

        g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(op.vertex(0)));
        e->setMeasurement(obs);

        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
        e->setInformation(Info);

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(deltaStereo);

        e->fx = frame->fx;
        e->fy = frame->fy;
        e->cx = frame->cx;
        e->cy = frame->cy;
        e->bf = frame->mbf;

        return e;
    }

} //namespace ORB_SLAM
