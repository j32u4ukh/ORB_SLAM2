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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"

namespace ORB_SLAM2
{

class LoopClosing;

class Optimizer
{
public:
    static const float thHuber2D;
    static const float thHuber3D;
    static const float thHuberMono;
    static const float thHuberStereo;
    static const float deltaMono;
    static const float deltaStereo;
    static const Eigen::Matrix<double, 7, 7> matLambda;
    static const int start_idx;
    static const int end_idx;

    void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                 const bool bRobust = true);
    void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                       const unsigned long nLoopKF=0, const bool bRobust = true);
    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);
    int static PoseOptimization(Frame* pFrame, const int idx=0);

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale);

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);

    // ==================================================
    // 自己封裝的函式
    // ==================================================

    // **********
    static inline void addKeyFramePoses(const vector<KeyFrame *> &vpKFs, g2o::SparseOptimizer &op,
                                        unsigned long &maxKFid);

    static inline void addMapPoints(const vector<MapPoint *> &vpMP, g2o::SparseOptimizer &op,
                                    const unsigned long maxKFid,
                                    const bool bRobust, vector<bool> &vbNotIncludedMP);

    static inline void updateKeyFramePoses(const vector<KeyFrame *> &vpKFs, g2o::SparseOptimizer &op,
                                           const unsigned long nLoopKF);

    static inline void updateMapPoints(const vector<MapPoint *> &vpMP, vector<bool> &vbNotIncludedMP,
                                       g2o::SparseOptimizer &op, long unsigned int &maxKFid,
                                       const unsigned long nLoopKF);

    static inline void extractLocalKeyFrames(list<KeyFrame *> &lLocalKeyFrames, KeyFrame *pKF);

    static inline void extractLocalMapPoints(list<MapPoint *> &lLocalMapPoints,
                                             const list<KeyFrame *> &lLocalKeyFrames, 
                                             const KeyFrame *pKF);

    static inline void extractFixedCameras(list<KeyFrame *> &lFixedCameras,
                                           const list<MapPoint *> &lLocalMapPoints, 
                                           const KeyFrame *pKF);

    static inline void addLocalKeyFrames(list<KeyFrame *> &lLocalKeyFrames, g2o::SparseOptimizer &op,
                                         long unsigned int &maxKFid);

    static inline void addFixedCameras(list<KeyFrame *> &lFixedCameras, g2o::SparseOptimizer &op,
                                       long unsigned int &maxKFid);

    static inline void addLocalMapPoints(list<MapPoint *> &lLocalMapPoints, g2o::SparseOptimizer &op,
                                         long unsigned int &maxKFid, KeyFrame *pKF,
                                         vector<g2o::EdgeSE3ProjectXYZ *> &vpEdgesMono,
                                         vector<KeyFrame *> &vpEdgeKFMono,
                                         vector<MapPoint *> &vpMapPointEdgeMono,
                                         vector<g2o::EdgeStereoSE3ProjectXYZ *> &vpEdgesStereo,
                                         vector<KeyFrame *> &vpEdgeKFStereo,
                                         vector<MapPoint *> &vpMapPointEdgeStereo);

    static inline void filterMonoLocalMapPoints(vector<g2o::EdgeSE3ProjectXYZ *> &vpEdgesMono,
                                                vector<MapPoint *> &vpMapPointEdgeMono);

    static inline void markEarseMono(vector<pair<KeyFrame *, MapPoint *>> &vToErase,
                                     vector<MapPoint *> &vpMapPointEdgeMono,
                                     vector<g2o::EdgeSE3ProjectXYZ *> &vpEdgesMono,
                                     vector<KeyFrame *> &vpEdgeKFMono);

    static inline void executeEarsing(vector<pair<KeyFrame *, MapPoint *>> &vToErase);

    static inline void updateLocalKeyFrames(g2o::SparseOptimizer &op, list<KeyFrame *> &lLocalKeyFrames);

    static inline void updateLocalMapPoints(g2o::SparseOptimizer &op, list<MapPoint *> lLocalMapPoints,
                                            const unsigned long maxKFid);

    // ***** Optimizer::OptimizeSim3 *****
    static inline void addSim3MapPointsAndKeyPoints(KeyFrame *pKF1, KeyFrame *pKF2, const bool bFixScale,
                                                    g2o::Sim3 &g2oS12, g2o::SparseOptimizer &op,
                                                    const int N, const float th2, int &nCorrespondences,
                                                    vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                                    vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                                    vector<size_t> &vnIndexEdge,
                                                    vector<MapPoint *> &vpMatches1);

    static inline void addSim3KeyPoints(const int i, const int i2, const int id1, const int id2,
                                        const float deltaHuber, KeyFrame *pKF1, KeyFrame *pKF2,
                                        g2o::SparseOptimizer &op,
                                        vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                        vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                        vector<size_t> &vnIndexEdge);

    static inline int filterSim3Outlier(g2o::SparseOptimizer &op, const float th2,
                                        vector<MapPoint *> &vpMatches1,
                                        vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                        vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                        vector<size_t> &vnIndexEdge);

    static inline int filterSim3Inlier(g2o::SparseOptimizer &op, const float th2,
                                       vector<MapPoint *> &vpMatches1,
                                       vector<g2o::EdgeSim3ProjectXYZ *> &vpEdges12,
                                       vector<g2o::EdgeInverseSim3ProjectXYZ *> &vpEdges21,
                                       vector<size_t> &vnIndexEdge);

    // ***** Optimizer::OptimizeEssentialGraph *****

    static inline void addEssentialSim3(g2o::SparseOptimizer &op, KeyFrame *pLoopKF,
                                        const vector<KeyFrame *> &vpKFs,
                                        const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vScw,
                                        const unsigned int nMaxKFid, const bool &bFixScale);

    static inline void addEssentialLoopConnections(g2o::SparseOptimizer &op,
                                                   set<pair<long unsigned int, long unsigned int>> &sInsertedEdges,
                                                   const map<KeyFrame *, set<KeyFrame *>> &LoopConnections,
                                                   vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vScw,
                                                   const int minFeat, KeyFrame *pLoopKF, KeyFrame *pCurKF);

    static inline void addEssentialEdges(const vector<KeyFrame *> vpKFs, g2o::SparseOptimizer &op,
                                         const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                         vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw,
                                         const int minFeat,
                                         set<pair<long unsigned int, long unsigned int>> sInsertedEdges);

    static inline void addLoopEdges(KeyFrame *pKF, g2o::SparseOptimizer &op, const g2o::Sim3 Swi,
                                    const LoopClosing::KeyFrameAndPose &NonCorrectedSim3, const int nIDi,
                                    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw);

    static inline void addCovisibilityEdges(KeyFrame *pKF, KeyFrame *pParentKF, const int minFeat,
                                            set<pair<long unsigned int, long unsigned int>> sInsertedEdges,
                                            const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                            const g2o::Sim3 Swi, g2o::SparseOptimizer &op, const int nIDi,
                                            vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw);

    static inline void updateEssentialPoses(const vector<KeyFrame *> &vpKFs, g2o::SparseOptimizer &op,
                                            vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vCorrectedSwc);
    
    static inline void updateEssentialMapPoints(const vector<MapPoint *> vpMPs, KeyFrame *pCurKF,
                                                const vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vScw,
                                                const vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> &vCorrectedSwc);

    // ***** Optimizer::PoseOptimization *****

    static inline void addPoseOptimization(Frame *pFrame, int &nInitialCorrespondences,
                                           g2o::SparseOptimizer &op, vector<size_t> &vnIndexEdgeMono,
                                           vector<g2o::EdgeSE3ProjectXYZOnlyPose *> &vpEdgesMono,
                                           vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo,
                                           vector<size_t> vnIndexEdgeStereo, const int idx=0);

    static inline void addPoseOptimizationMonoEdges(vector<g2o::EdgeSE3ProjectXYZOnlyPose *> &vpEdgesMono,
                                                    vector<size_t> &vnIndexEdgeMono, Frame *pFrame,
                                                    const float chi2Mono[], const size_t it, int &nBad, 
                                                    const int idx=0);

    static inline void addPoseOptimizationStereoEdges(Frame *pFrame,
                                                      vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> &vpEdgesStereo,
                                                      vector<size_t> &vnIndexEdgeStereo,
                                                      const float chi2Stereo[], const size_t it, int &nBad);

    static inline int addPoseOptimizationEdges(g2o::VertexSE3Expmap *vSE3, Frame *pFrame,
                                               g2o::SparseOptimizer &op,
                                               vector<g2o::EdgeSE3ProjectXYZOnlyPose *> &vpEdgesMono,
                                               vector<size_t> &vnIndexEdgeMono,
                                               vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> &vpEdgesStereo,
                                               vector<size_t> &vnIndexEdgeStereo,
                                               const int idx=0);

    // ********************************************************************************
    static inline g2o::EdgeSE3ProjectXYZ *addEdgeSE3ProjectXYZ(g2o::SparseOptimizer &op,
                                                               const cv::KeyPoint kpUn,
                                                               const KeyFrame *pKF,
                                                               int v0, int v1, bool bRobust, double delta);

    static inline g2o::VertexSE3Expmap * addVertexSE3Expmap(g2o::SparseOptimizer &op, cv::Mat pose, 
                                                            const int id, const bool fixed);    

    static inline g2o::EdgeSim3* addEdgeSim3(g2o::SparseOptimizer &op, g2o::Sim3 sim3, 
                                             const int v0, const int v1);

    static inline g2o::VertexSBAPointXYZ *newVertexSBAPointXYZ(cv::Mat pos, const int id);

    static inline g2o::EdgeSE3ProjectXYZOnlyPose *newEdgeSE3ProjectXYZOnlyPose(g2o::SparseOptimizer &op,
                                                                               Frame *frame,
                                                                               const cv::KeyPoint kpUn);

    static inline double computeError(){
        
    }                                                                           

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    static inline void filterStereoLocalMapPoints(vector<g2o::EdgeStereoSE3ProjectXYZ *> &vpEdgesStereo,
                                                  vector<MapPoint *> &vpMapPointEdgeStereo);

    static inline void markEarseStereo(vector<pair<KeyFrame *, MapPoint *>> &vToErase,
                                       vector<MapPoint *> &vpMapPointEdgeStereo,
                                       vector<g2o::EdgeStereoSE3ProjectXYZ *> &vpEdgesStereo,
                                       vector<KeyFrame *> &vpEdgeKFStereo);

    static inline g2o::EdgeStereoSE3ProjectXYZ *addEdgeStereoSE3ProjectXYZ(g2o::SparseOptimizer &op,
                                                                           const cv::KeyPoint kpUn,
                                                                           const KeyFrame *pKF,
                                                                           const size_t kp_idx,
                                                                           const int v0,
                                                                           const int v1,
                                                                           bool bRobust, double delta);

    static inline g2o::EdgeStereoSE3ProjectXYZOnlyPose* 
    newEdgeStereoSE3ProjectXYZOnlyPose(g2o::SparseOptimizer &op, Frame *frame, 
                                       const cv::KeyPoint kpUn, const float kp_ur);  
                                                                         
};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
