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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
#include "Sim3Solver.h"
#include "KeyFrameDatabase.h"
#include "ORBmatcher.h"

#include <thread>
#include <mutex>

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;


class LoopClosing
{
public:

    typedef pair<set<KeyFrame*>,int> ConsistentGroup;
    
    // typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
    //     Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;
    typedef map<KeyFrame*, 
                g2o::Sim3, 
                std::less<KeyFrame*>, 
                Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3> > > KeyFrameAndPose;
    // 參考：https://blog.csdn.net/lixujie666/article/details/90023059

    /* const KeyFrame* V.S. const KeyFrame *const
    之前的 const KeyFrame* 表示的指針意義為：此指針指向的 KeyFame 是一個 const 值，
    改後的 const KeyFrame *const表示的指針意義為：此指針為一個 const 指針，而 KeyFrame 亦為 const 值。
    這樣便能符合分配器必須為為 std::pair<const Key, 值> 的類型的要求。

    註：上方改法和這裡的說明略有不同
    參考：https://zhuanlan.zhihu.com/p/218019316

    template < class Key,                                 // multimap::key_type
           class T,                                       // multimap::mapped_type
           class Compare = less<Key>,                     // multimap::key_compare
           class Alloc = allocator<pair<const Key,T> >    // multimap::allocator_type
           > class multimap;

    * Key: 鍵的類型。地圖中的每個元素都由其鍵值標識。別名為成員類型 multimap::key_type。
    
    * T: 映射值的類型。多重映射中的每個元素都存儲一些數據作為其映射值。別名為成員類型 multimap::mapped_type。
    
    * Compare: 
    一個二元謂詞，將兩個元素鍵作為參數並返回一個布爾值。表達式 comp(a,b)，其中 comp 是這種類型的對象，a 和 b 是元素鍵，
    如果在函數定義的嚴格弱排序中 a 被認為在 b 之前，則應返回 true。
    multimap 對象使用此表達式來確定元素在容器中遵循的順序以及兩個元素鍵是否等效（通過反射性比較它們：
    如果 !comp(a,b) && !comp(b,a)，它們是等效的）。
    這可以是函數指針或函數對象（參見構造函數示例）。這默認為 less<T>，它返回與應用小於運算符 (a<b) 相同的值。
    別名為成員類型 multimap::key_compare。
    
    * Alloc: 
    用於定義存儲分配模型的分配器對象的類型。默認情況下使用分配器類模板，它定義了最簡單的內存分配模型，並且與值無關。
    別名為成員類型 multimap::allocator_type。

    參考：https://www.cplusplus.com/reference/map/multimap/
    */

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale, 
                const int &idx);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(unsigned long nLoopKF);

    // 『執行續 RunGlobalBundleAdjustment』是否正在執行
    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    bool hasNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;

    // LoopClosing 執行續是否已有效停止
    bool mbFinished;

    std::mutex mMutexFinish;

    inline float findLeastSimilarScore();
    inline void updateConsistentGroups(vector<KeyFrame *> &vpCandidateKFs);
    inline int newSim3Solvers(const int nInitialCandidates,
                              vector<bool> &vbDiscarded, ORBmatcher &matcher,
                              vector<vector<MapPoint *>> &vvpMapPointMatches,
                              vector<Sim3Solver *> &vpSim3Solvers);
    inline bool callOptimizeSim3(int &nCandidates, const int nInitialCandidates,
                                 vector<bool> &vbDiscarded, ORBmatcher &matcher,
                                 vector<vector<MapPoint *>> &vvpMapPointMatches,
                                 vector<Sim3Solver *> &vpSim3Solvers);
    inline void searchByProjection(ORBmatcher &matcher);
    inline void whetherCorrectedSim3(const cv::Mat &Twc, KeyFrameAndPose &CorrectedSim3,
                                     KeyFrameAndPose &NonCorrectedSim3);
    inline void scaledCorrectedSim3(KeyFrameAndPose &CorrectedSim3, 
                                    KeyFrameAndPose &NonCorrectedSim3);
    inline void updateMatchedMapPoints();
    inline void updateLoopConnections(map<KeyFrame *, set<KeyFrame *>> &LoopConnections);
    inline void updateChildPose(list<KeyFrame *> &lpKFtoCheck, const unsigned long nLoopKF);
    inline void updateMapPointsPosition(const unsigned long nLoopKF);

    Map* mpMap;
    Tracking* mpTracker;

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* mpCurrentKF;
    KeyFrame* mpMatchedKF;

    // ConsistentGroup: pair<set<KeyFrame*>,int>
    std::vector<ConsistentGroup> mvConsistentGroups;

    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;

    // mvpCurrentMatchedPoints：根據已配對的地圖點與關鍵幀，再次匹配成功後找到的『地圖點』
    std::vector<MapPoint*> mvpCurrentMatchedPoints;
    
    std::vector<MapPoint*> mvpLoopMapPoints;
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;

    bool mnFullBAIdx;

    const int* index;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
