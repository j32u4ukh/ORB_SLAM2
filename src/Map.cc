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

#include "Map.h"

#include <mutex>

namespace ORB_SLAM2
{
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0)
    {
    }

    /// NOTE: 20210829
    // 將『關鍵幀 pKF』加到地圖的『關鍵幀陣列』中
    void Map::AddKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);

        // 更新最新關鍵幀序號
        if (pKF->mnId > mnMaxKFid)
        {
            mnMaxKFid = pKF->mnId;
        }
    }

    void Map::AddMapPoint(MapPoint *pMP)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    // 清除『地圖點 pMP』
    void Map::EraseMapPoint(MapPoint *pMP)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);

        /// TODO: This only erase the pointer. Delete the MapPoint
    }

    // 返回地圖中的關鍵幀數量
    long unsigned int Map::getInMapKeyFrameNumber()
    {
        unique_lock<mutex> lock(mMutexMap);

        // 地圖中的關鍵幀數量
        return mspKeyFrames.size();
    }

    void Map::EraseKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    // 取出所有『關鍵幀』
    vector<KeyFrame *> Map::GetAllKeyFrames()
    {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    // 取出所有『地圖點』
    vector<MapPoint *> Map::GetAllMapPoints()
    {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::GetMaxKFid()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    // 增加『重要變革的索引值』
    void Map::InformNewBigChange()
    {
        unique_lock<mutex> lock(mMutexMap);
        mnBigChangeIdx++;
    }

    long unsigned int Map::getInMapMapPointNumber()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    // 設置參考用地圖點
    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
    {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    // 回傳重要變革的索引值
    int Map::GetLastBigChangeIdx()
    {
        unique_lock<mutex> lock(mMutexMap);

        // 重要變革的索引值
        return mnBigChangeIdx;
    }

    vector<MapPoint *> Map::GetReferenceMapPoints()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    void Map::clear()
    {
        set<MapPoint *>::iterator sit, send = mspMapPoints.end();

        for (sit = mspMapPoints.begin(); sit != send; sit++){
            delete *sit;
        }

        set<KeyFrame*>::iterator s_kf_it, s_kf_end = mspKeyFrames.end();

        for (s_kf_it = mspKeyFrames.begin(); s_kf_it != s_kf_end; sit++){
            delete *s_kf_it;
        }

        mspMapPoints.clear();
        mspKeyFrames.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpKeyFrameOrigins.clear();
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    
} //namespace ORB_SLAM
