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

    /// NOTE: 20210830
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
    // 自定義函式
    // ==================================================

    void Map::updateLogOdd(const Eigen::Vector3d origin, const Eigen::Vector3d endpoint)
    {
        // LOCK!!!!
        unique_lock<mutex> lock(mMutexMap);
        std::cout << "Start Map::updateLogOdd" << std::endl;

        std::set<ORB_SLAM2::MapPoint*>::iterator mp_it, mp_end = mspMapPoints.end();
        MapPoint* mp;
        Eigen::Vector3d mp_vector, ray = endpoint - origin;

        for(mp_it = mspMapPoints.begin(); mp_it != mp_end; mp_it++)
        {
            mp = *mp_it;

            if(!mp)
            {
                continue;
            }

            mp_vector = Converter::toVector3d(mp->GetWorldPos()) - origin;

            // 地圖點 mp 在此次觀察到的地圖點 endpoint 的射線上，表示地圖點 mp 應該已不在原本的位置
            if(isOnRay(ray, mp_vector))
            {
                if(mp_vector.norm() == ray.norm())
                {                    
                    mp->hit();

                    std::cout << "Map::updateLogOdd hit\n"
                          << "origin: " << origin.transpose()
                          << ", endpoint: " << endpoint.transpose()
                          << "\nmp: " << Converter::toVector3d(mp->GetWorldPos()).transpose()
                          << ", mp_vector: " << mp_vector.transpose()
                          << ", getHitLog: " << mp->getHitLog()
                          << ", getHitProb: " << mp->getHitProb()
                          << std::endl;
                }
                else
                {
                    mp->miss();

                    std::cout << "Map::updateLogOdd miss\n"
                          << "origin: " << origin.transpose()
                          << ", endpoint: " << endpoint.transpose()
                          << "\nmp: " << Converter::toVector3d(mp->GetWorldPos()).transpose()
                          << ", mp_vector: " << mp_vector.transpose()
                          << ", getHitLog: " << mp->getHitLog()
                          << ", getHitProb: " << mp->getHitProb()
                          << std::endl;
                }
            }
        }

        std::cout << "End Map::updateLogOdd" << std::endl;
    }

    bool Map::isOnRay(const Eigen::Vector3d ray, const Eigen::Vector3d vector)
    {
        if(vector.norm() > ray.norm())
        {
            return false;
        }

        return ray.normalized() == vector.normalized();
    }

    template<class Archive>
    void Map::serialize(Archive &ar, const unsigned int version)
    {
        // don't save mutex
        unique_lock<mutex> lock_MapUpdate(mMutexMapUpdate);
        unique_lock<mutex> lock_Map(mMutexMap);
        ar & mspMapPoints;
        ar & mvpKeyFrameOrigins;
        ar & mspKeyFrames;
        ar & mvpReferenceMapPoints;
        ar & mnMaxKFid & mnBigChangeIdx;
    }
    
    template void Map::serialize(boost::archive::binary_iarchive&, const unsigned int);
    template void Map::serialize(boost::archive::binary_oarchive&, const unsigned int);
 
} //namespace ORB_SLAM
