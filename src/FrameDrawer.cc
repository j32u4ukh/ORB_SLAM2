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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>

namespace ORB_SLAM2
{
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    FrameDrawer::FrameDrawer(Map *pMap, bool bReuseMap) : mpMap(pMap)
    {
        if (bReuseMap)
        {
            mState = Tracking::LOST;
        }
        else
        {
            mState = Tracking::SYSTEM_NOT_READY;
        }

        img_buffer = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    // 在灰階圖片上標注出特徵點的位置，並根據當前狀態，將文字寫在下方的黑色區域
    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat img;

        // Initialization: KeyPoints in reference frame
        vector<cv::KeyPoint> vIniKeys;     

        // Initialization: correspondeces with reference keypoints
        vector<int> vMatches;             

        // KeyPoints in current frame 
        vector<cv::KeyPoint> vCurrentKeys; 
        
        // Tracked visual odometry in current frame
        vector<bool> vbVO;

        // Tracked MapPoints in current frame
        vector<bool> vbMap;
        
        // Tracking state
        int state;                         

        // Copy variables within scoped mutex
        // 暫停執行續 mMutex、取出繪圖用數值、恢復執行續 mMutex
        {
            unique_lock<mutex> lock(mMutex);
            state = mState;

            if (mState == Tracking::SYSTEM_NOT_READY)
            {
                mState = Tracking::NO_IMAGES_YET;
            }

            img_buffer.copyTo(img);

            if (mState == Tracking::NOT_INITIALIZED)
            {
                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
            }
            else if (mState == Tracking::OK)
            {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
            }
            else if (mState == Tracking::LOST)
            {
                vCurrentKeys = mvCurrentKeys;
            }
        } // destroy scoped mutex -> release mutex

        // this should be always true
        if (img.channels() < 3) {
            cvtColor(img, img, CV_GRAY2BGR);
        }

        // Draw
        // INITIALIZING
        if (state == Tracking::NOT_INITIALIZED) 
        {
            for (unsigned int i = 0; i < vMatches.size(); i++)
            {
                if (vMatches[i] >= 0)
                {
                    cv::line(img, 
                             vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt,
                             cv::Scalar(0, 255, 0));
                }
            }
        }

        // TRACKING
        else if (state == Tracking::OK) 
        {
            mnTracked = 0;
            mnTrackedVO = 0;
            const float r = 5;
            const int n = vCurrentKeys.size();

            for (int i = 0; i < n; i++)
            {
                if (vbVO[i] || vbMap[i])
                {
                    cv::Point2f pt1, pt2;
                    pt1.x = vCurrentKeys[i].pt.x - r;
                    pt1.y = vCurrentKeys[i].pt.y - r;
                    pt2.x = vCurrentKeys[i].pt.x + r;
                    pt2.y = vCurrentKeys[i].pt.y + r;

                    // This is a match to a MapPoint in the map
                    // 若匹配到地圖點
                    if (vbMap[i])
                    {
                        // 地圖點對應的特徵點的區域畫上綠色的矩形
                        cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));

                        // 特徵點的周圍則劃上半徑為 2 的綠色圓圈
                        cv::circle(img, vCurrentKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);

                        mnTracked++;
                    }

                    // This is match to a "visual odometry" MapPoint created in the last frame
                    // 若為『視覺里程計』
                    else 
                    {
                        // 對應的特徵點的區域畫上紅色的矩形
                        cv::rectangle(img, pt1, pt2, cv::Scalar(255, 0, 0));

                        // 特徵點的周圍則劃上半徑為 2 的紅色圓圈
                        cv::circle(img, vCurrentKeys[i].pt, 2, cv::Scalar(255, 0, 0), -1);

                        mnTrackedVO++;
                    }
                }
            }
        }

        cv::Mat imWithInfo;

        // 根據當前狀態，將文字寫在下方的黑色區域
        DrawTextInfo(img, state, imWithInfo);

        return imWithInfo;
    }

    // 根據當前狀態，將文字寫在下方的黑色區域
    void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
    {
        stringstream s;
        
        if (nState == Tracking::NO_IMAGES_YET){
            s << " WAITING FOR IMAGES";
        }
        else if (nState == Tracking::NOT_INITIALIZED){
            s << " TRYING TO INITIALIZE ";
        }
        else if (nState == Tracking::OK)
        {
            if (!mbOnlyTracking){
                s << "SLAM MODE |  ";
            }
            else{
                s << "LOCALIZATION | ";
            }

            int nKFs = mpMap->getInMapKeyFrameNumber();
            int nMPs = mpMap->getInMapMapPointNumber();
            s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;

            if (mnTrackedVO > 0){
                s << ", + VO matches: " << mnTrackedVO;
            }
        }
        else if (nState == Tracking::LOST)
        {
            s << " TRACK LOST. TRYING TO RELOCALIZE ";
        }
        else if (nState == Tracking::SYSTEM_NOT_READY)
        {
            s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        // 呈現文字的圖片，為追蹤中的圖片再加高數個像素，將文字寫在多出來的黑色區域
        imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
        im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
        imText.rowRange(im.rows, imText.rows) = 
                                            cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
        
        /* void cv::putText(
            // 待繪制的圖像
		    cv::Mat& img,

            // 待繪制的文字 
		    const string& text, 

            // 文本框的左下角
            cv::Point origin, 

            // 字體 (如cv::FONT_HERSHEY_PLAIN)
            int fontFace, 

            // 尺寸因子，值越大文字越大
            double fontScale, 

            // 線條的顏色（RGB） cv::Scalar(255, 255, 255) 白色
            cv::Scalar color, 

            // 線條寬度
            int thickness = 1, 

            // 線型（4鄰域或8鄰域，默認8鄰域）
            int lineType = 8, 

            // true='origin at lower left'
            bool bottomLeftOrigin = false 
	    );*/
        cv::putText(imText, s.str(), 
                    cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, 
                    cv::Scalar(255, 255, 255), 1, 8);
    }

    // 初始化後，根據是否被關鍵幀觀察到，將當前幀的所有關鍵點區分為『地圖點』或『視覺里程計』
    void FrameDrawer::Update(Tracking *pTracker)
    {
        unique_lock<mutex> lock(mMutex);

        // 將 Tracking 當中的灰階圖片複製給 img_buffer 用於至地圖
        pTracker->gray.copyTo(img_buffer);

        // 當前幀的所有關鍵點
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;

        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N, false);
        mvbMap = vector<bool>(N, false);
        mbOnlyTracking = pTracker->mbOnlyTracking;
        
        // 第 2 幀時 pTracker->mLastProcessedState 才會是 Tracking::NOT_INITIALIZED
        if (pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED)
        {
            mvIniKeys = pTracker->mInitialFrame.mvKeys;
            mvIniMatches = pTracker->mvIniMatches;
        }
        
        else if (pTracker->mLastProcessedState == Tracking::OK)
        {
            MapPoint *pMP;

            for (int i = 0; i < N; i++)
            {
                pMP = pTracker->mCurrentFrame.mvpMapPoints[i];

                if (pMP)
                {
                    if (!pTracker->mCurrentFrame.mvbOutlier[i])
                    {
                        // 若地圖點被至少 1 個關鍵幀觀察到
                        if (pMP->beObservedNumber() > 0){
                            
                            // 地圖中匹配成功的地圖點
                            mvbMap[i] = true;
                        }
                        else{
                            // 視覺里程計中匹配成功的地圖點
                            mvbVO[i] = true;
                        }
                    }
                }
            }
        }

        mState = static_cast<int>(pTracker->mLastProcessedState);
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    
} //namespace ORB_SLAM
