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

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>
#include <unistd.h>
#include <mutex>

using namespace std;

namespace ORB_SLAM2
{
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    // 相機標定 與 建構 ORBextractor 物件等
    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer,
                       MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB,
                       const string &strSettingPath, const int sensor) : 
                       mpSystem(pSys), mpORBVocabulary(pVoc), mpFrameDrawer(pFrameDrawer),
                       mpMapDrawer(pMapDrawer), mpMap(pMap), mpKeyFrameDB(pKFDB), mSensor(sensor),
                       mState(NO_IMAGES_YET), mbOnlyTracking(false), mbVO(false), mpViewer(NULL),
                       mpInitializer(static_cast<Initializer *>(NULL)), mnLastRelocFrameId(0)
    {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        // 相機內參
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];

        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }

        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];
        float fps = fSettings["Camera.fps"];

        if (fps == 0)
        {
            fps = 30;
        }

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl
             << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;

        if (DistCoef.rows == 5)
        {
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        }

        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;

        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
        {
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        }
        else
        {
            cout << "- color order: BGR (ignored if grayscale)" << endl;
        }

        // Load ORB parameters
        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
        {
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, 
                                                                                fIniThFAST, fMinThFAST);
        }
        else if (sensor == System::MONOCULAR)
        {
            mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, 
                                                                                fIniThFAST, fMinThFAST);
        }

        cout << endl
             << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        if (sensor == System::STEREO || sensor == System::RGBD)
        {
            mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
            cout << endl
                 << "Depth Threshold (Close/Far Points): " << mThDepth << endl;

            if (sensor == System::RGBD)
            {
                mDepthMapFactor = fSettings["DepthMapFactor"];

                if (fabs(mDepthMapFactor) < 1e-5)
                {
                    mDepthMapFactor = 1;
                }
                else
                {
                    mDepthMapFactor = 1.0f / mDepthMapFactor;
                }
            }
        }
    }

    // **************************************************
    
    // 設置是否僅追蹤不建圖
    void Tracking::InformOnlyTracking(const bool &flag)
    {
        // 是否僅追蹤不建圖
        mbOnlyTracking = flag;
    }

    void Tracking::Reset()
    {
        /* 被呼叫的可能情境：
        1. 關鍵幀所觀察到的地圖點的深度為負數
        2. 至少被 1 個關鍵幀所觀察到的地圖點不足 100 個
        */

        cout << "System Reseting" << endl;

        if (mpViewer)
        {
            mpViewer->RequestStop();

            while (!mpViewer->isStopped()){
                usleep(3000);
            }
        }

        // Reset Local Mapping
        cout << "Reseting Local Mapper...";

        // 請求清空『新關鍵幀容器』以及『最近新增的地圖點』
        mpLocalMapper->RequestReset();
        
        cout << " done" << endl;

        // Reset Loop Closing
        cout << "Reseting Loop Closing...";

        // 請求重置狀態
        mpLoopClosing->RequestReset();
        
        cout << " done" << endl;

        // Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

        // Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        if (mpViewer){
            mpViewer->Release();
        }
    }

    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
    {
        mImGray = im;

        // 根據輸入影像的類型，轉換所需的灰階影像
        if (mImGray.channels() == 3)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            }
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            }
        }

        // mState 初始化時（第 1 幀）是 NO_IMAGES_YET
        // 建構當前幀 Frame 物件
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray, timestamp,
                                  mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        }
        else
        {
            mCurrentFrame = Frame(mImGray, timestamp,
                                  mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        }

        // 進行初始化
        Track();

        return mCurrentFrame.mTcw.clone();
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
    {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer)
    {
        mpViewer = pViewer;
    }

    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, 
                                      const double &timestamp)
    {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        mCurrentFrame = Frame(mImGray, 
                              imGrayRight, 
                              timestamp, 
                              mpORBextractorLeft, 
                              mpORBextractorRight, 
                              mpORBVocabulary, 
                              mK, 
                              mDistCoef, 
                              mbf, 
                              mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp)
    {
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if (mImGray.channels() == 3)
        {
            if (mbRGB){
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            }
            else{
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            }
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB){
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            }
            else{
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
            }
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F){
            imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);
        }

        mCurrentFrame = Frame(mImGray, 
                              imDepth, 
                              timestamp, 
                              mpORBextractorLeft, 
                              mpORBVocabulary, 
                              mK, 
                              mDistCoef, 
                              mbf, 
                              mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track()
    {
        /* ORB-SLAM 使用了三種方式來估計相機的位姿。
        1. 勻速模型：如果一切正常就使用勻速運動模型來粗略的估計相機位姿，在通過優化提高定位精度。
        2. 詞袋重定位：如果跟丟了，就通過詞袋模型進行重定位，重新確認參考關鍵幀。
        3. 參考關鍵幀：當使用勻速運動模型不能有效估計相機位姿，或者因為重定位的發生導致沒有速度估計或者幀 ID 發生跳轉時，
        ORB-SLAM 就會通過參考關鍵幀來估計位姿。*/

        // 第一張影像時，mState 會是 NO_IMAGES_YET
        if (mState == NO_IMAGES_YET)
        {
            // 第二張影像時，mState 會是 NOT_INITIALIZED
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        // 第一張影像時，mState 在前面被改為 NOT_INITIALIZED，而進入下方區塊，進行地圖初始化
        // Monocular 在前兩幀都會進入這個區塊進入初始化階段          
        if (mState == NOT_INITIALIZED)
        {
            if (mSensor == System::STEREO || mSensor == System::RGBD)
            {
                StereoInitialization();
            }
            else
            {
                // 第二幀時會進入 CreateInitialMapMonocular 將 mState 改為 OK
                MonocularInitialization();
            }

            mpFrameDrawer->Update(this);

            if (mState != OK)
            {
                return;
            }
        }

        // 地圖已初始化，開始估計相機位姿
        else
        {
            // System is initialized. Track Frame.
            bool bOK;

            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            // mbOnlyTracking 區分純定位和建圖兩種工作模式
            // 建圖模式
            if (!mbOnlyTracking)
            {
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.
                // 當系統狀態 mState 處於 OK 時，意味著當前的視覺里程計成功地跟上了相機的運動。
                // 如果跟丟了就只能進行重定位了。

                if (mState == OK)
                {
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    // 更新前一幀的地圖點，更換為被較多關鍵幀觀察到的地圖點
                    CheckReplacedInLastFrame();

                    // 速度估計丟失了，幀 ID 也可能因為重定位而向前发生了跳轉
                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                    {
                        // 利用詞袋模型，快速將『當前幀』與『參考關鍵幀』進行特徵點匹配，更新『當前幀』匹配的地圖點，
                        // 並返回數量是否足夠多
                        bOK = TrackReferenceKeyFrame();
                    }
                    else
                    {
                        /* 
                        如果上一幀圖像成功的跟上了相機的運動，並成功的估計了速度，就以勻速運動模型來估計相機的位姿。
                        並以此為參考，把上一幀中觀測到的地圖點投影到當前幀上，並一個比較小的範圍里搜索匹配的特征點。
                        如果找不到足夠多的匹配特征點，就適當的放大搜索半徑。
                        找到了足夠多的匹配特征點後，就進行一次優化提高相機位姿估計的精度。
                        
                        實現了基於勻速運動模型的跟蹤定位方法，假設當前幀的特徵點和前一幀位於差不多的位置，
                        進行兩幀之間特徵點的匹配，將匹配到的地圖點設置給當前幀，並返回是否匹配到足夠的點數。
                        */                        
                        bOK = TrackWithMotionModel();

                        // 若未能匹配到特徵點，則利用參考關鍵幀重新進行估計位姿
                        if (!bOK)
                        {
                            // 利用詞袋模型，快速將『當前幀』與『參考關鍵幀』進行特徵點匹配，
                            // 更新『當前幀』匹配的地圖點，並返回數量是否足夠多
                            bOK = TrackReferenceKeyFrame();
                        }
                    }
                }

                // 當系統狀態 mState 處於 OK 時，意味著當前的視覺里程計成功地跟上了相機的運動。
                // 無法成功估計位姿，意味著我們跟丟了，需要進行重定位
                else
                {
                    // 將有相同『重定位詞』的關鍵幀篩選出來後，選取有足夠多內點的作為重定位的參考關鍵幀，
                    // 並返回是否成功重定位
                    bOK = Relocalization();
                }
            }

            // 定位模式
            else
            {
                // Localization Mode: Local Mapping is deactivated

                if (mState == LOST)
                {
                    bOK = Relocalization();
                }
                else
                {
                    if (!mbVO)
                    {
                        // In last frame we tracked enough MapPoints in the map

                        if (!mVelocity.empty())
                        {
                            bOK = TrackWithMotionModel();
                        }
                        else
                        {
                            bOK = TrackReferenceKeyFrame();
                        }
                    }
                    else
                    {
                        // In last frame we tracked mainly "visual odometry" points.

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint *> vpMPsMM;
                        vector<bool> vbOutMM;
                        cv::Mat TcwMM;

                        if (!mVelocity.empty())
                        {
                            bOKMM = TrackWithMotionModel();
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.mTcw.clone();
                        }

                        bOKReloc = Relocalization();

                        if (bOKMM && !bOKReloc)
                        {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;

                            if (mbVO)
                            {
                                for (int i = 0; i < mCurrentFrame.N; i++)
                                {
                                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                    {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        }
                        else if (bOKReloc)
                        {
                            mbVO = false;
                        }

                        bOK = bOKReloc || bOKMM;
                    }
                }
            }

            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            /* 通過勻速運動模型或者重定位，我們已經得到了相機位姿的初始估計，同時找到了一個與當前幀初步匹配的地圖點集合。
            此時，我們可以把地圖投影到當前幀上找到更多的匹配地圖點。進而再次優化相機的位姿，得到更為精準的估計。
            為了限制這一搜索過程的覆雜度，ORB-SLAM 只對一個局部地圖進行投影。這個局部地圖包含了兩個關鍵幀集合 K1、K2，
            K1 中的關鍵幀都與當前幀有共視的地圖點，K2 則是 K1 的元素在共視圖中的臨接節點。*/
            if (!mbOnlyTracking)
            {
                // 在建圖模式下，Tracking 根據局部變量 bOK 控制是否進行局部地圖的更新。
                // 它反映了是否成功的估計了相機位姿，若沒有則意味著當前視覺里程計跟丟了，再進行局部地圖更新就沒有意義了。
                if (bOK)
                {
                    /* 函數 TrackLocalMap 具體完成了局部地圖的更新以及相機位姿的進一步優化工作，
                    它將返回一個布爾數據表示更新和優化工作是否成功。
                    只有成功地完成了這一項任務，我們才能認為視覺里程計真正完成了位姿跟蹤的操作。
                    
                    找出當前幀的『共視關鍵幀』以及其『已配對地圖點』，確保這些地圖點至少被 1 個關鍵幀觀察到，
                    且重投影後的內點足夠多
                    */                    
                    bOK = TrackLocalMap();
                }
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map.
                // We cannot retrieve a local map and therefore we do not perform TrackLocalMap().
                // Once the system relocalizes the camera we will use the local map again.
                if (bOK && !mbVO)
                {
                    bOK = TrackLocalMap();
                }
            }

            if (bOK)
            {
                mState = OK;
            }
            else
            {
                mState = LOST;
            }

            // Update drawer
            // 初始化後，根據是否被關鍵幀觀察到，將當前幀的所有關鍵點區分為『地圖點 MapPoint』或『視覺里程計 VO』
            mpFrameDrawer->Update(this);

            // If tracking were good, check if we insert a keyframe
            // 如果此時局部變量 bOK 仍然為真，說明視覺里程計是跟上了相機的運動。
            if (bOK)
            {
                // Update motion model
                if (!mLastFrame.mTcw.empty())
                {
                    // 前一幀的轉換矩陣
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);

                    // 旋轉矩陣（相機座標 → 世界座標）
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));

                    // 取得相機中心世界座標
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));

                    // 與當前幀的位姿齊次矩陣相乘，求得當前幀相對於上一幀的位姿變換，並將結果保留在 mVelocity 中。
                    // mCurrentFrame.mTcw = world to current; LastTwc = last to world
                    // 因此 mCurrentFrame.mTcw * LastTwc = last to current
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                }
                else
                {
                    mVelocity = cv::Mat();
                }

                // 設置當前幀的位姿
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    if (pMP)
                    {
                        // 若該地圖點未被任一關鍵幀觀察到，則從當前幀的地圖點中移除
                        /// NOTE: 這些點在 mpFrameDrawer 中被標注為『視覺里程計 VO』
                        if (pMP->Observations() < 1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                    }
                }

                // Delete temporal MapPoints
                // mlpTemporalPoints 和『單目模式』與『建圖模式』無關，暫時跳過
                list<MapPoint *>::iterator lit, lend = mlpTemporalPoints.end();

                for (lit = mlpTemporalPoints.begin(); lit != lend; lit++)
                {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }

                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                // 判定是否生成關鍵幀
                if (NeedNewKeyFrame())
                {
                    // 生成關鍵幀
                    CreateNewKeyFrame();
                }

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    {
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
                }
            }

            // Reset if the camera get lost soon after initialization
            // 如果跟丟了，而且當前系統還沒有足夠多的關鍵幀，將重置系統，重新初始化
            if (mState == LOST)
            {
                // 地圖中的關鍵幀數量不足 6 個
                if (mpMap->KeyFramesInMap() <= 5)
                {
                    cout << "Track lost soon after initialisation, reseting..." << endl;
                    mpSystem->Reset();
                    return;
                }
            }

            // 更新當前幀的狀態和對象，方便下次叠代
            if (!mCurrentFrame.mpReferenceKF)
            {
                mCurrentFrame.mpReferenceKF = mpReferenceKF;
            }

            // 更新上一幀的狀態和對象，方便下次叠代
            mLastFrame = Frame(mCurrentFrame);
        }

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        // 根據當前幀的位姿估計是否存在，更新 Tracking 對象的一些狀態
        if (!mCurrentFrame.mTcw.empty())
        {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }
    }

    void Tracking::StereoInitialization()
    {
        // 檢查當前幀的特征點數量，如果太少就放棄了
        if (mCurrentFrame.N > 500)
        {
            // Set Frame pose to the origin
            // 將當前幀的姿態設定到原點上，並以此構建關鍵幀添加到地圖對象 mpMap 中。
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            // 檢查當前幀的特征點，如果有深度信息，就依此信息將之還原到 3D 物理世界中，新建地圖點並將之與關鍵幀關聯上。
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];

                if (z > 0)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }

            cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

            // ==================================================
            // ===== 更新系統的相關變量和狀態 =====
            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            // 將 mvpLocalMapPoints 設置為參考用地圖點
            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
            // ==================================================
        }
    }

    void Tracking::MonocularInitialization()
    {
        /* 單目相機整個初始化過程可以總結為六個步驟：
        1. 計算兩幀圖像的 ORB 特征點，並進行匹配；
        2. 在兩個線程中以RANSAC策略並行的計算單應矩陣和基礎矩陣；
        3. 根據評判標準在單應矩陣和基礎矩陣之間選擇一個模型；
        4. 根據選定的模型分解相機的旋轉矩陣和平移向量，並對匹配的特征點進行三角化；
        5. 建立關鍵幀和地圖點，進行完全BA(full BA)優化；
        6. 以參考幀下深度的中位數為基準建立基礎尺度；
        */

        // 檢查指針 mpInitializer，如果它是個空指針就以當前幀作為參考具例化 mpInitializer。
        if (!mpInitializer)
        {
            // Set Reference Frame
            // 為了保證初始化的品質，只有當圖像的特征點數量超過了 100 個，我們才進行初始化操作。
            if (mCurrentFrame.mvKeys.size() > 100)
            {
                // 在具例化 mpInitializer 之前， 先以當前幀作為整個系統的初始幀，記錄在 mInitialFrame，
                // 同時用 mLastFrame 記錄當前幀。
                // 參數為 Frame 時，表示此為複製建構，之前在 Frame 當中初始化的物件直接複製一份，不須再初始化一次
                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);

                // 將扭曲校正後的特征點保存到 mvbPrevMatched 中
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());

                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                {
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
                }

                if (mpInitializer)
                {
                    delete mpInitializer;
                }

                // 根據當前幀具例化初始化器
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                // 將 mvIniMatches 中所有的特征點匹配值都置為 -1，表示它們都還沒有匹配。
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }
        }

        // 如果 mpInitializer 已經被具例化了，說明我們已經有了一個參考幀。也就是已經第二幀了。
        else
        {
            // Try to initialize
            // 我們再次檢查當前幀的特征點數量，如果太少將銷毀參考幀，重新構建mpInitializer。
            if ((int)mCurrentFrame.mvKeys.size() <= 100)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // Find correspondences
            // 創建了一個 ORB 特征點匹配器
            ORBmatcher matcher(0.9, true);

            // 通過接口 SearchForInitialization 針對初始化進行特征匹配
            // 初始化時，將 Frame 中的已校正關鍵點加入 mvbPrevMatched
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame,
                                                           mvbPrevMatched, mvIniMatches, 100);

            // Check if there are enough correspondences
            // 如果匹配點數量少於 100 個， 認為匹配的特征點數量太少，不適合初始化。
            if (nmatches < 100)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            // Current Camera Rotation
            cv::Mat Rcw;

            // Current Camera Translation
            cv::Mat tcw;

            // Triangulated Correspondences (mvIniMatches)
            vector<bool> vbTriangulated;

            // 通過初始化器 mpInitializer 的接口 Initialize 完成初始化操作，估計相機的旋轉量和平移量，
            // 並對特征點進行三角化形成初始的地圖點。這些結果將被保存在 Rcw, tcw, mvIniP3D, vbTriangulated 中。
            // 同時利用『單應性矩陣』和『基礎矩陣』估計空間點 vP3D（過程中包含估計旋轉和平移），挑選較佳的估計結果
            // 並確保『兩相機間有足夠的夾角，且分別相機上的重投影誤差都足夠小』，返回是否順利估計
            // 估計 旋轉 Rcw, 平移 tcw, 空間點位置 mvIniP3D
            if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, 
                                          mvIniP3D, vbTriangulated))
            {
                // 完成了初始化，但並不是所有匹配的特征點都能夠成功進行三角化的。
                // 所以還需要根據 vbTriangulated 進一步的篩除未成功三角化的點。
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
                {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);

                // Tcw 的旋轉矩陣
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));

                // Tcw 的平移
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));

                // 將 mCurrentFrame 的位姿設為 Tcw
                mCurrentFrame.SetPose(Tcw);

                // 調用函數 CreateInitialMapMonocular 來完成地圖的初始化工作。
                CreateInitialMapMonocular();
            }
        }
    }

    void Tracking::CreateInitialMapMonocular()
    {
        // Create KeyFrames
        // 先根據參考幀和當前幀創建兩個關鍵幀(KeyFrame)
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // 更新詞袋
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        // 將兩個關鍵幀插入地圖中
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to keyframes
        // 根據成功三角化的特征點創建地圖點(MapPoint)，並建立起地圖點與關鍵幀之間的可視關系。
        for (size_t i = 0; i < mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] < 0)
            {
                continue;
            }

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            // 關鍵幀紀錄觀察到的地圖點
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            // 地圖點紀錄被哪些關鍵幀觀察到
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // 以『所有描述這個地圖點的描述子的集合』的中心描述子，作為地圖點的描述子
            pMP->ComputeDistinctiveDescriptors();

            // 利用所有觀察到這個地圖點的關鍵幀來估計關鍵幀們平均指向的方向，以及該地圖點可能的深度範圍(最近與最遠)
            pMP->UpdateNormalAndDepth();

            // Fill Current Frame structure
            // 當前幀的第 i 個匹配成功的特徵點索引值 j，其所觀察到的第 j 個地圖點是 pMP 
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;

            // 觀察到第 j 個地圖點，因此不會是 Outlier
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            // Add to Map
            mpMap->AddMapPoint(pMP);
        }

        // Update Connections
        // 根據地圖點和關鍵幀之間的關系提取出共視圖(covisibility graph)和基圖(essential graph)
        // 更新關鍵幀的連接關系
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // 最後進行一次全局的 BA 優化，就完成了初始地圖的構建。
        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        // 對於單目而言尺度仍然是缺失的，所以還須進一步的以當前地圖點深度的中位數為基礎建立基礎尺度，
        // 以後的地圖重建工作都是在這個基礎的尺度上進行的
        // ComputeSceneMedianDepth 取得當前關鍵幀的座標系之下，關鍵幀觀察到的所有地圖點的深度中位數
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);

        // 深度中位數的倒數
        float invMedianDepth = 1.0f / medianDepth;

        // 檢查該深度（深度中位數）以及當前關鍵幀跟蹤上的地圖點數量
        // pKFcur->TrackedMapPoints(1) 有多少個地圖點，至少被 1 個關鍵幀所觀察到的
        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
        {
            cout << "Wrong initialization, reseting..." << endl;

            // 暫停地圖的呈現，直到 Tracking 再次喚醒 Viewer 的執行續
            Reset();

            return;
        }

        // Scale initial baseline
        // 初始化基線和地圖點的尺度
        cv::Mat Tc2w = pKFcur->GetPose();

        // 轉換矩陣的平移部份 乘上 深度中位數的倒數，用於控制地圖規模
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;

        // 更新位姿
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();

        for(MapPoint *pMP : vpAllMapPoints){

            if (pMP)
            {
                // 地圖點的位置 乘上 深度中位數的倒數，用於控制地圖規模
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);

        // 從地圖中取得所有地圖點
        mvpLocalMapPoints = mpMap->GetAllMapPoints();

        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        /// NOTE: 推測『mvpLocalMapPoints 從 mpMap 取出，再傳入 mpMap』，是為了
        /// 將 mspMapPoints 複製一份給 mvpReferenceMapPoints 來繪製地圖，兩者於不同執行續的管理之下，
        /// 繪製地圖的同時，mspMapPoints 可以繼續新增、刪減地圖點，而不影響繪圖
        // 將 mvpLocalMapPoints 設置為參考用地圖點
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
    }

    // 更新前一幀的地圖點，更換為被較多關鍵幀觀察到的地圖點
    void Tracking::CheckReplacedInLastFrame()
    {
        for(MapPoint *pMP : mLastFrame.mvpMapPoints){

            if (pMP)
            {
                // 更換為被較多關鍵幀觀察到的地圖點
                MapPoint *pRep = pMP->GetReplaced();

                if (pRep)
                {
                    pMP = pRep;
                }
            }
        }
    }

    // 利用詞袋模型，快速將『當前幀』與『參考關鍵幀』進行特徵點匹配，更新『當前幀』匹配的地圖點，並返回數量是否足夠多
    bool Tracking::TrackReferenceKeyFrame()
    {
        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        // 構建ORB特征匹配器
        ORBmatcher matcher(0.7, true);

        vector<MapPoint *> vpMapPointMatches;

        // 利用詞袋模型，快速將當前幀與參考關鍵幀進行特征點匹配，
        // 匹配到的地圖點保存在臨時的容器 vpMapPointMatches 中，nmatches 將獲得匹配的特征點數量。
        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

        // 如果匹配的點太少就認為跟丟了。
        if (nmatches < 15)
        {
            return false;
        }

        // 用 vpMapPointMatches 更新當前幀匹配的地圖點
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;

        // 使用上一幀的位姿變換作為當前幀的優化初值
        mCurrentFrame.SetPose(mLastFrame.mTcw);

        // 優化『mCurrentFrame 觀察到的地圖點』的位置，以及 mCurrentFrame 的位姿估計，並返回優化後的內點個數
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        MapPoint *pMP;

        // 對野點(Outlier)進行篩選
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                // 若地圖點為 Outlier
                if (mCurrentFrame.mvbOutlier[i])
                {
                    pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }

                // 若地圖點被至少 1 個關鍵幀觀察到
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                {
                    nmatchesMap++;
                }
            }
        }

        // 實際匹配到的地圖點數量是否足夠多
        return nmatchesMap >= 10;
    }

    // 和『單目模式』與『建圖模式』無關，暫時跳過
    void Tracking::UpdateLastFrame()
    {
        // Update pose according to reference keyframe
        // 取出前一幀的參考關鍵幀
        KeyFrame *pRef = mLastFrame.mpReferenceKF;

        // 取出『前一幀和其參考關鍵幀之間的位姿轉換』，即轉換矩陣
        cv::Mat Tlr = mlRelativeFramePoses.back();

        // 參考關鍵幀位姿 乘上 『參考關鍵幀到前一幀的轉換矩陣』，得到前一幀的位姿
        mLastFrame.SetPose(Tlr * pRef->GetPose());

        // 若為單目模式，直接返回
        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking){
            return;
        }

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);

        for (int i = 0; i < mLastFrame.N; i++)
        {
            float z = mLastFrame.mvDepth[i];

            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (vDepthIdx.empty())
            return;

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;

        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
            }

            if (bCreateNew)
            {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }
    }

    // 實現了基於勻速運動模型的跟蹤定位方法，假設當前幀的特徵點和前一幀位於差不多的位置，進行兩幀之間特徵點的匹配，
    // 將匹配到的地圖點設置給當前幀，並返回是否匹配到足夠的點數
    bool Tracking::TrackWithMotionModel()
    {
        // 第一個參數是一個接受最佳匹配的系數，只有當最佳匹配點的漢明距離小於次加匹配點距離的 0.9 倍時才接收匹配點，
        // 第二個參數表示匹配特征點時是否考慮方向。
        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        // 在純定位模式下構建一個視覺里程計，對於建圖模式作用不大
        // 和『單目模式』與『建圖模式』無關，暫時跳過
        UpdateLastFrame();

        /* 如果上一幀圖像成功的跟上了相機的運動，並成功的估計了速度，就以勻速運動模型來估計相機的位姿。
        所謂的勻速運動模型，就是假設從上一幀到當前幀的這段時間里機器人的線速度和角速度都沒有變化，
        直接通過速度和時間間隔估計前後兩幀的相對位姿變換，再將之直接左乘到上一幀的位姿上，從而得到當前幀的位姿估計。*/
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        // 將當前幀的地圖點設置為 NULL
        // fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
        //      static_cast<MapPoint *>(NULL));
        mCurrentFrame.resetMappoints();

        // Project points seen in previous frame
        int th;

        if (mSensor != System::STEREO)
        {
            th = 15;
        }
        else
        {
            th = 7;
        }

        // 把上一幀中觀測到的地圖點投影到當前幀上，並一個比較小的範圍里搜索匹配的特征點。
        // 這個接口有四個參數，前兩個分別是當前幀和上一幀。
        // 第三個參數 th 是一個控制搜索半徑的參數，最後一個參數用於判定是否為單目相機。
        // 尋找 CurrentFrame 當中和 LastFrame 特徵點對應的位置，形成 CurrentFrame 的地圖點，並返回匹配成功的個數
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th,
                                                  mSensor == System::MONOCULAR);

        // If few matches, uses a wider window search
        // 如果找不到足夠多的匹配特征點，就適當的放大搜索半徑(th -> 2 * th)。
        if (nmatches < 20)
        {
            // fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
            //      static_cast<MapPoint *>(NULL));
            mCurrentFrame.resetMappoints();
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th,
                                                  mSensor == System::MONOCULAR);
        }

        // 若擴大搜索半徑後，仍找不到足夠多的匹配特征點，則返回 false，表示『基於勻速運動模型的跟蹤定位方法』失敗了
        if (nmatches < 20)
        {
            return false;
        }

        // Optimize frame pose with all matches
        // 找到了足夠多的匹配特征點後，就進行一次優化提高相機位姿估計的精度。
        // 優化『pFrame 觀察到的地圖點』的位置，以及 pFrame 的位姿估計，並返回優化後的內點個數
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;

        // 進一步的檢查了當前幀看到的各個地圖點，拋棄了那些外點(outlier)。
        // 它定義了一個局部整型 nmatchesMap 用於記錄當前幀實際看到的地圖點數量，並在一個 for 循環中遍歷所有特征點。
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                /* 當前幀對象的容器 mvpMapPoints 和 mvbOutlier 與特征點是一一對應的， 
                mvpMapPoints 記錄了各個特征點所對應的地圖點指針，如果沒有對應地圖點則為 NULL。
                mvbOutlier 記錄了各個特征點是否為外點(在優化的時候會更新這個狀態)，若是野點則直接拋棄之。*/
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    // 將被認定為外點（第 i 個特徵點）對應的地圖點拋棄
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);

                    // 由於沒有地圖點，因此第 i 個特徵點也不再被認定為外點
                    mCurrentFrame.mvbOutlier[i] = false;

                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }

                // 若 mCurrentFrame 所觀察到的第 i 個地圖點，被至少 1 個關鍵幀觀察到
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                {
                    nmatchesMap++;
                }
            }
        }

        // 在最後退出的時候，檢查剩下多少匹配的地圖點，如果太少就認為跟丟了。
        if (mbOnlyTracking)
        {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        // 檢查 mCurrentFrame 匹配到的特徵點（其對應的地圖點被至少 1 個關鍵幀觀察到）是否足夠多（至少 10 個）
        return nmatchesMap >= 10;
    }

    // 找出當前幀的『共視關鍵幀』以及其『已配對地圖點』，確保這些地圖點至少被 1 個關鍵幀觀察到，且重投影後的內點足夠多
    bool Tracking::TrackLocalMap()
    {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.
        /* 可以分為三個階段：
        1. 先更新局部地圖並進行優化，
        2. 再統計優化位姿後仍然匹配的地圖點數量，
        3. 最後根據匹配的地圖點數量判定局部地圖更新任務是否成功。*/

        // 用於計算關鍵幀集合 K1、K2。
        // 設置參考用地圖點，更新當前幀的『共視關鍵幀』以及『共視關鍵幀的共視關鍵幀』，以及其『已配對地圖點』
        UpdateLocalMap();

        /* 通過 UpdateLocalKeyFrames 得到的局部地圖中的地圖點是很多的， Tracking 還需要通過
        成員函數 SearchLocalPoints 來對這些地圖點進行進一步的篩選。
        根據由 K1、K2 構成的局部地圖，將局部地圖中的地圖點投影到當前幀上，根據視角以及深度信息篩選匹配的地圖點。*/
        SearchLocalPoints();

        // 優化『pFrame 觀察到的地圖點』的位置，以及 pFrame 的位姿估計，並返回優化後的內點個數
        Optimizer::PoseOptimization(&mCurrentFrame);
        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        // 在進一步位姿優化之後，統計仍然匹配的地圖點數量
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            // 如果位姿優化之後仍然能夠匹配到地圖點，就認為在該幀中找到了地圖點
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    // 相應的調用地圖點接口 IncreaseFound 增加 mnFound 計數（實際能觀察到該地圖點的特徵點數）。
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();

                    // 純定位模式
                    if (mbOnlyTracking)
                    {
                        mnMatchesInliers++;
                    }

                    // 建圖模式
                    else
                    {
                        // 若這個地圖點，至少被 1 個關鍵幀觀察到
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        {
                            mnMatchesInliers++;
                        }
                        
                    }
                }
                else if (mSensor == System::STEREO)
                {
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        // 『自上一幀到現在的幀數，少於創建一關鍵幀所需的幀數』且『匹配到的內點不足 50 點』，則表示重定位失敗。
        // 由於發生重定位，因此這裡設置了較嚴苛的條件（內點至少 50 點），
        // 而本來只要求位姿估計優化前至少 20 點，優化後至少 10 點。
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        {
            return false;
        }

        // 內點至少 30 點，才算重定位成功
        return mnMatchesInliers >= 30;
    }

    // 判定是否生成關鍵幀
    bool Tracking::NeedNewKeyFrame()
    {
        /* ORB-SLAM 論文中說要插入新的關鍵幀需要滿足如下的幾個條件：
        1. 如果發生了重定位，那麽需要在 20 幀之後才能添加新的關鍵幀。保證了重定位的效果。
        2. LOCAL MAPPING 線程處於空閑(idle)的狀態，或者距離上次插入關鍵幀已經過去了 20 幀。
           保證 LOCAL MAPPING 盡快的處理。
        3. 當前幀至少保留了 50 個匹配的地圖點。保證了軌跡跟蹤的效果。
        4. 當前幀跟蹤的地圖點數量少於參考幀的 90% 。是一個比較嚴苛的約束可以適當的降低關鍵幀數量。
        */

        // 與定位模式無關
        if (mbOnlyTracking)
        {
            return false;
        }

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        // 如果 LOOP CLOSURE 線程檢測到了閉環，那麽 LOCAL MAPPING 將被暫停，此時也不會插入新的關鍵幀的。
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        {
            return false;
        }

        // 地圖中的關鍵幀數量
        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        // mMaxFrames 在 Tracking 的構造函數里的賦值是相機的幀率，它是可以通過配置文件中的字段 Camera.fps 調整的
        // 若創建關鍵幀所需的幀數尚不足，或地圖中的關鍵幀數量超過上限，則返回
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        {
            return false;
        }

        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;

        if (nKFs <= 2)
        {
            // 要追蹤參考關鍵幀當中地圖點
            nMinObs = 2;
        }

        // 有多少個地圖點，是被足夠多的關鍵幀所觀察到的
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        // 是否接受關鍵幀（此時 LOCAL MAPPING 線程是否處於空閑的狀態）
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;

        // 與單目無關，暫時跳過
        if (mSensor != System::MONOCULAR)
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    {
                        nTrackedClose++;
                    }
                    else
                    {
                        nNonTrackedClose++;
                    }
                }
            }
        }

        // 單目的 bNeedToInsertClose 應該會是 false，因為直接跳過上面那個區塊
        bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Thresholds
        float thRefRatio = 0.75f;

        if (nKFs < 2)
        {
            thRefRatio = 0.4f;
        }

        // 條件 4 要求『當前幀跟蹤的地圖點數量少於參考幀的 90%（thRefRatio）』
        if (mSensor == System::MONOCULAR)
        {
            thRefRatio = 0.9f;
        }

        // 條件2. LOCAL MAPPING 線程處於空閑(idle)的狀態（bLocalMappingIdle），
        //       或者距離上次插入關鍵幀已經過去了 20 幀。保證 LOCAL MAPPING 盡快的處理。
        // c1a 和 c1b 其實是對 條件 2 的實現
        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;

        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);

        // Condition 1c: tracking is weak
        // c1c 對於單目沒有意義
        const bool c1c = mSensor != System::MONOCULAR &&
                         (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);

        // 條件4. 當前幀跟蹤的地圖點數量少於參考幀的 90% 。是一個比較嚴苛的約束可以適當的降低關鍵幀數量。
        // c2 是對 條件 4 的實現
        // 變量 bNeedToInsertClose 並不是對單目的限制，c2 最後的與條件里其實放寬了對 條件 3 的限制
        // Condition 2: Few tracked points compared to reference keyframe.
        // Lots of visual odometry compared to map matches.
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
                         mnMatchesInliers > 15);

        // 在滿足了條件 1, 2, 3, 4 後，就應當返回是否新建關鍵幀的判定結果
        if ((c1a || c1b || c1c) && c2)
        {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            // 若 LOCAL MAPPING 線程處於空閑(idle)的狀態
            if (bLocalMappingIdle)
            {
                return true;
            }

            // 若 LOCAL MAPPING 線程處於繁忙的狀態
            else
            {
                // 如果 LOCAL MAPPING 線程並沒有處於空閑狀態的話，貿然插入新的關鍵幀可能對其局部 BA 優化有影響，
                // 所以只是發送了一個信號要求 LOCAL MAPPING 暫停 BA 優化。
                mpLocalMapper->InterruptBA();

                // 和單目無關，暫時跳過
                if (mSensor != System::MONOCULAR)
                {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                    {
                        return true;
                    }
                }
            }
        }
        
        return false;
    }

    // 生成關鍵幀
    void Tracking::CreateNewKeyFrame()
    {
        // 首先檢查 LOCAL MAPPING 是否可以插入新關鍵幀
        if (!mpLocalMapper->SetNotStop(true))
        {
            return;
        }

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // 更新當前的參考關鍵幀和當前幀的參考關鍵幀
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        // 處理非單目，故暫時跳過
        if (mSensor != System::MONOCULAR)
        {
            mCurrentFrame.UpdatePoseMatrices();

            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            vector<pair<float, int>> vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);

            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];

                if (z > 0)
                {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty())
            {
                sort(vDepthIdx.begin(), vDepthIdx.end());
                int nPoints = 0;

                for (size_t j = 0; j < vDepthIdx.size(); j++)
                {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1)
                    {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    if (bCreateNew)
                    {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                        pNewMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    }
                    else
                    {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                        break;
                }
            }
        }

        // 通知 mpLocalMapper 有新的關鍵幀插入，將新生成的關鍵幀提供給了 LocalMapping 對象
        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        // 更新成員變量 mnLastKeyFrameId 和 mpLastKeyFrame
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    void Tracking::SearchLocalPoints()
    {
        // Do not search map points already matched
        vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin();
        vector<MapPoint *>::iterator vend = mCurrentFrame.mvpMapPoints.end();
        
        /* 遍歷了當前幀的所有地圖點，增加可視幀計數，同時更新其成員變量 mnLastFrameSeen。
        這些地圖點是位姿估計時得到的與當前幀匹配的地圖點，它們有可能出現在局部地圖關鍵幀集合 K1、K2 中。 
        由於它們已經是匹配的了，所以不再需要進行投影篩選了，因此更新其成員變量 mnLastFrameSeen。*/
        for (; vit != vend; vit++)
        {
            MapPoint *pMP = *vit;

            if (pMP)
            {
                if (pMP->isBad())
                {
                    *vit = static_cast<MapPoint *>(NULL);
                }
                else
                {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        // 記錄了通過篩選的地圖點數量
        int nToMatch = 0;

        // Project points in frame and check its visibility
        // 對局部地圖中的地圖點進行投影篩選
        for(MapPoint *pMP : mvpLocalMapPoints){

            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            {
                continue;
            }

            if (pMP->isBad())
            {
                continue;
            }

            // Project (this fills MapPoint variables for matching)
            /* 通過當前幀的接口函數 isInFrustum 完成實際的投影篩選工作。
            這個函數有兩個參數，第一個參數就是待篩的地圖點指針，
            第二個參數則是拒絕待篩地圖點的視角余弦閾值，這里是 0.5 = cos60◦。*/
            if (mCurrentFrame.isInFrustum(pMP, 0.5))
            {
                pMP->IncreaseVisible();
                nToMatch++;
            }
        }

        // 如果最後發現有地圖點通過了篩選，就對當前幀進行一次投影特征匹配，擴展匹配地圖點。
        if (nToMatch > 0)
        {
            ORBmatcher matcher(0.8);

            // 這里定義的局部參數 th 限定了投影搜索範圍，該值越大則搜索範圍就越大。
            int th = 1;

            if (mSensor == System::RGBD)
            {
                th = 3;
            }

            // If the camera has been relocalised recently, perform a coarser search
            // 發生了重定位，此時需要適當的放大搜索範圍
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            {
                th = 5;
            }

            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    // 設置參考用地圖點，更新當前幀的『共視關鍵幀』以及『共視關鍵幀的共視關鍵幀』，以及其『已配對地圖點』
    void Tracking::UpdateLocalMap()
    {
        /* 要更新的這個局部地圖包含了兩個關鍵幀集合 K1、K2，
        K1 中的關鍵幀都與當前幀有共視的地圖點，
        K2 則是 K1 的元素在共視圖中的臨接節點。
        可以通過遍歷共視圖的臨接表來計算局部地圖*/

        // 將 mvpLocalMapPoints 設置為參考用地圖點
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        // 計算關鍵幀集合K1、K2
        // 更新 mvpLocalKeyFrames 為當前幀的『共視關鍵幀』以及『共視關鍵幀的共視關鍵幀』
        UpdateLocalKeyFrames();

        // 搜索局部地圖中的地圖點
        // 更新當前幀的『共視關鍵幀』以及『共視關鍵幀的共視關鍵幀』的『已配對地圖點』
        UpdateLocalPoints();
    }

    // 更新當前幀的『共視關鍵幀』以及『共視關鍵幀的共視關鍵幀』的『已配對地圖點』
    void Tracking::UpdateLocalPoints()
    {
        mvpLocalMapPoints.clear();

        // UpdateLocalKeyFrames 當中更新的當前幀的『共視關鍵幀』以及『共視關鍵幀的共視關鍵幀』
        /* 遍歷剛剛計算的關鍵幀集合 K1、K2，把它們的地圖點一個個的都給摳出來，放到容器 mvpLocalMapPoints 中。
        通過地圖點的成員變量 mnTrackReferenceForFrame 來防止重覆添加地圖點。*/
        for(KeyFrame *pKF : mvpLocalKeyFrames){

            // 『共視關鍵幀 pKF』的已配對地圖點
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            // 遍歷『共視關鍵幀 pKF』的已配對地圖點
            for(MapPoint *pMP : vpMPs){

                if (!pMP)
                {
                    continue;
                }

                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                {
                    continue;
                }

                if (!pMP->isBad())
                {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    // 更新 mvpLocalKeyFrames 為當前幀的『共視關鍵幀』以及『共視關鍵幀的共視關鍵幀』
    void Tracking::UpdateLocalKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        // 紀錄各『關鍵幀』分別觀察到幾次『當前幀』的地圖點，描述了共視關係
        map<KeyFrame *, int> keyframeCounter;

        MapPoint *pMP;

        // 遍歷了當前幀的每一個地圖點
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                pMP = mCurrentFrame.mvpMapPoints[i];

                if (!pMP->isBad())
                {
                    // 取得觀察到『地圖點 pMP』的關鍵幀，以及是與該關鍵幀的哪個特徵點相對應
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();

                    // 遍歷這些關鍵幀，累積共視地圖點數量。
                    for(pair<KeyFrame *, size_t> obs : observations){

                        keyframeCounter[obs.first]++;
                    }
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        // 如果容器 keyframeCounter 為空，意味著沒有找到與當前幀具有共視關系的關鍵幀，集合 K1、K2 為空，直接返回。
        if (keyframeCounter.empty())
        {
            return;
        }

        // 創建了臨時變量 pKFmax 和 max 用於記錄與當前幀具有最多共視地圖點的關鍵幀和共視地圖點數量
        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        // 紀錄有觀察到相同點的關鍵幀
        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map.
        // Also check which keyframe shares most points
        // 遍歷局部 map 容器 keyframeCounter 中的所有關鍵幀，將之保存到成員容器 mvpLocalKeyFrames 中
        for(pair<KeyFrame *, int> kf_counter : keyframeCounter){

            KeyFrame *pKF = kf_counter.first;

            if (pKF->isBad()){
                continue;
            }

            int counter = kf_counter.second;

            // it->second：觀察到當前幀地圖點的次數
            // 更新觀察到最多次的關鍵幀，及其次數
            if (counter > max)
            {
                // 更新 pKFmax 和 max
                max = counter;
                pKFmax = pKF;
            }

            // 取出有觀察到相同點的關鍵幀
            mvpLocalKeyFrames.push_back(pKF);

            // 紀錄『提供哪一幀作為參考幀』
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }

        // Include also some not-already-included keyframes that are neighbors to
        // already-included keyframes
        // 計算集合 K2，遍歷集合 K1 中的所有元素，獲取它們在共視圖中的臨接節點。
        vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin();
        vector<KeyFrame *>::const_iterator itEndKF = mvpLocalKeyFrames.end();

        KeyFrame *pParent;

        // 遍歷共視關鍵幀
        for (; itKF != itEndKF; itKF++)
        {
            // Limit the number of keyframes
            // 為了節約計算資源，ORB-SLAM2 將集合 K1、K2 中的關鍵幀數量限制在了 80 幀。
            if (mvpLocalKeyFrames.size() > 80)
            {
                break;
            }

            // 共視關鍵幀 pKF
            KeyFrame *pKF = *itKF;

            // 獲取考察關鍵幀的共視圖臨接節點，在調用的時候為之傳遞了一個參數 10，表示獲取最多 10 個臨接關鍵幀
            // GetBestCovisibilityKeyFrames：根據觀察到的地圖點數量排序的共視關鍵幀
            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for(KeyFrame *pNeighKF : vNeighs){

                if (!pNeighKF->isBad())
                {
                    // 若未曾作為當前幀的『參考關鍵幀』
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        // 把共視關鍵幀放入容器 mvpLocalKeyFrames 中
                        mvpLocalKeyFrames.push_back(pNeighKF);

                        // 更新每個關鍵幀的成員變量 mnTrackReferenceForFrame 為當前幀的 ID，
                        // 以防止重覆添加某個關鍵幀
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            // 取出與『共視關鍵幀 pKF』有高度共視程度的關鍵幀
            const set<KeyFrame *> spChilds = pKF->GetChilds();

            for(KeyFrame *pChildKF : spChilds){

                if (!pChildKF->isBad())
                {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        // 將有高度共視程度的關鍵幀加入 mvpLocalKeyFrames 進行管理
                        mvpLocalKeyFrames.push_back(pChildKF);

                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            // 取得『共視關鍵幀 pKF』的父關鍵幀
            pParent = pKF->GetParent();

            if (pParent)
            {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    // 將父關鍵幀加入 mvpLocalKeyFrames 進行管理
                    mvpLocalKeyFrames.push_back(pParent);

                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;

                    break;
                }
            }
        }

        // 更新當前幀的參考關鍵幀為 K1 中具有最多共視地圖點的關鍵幀
        if (pKFmax)
        {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    // 將有相同『重定位詞』的關鍵幀篩選出來後，選取有足夠多內點的作為重定位的參考關鍵幀，並返回是否成功重定位
    bool Tracking::Relocalization()
    {
        // Compute Bag of Words Vector
        // 將當前幀轉換成詞袋
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        // 先遍歷一下所有的關鍵幀篩選出具有共享詞的那些，再通過共享詞的數量以及 BoW 相似性得分，在數據庫粗選幾個候選幀
        // 當前幀的詞袋模型包含的所有關鍵幀（擁有相同的『重定位詞』）
        // 篩選出『重定位詞較多』、『BoW 相似性得分較高』的關鍵幀
        // 這裡的分數同時考慮了其他觀察到相同地圖點的關鍵幀的 BoW 相似性得分
        vector<KeyFrame *> vpCandidateKFs =
            mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        if (vpCandidateKFs.empty())
        {
            return false;
        }

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        // 各個共視關鍵幀的 PnP 求解器
        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *>> vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        // 是否丟棄
        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0, nmatches;
        KeyFrame *pKF;

        /* 用 ORB 匹配器遍歷一下所有的候選關鍵幀，容器 vpPnPsolvers 就是用來記錄各個候選幀的求解器的，
        vvpMapPointMatches 則用於保存各個候選幀與當前幀的匹配關鍵點，vbDiscarded 標記了對應候選幀
        是否因為匹配點數量不足而被拋棄。*/
        for (int i = 0; i < nKFs; i++)
        {
            // 取出第 i 個共視關鍵幀
            pKF = vpCandidateKFs[i];

            if (pKF->isBad())
            {
                vbDiscarded[i] = true;
            }
            else
            {
                // 利用詞袋模型，快速匹配兩幀同時觀察到的地圖點 vector<MapPoint *> vvpMapPointMatches[i]
                nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);

                // 配對數量不足（少於 15 點），標記該關鍵幀為要丟棄
                if (nmatches < 15)
                {
                    vbDiscarded[i] = true;
                    continue;
                }

                // 當有足夠多的匹配點時為之創建一個 PnP 求解器
                else
                {
                    // vvpMapPointMatches[i]：當前幀與『第 i 個共視關鍵幀』共同觀察到的地圖點
                    // 利用 PnP 求解當前幀與『第 i 個共視關鍵幀』之間的位姿轉換
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        // 在進行新的篩選之前，先創建了一個標識重定位是否成功的布爾變量 bMatch，
        // 和一個用於對候選幀的關鍵點進行投影匹配的 ORB 匹配器 matcher2。
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        // while 循環用於推進位姿估計的優化叠代，for 循環用於遍歷候選關鍵幀。
        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nKFs; i++)
            {
                if (vbDiscarded[i])
                {
                    continue;
                }

                // 記錄了候選幀中成功匹配上的地圖點
                vector<bool> vbInliers;

                // 記錄了匹配點的數量
                int nInliers;

                // 用於標記 PnP 求解是否達到了最大叠代次數
                bool bNoMore;

                // Perform 5 Ransac Iterations
                // 針對每個關鍵幀先通過 PnP 求解器估計相機的位姿，結果保存在局部變量 Tcw 中。
                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                // 如果達到了最大叠代次數，那麽意味著通過 PnP 算法無法得到一個比較合理的位姿估計，
                // 所以才會叠代了那麽多次。因此需要拋棄該候選幀。
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                // 如果成功的求解了 PnP 問題，並得到了相機的位姿估計，那麽就進一步的對該估計進行優化
                if (!Tcw.empty())
                {
                    // 用剛剛計算得到的位姿估計（Tcw）來更新當前幀的位姿
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++)
                    {
                        // 若為內點，則加入當前幀進行管理
                        if (vbInliers[j])
                        {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        }
                        else{
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                        }
                    }
                    
                    // 優化『pFrame 觀察到的地圖點』的位置，以及 pFrame 的位姿估計，並返回優化後的內點個數
                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    // 局部變量 nGood 評價了匹配程度，如果太低就結束當此叠代。
                    if (nGood < 10)
                    {
                        continue;
                    }

                    for (int io = 0; io < mCurrentFrame.N; io++)
                    {
                        if (mCurrentFrame.mvbOutlier[io])
                        {
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);
                        }
                    }

                    // If few inliers, search by projection in a coarse window and optimize again
                    // 如果內點數量比較少，就以一個較大的窗口將候選幀的地圖點投影到當前幀上獲取更多的可能匹配點，
                    // 並重新進行優化。
                    if (nGood < 50)
                    {
                        // 利用較大的搜索半徑 10 進行再次配對
                        // 尋找 CurrentFrame 當中和『關鍵幀 vpCandidateKFs[i]』的特徵點對應的位置，
                        // 形成 CurrentFrame 的地圖點，並返回匹配成功的個數
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i],
                                                                      sFound, 10, 100);

                        if (nadditional + nGood >= 50)
                        {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in
                            // a narrower window the camera has been already optimized with many points
                            // 如果內點數量得到了增加但仍然不夠多，就。。
                            if (nGood > 30 && nGood < 50)
                            {
                                sFound.clear();

                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                {
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                    {
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                    }
                                }

                                // 再次將候選幀的地圖點投影到當前幀上搜索匹配點，只是這次投影的窗口（64）比較小
                                // 窗口小，形成的候選網格則會增加，也就是搜索細緻度增加了
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, 
                                                                          vpCandidateKFs[i], 
                                                                          sFound, 3, 64);

                                // Final optimization
                                // 產生足夠的配對點
                                if (nGood + nadditional >= 50)
                                {
                                    // 再進行一次優化
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                    {
                                        if (mCurrentFrame.mvbOutlier[io])
                                        {
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // If the pose is supported by enough inliers stop ransacs and continue
                    // 如果找到一個候選幀經過一次次的優化之後，具有足夠多的匹配點，就認為重定位成功，退出循環叠代。
                    if (nGood >= 50)
                    {
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        // 最後根據是否成功找到匹配關鍵幀返回重定位是否成功。
        if (bMatch)
        {
            mnLastRelocFrameId = mCurrentFrame.mnId;
        }

        return bMatch;
    }

    void Tracking::ChangeCalibration(const string &strSettingPath)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];

        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }
} //namespace ORB_SLAM
