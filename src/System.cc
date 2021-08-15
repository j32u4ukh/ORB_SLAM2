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

#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>

namespace ORB_SLAM2
{
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, 
                   const bool bUseViewer) : 
                   mSensor(sensor), mpViewer(static_cast<Viewer *>(NULL)), mbReset(false), 
                   mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false)
    {
        // Output welcome message
        cout << endl
             << "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl
             << "This program comes with ABSOLUTELY NO WARRANTY;" << endl
             << "This is free software, and you are welcome to redistribute it" << endl
             << "under certain conditions. See LICENSE.txt." << endl
             << endl;

        cout << "Input sensor was set to: ";

        if (mSensor == MONOCULAR)
        {
            cout << "Monocular" << endl;
        }
        else if (mSensor == STEREO)
        {
            cout << "Stereo" << endl;
        }
        else if (mSensor == RGBD)
        {
            cout << "RGB-D" << endl;
        }

        //Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);

        if (!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        //Load ORB Vocabulary
        cout << endl
             << "Loading ORB Vocabulary. This could take a while..." << endl;

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        // 此問題為 C++ 版本兼容問題，可利用上方 COMPILEDWITHC11 根據 C++ 版本不同使用不同程式碼
        // COMPILEDWITHC11 則在 CMakeLists.txt 當中作定義
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);

        if (!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double loading_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        cout << "Vocabulary loaded! Cost " << loading_time << " s." << endl << endl;

        // Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        // Create the Map
        mpMap = new Map();

        // Create Drawers. These are used by the Viewer
        mpFrameDrawer = new FrameDrawer(mpMap);
        mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

        // Initialize the Tracking thread
        // (it will live in the main thread of execution, the one that called this constructor)
        mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                                 mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

        //Initialize the Local Mapping thread and launch
        mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
        mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

        //Initialize the Loop Closing thread and launch
        mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
        mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

        //Initialize the Viewer thread and launch
        if (bUseViewer)
        {
            mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
            mptViewer = new thread(&Viewer::Run, mpViewer);
            mpTracker->SetViewer(mpViewer);
        }

        // Set pointers between threads
        // 配置三個任務對象的相互指針，讓它們能夠互相通信。
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpTracker->SetLoopClosing(mpLoopCloser);

        mpLocalMapper->SetTracker(mpTracker);
        mpLocalMapper->SetLoopCloser(mpLoopCloser);

        mpLoopCloser->SetTracker(mpTracker);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);
    }
    
    // 啟用定位模式
    void System::ActivateLocalizationMode()
    {
        unique_lock<mutex> lock(mMutexMode);

        // 使用定位模式
        mbActivateLocalizationMode = true;
    }

    // 不啟用定位模式
    void System::DeactivateLocalizationMode()
    {
        unique_lock<mutex> lock(mMutexMode);

        // 不啟用定位模式
        mbDeactivateLocalizationMode = true;
    }

    void System::Reset()
    {
        unique_lock<mutex> lock(mMutexReset);
        mbReset = true;
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
    {
        if (mSensor != MONOCULAR)
        {
            cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." 
                 << endl;
                 
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);

            // 是否為定位模式
            if (mbActivateLocalizationMode)
            {
                // LocalMapping::Run & Tracking::NeedNewKeyFrame & Optimizer::LocalBundleAdjustment 
                // 將被暫時停止
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);

                // 關閉定位模式
                mbActivateLocalizationMode = false;
            }

            if (mbDeactivateLocalizationMode)
            {
                // 設置是否僅追蹤不建圖
                mpTracker->InformOnlyTracking(false);

                // 清空『新關鍵幀容器』
                mpLocalMapper->Release();

                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);

            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

        unique_lock<mutex> lock2(mMutexState);

        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

        return Tcw;
    }

    // 關閉系統
    void System::Shutdown()
    {
        // 請求結束 LocalMapping 執行續
        mpLocalMapper->RequestFinish();

        // 請求結束 LoopClosing 執行續
        mpLoopCloser->RequestFinish();

        if (mpViewer)
        {
            // 請求結束 Viewer 執行續
            mpViewer->RequestFinish();

            while (!mpViewer->isFinished())
            {
                usleep(5000);
            }

            // ===== Add by myself =====
            delete mpViewer;
            mpViewer = static_cast<Viewer *>(NULL);
            // =========================
        }

        // Wait until all thread have effectively stopped
        // 等待所有線程確實停止
        while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || 
                mpLoopCloser->isRunningGBA())
        {
            usleep(5000);
        }

        if (mpViewer)
        {
            pangolin::BindToContext("ORB-SLAM2: Map Viewer");
        }

        int n_kf = mpMap->KeyFramesInMap();
        int n_mappoint = mpMap->MapPointsInMap();
        cout << "#KeyFrame: " << n_kf << endl;
        cout << "#MapPoint: " << n_mappoint << endl;
    }

    // For Monocular
    void System::SaveKeyFrameTrajectoryTUM(const string &filename)
    {
        cout << endl
             << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        //cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for(KeyFrame *pKF : vpKFs){

            if (pKF->isBad()){
                continue;
            }

            cv::Mat R = pKF->GetRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();

            f << setprecision(6) << pKF->mTimeStamp 
              << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " 
              << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }

        f.close();
        cout << endl << "trajectory saved!" << endl;
    }

    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
    {
        if (mSensor != STEREO)
        {
            cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);

            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);

            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
    {
        if (mSensor != RGBD)
        {
            cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);

            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    bool System::MapChanged()
    {
        static int n = 0;
        int curn = mpMap->GetLastBigChangeIdx();

        if (n < curn)
        {
            n = curn;

            return true;
        }
        else{
            return false;
        }
    }

    void System::SaveTrajectoryTUM(const string &filename)
    {
        cout << endl
             << "Saving camera trajectory to " << filename << " ..." << endl;

        if (mSensor == MONOCULAR)
        {
            cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
            return;
        }

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        list<bool>::iterator lbL = mpTracker->mlbLost.begin();

        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                                     lend = mpTracker->mlRelativeFramePoses.end();
             lit != lend; lit++, lRit++, lT++, lbL++)
        {
            if (*lbL)
                continue;

            KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while (pKF->isBad())
            {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            vector<float> q = Converter::toQuaternion(Rwc);

            f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
        f.close();
        cout << endl
             << "trajectory saved!" << endl;
    }

    void System::SaveTrajectoryKITTI(const string &filename)
    {
        cout << endl
             << "Saving camera trajectory to " << filename << " ..." << endl;
        if (mSensor == MONOCULAR)
        {
            cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
            return;
        }

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++)
        {
            ORB_SLAM2::KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            while (pKF->isBad())
            {
                //  cout << "bad parent" << endl;
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " " << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " " << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
        }
        f.close();
        cout << endl
             << "trajectory saved!" << endl;
    }

    int System::GetTrackingState()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackingState;
    }

    vector<MapPoint *> System::GetTrackedMapPoints()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedMapPoints;
    }

    vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedKeyPointsUn;
    }

} //namespace ORB_SLAM
