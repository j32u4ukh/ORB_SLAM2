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

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;

// ==================================================

// ==================================================
// 以上為管理執行續相關函式
// ==================================================

void loadImages(const string &path, vector<string> &files, vector<double> &time_stamps,
                const size_t start = 0, const size_t end = 1000000);

// ==================================================
// 以下為非單目相關函式
// ==================================================

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        // 以 KITTI Dataset 為例
        // argc[0] ./Examples/Monocular/mono_kitti
        // argc[1] Vocabulary/ORBvoc.txt
        // argc[2] Examples/Monocular/KITTIX.yaml
        // argc[3] PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
        // argc[4] [1|0](save map?)
        cerr << endl
             << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence [1|0](save map?)" 
             << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;

    // 以 KITTI Dataset 為例
    // argc[3] PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
    loadImages(string(argv[3]), vstrImageFilenames, vTimestamps, 0, 5000);
    // loadImages(string(argv[3]), vstrImageFilenames, vTimestamps, 0, 1650);
    
    int nImages = vstrImageFilenames.size();

    // 以 KITTI Dataset 為例
    // argc[1] Vocabulary/ORBvoc.txt
    // argc[2] Examples/Monocular/KITTIX.yaml
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true, (bool)atoi(argv[4]));
    cv::Mat im;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point main_t1 = std::chrono::steady_clock::now();
#else
        // 此問題為 C++ 版本兼容問題，可利用上方 COMPILEDWITHC11 根據 C++ 版本不同使用不同程式碼
        // COMPILEDWITHC11 則在 CMakeLists.txt 當中作定義
        std::chrono::monotonic_clock::time_point main_t1 = std::chrono::monotonic_clock::now();
#endif

    double tframe, ttrack, T;

    // Main loop
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        // 此問題為 C++ 版本兼容問題，可利用上方 COMPILEDWITHC11 根據 C++ 版本不同使用不同程式碼
        // COMPILEDWITHC11 則在 CMakeLists.txt 當中作定義
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe, ni);

        // cout << "Current index: " << ni << endl;

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        T = 0;

        if (ni < nImages - 1)
        {
            T = vTimestamps[ni + 1] - tframe;
        }
        else if (ni > 0)
        {
            T = tframe - vTimestamps[ni - 1];
        }

        /// NOTE: 有時會發生等待很長一段時間，或許跟這裡也有關系？
        if (ttrack < T)
        {
            usleep((T - ttrack) * 1e6);
        }
    }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point main_t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point main_t2 = std::chrono::monotonic_clock::now();
#endif

    // SLAM.buildOctomap();

    // Stop all threads
    SLAM.Shutdown();

    // ===== Tracking time statistics =====

    // 排序每一幀花費的時間，將用於找出中位數，以衡量每幀追蹤時間
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float tracking_time = 0;
    double total = std::chrono::duration_cast<std::chrono::duration<double>>(main_t2 - main_t1).count();

    for(float time : vTimesTrack){

        tracking_time += time;
    }

    int minute = (int)(total / 60.0);
    double second = total - 60.0 * minute;

    cout << "Total time: " << minute << ":" << second << endl;
    cout << "Median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "Mean tracking time: " << tracking_time / nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void loadImages(const string &path, vector<string> &files, vector<double> &time_stamps,
                const size_t start, const size_t end)
{
    ifstream fTimes;
    string strPathTimeFile = path + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    vector<double> temp_stamps;

    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);

        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            // time_stamps.push_back(t);
            temp_stamps.push_back(t);
        }
    }

    // left: image_0; right: image_1
    // 由於此專案為單目相機，因此只使用其中一邊的圖像
    string strPrefixLeft = path + "/image_0/";

    const size_t nTimes = temp_stamps.size();
    size_t i, len = min(end, nTimes);
    // files.resize(nTimes);

    for (i = start; i < len; i++)
    {
        time_stamps.push_back(temp_stamps[i]);

        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        // files[i] = strPrefixLeft + ss.str() + ".png";
        files.push_back(strPrefixLeft + ss.str() + ".png");
    }
}
