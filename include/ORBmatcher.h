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

#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <tuple>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace ORB_SLAM2
{

    class ORBmatcher
    {
    public:
        // 第一個參數是一個接受最佳匹配的系數，只有當最佳匹配點的漢明距離小於次加匹配點距離的 nnratio 倍時才接收匹配點，
        // 第二個參數表示匹配特征點時是否考慮方向。
        ORBmatcher(float nnratio = 0.6, bool checkOri = true);

        // Computes the Hamming distance between two ORB descriptors
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
        // Used to track the local map (Tracking)
        int SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th = 3);

        // Project MapPoints tracked in last frame into the current frame and search matches.
        // Used to track from previous frame (Tracking)
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

        // Project MapPoints seen in KeyFrame into the Frame and search matches.
        // Used in relocalisation (Tracking)
        int SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const std::set<MapPoint *> &sAlreadyFound, const float th, const int ORBdist);

        // Project MapPoints using a Similarity Transformation and search matches.
        // Used in loop detection (Loop Closing)
        int SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpMatched, int th);

        // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
        // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
        // Used in Relocalisation and Loop Detection
        int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);
        int SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);

        // Matching for the Map Initialization (only used in the monocular case)
        int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);

        // Matching to triangulate new MapPoints. Check Epipolar Constraint.
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                   std::vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo);

        // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
        // In the stereo and RGB-D case, s12=1
        int SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

        // Project MapPoints into KeyFrame and search for duplicated MapPoints.
        int Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th = 3.0);

        // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
        int Fuse(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

        template <class T, class D>
        inline int convergenceMatched(int n_match, std::vector<int> *rot_hist,
                                      std::vector<T> &v_matched, D default_value);

        inline void updateRotHist(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                                  const float factor, const int idx, std::vector<int> *rot_hist);

        inline std::tuple<bool, float, float, float> getPixelCoordinates(KeyFrame *pKF, cv::Mat sp3Dc,
                                                                         const float fx, const float fy,
                                                                         const float cx, const float cy);

        inline std::tuple<bool, float, float, float> getPixelCoordinatesStereo(KeyFrame *pKF, 
                                                                               cv::Mat sp3Dc,
                                                                               const float bf,
                                                                               const float fx, 
                                                                               const float fy,
                                                                               const float cx, 
                                                                               const float cy);

        inline std::tuple<bool, float> isValidDistance(MapPoint *pMP, cv::Mat p3Dw, cv::Mat Ow);
        inline std::tuple<bool, float> isValidDistanceSim3(MapPoint *pMP, cv::Mat sp3Dc);
        inline std::tuple<bool, cv::KeyPoint, int> checkFuseTarget(KeyFrame *pKF, const size_t idx,
                                                                   int nPredictedLevel);
        inline void updateFuseTarget(KeyFrame *pKF, const size_t idx, const cv::Mat dMP,
                                     int &bestDist, int &bestIdx);

    public:
        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;

    protected:
        bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

        float RadiusByViewingCos(const float &viewCos);

        void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

        float mfNNratio;
        bool mbCheckOrientation;
    };

} // namespace ORB_SLAM

#endif // ORBMATCHER_H
