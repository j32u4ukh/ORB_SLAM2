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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, 
          ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, 
          cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, 
          ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, 
          const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, const cv::Mat &img, 
          ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, 
          const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    inline void resetMappoints()
    {
        fill(mvpMapPoints.begin(), mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    }

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    // 旋轉矩陣（相機座標 → 世界座標）
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // 取得圖片顏色資訊
    cv::Mat getColorImage();

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat K;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // ORB 特徵點個數（Number of KeyPoints.）
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;

    // 存入校正後的關鍵點（推測是像素坐標）
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // ==================================================
    // 詞典是一個事先訓練好的分類樹(System.cc 當中的 mpVocabulary), 而 BOW 特徵有兩種
    // 在利用幀間所有特徵點比對初始化地圖點以後, 後面的幀間比對都採用 Feature vector 進行, 
    // 而不再利用所有特徵點的 descriptor 兩兩比對
    // 這裡只是利用 mpVocabulary 根據 Frame 的描述子，將它們分別劃分到這個分類樹的各個節點，
    // 並將分類結果紀錄在 mBowVec 和 mFeatVec 當中
    // ================================================== 
    // 分類樹中 leaf 的數值與權重(葉) Bag of Words Vector structures.
    // BowVector == std::map<WordId, WordValue>
    DBoW2::BowVector mBowVec;

    // FeatureVector == std::map<NodeId, std::vector<unsigned int> >
    // 以一張圖片的每個特徵點在詞典某一層節點下爲條件進行分組，用來加速圖形特徵匹配——
    // 兩兩圖像特徵匹配只需要對相同 NodeId 下的特徵點進行匹配就好。
    // 是分類樹中 leaf 的 id 值與對應輸入 ORB 特徵列表的特徵序號.
    DBoW2::FeatureVector mFeatVec;
    // ==================================================

    // ORB descriptor, each row associated to a keypoint.
    // 每一 row 代表一個關鍵點的描述子，各個層級的描述子拼接而成共同描述這個影像的描述子
    cv::Mat mDescriptors;

    // 每一 row 代表一個關鍵點的描述子
    cv::Mat mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
 
    /* 每個格子分配的特征點數，將圖像分成格子，保證提取的特征點比較均勻

    目前看起來 mGrid[i][j] 會是一個 std::vector，紀錄網格 (i, j) 所包含的'關鍵點的索引值'
    由 Frame::GetFeaturesInArea 當中 const vector<size_t> vCell = mGrid[ix][iy]; 可知 */
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;

    // log(mfScaleFactor)
    float mfLogScaleFactor;

    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // 旋轉矩陣（world to camera）
    cv::Mat mRcw;

    // 旋轉矩陣（camera to world）
    cv::Mat mRwc;

    // 位姿中的平移（world to camera）
    cv::Mat mtcw;

    // 位姿中的平移（camera to world） == mtwc
    cv::Mat mOw; 

    // 影像顏色資訊
    cv::Mat color_image;
};

}// namespace ORB_SLAM

#endif // FRAME_H
