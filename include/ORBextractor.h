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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
//#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>


namespace ORB_SLAM2
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    // 特徵點們
    std::vector<cv::KeyPoint> vKeys;

    // 區域邊界點
    cv::Point2i UL, UR, BL, BR;

    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;
    }

    float inline GetScaleFactor(){
        return scaleFactor;
    }

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    // 儲存影像金字塔各層級的影像
    std::vector<cv::Mat> mvImagePyramid;

    // ExtractorNode &node
    // list<ExtractorNode> &list_node
    // int nToExpand
    // vector<pair<int, ExtractorNode *>> &size_node_list
    int addContainPoints(ExtractorNode &node, std::list<ExtractorNode> &list_node, int n_to_expand, 
                         std::vector<std::pair<int, ExtractorNode *>> &size_node_list);

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    inline void computeKeyPoints(std::vector<std::vector<cv::KeyPoint>> &all_keypoints, 
                                        const float W, const int level);
    inline void computeFastFeature(const int i, const int level, const int n_col,
                                          const int min_x, const int max_x, 
                                          const int min_y, const int max_y, 
                                          const int cell_w, const int cell_h,
                                          std::vector<cv::KeyPoint> &distribute_keys);

    static float IC_Angle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max);
    static void computeOrbDescriptor(const cv::KeyPoint &kpt, const cv::Mat &img, 
                                     const cv::Point *pattern, uchar *desc);
    static void computeOrientation(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, 
                                   const std::vector<int> &umax);
    static void computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, 
                                   cv::Mat &descriptors, const std::vector<cv::Point> &pattern);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern;

    // 影像金字塔各層級全部的特徵數總和（至少，可能超過）
    int nfeatures;

    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    // 用於 IC_Angle 的計算當中
    std::vector<int> umax;

    // 1.0, 1.2, 1.44, 1.728, 2.0736
    std::vector<float> mvScaleFactor;

    // 1.0, 0.8333, 0.6944, 0.5787, 0.4823
    std::vector<float> mvInvScaleFactor;
    
    // 1.0, 1.44, 2.0736, 2.9860, 4.2998
    std::vector<float> mvLevelSigma2;

    // 1.0 , 0.6944, 0.4823, 0.3349, 0.2326
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

