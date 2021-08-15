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

#include "Converter.h"

namespace ORB_SLAM2
{
    // ==================================================

    // ==================================================
    // 以上為管理執行續相關函式
    // ==================================================

    /****************************************************************************
     * 函數：Converter::toDescriptorVector()
     * 功能：將描述子轉換為以向量的形式描述
     * 輸入：const cv::Mat &Descriptors -- 描述子
     * 輸出：無
     * 返回：std::vector<cv::Mat> -- 每一個cv::Mat代表了描述子的每一行
     * 其他：
    *****************************************************************************/
    std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
    {
        std::vector<cv::Mat> vDesc;
        vDesc.reserve(Descriptors.rows);

        for (int j = 0; j < Descriptors.rows; j++){
            vDesc.push_back(Descriptors.row(j));
        }

        return vDesc;
    }

    /****************************************************************************
     * 函數：Converter::toSE3Quat()
     * 功能：將opencv的Mat數據類型轉換為g2o的李代數se3
     * 輸入：const cv::Mat &cvT -- 輸入矩陣
     * 輸出：無
     * 返回：g2o::SE3Quat -- g2o的李代數
     * 其他：
    *****************************************************************************/
    g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
    {
        Eigen::Matrix<double, 3, 3> R;
        R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
             cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2), 
             cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

        Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

        return g2o::SE3Quat(R, t);
    }

    /****************************************************************************
     * 函數：Converter::toVector3d()
     * 功能：將Opencv的Mat轉化為Eigen的Matrix
     * 輸入：const cv::Mat &cvVector -- 輸入向量
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector)
    {
        Eigen::Matrix<double, 3, 1> v;
        v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

        return v;
    }

    /****************************************************************************
     * 函數：Converter::toCvMat()
     * 功能：將g2o的李代數se3轉化為opencv的mat
     * 輸入：const g2o::SE3Quat &SE3 -- 輸入矩陣
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：
    *****************************************************************************/
    cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
    {
        Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
        return toCvMat(eigMat);
    }

    /****************************************************************************
     * 函數：Converter::toCvMat()
     * 功能：將Eigen的Matrix轉化為Opencv的Mat
     * 輸入：const Eigen::Matrix<double,4,4> &m -- 輸入矩陣
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m)
    {
        cv::Mat cvMat(4, 4, CV_32F);

        for (int i = 0; i < 4; i++){

            for (int j = 0; j < 4; j++){

                cvMat.at<float>(i, j) = m(i, j);
            }
        }

        return cvMat.clone();
    }

    /****************************************************************************
     * 函數：Converter::toCvMat()
     * 功能：將Eigen的Matrix轉化為Opencv的Mat
     * 輸入：const Eigen::Matrix<double,3,1> &m -- 輸入矩陣
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &m)
    {
        cv::Mat cvMat(3, 1, CV_32F);

        for (int i = 0; i < 3; i++){
            
            cvMat.at<float>(i) = m(i);
        }

        return cvMat.clone();
    }

    /****************************************************************************
     * 函數：Converter::toMatrix3d()
     * 功能：將Opencv的Mat轉化為Eigen的Matrix
     * 輸入：const cv::Mat &cvMat3 -- 輸入矩陣
     * 輸出：無
     * 返回：Eigen::Matrix<double,3,3> -- 輸出矩陣
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat3)
    {
        Eigen::Matrix<double, 3, 3> M;

        M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
             cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
             cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

        return M;
    }

    /****************************************************************************
     * 函數：Converter::toCvMat()
     * 功能：將g2o的仿射矩陣轉化為opencv的mat
     * 輸入：const g2o::Sim3 &Sim3 -- 輸入矩陣
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：
    *****************************************************************************/
    cv::Mat Converter::toCvMat(const g2o::Sim3 &Sim3)
    {
        Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = Sim3.translation();
        double s = Sim3.scale();

        return toCvSE3(s * eigR, eigt);
    }

    /****************************************************************************
     * 函數：Converter::toCvMat()
     * 功能：將Eigen的Matrix轉化為Opencv的Mat
     * 輸入：const Eigen::Matrix<double,3,3> &R -- 旋轉矩陣
     *      const Eigen::Matrix<double,3,1> &t -- 平移矩陣
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R, 
                               const Eigen::Matrix<double, 3, 1> &t)
    {
        cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cvMat.at<float>(i, j) = R(i, j);
            }
        }
        
        for (int i = 0; i < 3; i++)
        {
            cvMat.at<float>(i, 3) = t(i);
        }

        return cvMat.clone();
    }
    
    // ==================================================
    // 以下為非單目相關函式
    // ==================================================

    /****************************************************************************
     * 函數：Converter::toCvMat()
     * 功能：將Eigen的Matrix轉化為Opencv的Mat
     * 輸入：const Eigen::Matrix3d &m -- 輸入矩陣
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
    {
        cv::Mat cvMat(3, 3, CV_32F);

        for (int i = 0; i < 3; i++){

            for (int j = 0; j < 3; j++){

                cvMat.at<float>(i, j) = m(i, j);
            }
        }

        return cvMat.clone();
    }

    /****************************************************************************
     * 函數：Converter::toVector3d()
     * 功能：將Opencv的Mat轉化為Eigen的Matrix
     * 輸入：const cv::Mat &cvVector -- 輸入向量
     * 輸出：無
     * 返回：cv::Mat -- 位姿矩陣
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &cvPoint)
    {
        Eigen::Matrix<double, 3, 1> v;
        v << cvPoint.x, cvPoint.y, cvPoint.z;

        return v;
    }


    /****************************************************************************
     * 函數：Converter::toQuaternion()
     * 功能：將Opencv的Mat轉化為四元數
     * 輸入：const cv::Mat &M -- 輸入矩陣
     * 輸出：無
     * 返回：std::vector<float> -- 四元數
     * 其他：其實當前有更好的方法！
    *****************************************************************************/
    std::vector<float> Converter::toQuaternion(const cv::Mat &M)
    {
        Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
        Eigen::Quaterniond q(eigMat);

        std::vector<float> v(4);
        v[0] = q.x();
        v[1] = q.y();
        v[2] = q.z();
        v[3] = q.w();

        return v;
    }

} //namespace ORB_SLAM
