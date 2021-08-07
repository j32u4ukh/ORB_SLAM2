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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{

    MapDrawer::MapDrawer(Map *pMap, const string &strSettingPath) : mpMap(pMap)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
    }

    // 將當前觀察到的點畫成紅色，過去觀察到的地圖點畫成黑色
    void MapDrawer::DrawMapPoints()
    {
        // 從『地圖 mpMap』中取出所有地圖點
        const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();

        // 從『地圖 mpMap』中取出所有參考地圖點（當前觀察到的點）
        const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();

        set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

        if (vpMPs.empty()){
            return;
        }

        glPointSize(mPointSize);

        // 繪畫點模式
        glBegin(GL_POINTS);

        // 設置為黑色 -> 過去觀察到的點
        glColor3f(0.0, 0.0, 0.0);

        for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
        {
            if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i])){
                continue;
            }

            cv::Mat pos = vpMPs[i]->GetWorldPos();
            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        }

        glEnd();

        glPointSize(mPointSize);
        glBegin(GL_POINTS);

        // 設置為紅色 -> 當前觀察到的點
        glColor3f(1.0, 0.0, 0.0);

        set<MapPoint *>::iterator sit = spRefMPs.begin();
        set<MapPoint *>::iterator send = spRefMPs.end();

        for (; sit != send; sit++)
        {
            if ((*sit)->isBad()){
                continue;
            }

            cv::Mat pos = (*sit)->GetWorldPos();
            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        }

        glEnd();
    }

    // 畫出過去所有關鍵幀，並畫出和『共視關鍵幀』、『父關鍵幀』、『迴路關鍵幀』之間的連線
    void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
    {
        const float &w = mKeyFrameSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        // 從『地圖 mpMap』當中取得所有關鍵幀
        const vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();

        // 畫出過去所有關鍵幀（藍色）
        if (bDrawKF)
        {
            // 遍歷所有關鍵幀
            for(KeyFrame *pKF : vpKFs){
                // 由『關鍵幀 pKF』的相機座標系，轉換到世界座標系
                cv::Mat Twc = pKF->GetPoseInverse().t();

                // 紀錄當前位姿
                glPushMatrix();

                // 當前位姿，乘上『轉換矩陣 Twc』
                glMultMatrixf(Twc.ptr<GLfloat>(0));

                glLineWidth(mKeyFrameLineWidth);

                // 設為藍色
                glColor3f(0.0f, 0.0f, 1.0f);

                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(w, h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, h, z);

                glVertex3f(w, h, z);
                glVertex3f(w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(-w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(w, h, z);

                glVertex3f(-w, -h, z);
                glVertex3f(w, -h, z);
                glEnd();

                glPopMatrix();
            }

            // for (size_t i = 0; i < vpKFs.size(); i++)
            // {
            //     KeyFrame *pKF = vpKFs[i];

            //     // 由『關鍵幀 pKF』的相機座標系，轉換到世界座標系
            //     cv::Mat Twc = pKF->GetPoseInverse().t();

            //     // 紀錄當前位姿
            //     glPushMatrix();

            //     // 當前位姿，乘上『轉換矩陣 Twc』
            //     glMultMatrixf(Twc.ptr<GLfloat>(0));

            //     glLineWidth(mKeyFrameLineWidth);

            //     // 設為藍色
            //     glColor3f(0.0f, 0.0f, 1.0f);

            //     glBegin(GL_LINES);
            //     glVertex3f(0, 0, 0);
            //     glVertex3f(w, h, z);
            //     glVertex3f(0, 0, 0);
            //     glVertex3f(w, -h, z);
            //     glVertex3f(0, 0, 0);
            //     glVertex3f(-w, -h, z);
            //     glVertex3f(0, 0, 0);
            //     glVertex3f(-w, h, z);

            //     glVertex3f(w, h, z);
            //     glVertex3f(w, -h, z);

            //     glVertex3f(-w, h, z);
            //     glVertex3f(-w, -h, z);

            //     glVertex3f(-w, h, z);
            //     glVertex3f(w, h, z);

            //     glVertex3f(-w, -h, z);
            //     glVertex3f(w, -h, z);
            //     glEnd();

            //     glPopMatrix();
            // }
        }

        if (bDrawGraph)
        {
            glLineWidth(mGraphLineWidth);

            // 設置為淺綠色
            glColor4f(0.0f, 1.0f, 0.0f, 0.6f);

            glBegin(GL_LINES);

            for(KeyFrame *pKF : vpKFs){
                // Covisibility Graph
                // 取得『關鍵幀 vpKFs[i]』的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序，
                // 且觀察到的地圖點數量「大於」 100）』
                const vector<KeyFrame *> vCovKFs = pKF->GetCovisiblesByWeight(100);

                // 取得『關鍵幀 vpKFs[i]』的相機中心
                cv::Mat Ow = pKF->GetCameraCenter();
                
                if (!vCovKFs.empty())
                {
                    // 遍歷『關鍵幀 vpKFs[i]』的『共視關鍵幀』
                    for(KeyFrame * kf : vCovKFs){

                        if (kf->mnId < pKF->mnId){
                            continue;
                        }

                        // 取得『共視關鍵幀 (*vit)』的相機中心
                        cv::Mat Ow2 = kf->GetCameraCenter();

                        // 現在為線段模式(GL_LINES)，因此是相機中心之間的連線
                        glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                        glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
                    }
                }

                // Spanning tree
                // 取得『關鍵幀 vpKFs[i]』的父關鍵幀
                KeyFrame *pParent = pKF->GetParent();

                if (pParent)
                {
                    // 取得父關鍵幀的相機中心
                    cv::Mat Owp = pParent->GetCameraCenter();

                    glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                    glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
                }

                // Loops
                // 取得形成迴路的關鍵幀
                set<KeyFrame *> sLoopKFs = pKF->GetLoopEdges();

                for(KeyFrame * kf : sLoopKFs){
                    if (kf->mnId < pKF->mnId){
                        continue;
                    }

                    // 取得迴路關鍵幀的相機中心
                    cv::Mat Owl = kf->GetCameraCenter();

                    glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                    glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
                }
            }

            // for (size_t i = 0; i < vpKFs.size(); i++)
            // {
            //     // Covisibility Graph
            //     // 取得『關鍵幀 vpKFs[i]』的『已連結關鍵幀（根據觀察到的地圖點數量由大到小排序，
            //     // 且觀察到的地圖點數量「大於」 100）』
            //     const vector<KeyFrame *> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);

            //     // 取得『關鍵幀 vpKFs[i]』的相機中心
            //     cv::Mat Ow = vpKFs[i]->GetCameraCenter();
                
            //     if (!vCovKFs.empty())
            //     {
            //         vector<KeyFrame *>::const_iterator vit = vCovKFs.begin();
            //         vector<KeyFrame *>::const_iterator vend = vCovKFs.end();
            //         // 遍歷『關鍵幀 vpKFs[i]』的『共視關鍵幀』
            //         for (; vit != vend; vit++)
            //         {
            //             if ((*vit)->mnId < vpKFs[i]->mnId){
            //                 continue;
            //             }
            //             // 取得『共視關鍵幀 (*vit)』的相機中心
            //             cv::Mat Ow2 = (*vit)->GetCameraCenter();
            //             // 現在為線段模式(GL_LINES)，因此是相機中心之間的連線
            //             glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
            //             glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
            //         }
            //     }

            //     // Spanning tree
            //     // 取得『關鍵幀 vpKFs[i]』的父關鍵幀
            //     KeyFrame *pParent = vpKFs[i]->GetParent();

            //     if (pParent)
            //     {
            //         // 取得父關鍵幀的相機中心
            //         cv::Mat Owp = pParent->GetCameraCenter();

            //         glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
            //         glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
            //     }

            //     // Loops
            //     // 取得形成迴路的關鍵幀
            //     set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();

            //     set<KeyFrame *>::iterator sit = sLoopKFs.begin();
            //     set<KeyFrame *>::iterator send = sLoopKFs.end();

            //     for (; sit != send; sit++)
            //     {
            //         if ((*sit)->mnId < vpKFs[i]->mnId){
            //             continue;
            //         }

            //         // 取得迴路關鍵幀的相機中心
            //         cv::Mat Owl = (*sit)->GetCameraCenter();

            //         glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
            //         glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
            //     }
            // }

            glEnd();
        }
    }

    // 畫出當前相機位姿
    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
    {
        const float &w = mCameraSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        /* glPushMatrix() 將當前矩陣 push 入一個 stack，反之 glPopMatrix() 取出 stack 頂端的矩陣，
        因此我們就可以"記憶 gluLookAt() 後的 CM 長相"，並且在所有轉換都做完之後，
        又再度回到" gluLookAt() 後的 CM 長像"，而且重點是這兩個函數是硬體實作且 hard dependent，速度更快，
        因此可以用一句話形容： glPushMatrix() 是記住自己現在的位置，而 glPopMatrix() 是回到之前記住的位置!! 
        
        參考：http://ppb440219.blogspot.com/2012/01/opengl-glpushmatrix-glpopmatrix.html */

        // 紀錄當前位姿
        glPushMatrix();

        // 將位姿從原本的相機座標系，轉換為世界座標系
#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);

        // 設置為綠色
        glColor3f(0.0f, 1.0f, 0.0f);

        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(w, h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, h, z);

        glVertex3f(w, h, z);
        glVertex3f(w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(-w, -h, z);

        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);

        glVertex3f(-w, -h, z);
        glVertex3f(w, -h, z);
        glEnd();

        // 取出紀錄的位姿，以還原成之前的位姿
        glPopMatrix();
    }

    // 設置當前幀的位姿
    void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
    {
        unique_lock<mutex> lock(mMutexCamera);
        mCameraPose = Tcw.clone();
    }

    // 根據當前相機位姿，更新 OpenGlMatrix M 的值
    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
    {
        // 若當前相機位姿不為空
        if (!mCameraPose.empty())
        {
            cv::Mat Rwc(3, 3, CV_32F);
            cv::Mat twc(3, 1, CV_32F);

            {
                unique_lock<mutex> lock(mMutexCamera);

                // 取得當前相機的旋轉矩陣
                Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();

                // 取得當前相機的平移向量
                twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
            }

            // 當前相機的位姿，存入 OpenGlMatrix M
            M.m[0] = Rwc.at<float>(0, 0);
            M.m[1] = Rwc.at<float>(1, 0);
            M.m[2] = Rwc.at<float>(2, 0);
            M.m[3] = 0.0;

            M.m[4] = Rwc.at<float>(0, 1);
            M.m[5] = Rwc.at<float>(1, 1);
            M.m[6] = Rwc.at<float>(2, 1);
            M.m[7] = 0.0;

            M.m[8] = Rwc.at<float>(0, 2);
            M.m[9] = Rwc.at<float>(1, 2);
            M.m[10] = Rwc.at<float>(2, 2);
            M.m[11] = 0.0;

            M.m[12] = twc.at<float>(0);
            M.m[13] = twc.at<float>(1);
            M.m[14] = twc.at<float>(2);
            M.m[15] = 1.0;
        }

        // 若當前相機位姿是空的
        else
        {
            // OpenGlMatrix M 設為單位矩陣
            M.SetIdentity();
        }
    }

} //namespace ORB_SLAM
