/*
 * map save/load extension for ORB_SLAM2
 * This header contains boost headers needed by serialization
 *
 * object to save:
 *   - KeyFrame
 *   - KeyFrameDatabase
 *   - Map
 *   - MapPoint
 */
#ifndef BOOST_ARCHIVER_H
#define BOOST_ARCHIVER_H

#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

// set serialization needed by KeyFrame::mspChildrens ...
#include <boost/serialization/map.hpp>

// map serialization needed by KeyFrame::mConnectedKeyFrameWeights ...
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/base_object.hpp>

// base object needed by DBoW2::BowVector and DBoW2::FeatureVector
#include <opencv2/core/core.hpp>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)
BOOST_SERIALIZATION_SPLIT_FREE(::cv::Vec3b)
namespace boost
{
    namespace serialization
    {
        /*
        「Serialization」（序列化）在程式語言裡面，基本上是一種用來把資料結構或是物件的狀態，
        轉換成可儲存、可交換的格式的一種方法。透過這樣的功能，可以把物件的狀態儲存下、之後再還原回來。

        可以使用 Boost serialization 進行序列化處理的資料，包括了：
        * C++ 原生型別（primitive type）
        * 有定義 serialize() 這個成員函式的類別（class）
            * 或是有對應的全域函式也可以
        * 可序列化型別的指標、參考、C++ 原生陣列
        
        要使用時，都需要 include 對應的 header 檔；
        以「text_woarchive」來說，他的 header 檔就是「<boost/archive/text_woarchive.hpp>」。
        
        而在操作上，archive 的類別基本上是把標準函式庫既有的 stream 作封包，在使用上就接近本來的 stream 一樣，
        可以透過 << 和 >> 來做輸出和輸入，相當地簡單。
        不過這邊可能要注意的，是 binary 的 archive 基本上會受到平台實作的影響，所以在跨平台的時候可能會有問題。

        ================================================================================
        「讓自訂型別支援 Boost serialization」
        ================================================================================
        寫成 template 的形式，其中 Archive 可以是所有輸入輸出的 archive。

        透過 & 這個運算子來統一輸入（<<）以及輸入（>>）的介面；
        當 Archive 這個 template 型別屬於輸出的 Archive 的時候，他就會輸出資料、
        而當 Archive 是輸入的 archive 的時候，就會自動變成輸入。

        接下來只要把要儲存／讀取的資料（這邊是所有的成員資料），透過 & 這個運算子來給 Archive 的物件 ar 做處理就可以了。

        這邊唯一的需求，就是這些資料必須要可以序列化了～如果遇到不能序列化的資料型別，
        就需要另外幫他撰寫 serialize() 這樣的函式了。

        如果選擇在類別中定義成員函式 serialize() 的時候，考慮到這個函式的特殊性，
        並不適合做為一個外部可以直接呼叫的 public 函式。

        所以這邊比較好的做法，應該是將它改為 private、並透過設定 friend class、
        來讓它可以被 Boost serialization 提供的 archive 呼叫，理論上會比較安全。

        ##### 讓既有的型別支援序列化 #####

        上面的方法，是在類別裡面加入 serialize() 這樣的函式，讓他支援 Boost Serialization。
        但是實際上，很多時候，並不一定可以修改類別的內容、也就是沒辦法加入這樣的成員函式。

        為了對應這樣的狀況，Boost Serialization 也有提供「非侵入式」（non Intrusive）的方案可以使用。
        要使用非侵入式的方案，基本上就是把 serialize() 這個函式，
        改寫成在 boost::serialization 這個 namespace 下的全域函式，這樣就可以了。

        例如上面 CURL 的例子，就可以改成：

        class CURL
        {
            public:
            std::string   sHost;
            unsigned int  sPort;
            std::string   sPath;
        };
            
        namespace boost 
        {
            namespace serialization 
            {
                template<class Archive>
                void serialize(Archive& ar, CURL& rURL, const unsigned int version)
                {
                    ar& rURL.sHost;
                    ar& rURL.sPort;
                    ar& rURL.sPath;
                }
            }
        }

        ================================================================================

        ##### 陣列、其他 STL 型別 #####

        在原生矩陣（Array）的部分，其實 Boost serialization 是可以直接支援的。
        而如果是標準函式庫裡面的容器、像是 vector、list 之類的型別，在要序列化之前，
        就需要 include <boost/serialization/vector.hpp> 這個檔案，
        否則會出現沒有serialize() 這個成員函式的錯誤。

        而在 boost/serialization/ 這個資料夾下，也還有很多其他型別用的 header 檔，
        如果遇到不能序列化的型別，也可以先來這邊看看有沒有對應的檔案。

        至於其他 Boost 函式庫的類別，有的則是藏在各自的目錄下，有需要可能也得自己挖看看了。
        例如 boost UUID 的序列化程式就是 boost/uuid/uuid_serialize.hpp 這個檔案。

        ================================================================================
        「繼承的類別」
        ================================================================================
        
        如果再遇到類別有繼承的時候，官方是有說「不要」直接去呼叫 base class 的 serialize() 函式；
        雖然可以好像可以用，但是實際上卻可能會跳過一些內部的機制。

        所以在這個時候，官方的建議是，把永遠把 serialize() 這個成員函式寫成 private 的，
        然後透過 boost::serialization::base_object<>() 來處理 base class 的資料。
        序列化衍生類別時，會先序列化其 父類別/基本類別，再序列化衍生類別才有的內容。

        // definition of serialize function
        template<class Archive>
        void serialize(Archive & archive, const unsigned int version){

            // invoke serialization of the base class 
            ar & boost::serialization::base_object<base_class_of_T>(*this);

            // save/load class member variables
            ar & member1;
            ar & member2;
        }
        
        下面就是一個繼承上面的 CURL 的例子：

        class CAuth : public CURL
        {
            public:
            std::string sAccount;
            std::string sPassword;
            
            private:
            friend class boost::serialization::access;
            
            template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar& boost::serialization::base_object<CURL>(*this);
                ar& sAccount;
                ar& sPassword;
            }
        };

        ================================================================================


        ================================================================================
        「將儲存和讀取分開處理」
        ================================================================================
        
        根據 Boost serialization 的設計下，主要是透過 serialize() 這樣的單一函式，來解決輸入、輸出的功能，
        並避免可能的兩個方向不一致的可能性。但是，有的時候，我們可能還是會需要輸入、輸出兩邊是不一樣的狀況。

        這個時候，除非要去直接修改 CURL 這個類別，基本上就沒辦法靠單一的 serialize() 來同時處理輸出和輸入。

        如果遇到這樣的情況，Boost serialization 也提供了把 serialize() 拆分成 save() / load() 兩個函式的功能，
        讓開發者可以使用。比如說，這邊就可以寫成：

        namespace boost 
        {
            namespace serialization 
            {
                template<class Archive>
                void save(Archive& ar, const CURL& t, unsigned int version)
                {
                    ar& t.getHost();
                    ar& t.getPort();
                    ar& t.getPath();
                }
            
                template<class Archive>
                void load(Archive& ar, CURL& t, unsigned int version)
                {
                    std::string sVal;
                    unsigned int iVal;
                    ar& sVal;  t.setHost(sVal);
                    ar& iVal;  t.setPort(iVal);
                    ar& sVal;  t.setPath(sVal);
                }
            }
        }
        
        BOOST_SERIALIZATION_SPLIT_FREE(CURL);

        這邊的作法，基本上就是和撰寫 serialize() 的方法一樣，不過這邊把程式分成 save() / load() 兩個版本。
        最後，則是要再透過 BOOST_SERIALIZATION_SPLIT_FREE() 這個巨集（定義在 
        <boost/serialization/split_free.hpp>），告訴 Boost Serialization CURL 這個類別需要使用拆開的版本，
        然後就可以正常使用了～

        參考網站：
        https://kheresy.wordpress.com/2020/03/25/boost-serialization-p1/
        https://kheresy.wordpress.com/2020/03/27/boost-serialization-p2/
        */

        /* serialization for DBoW2 BowVector */
        template<class Archive>
        void serialize(Archive &ar, DBoW2::BowVector &BowVec, const unsigned int file_version)
        {
            // ar & boost::serialization::base_object<DBoW2::BowVector::super>(BowVec);
            ar & boost::serialization::base_object<std::map<DBoW2::WordId, DBoW2::WordValue>>(BowVec);
        }

        /* serialization for DBoW2 FeatureVector */
        template<class Archive>
        void serialize(Archive &ar, DBoW2::FeatureVector &FeatVec, const unsigned int file_version)
        {
            // ar & boost::serialization::base_object<DBoW2::FeatureVector::super>(FeatVec);
            ar & boost::serialization::base_object<std::map<DBoW2::NodeId, std::vector<unsigned int> >>(FeatVec);
        }

        /* serialization for CV KeyPoint */
        template<class Archive>
        void serialize(Archive &ar, ::cv::KeyPoint &kf, const unsigned int file_version)
        {
            ar & kf.angle;
            ar & kf.class_id;
            ar & kf.octave;
            ar & kf.response;
            ar & kf.response;
            ar & kf.pt.x;
            ar & kf.pt.y;
        }

        /* serialization for CV Mat */
        template<class Archive>
        void save(Archive &ar, const ::cv::Mat &m, const unsigned int file_version)
        {
            cv::Mat m_ = m;

            if (!m.isContinuous())
            {
                m_ = m.clone();
            }
            
            size_t elem_size = m_.elemSize();
            size_t elem_type = m_.type();
            ar & m_.cols;
            ar & m_.rows;
            ar & elem_size;
            ar & elem_type;

            const size_t data_size = m_.cols * m_.rows * elem_size;

            ar & boost::serialization::make_array(m_.ptr(), data_size);
        }

        template<class Archive>
        void load(Archive & ar, ::cv::Mat& m, const unsigned int version)
        {
            int cols, rows;
            size_t elem_size, elem_type;

            ar & cols;
            ar & rows;
            ar & elem_size;
            ar & elem_type;

            m.create(rows, cols, elem_type);
            size_t data_size = m.cols * m.rows * elem_size;

            ar & boost::serialization::make_array(m.ptr(), data_size);
        }

        /* serialization for CV Vec3b */
        template<class Archive>
        void save(Archive &ar, const ::cv::Vec3b &color, const unsigned int file_version)
        {
            uchar b, g, r;

            b = color[0];
            g = color[1];
            r = color[2];

            ar & b;
            ar & g;
            ar & r;
        }

        template<class Archive>
        void load(Archive & ar, ::cv::Vec3b &color, const unsigned int version)
        {
            uchar b, g, r;
            
            ar & b;
            ar & g;
            ar & r;

            color = cv::Vec3b(b, g, r);
        }
    }
}
// TODO: boost::iostream zlib compressed binary format
#endif // BOOST_ARCHIVER_H
