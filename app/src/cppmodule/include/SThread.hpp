#include <vector>

class SThread{
    private:
        size_t m_start;
        size_t m_end;
        std::vector<cv::Mat> m_dstImg;

    public:
        SThread(){};

        inline void Init(size_t start, size_t end){
            m_start = start;
            m_end = end;

            size_t dstSize = end - start;

            if (dstSize > 0){
                m_dstImg.resize(dstSize);
            }else{
                std::cerr << "Warning: dstSize is o or minus. start: " << start << ". end: " << end << ". dstSize: " << dstSize << std::endl;
            }
        }

        inline size_t GetStart(){
            return m_start;
        }

        inline size_t GetEnd(){
            return m_end;
        }

        inline void SetDstImg(size_t pos, const cv::Mat& img){
            if (pos >= 0 && pos < m_dstImg.size()){
                m_dstImg[pos] = img;
            }else{
                std::cerr << "Warning: pos is out of range m_dstImages. pos: " << pos << ". m_dstImages.size(): " << m_dstImg.size() << std::endl;
            }
        }

        inline const std::vector<cv::Mat>& GetDstImg() const{
            return m_dstImg;
        }
};