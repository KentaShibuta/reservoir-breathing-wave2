#include <vector>

class SCirculationBuffer{
    private:
        size_t m_pos;
        size_t m_window_size;
        size_t m_nx;
        std::vector<std::vector<float>> m_data;

    public:
        SCirculationBuffer(){};

        inline void Init(size_t window_size, size_t N_x){
            m_pos = 0;
            auto mat_window = std::make_unique<std::vector<std::vector<float>>>(window_size, std::vector<float>(N_x, 0.0f));
            m_data = std::move(*mat_window);
            m_window_size = window_size;
            m_nx = N_x;
        }

        inline void Update(const std::vector<float> &x){
            size_t row_no = 0;
            for (const auto &elem : x){
                m_data[m_pos][row_no] = elem;
                row_no++;
            }

            m_pos = m_pos < m_window_size-1 ? m_pos + 1 : 0;
        }

        inline void GetAverage(std::vector<float> &output){
            float inv_window_size = 1.0f / m_window_size;
            auto average = std::make_unique<std::vector<float>>(m_nx, 0.0);
            
            for (size_t i = 0; i < m_nx; i++){
                float tmp = 0.0f;
                for (size_t w = 0; w < m_window_size; w++){
                    tmp += m_data[w][i];

                    // 末尾の要素まで足しこんだ後に、window sizeで割って平均を計算する
                    if (w == m_window_size - 1){
                        tmp *= inv_window_size;
                        output[i] = tmp;
                    }
                }
            }
        }

        inline void Print(){
            for (const auto& row : m_data){
                for (const auto& elem : row){
                    std::cout << elem << " ";
                }
                std::cout << std::endl;
            }
        }
};