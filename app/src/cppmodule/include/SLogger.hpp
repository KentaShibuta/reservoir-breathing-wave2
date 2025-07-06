#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

inline void init_logger(const std::string &log_name) {
    // すでに存在する場合はスキップ
    if (spdlog::get(log_name) == nullptr) {
        auto logger = spdlog::basic_logger_mt(
            log_name,
            "./log/cpp_esn.log",
            true // 上書きモード
        ); 
        logger->set_level(spdlog::level::debug);
        //logger->set_pattern("%Y-%m-%d %H:%M:%S [%n] [%^%L%$] %v"); 
    }
}