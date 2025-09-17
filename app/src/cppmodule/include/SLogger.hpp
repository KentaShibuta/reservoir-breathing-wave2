#ifndef SLOGGER_H_
#define SLOGGER_H_

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

inline void init_logger(const std::string &logDir, const std::string &logName) {
    // すでに存在する場合はスキップ
    if (spdlog::get(logName) == nullptr) {
        std::string logPath = logDir + "/" + logName + ".log";
        auto logger = spdlog::basic_logger_mt(
            logName,
            logPath,
            true // 上書きモード
        );
        logger->set_level(spdlog::level::debug);
        //logger->set_pattern("%Y-%m-%d %H:%M:%S [%n] [%^%L%$] %v");
    }
}

#endif // SLOGGER_H_
