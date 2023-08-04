#pragma once

#include <chrono>
#include <string>

#include "logging.h"

namespace quik_fix {
namespace internal {

template <typename TDuration = std::chrono::microseconds> class ChronoTimer {
private:
  std::string _block_name;
  std::chrono::system_clock::time_point _tic;
  const char *const _unit;
  const char *_file;
  int _line;

public:
  ChronoTimer(std::string &&block_name, const char *const unit,
              const char *const file, const int line)
      : _block_name(std::move(block_name)),
        _tic(std::chrono::system_clock::now()), _unit(unit), _file(file),
        _line(line) {}
  ~ChronoTimer() {
    std::chrono::system_clock::time_point toc =
        std::chrono::system_clock::now();
    QF_LOG_INFO_AT(_file, _line)
        << "Total time for block=" << _block_name << ": "
        << QF_HL(std::chrono::duration_cast<TDuration>(toc - _tic).count())
        << " "
        << "(" << _unit << ")";
  }
  operator bool() const { return true; }
};

} // namespace internal
} // namespace quik_fix

#define WITH_CHRONO_US_TIMER(block_name)                                       \
  if (quik_fix::internal::ChronoTimer<std::chrono::microseconds> timer =       \
          quik_fix::internal::ChronoTimer<std::chrono::microseconds>(          \
              block_name, "us", __FILE__, __LINE__))

#define WITH_CHRONO_MS_TIMER(block_name)                                       \
  if (quik_fix::internal::ChronoTimer<std::chrono::milliseconds> timer =       \
          quik_fix::internal::ChronoTimer<std::chrono::milliseconds>(          \
              block_name, "ms", __FILE__, __LINE__))

#define WITH_CHRONO_S_TIMER(block_name)                                        \
  if (quik_fix::internal::ChronoTimer<std::chrono::seconds> timer =            \
          quik_fix::internal::ChronoTimer<std::chrono::seconds>(               \
              block_name, "s", __FILE__, __LINE__))
