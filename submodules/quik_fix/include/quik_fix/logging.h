#pragma once

#include <iostream>
#include <sstream>

#if defined(QF_LOG_PY_FRAME)
#include <Python.h>
#endif

#define QF_BOLD(str_or_obj) "\033[1m" << str_or_obj << "\033[0m"
#define QF_EMPH(str_or_obj) "\033[1;4m" << str_or_obj << "\033[0m"
#define QF_POSITIVE(str_or_obj) "\033[1;32;7m" << str_or_obj << "\033[0m"
#define QF_NEGATIVE(str_or_obj) "\033[1;31;7m" << str_or_obj << "\033[0m"
#define QF_HL(str_or_obj) "\033[1;33;7m" << str_or_obj << "\033[0m"

namespace quik_fix {
namespace internal {
class Logger;
}
} // namespace quik_fix

template <typename T>
inline quik_fix::internal::Logger &
operator<<(quik_fix::internal::Logger &logger, const T &item);

namespace quik_fix {

static std::ostream &(*endl)(std::ostream &) =
    std::endl<char, std::char_traits<char>>;

namespace internal {

class Logger {
private:
  std::ostream &_out;
  std::string _line_prefix;

protected:
  const bool _predicate;

public:
  Logger(std::ostream &out, const char *const file, const int line,
         const char level, const bool predicate = true)
      : _out(out), _predicate(predicate) {
    if (_predicate) {
      std::ostringstream strout;
      strout << "[quik_fix :: " << file << ":" << line;
#if defined(QF_LOG_PY_FRAME)
      PyThreadState *py_thread_state = nullptr;
      try {
        py_thread_state = PyThreadState_GET();
      } catch (...) {
        // In the case of failure, get the thread state from the main thread.
        PyInterpreterState *py_main_interpreter_state =
            PyInterpreterState_Main();
        py_thread_state =
            PyInterpreterState_ThreadHead(py_main_interpreter_state);
      }
      if (py_thread_state) {
        PyFrameObject *const py_frame = py_thread_state->frame;
        if (py_frame) {
          int __PY_LINE__ =
              PyCode_Addr2Line(py_frame->f_code, py_frame->f_lasti);
          const char *const __PY_FILE__ =
              PyUnicode_AsUTF8(py_frame->f_code->co_filename);
          strout << " (" << __PY_FILE__ << ":" << __PY_LINE__ << ")";
        }
      }
#endif
      strout << ", ";
      if (&out == &std::cerr) {
        strout << QF_NEGATIVE(level);
      } else {
        strout << level;
      }
      strout << "] ";
      _line_prefix = strout.str();
      out << _line_prefix;
    }
  }
  ~Logger() {
    if (_predicate) {
      _out << std::endl;
    }
  }
  virtual operator bool() const { return true; }
  template <typename T>
  friend Logger & ::operator<<(Logger &logger, const T &item);
};

class Asserter final : public Logger {

public:
  Asserter(std::ostream &out, const char *const file, const int line,
           const char level, const bool cond)
      : Logger(out, file, line, level, !cond) {}
  ~Asserter() {
    if (_predicate) {
      std::cerr << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  operator bool() const final { return _predicate; }
};

} // namespace internal
} // namespace quik_fix

template <typename T>
inline quik_fix::internal::Logger &
operator<<(quik_fix::internal::Logger &logger, const T &item) {
  logger._out << item;
  return logger;
}

// Newline Characters
template <>
inline quik_fix::internal::Logger &
operator<<(quik_fix::internal::Logger &logger, const char &c) {
  logger._out << c;
  if (c == '\n') {
    logger._out << logger._line_prefix << "⮓ ";
  }
  return logger;
}

template <>
inline quik_fix::internal::Logger &
operator<<(quik_fix::internal::Logger &logger, const std::string &str) {
  std::istringstream strin(str);
  std::string token;
  token.reserve(str.size());

  std::getline(strin, token, '\n');
  logger._out << token;
  while (std::getline(strin, token, '\n')) {
    logger._out << std::endl << logger._line_prefix << "⮓ " << token;
  }
  // In case if the last character is a newline character.
  if (!str.empty() && str[str.size() - 1] == '\n') {
    logger._out << std::endl << logger._line_prefix << "⮓ ";
  }
  return logger;
}

using endl_symbol_t = std::ostream &(*)(std::ostream &);

template <>
inline quik_fix::internal::Logger &
operator<<(quik_fix::internal::Logger &logger,
           const endl_symbol_t &endl_symbol) {
  logger._out << endl_symbol;
  if (endl_symbol == quik_fix::endl) {
    logger._out << logger._line_prefix << "⮓ ";
  }
  return logger;
}

#define QF_CHECK_AT(cond, file, line)                                          \
  if (auto _asserter =                                                         \
          quik_fix::internal::Asserter(std::cerr, file, line, 'E', (cond)))    \
  _asserter

#define QF_CHECK(cond) QF_CHECK_AT(cond, __FILE__, __LINE__)

#define QF_LOG_INFO_AT(file, line)                                             \
  if (auto _logger = quik_fix::internal::Logger(std::cout, file, line, 'I'))   \
  _logger

#define QF_LOG_INFO QF_LOG_INFO_AT(__FILE__, __LINE__)

#define QF_WARN_AT(file, line)                                                 \
  if (auto _logger = quik_fix::internal::Logger(std::cerr, file, line, 'W'))   \
  _logger

#define QF_WARN QF_WARN_AT(__FILE__, __LINE__)
