#pragma once

#include <algorithm>
#include <functional>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <getopt.h>

#include "quik_fix/logging.h"

namespace quik_fix {

template <typename T> T parseOptArg(const char *const optarg) {
  return T(optarg);
}

template <> int parseOptArg<int>(const char *const optarg) {
  return std::stoi(optarg);
}

template <typename T>
void _parseOptArgUnifiedWrapper(const char *const optarg, void *const argv) {
  T *const casted_argv = static_cast<T *>(argv);
  *casted_argv = parseOptArg<T>(optarg);
}

class Argument {
private:
  std::string _arg_str;

  static std::vector<option> _long_options;
  static std::unordered_map<
      size_t, std::tuple<const char *, void *,
                         std::function<void(const char *const, void *const)>>>
      _opt_idx_to_argv_and_setter;

  static void _UnderscoreToDash(std::string &arg_str) {
    std::replace(arg_str.begin(), arg_str.end(), '_', '-');
  }

  static void _AddStoreTrueArgument(const char *const arg_cstr,
                                    int *const argv) {
    _long_options.push_back({arg_cstr, no_argument, argv, 1});
    QF_LOG_INFO << "Adding argument --" << _long_options.back().name;
  }

  template <typename T>
  static void _AddRequiredArgument(const char *const arg_cstr, T *const argv,
                                   const char short_opt) {
    _long_options.push_back({arg_cstr, required_argument, nullptr, short_opt});
    if (short_opt != 0) {
      QF_LOG_INFO << "Adding argument --" << arg_cstr << "/-" << short_opt;
    } else {
      QF_LOG_INFO << "Adding argument --" << arg_cstr;
    }
    auto emplace_ret = _opt_idx_to_argv_and_setter.emplace(
        short_opt == 0 ? _long_options.size() - 1
                       : static_cast<size_t>(short_opt),
        std::make_tuple(arg_cstr, argv, _parseOptArgUnifiedWrapper<T>));

    QF_CHECK(emplace_ret.second) << "Short option=" << emplace_ret.first->first
                                 << " has already been inserted before";
  }

public:
  Argument(const char *const arg_cstr, int *const argv) : _arg_str(arg_cstr) {
    _UnderscoreToDash(_arg_str);
    _AddStoreTrueArgument(_arg_str.c_str(), argv);
  }

  template <typename T>
  Argument(const char *const arg_cstr, T *const argv, const char short_opt)
      : _arg_str(arg_cstr) {
    _UnderscoreToDash(_arg_str);
    _AddRequiredArgument<T>(_arg_str.c_str(), argv, short_opt);
  }

  static void parseArguments(const int argc, char **const argv) {
    _long_options.push_back({"help", no_argument, nullptr, 'h'});
    _long_options.push_back({nullptr, 0, nullptr, 0});

    int opt_ind, opt_idx;
    std::ostringstream short_opts_strout;

    for (const option &opt : _long_options) {
      if (opt.val != 0) {
        short_opts_strout << static_cast<char>(opt.val);
        if (opt.has_arg == required_argument) {
          short_opts_strout << ":";
        }
      }
    }

    while ((opt_ind = getopt_long(argc, argv, short_opts_strout.str().c_str(),
                                  _long_options.data(), &opt_idx)) != -1) {
      QF_CHECK(opt_ind != '?') << "Unknown option detected. Use --help/-h for "
                                  "all the available options.";
      if (opt_ind == 'h') {
        QF_LOG_INFO << "Usage:";
        for (const option &opt : _long_options) {
          if (opt.name == nullptr) {
            break;
          }
          std::ostringstream strout;
          strout << "\t--" << opt.name;
          if (opt.val != 0 && opt.val != 1) {
            strout << "/-" << static_cast<char>(opt.val);
          }
          if (opt.has_arg == required_argument) {
            strout << " xxx";
          }
          QF_LOG_INFO << strout.str();
        }
        exit(EXIT_SUCCESS);
      }
      if (_long_options.at(opt_idx).flag != nullptr) {
        QF_LOG_INFO << "Flagging option --" << _long_options.at(opt_idx).name;
        continue;
      }
      auto &argv_and_setter =
          _opt_idx_to_argv_and_setter.at(opt_ind == 0 ? opt_idx : opt_ind);
      std::ostringstream opt_strout;
      if (opt_ind == 0) {
        opt_strout << "--" << _long_options.at(opt_idx).name;
      } else {
        opt_strout << "--" << std::get<0>(argv_and_setter) << "/-"
                   << static_cast<char>(opt_ind);
      }
      std::get<2>(argv_and_setter)(optarg, std::get<1>(argv_and_setter));
      QF_LOG_INFO << "Passing value \"" << optarg << "\" to option "
                  << opt_strout.str();
    }
  }
};

std::vector<option> Argument::_long_options;
std::unordered_map<size_t,
                   std::tuple<const char *, void *,
                              std::function<void(const char *const, void *)>>>
    Argument::_opt_idx_to_argv_and_setter;

} // namespace quik_fix

#define ADD_STORE_TRUE_ARGUMENT(arg)                                           \
  static int arg = 0;                                                          \
  quik_fix::Argument __##arg(#arg, &arg)

#define ADD_REQUIRED_ARGUMENT(type, arg, default)                              \
  static type arg(default);                                                    \
  quik_fix::Argument __##arg(#arg, &arg, 0)

#define ADD_REQUIRED_ARGUMENT_WITH_SHORT_OPT(type, arg, short_opt, default)    \
  static type arg(default);                                                    \
  quik_fix::Argument __##arg(#arg, &arg, short_opt)
