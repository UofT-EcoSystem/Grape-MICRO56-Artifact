#pragma once

#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
  out << "[";
  for (const auto &elem : vec) {
    out << elem << ", ";
  }
  out << "]";
  return out;
}

template <typename K, typename V, typename Hash, typename Pred>
inline std::ostream &
operator<<(std::ostream &out,
           const std::unordered_map<K, V, Hash, Pred> &hash_table) {
  out << "{" << std::endl;
  for (const auto &elem : hash_table) {
    out << "  " << elem.first << " : " << elem.second << ", " << std::endl;
  }
  out << "}";
  return out;
}

template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &out,
                                const std::pair<T1, T2> &pair) {
  out << "(" << pair.first << ", " << pair.second << ")";
  return out;
}

namespace internal {

template <std::size_t...> struct Seq {};

template <std::size_t N, std::size_t... TIs>
struct SeqGenerator : SeqGenerator<N - 1, N - 1, TIs...> {};

template <std::size_t... TIs> struct SeqGenerator<0, TIs...> : Seq<TIs...> {};

template <class Tuple, std::size_t... TIs>
void _unpack_tuple_in_ostream(std::ostream &out, const Tuple &t, Seq<TIs...>) {
  using kirby = int[];
  (void)kirby{0,
              (void(out << (TIs == 0 ? "" : ", ") << std::get<TIs>(t)), 0)...};
}

} // namespace internal

template <typename... TArgs>
inline std::ostream &operator<<(std::ostream &out,
                                const std::tuple<TArgs...> &tuple) {
  out << "(";
  internal::_unpack_tuple_in_ostream(
      out, tuple, internal::SeqGenerator<sizeof...(TArgs)>());
  out << ")";
  return out;
}
