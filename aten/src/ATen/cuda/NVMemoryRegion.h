#pragma once

#include <cstdlib>

#include <memory>

namespace at {
namespace cuda {

using HostMemoryRegion_t = std::unique_ptr<void, void (*)(void*)>;

size_t getZeroCompressedUInt32SizeForHostMemoryRegion(
    HostMemoryRegion_t&& host_memory_region,
    const size_t size);

struct NVMemoryRegion {
  void* dptr = nullptr;
  size_t size = 0;

  NVMemoryRegion() = default;
  /// @brief Construct a view of a memory region.
  /// @param dptr
  /// @param size
  NVMemoryRegion(void* const dptr, const size_t size)
      : dptr(dptr), size(size) {}
  NVMemoryRegion(const size_t size_in_bytes);
  ~NVMemoryRegion();

 private:
  bool _free_needed = false;
  NVMemoryRegion(const NVMemoryRegion&) = delete;
  NVMemoryRegion& operator=(const NVMemoryRegion&) = delete;

 public:
  NVMemoryRegion(NVMemoryRegion&&) noexcept;
  NVMemoryRegion& operator=(NVMemoryRegion&&) noexcept;

  /// @brief Get the starting position of the trailing zeroes.
  /// @return
  size_t getZeroCompressedUInt32Size() const;

  /// @brief Copy the memory region to the host.
  /// @param fetch_from_gpu Whether to only reserve the region or copy the data
  /// values as well.
  /// @return A unique pointer to the allocated memory region on the host side
  HostMemoryRegion_t toCPU(const bool fetch_from_gpu = true) const;

  /// @brief Dump the memory region to a plain text/binary file.
  /// @param filename
  /// @param as_plain_text
  /// @param width
  void dump(
      const std::string& filename,
      const bool as_plain_text = true,
      const size_t width = 128) const;
  /// @brief Restore the memory region from a binary file.
  /// @param filename
  void load(const std::string& filename);
};

} // namespace cuda
} // namespace at
