// <bojian/DynamicCUDAGraph>
#pragma once

#include <array>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>

#include <boost/filesystem.hpp>
#include <cuda_runtime.h>
#include <dmlc/logging.h>
#include <helper_cuda.h>

namespace at {
namespace cuda {

struct ZeroCompressedPtr;

#if !defined(NATIVE_PLAYGROUND_BUILD)
struct CUDAGraph;
#endif

class
#if !defined(NATIVE_PLAYGROUND_BUILD)
    TORCH_CUDA_CPP_API
#endif
        NVPMAAllocCapturer {
 private:
  static constexpr unsigned int _C_DEVICE_PCI_ID_LEN = 128;
  enum class _CapturePMAAllocMode {
    kDefault = 0,
    kRecord = 1,
    kReplay = 2,
    kReplayNextImm = 3,
    kProbeNextImm = 4, // Probe the allocation size, but do NOT perform any real
                       // allocations.
    kClearRecords = 5,
    kEnd = 6
  };
  size_t _pma_alloc_size = 0;
  void* _shadow_ptr = nullptr;
  // std::vector<void*> _shadow_residuals;

  static std::array<char, _C_DEVICE_PCI_ID_LEN> _getCurrentDevicePCIBusId() {
    std::array<char, _C_DEVICE_PCI_ID_LEN> current_device_pci_id;
    int current_device;
    checkCudaErrors(cudaGetDevice(&current_device));
    checkCudaErrors(cudaDeviceGetPCIBusId(
        current_device_pci_id.data(), _C_DEVICE_PCI_ID_LEN, current_device));
    return current_device_pci_id;
  }

  static bool _writeToDrvFile(
      const _CapturePMAAllocMode capture_pma_alloc_mode) {
    using namespace boost::filesystem;

    path nv_drv_gpus_dir("/proc/driver/nvidia/gpus/");
    path nv_capture_pma_alloc_file(
        nv_drv_gpus_dir / _getCurrentDevicePCIBusId().data() /
        "capture_pma_alloc");
    if (is_regular_file(nv_capture_pma_alloc_file.string())) {
      std::ofstream fout(nv_capture_pma_alloc_file.string());

      LOG(INFO) << "Writing " << static_cast<int>(capture_pma_alloc_mode)
                << " to file=" << nv_capture_pma_alloc_file.string();
      fout << static_cast<int>(capture_pma_alloc_mode) << std::endl;
      fout.close();
      return true;
    }
    LOG(INFO) << "File=" << nv_capture_pma_alloc_file.string()
              << " has not been found";
    return false;
  }

  static std::pair<size_t, unsigned> _queryRecordedPMAAllocSizes() {
    using namespace boost::filesystem;

    size_t pma_alloc_size = 0;
    unsigned num_residuals = 0;

    path nv_drv_gpus_dir("/proc/driver/nvidia/gpus/");
    path nv_capture_pma_alloc_file(
        nv_drv_gpus_dir / _getCurrentDevicePCIBusId().data() /
        "capture_pma_alloc");
    FILE* fin = fopen(nv_capture_pma_alloc_file.string().c_str(), "r");

    if (fin == NULL) {
      return std::make_pair<size_t, unsigned>(0, 0);
    }
    if (fscanf(fin, "%ld %d", &pma_alloc_size, &num_residuals) == 0) {
      fclose(fin);
      return std::make_pair<size_t, unsigned>(0, 0);
    }
    fclose(fin);

    LOG(INFO) << "Returning pma_alloc_size=" << pma_alloc_size
              << ", num_residuals=" << num_residuals;
    return std::make_pair(pma_alloc_size, num_residuals);
  }

  size_t _getTrailingZerosStartPos() const;

  static void* _sLastGraphResidualPtr;
  static size_t _sLastGraphResidualSize;

  static void* _getShadowHostPtr(
      void* shadow_ptr,
      const size_t pma_alloc_size,
      const bool fetch_from_gpu) {
    if (!pma_alloc_size) {
      return nullptr;
    }
    void* shadow_host_ptr = malloc(pma_alloc_size);
    if (fetch_from_gpu) {
      checkCudaErrors(cudaMemcpy(
          shadow_host_ptr, shadow_ptr, pma_alloc_size, cudaMemcpyDeviceToHost));
    }
    return shadow_host_ptr;
  }

  static std::vector<size_t> _sQueriedPMAAllocSizes;

  NVPMAAllocCapturer(NVPMAAllocCapturer&&) noexcept = delete;
  NVPMAAllocCapturer& operator=(NVPMAAllocCapturer&&) noexcept = delete;

 public:
  NVPMAAllocCapturer() = default;
  NVPMAAllocCapturer(const NVPMAAllocCapturer&) = default;
  NVPMAAllocCapturer& operator=(const NVPMAAllocCapturer&) = default;

  static void setToRecordMallocs() {
    LOG(INFO) << "Recording ...";
    _writeToDrvFile(_CapturePMAAllocMode::kClearRecords);
    _writeToDrvFile(_CapturePMAAllocMode::kRecord);
  }
  static void setToShadowNextImmMalloc() {
    LOG(INFO) << "Shadowing the next immediate malloc ...";
    _writeToDrvFile(_CapturePMAAllocMode::kReplayNextImm);
  }
  static void setToShadowMallocs() {
    LOG(INFO) << "Shadowing ALL the subsequent mallocs ...";
    _writeToDrvFile(_CapturePMAAllocMode::kReplay);
  }
  static void resetToDefault() {
    LOG(INFO) << "Restoring to default";
    _writeToDrvFile(_CapturePMAAllocMode::kDefault);
  }

#define NV_HUGE_PAGE_SIZE (2 * 1024 * 1024)

  void materialize() {
    // CHECK(_pma_alloc_size == 0) << "Double materialization is not allowed";
    unsigned num_residuals;
    std::tie(_pma_alloc_size, num_residuals) = _queryRecordedPMAAllocSizes();
    if (!_pma_alloc_size) {
      return;
    }
    // setToShadowMallocs();
    setToShadowNextImmMalloc();
    checkCudaErrors(cudaMalloc(&_shadow_ptr, _pma_alloc_size));
    // The following free call will have no effect, since the driver will not be
    // able to locate the shadow memory handle in the resource tree. This
    // prevents catastrophic behavior such as destroying the memory allocations
    // that the shadow pointer refers to.
    //
    // checkCudaErrors(cudaFree(*_shadow_ptr));
    // _shadow_residuals.resize(num_residuals, nullptr);
    // for (size_t shadow_residual_id = 0; shadow_residual_id < num_residuals;
    //      ++shadow_residual_id) {
    //   checkCudaErrors(cudaMalloc(
    //       &_shadow_residuals[shadow_residual_id], NV_HUGE_PAGE_SIZE));
    // }
    // resetToDefault();
  }

#define MATERIALIZE_CAPTURER_WITH_STMT(capturer, stmt)    \
  do {                                                    \
    ::at::cuda::NVPMAAllocCapturer::setToRecordMallocs(); \
    checkCudaErrors(stmt);                                \
    (capturer).materialize();                             \
  } while (false)

#define REPLAY_MALLOCS_IN_STMT_USING_CAPTURER(stmt, capturer) \
  do {                                                        \
    ::at::cuda::NVPMAAllocCapturer::setToShadowMallocs();     \
    checkCudaErrors(stmt);                                    \
    (capturer).resetToDefault();                              \
  } while (false)

 private:
  static void _setToProbeNextImmMallocBegin() {
    LOG(INFO) << "Set to probe the next immediate malloc ...";
    _writeToDrvFile(_CapturePMAAllocMode::kClearRecords);
    _writeToDrvFile(_CapturePMAAllocMode::kProbeNextImm);
  }

  static void _setToProbeNextImmMallocEnd() {
    // The flag should be already unset to default by the driver, so do not need
    // to communicate with the driver file here.
    _sQueriedPMAAllocSizes.push_back(_queryRecordedPMAAllocSizes().first);
  }

#define _SCOPED_PROBE_FROM_STMT(stmt)                                \
  do {                                                               \
    ::at::cuda::NVPMAAllocCapturer::_setToProbeNextImmMallocBegin(); \
    stmt;                                                            \
    ::at::cuda::NVPMAAllocCapturer::_setToProbeNextImmMallocEnd();   \
  } while (false)

#define _PROBE_NEXT_IMM_MALLOC_FROM_STMT(stmt) \
  do {                                         \
    _SCOPED_PROBE_FROM_STMT(stmt);             \
    cudaGetLastError();                        \
  } while (false)

  static std::pair<size_t, size_t> _getLargestGraphAndClearCachedAllocSizes() {
    if (_sQueriedPMAAllocSizes.empty()) {
      return std::make_pair<size_t, size_t>(static_cast<size_t>(-1), 0);
    }
    size_t max_pma_alloc_size = _sQueriedPMAAllocSizes[0];
    size_t max_pma_alloc_size_id = 0;

    for (size_t pma_alloc_sizes_id = 1;
         pma_alloc_sizes_id < _sQueriedPMAAllocSizes.size();
         ++pma_alloc_sizes_id) {
      if (_sQueriedPMAAllocSizes[pma_alloc_sizes_id] > max_pma_alloc_size) {
        max_pma_alloc_size_id = pma_alloc_sizes_id;
      }
    }
    _sQueriedPMAAllocSizes.clear();
    return std::make_pair(max_pma_alloc_size_id, max_pma_alloc_size);
  }

 public:
#if defined(NATIVE_PLAYGROUND_BUILD)
  void reserveForCUDAGraphs(const std::vector<cudaGraph_t>& graphs);
#else
  static void MaterializeCUDAGraphs(
      std::vector<std::reference_wrapper<CUDAGraph>>& graphs);
#endif
  void calculateSparsityRatio() const;

  void* shadow_ptr() const {
    return _shadow_ptr;
  }
  size_t alloc_size() const {
    return _pma_alloc_size;
  }
  void dump(
      const std::string& filename,
      const bool as_plain_text = true,
      const size_t width = 128);
  void load(const std::string& filename);

  ZeroCompressedPtr compressZeros();
  void deflateZeros(const ZeroCompressedPtr& compressed_ptr);
};

} // namespace cuda
} // namespace at
