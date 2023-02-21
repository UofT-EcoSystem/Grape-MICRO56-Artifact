#include <memory>

#include <boost/filesystem.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#if !defined(BUILD_OUT_OF_PYTORCH_TREE)
#include "CUDAGraph.h"
#endif

#include "NVCompressor.h"
#include "NVPMAAllocCapturer.h"

#if defined(BUILD_OUT_OF_PYTORCH_TREE)
#include <quik_fix/logging.h>
#else
#include <dmlc/logging.h>
#define QF_LOG_INFO LOG(INFO)
#endif

namespace at {
namespace cuda {

bool NVPMAAllocCapturer::verbose = false;
void NVPMAAllocCapturer::setVerbosity(const bool _verbose) {
  verbose = _verbose;
}

std::array<char, NVPMAAllocCapturer::_C_DEVICE_PCI_ID_LEN> NVPMAAllocCapturer::
    _getCurrentDevicePCIBusId() {
  std::array<char, _C_DEVICE_PCI_ID_LEN> current_device_pci_id;
  int current_device;
  checkCudaErrors(cudaGetDevice(&current_device));
  checkCudaErrors(cudaDeviceGetPCIBusId(
      current_device_pci_id.data(), _C_DEVICE_PCI_ID_LEN, current_device));
  return current_device_pci_id;
}

const std::string& NVPMAAllocCapturer::_getNVDrvCapturePMAAllocAbsPathStr() {
  using namespace boost::filesystem;

  static path nv_drv_gpus_dir("/proc/driver/nvidia/gpus/");
  static path nv_capture_pma_alloc_file(
      nv_drv_gpus_dir / _getCurrentDevicePCIBusId().data() /
      "capture_pma_alloc");
  return nv_capture_pma_alloc_file.string();
}

bool NVPMAAllocCapturer::_writeToDrvFile(
    const NVCapturePMAAllocMode_t capture_pma_alloc_mode) {
  const std::string& nv_capture_pma_alloc_abs_path_str =
      _getNVDrvCapturePMAAllocAbsPathStr();
  if (boost::filesystem::is_regular_file(nv_capture_pma_alloc_abs_path_str)) {
    std::ofstream fout(nv_capture_pma_alloc_abs_path_str);
    if (verbose) {
      QF_LOG_INFO << "Writing ("
                  << NVCapturePMAAllocMode2CStr[static_cast<int>(
                         capture_pma_alloc_mode)]
                  << ") to file=" << nv_capture_pma_alloc_abs_path_str;
    }
    fout << static_cast<int>(capture_pma_alloc_mode) << std::endl;
    fout.close();
    return true;
  }
  QF_LOG_INFO << "File=" << nv_capture_pma_alloc_abs_path_str
              << " has not been found";
  return false;
}

void NVPMAAllocCapturer::setToRecordMallocs() {
  _writeToDrvFile(kClearRecords);
  _writeToDrvFile(kRecord);
}

void NVPMAAllocCapturer::setToShadowMallocs() {
  _writeToDrvFile(kReplay);
}

void NVPMAAllocCapturer::setToShadowNextMalloc() {
  _writeToDrvFile(kReplayNext);
}

void NVPMAAllocCapturer::setToProbeNextMalloc() {
  _writeToDrvFile(kClearRecords);
  _writeToDrvFile(kProbeNext);
}

void NVPMAAllocCapturer::setToShadowNextMallocAndStashResiduals() {
  _writeToDrvFile(kReplayNextAndStashResiduals);
}

void NVPMAAllocCapturer::setToShadowNextMallocAndAppendResiduals() {
  _writeToDrvFile(kReplayNextAndAppendResiduals);
}

void NVPMAAllocCapturer::setToShadowResiduals() {
  _writeToDrvFile(kReplayResiduals);
}

void NVPMAAllocCapturer::clearListOfResiduals() {
  _writeToDrvFile(kClearListOfResiduals);
}

void NVPMAAllocCapturer::setToRecordNextAndOverwrite() {
  _writeToDrvFile(kRecordNextAndOverwrite);
}

void NVPMAAllocCapturer::resetToDefault() {
  _writeToDrvFile(kDefault);
}

NVPMAQueryResult_t NVPMAAllocCapturer::queryRecordedPMAAllocSizes() {
  size_t pma_alloc_size = 0;
  uint32_t capture_pma_alloc_mode = 0;
  NVPMAQueryResult_t ret;

  const std::string& nv_capture_pma_alloc_abs_path_str =
      _getNVDrvCapturePMAAllocAbsPathStr();
  std::ifstream fin(nv_capture_pma_alloc_abs_path_str);

  fin >> capture_pma_alloc_mode;
  ret.current_mode =
      static_cast<NVCapturePMAAllocMode_t>(capture_pma_alloc_mode);
  QF_LOG_INFO << "Current PMAAllocCapturer mode="
              << NVCapturePMAAllocMode2CStr[capture_pma_alloc_mode];
  fin >> ret.current_residual_capacity;
  fin >> ret.current_residual_idx;
  while (fin >> pma_alloc_size) {
    QF_LOG_INFO << "Pushing pma_alloc_size=" << pma_alloc_size
                << " (=" << (pma_alloc_size * 1.0 / (1024 * 1024)) << " MB)";
    ret.record_entries.push_back(pma_alloc_size);
  }
  if (ret.record_entries.empty()) {
    ret.record_entries.push_back(0);
  }
  return ret;
}

std::vector<NVMemoryRegion> NVPMAAllocCapturer::residuals_snapshot;

void NVPMAAllocCapturer::updateResidualsSnapshot() {
  residuals_snapshot.clear();

  NVPMAQueryResult_t pma_query_result = queryRecordedPMAAllocSizes();
  QF_LOG_INFO << "Materializing " << pma_query_result.current_residual_capacity
              << " residuals";
  setToShadowResiduals();
  for (size_t residual_id = 0;
       residual_id < pma_query_result.current_residual_capacity;
       ++residual_id) {
    void* residual_dptr;
    checkCudaErrors(cudaMalloc(&residual_dptr, 2 * 1024 * 1024));
    residuals_snapshot.emplace_back(residual_dptr, 2 * 1024 * 1024);
  }
  resetToDefault();
}

void NVPMAAllocCapturer::materialize() {
  shadow_main_region.size = queryRecordedPMAAllocSizes().record_entries[0];
  if (!shadow_main_region.size) {
    return;
  }
  QF_LOG_INFO << "Materializing the capturer with a main region size="
              << shadow_main_region.size;
  setToShadowNextMalloc();
  shadow_main_region = NVMemoryRegion(shadow_main_region.size);
  // The following free call will have no effect, since the driver will not be
  // able to locate the shadow memory handle in the resource tree. This prevents
  // catastrophic behavior such as destroying the memory allocations that the
  // shadow pointer refers to.
  //
  //     checkCudaErrors(cudaFree(*_shadow_ptr));
  //
}

void NVPMAAllocCapturer::dump(
    const std::string& filename,
    const bool as_plain_text,
    const size_t width) {
  shadow_main_region.dump(filename, as_plain_text, width);
}
void NVPMAAllocCapturer::load(const std::string& filename) {
  shadow_main_region.load(filename);
}

} // namespace cuda
} // namespace at
