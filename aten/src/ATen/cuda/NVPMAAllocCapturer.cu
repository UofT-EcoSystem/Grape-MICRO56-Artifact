// <bojian/DynamicCUDAGraph>
#include <helper_cuda.h>

#if !defined(NATIVE_PLAYGROUND_BUILD)
#include "CUDAGeneratorImpl.h"
#include "CUDAGraph.h"
#endif

#include "NVPMAAllocCapturer.h"
#include "NVPMAAllocCompressor.h"

namespace at {
namespace cuda {

std::vector<size_t> NVPMAAllocCapturer::_sQueriedPMAAllocSizes;
void* NVPMAAllocCapturer::_sLastGraphResidualPtr = nullptr;
size_t NVPMAAllocCapturer::_sLastGraphResidualSize = 0;

#if defined(NATIVE_PLAYGROUND_BUILD)
void NVPMAAllocCapturer::reserveForCUDAGraphs(
    const std::vector<cudaGraph_t>& graphs) {
  LOG(INFO) << "Reserving memory spaces for " << graphs.size() << " CUDAGraphs";

  cudaGraphExec_t tmp_instance_workspace;

  for (const cudaGraph_t graph : graphs) {
    _PROBE_NEXT_IMM_MALLOC_FROM_STMT(cudaGraphInstantiate(
        &tmp_instance_workspace, graph, nullptr, nullptr, 0));
  }
  size_t max_pma_alloc_size_id, max_pma_alloc_size;
  std::tie(max_pma_alloc_size_id, max_pma_alloc_size) =
      _getLargestGraphAndClearCachedAllocSizes();
  if (max_pma_alloc_size_id == static_cast<size_t>(-1)) {
    return;
  }
  MATERIALIZE_CAPTURER_WITH_STMT(
      *this,
      cudaGraphInstantiate(
          &tmp_instance_workspace,
          graphs[max_pma_alloc_size_id],
          nullptr,
          nullptr,
          0));
}
#else
void NVPMAAllocCapturer::MaterializeCUDAGraphs(
    std::vector<std::reference_wrapper<CUDAGraph>>& graphs) {
  LOG(INFO) << "Compressing for the PyTorch's CUDAGraphs";
  LOG(INFO) << "Reserving memory spaces for " << graphs.size() << " CUDAGraphs";

  if (graphs.empty()) {
    LOG(WARNING) << "No graph to compress, directly exiting";
    return;
  }

  cudaGraphExec_t tmp_instance_workspace;
  std::vector<ZeroCompressedPtr> compressed_ptrs;
  void *yin_ptr, *yang_ptr, *curr_graph_ptr;

  for (size_t shape_id = 0; shape_id < graphs.size(); ++shape_id) {
    _PROBE_NEXT_IMM_MALLOC_FROM_STMT(cudaGraphInstantiate(
        &tmp_instance_workspace,
        graphs[shape_id].get().graph_,
        nullptr,
        nullptr,
        0));
  }
  size_t max_pma_alloc_size_id, max_pma_alloc_size;
  std::tie(max_pma_alloc_size_id, max_pma_alloc_size) =
      _getLargestGraphAndClearCachedAllocSizes();
  if (max_pma_alloc_size == 0) {
    return;
  }
  LOG(INFO) << "Graph ID with the maximum size: " << max_pma_alloc_size_id
            << " (" << max_pma_alloc_size << " bytes)";
  NVPMAAllocCapturer::setToRecordMallocs();
  checkCudaErrors(cudaMalloc(&yin_ptr, max_pma_alloc_size));
  checkCudaErrors(cudaMalloc(&yang_ptr, max_pma_alloc_size));
  NVPMAAllocCapturer::resetToDefault();

  NVPMAAllocCapturer capturer;

  for (size_t graph_id = 0; graph_id < graphs.size() * 2; ++graph_id) {
    size_t shape_id = graph_id / 2;
    curr_graph_ptr = graph_id % 2 == 0 ? yin_ptr : yang_ptr;

    checkCudaErrors(cudaMemset(curr_graph_ptr, 0, max_pma_alloc_size));
    if (_sLastGraphResidualPtr != nullptr) {
      checkCudaErrors(
          cudaMemset(_sLastGraphResidualPtr, 0, _sLastGraphResidualSize));
    }

    setToShadowNextImmMalloc();
    if (graph_id % 2 != 0) {
      graphs[shape_id].get().capture_end_epilog();
      graphs[shape_id].get().replay();
    } else {
      checkCudaErrors(cudaGraphInstantiate(
          &tmp_instance_workspace,
          graphs[shape_id].get().graph_,
          nullptr,
          nullptr,
          0));
      checkCudaErrors(
          cudaGraphLaunch(tmp_instance_workspace, getCurrentCUDAStream()));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    capturer._shadow_ptr = curr_graph_ptr;
    capturer._pma_alloc_size = max_pma_alloc_size;

    if (graph_id == 0) {
      capturer.dump("cuda_graph_sample");
    }
    ZeroCompressedPtr zero_compressed_main_metadata(
        std::move(capturer.compressZeros()));

    if (graph_id % 2 != 0) {
      std::string cuda_graph_metadata_dump_fname =
          "cuda_graph_shape" + std::to_string(shape_id);
      // capturer.dump(cuda_graph_metadata_dump_fname + "-main");

      graphs[shape_id].get()._main_capturer = capturer;
      graphs[shape_id].get()._zero_compressed_main_metadata =
          std::move(zero_compressed_main_metadata);

      if (_sLastGraphResidualPtr != nullptr) {
        capturer._shadow_ptr = _sLastGraphResidualPtr;
        capturer._pma_alloc_size = _sLastGraphResidualSize;
        // capturer.dump(cuda_graph_metadata_dump_fname + "-residual");

        ZeroCompressedPtr zero_compressed_residual_metadata(
            std::move(capturer.compressZeros()));

        if (zero_compressed_residual_metadata.compressed_size != 0) {
          graphs[shape_id].get()._residual_capturer = capturer;
          graphs[shape_id].get()._zero_compressed_residual_metadata =
              std::move(zero_compressed_residual_metadata);
        }
      }
      _sLastGraphResidualPtr = static_cast<char*>(curr_graph_ptr) +
          graphs[shape_id].get()._zero_compressed_main_metadata.compressed_size;
      _sLastGraphResidualSize = max_pma_alloc_size -
          graphs[shape_id].get()._zero_compressed_main_metadata.compressed_size;
    } else { // graph_id % 2 == 0
      _sLastGraphResidualPtr = static_cast<char*>(curr_graph_ptr) +
          zero_compressed_main_metadata.compressed_size;
      _sLastGraphResidualSize =
          max_pma_alloc_size - zero_compressed_main_metadata.compressed_size;
    } // graph_id % 2 != 0
  }
  LOG(INFO) << "Sanity checking on the compressed data";
  for (size_t shape_id = 0; shape_id < graphs.size(); ++shape_id) {
    graphs[shape_id].get().decompress();
    graphs[shape_id].get().replay();
    checkCudaErrors(cudaDeviceSynchronize());
  }
}
#endif

void NVPMAAllocCapturer::calculateSparsityRatio() const {
  void* shadow_host_ptr = _getShadowHostPtr(
      /*shadow_ptr=*/_shadow_ptr,
      /*pma_alloc_size=*/_pma_alloc_size,
      /*fetch_from_gpu=*/true);

  uint32_t* shadow_host_uint32_ptr = static_cast<uint32_t*>(shadow_host_ptr);
  size_t num_zeros = 0;
  for (size_t i = 0; i < _pma_alloc_size / sizeof(uint32_t); ++i) {
    if (shadow_host_uint32_ptr[i] == 0) {
      ++num_zeros;
    }
  }
  LOG(INFO) << "Sparsity Ratio: " << (_pma_alloc_size / sizeof(uint32_t))
            << " / " << num_zeros << " = "
            << (num_zeros * 100.0 / (_pma_alloc_size / sizeof(uint32_t)))
            << "%";
  free(shadow_host_ptr);
}

void NVPMAAllocCapturer::dump(
    const std::string& filename,
    const bool as_plain_text,
    const size_t width) {
  void* shadow_host_ptr = _getShadowHostPtr(
      /*shadow_ptr=*/_shadow_ptr,
      /*pma_alloc_size=*/_pma_alloc_size,
      /*fetch_from_gpu=*/true);
  std::ofstream fout(
      filename + "-main" + (as_plain_text ? ".txt" : ".bin"),
      as_plain_text ? std::ios::out : (std::ios::out | std::ios::binary));

  LOG(INFO) << "Dumping " << _pma_alloc_size
            << " bytes from the shadow pointer";

  if (as_plain_text) {
    uint32_t* shadow_host_uint32_ptr = static_cast<uint32_t*>(shadow_host_ptr);
    for (size_t i = 0; i < _pma_alloc_size / sizeof(uint32_t);) {
      for (size_t j = 0; i < _pma_alloc_size / sizeof(uint32_t) && j < width;
           ++j, ++i) {
        fout << std::hex << shadow_host_uint32_ptr[i] << std::dec;
      }
      fout << std::endl;
    }
    fout << std::endl;
  } else {
    fout.write(static_cast<const char*>(shadow_host_ptr), _pma_alloc_size);
  }
  fout.close();
  free(shadow_host_ptr);
}

void NVPMAAllocCapturer::load(const std::string& filename) {
  if (!_pma_alloc_size) {
    return;
  }
  void* shadow_host_ptr = _getShadowHostPtr(
      /*shadow_ptr=*/_shadow_ptr,
      /*pma_alloc_size=*/_pma_alloc_size,
      /*fetch_from_gpu=*/false);
  std::ifstream fin(filename + "-main.bin", std::ios::in | std::ios::binary);

  LOG(INFO) << "Loading " << _pma_alloc_size
            << " bytes into the shadow pointer";

  if (!fin) {
    LOG(WARNING) << "Failed to load into the shadow pointer";
    fin.close();
    return;
  }

  fin.read(static_cast<char*>(shadow_host_ptr), _pma_alloc_size);
  fin.close();
  checkCudaErrors(cudaMemcpy(
      _shadow_ptr, shadow_host_ptr, _pma_alloc_size, cudaMemcpyHostToDevice));
  free(shadow_host_ptr);
}

// static __device__ unsigned long long sTrailingZerosStartPos;

constexpr int C_NTHREADS_PER_BLOCK = 128;
constexpr size_t C_ZERO_COMPRESSION_PAGE_SIZE = 4096;
constexpr size_t C_NUM_CONSECUTIVE_ZEROS = 4;

__launch_bounds__(C_NTHREADS_PER_BLOCK) static __global__
    void _cudaGetTrailingZerosStartingPos(
        const uint32_t* const __restrict__ data,
        const size_t nelems,
        unsigned long long* const __restrict__ trailing_zeros_start_pos) {
  const unsigned int g_threadIdx = threadIdx.x + blockDim.x * blockIdx.x;

  if (g_threadIdx < nelems &&
      // In the case when
      //
      //     nelems - 1 - gthreadIdx <= sTrailingZerosStartPos
      //
      // there is no point in doing the updates.
      (nelems - 1 - g_threadIdx) > *trailing_zeros_start_pos) {
    uint32_t local_data = data[nelems - 1 - g_threadIdx];
    if (__any_sync(0xffffffff, local_data != 0)) {
      if (threadIdx.x == 0) {
        atomicMax(
            trailing_zeros_start_pos,
            static_cast<unsigned long long>(nelems - 1 - g_threadIdx + 1));
      }
    }
  }
}

// __launch_bounds__(C_NTHREADS_PER_BLOCK) static __global__
//     void _cudaDecompressZeros(
//         const uint32_t* const __restrict__ start_pos,
//         const size_t num_workers
//     ) {
//   const unsigned int g_threadIdx = threadIdx.x + blockDim.x * blockIdx.x;

//   if (g_threadIdx < num_workers) {

//   }
// }

size_t NVPMAAllocCapturer::_getTrailingZerosStartPos() const {
  unsigned long long trailing_zeros_start_pos_host,
      *trailing_zeros_start_pos_dev;

  checkCudaErrors(
      cudaMalloc(&trailing_zeros_start_pos_dev, sizeof(unsigned long long)));
  checkCudaErrors(
      cudaMemset(trailing_zeros_start_pos_dev, 0, sizeof(unsigned long long)));
  _cudaGetTrailingZerosStartingPos<<<
      (_pma_alloc_size / sizeof(uint32_t) + C_NTHREADS_PER_BLOCK - 1) /
          C_NTHREADS_PER_BLOCK,
      C_NTHREADS_PER_BLOCK>>>(
      static_cast<const uint32_t*>(_shadow_ptr),
      _pma_alloc_size / sizeof(uint32_t),
      trailing_zeros_start_pos_dev);
  checkCudaErrors(cudaMemcpy(
      &trailing_zeros_start_pos_host,
      trailing_zeros_start_pos_dev,
      sizeof(unsigned long long),
      cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(trailing_zeros_start_pos_dev));
  return trailing_zeros_start_pos_host;
}

struct DecompressEngine gDecompressEngine;

void NVPMAAllocCapturer::deflateZeros(const ZeroCompressedPtr& compressed_ptr) {
  CHECK(compressed_ptr.NotNull());

  LOG(INFO) << "main_ptr=" << compressed_ptr.dptr << " -> " << _shadow_ptr
            << " (" << compressed_ptr.compressed_size << ")";

  checkCudaErrors(cudaMemcpyAsync(
      _shadow_ptr,
      compressed_ptr.dptr,
      compressed_ptr.compressed_size,
      cudaMemcpyDeviceToDevice,
      gDecompressEngine.stream));
  checkCudaErrors(cudaMemsetAsync(
      static_cast<char*>(_shadow_ptr) + compressed_ptr.compressed_size,
      0,
      _pma_alloc_size - compressed_ptr.compressed_size,
      gDecompressEngine.stream));
}

ZeroCompressedPtr NVPMAAllocCapturer::compressZeros() {
  if (!_pma_alloc_size) {
    return ZeroCompressedPtr();
  }
  void* zero_compressed_ptr;
  size_t zero_compressed_uint32_size = _getTrailingZerosStartPos(),
         zero_compressed_size = zero_compressed_uint32_size * sizeof(uint32_t);

  // align the compressed size with the page size (4KB)
  // static_assert(C_ZERO_COMPRESSION_PAGE_SIZE % sizeof(uint32_t) == 0);
  // zero_compressed_size =
  //     (zero_compressed_size + C_ZERO_COMPRESSION_PAGE_SIZE - 1) /
  //     C_ZERO_COMPRESSION_PAGE_SIZE * C_ZERO_COMPRESSION_PAGE_SIZE;
  // size_t num_zero_compressed_pages =
  //     zero_compressed_size / C_ZERO_COMPRESSION_PAGE_SIZE;
  // CHECK(zero_compressed_size < _pma_alloc_size)
  //     << "Compressed size is large than the original PMA allocation size";
  // zero_compressed_uint32_size = zero_compressed_size / sizeof(uint32_t);

  LOG(INFO) << "Compressed Size (1st trial)=" << zero_compressed_size
            << ", Compression Ratio=" << std::setprecision(2)
            << _pma_alloc_size * 1.0 / zero_compressed_size;

  checkCudaErrors(cudaMalloc(&zero_compressed_ptr, zero_compressed_size));
  checkCudaErrors(cudaMemcpy(
      zero_compressed_ptr,
      _shadow_ptr,
      zero_compressed_size,
      cudaMemcpyDeviceToDevice));

  // std::unique_ptr<uint32_t[]> zero_compressed_host_ptr(
  //     new uint32_t[zero_compressed_uint32_size]);

  // checkCudaErrors(cudaMemcpy(
  //     zero_compressed_host_ptr.get(),
  //     zero_compressed_ptr,
  //     zero_compressed_size,
  //     cudaMemcpyDeviceToHost));

  // for (size_t page_id = 0; page_id < num_zero_compressed_pages; ++page_id) {
  //   size_t page_offset = page_id * C_ZERO_COMPRESSION_PAGE_SIZE;


  // }
  return ZeroCompressedPtr(zero_compressed_ptr, zero_compressed_size);
}

} // namespace cuda
} // namespace at
