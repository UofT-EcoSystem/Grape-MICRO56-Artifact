// clang-format off
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>

// <bojian/Grape>
#include "NVCompressor.h"
#include "NVPMAAllocCapturer.h"

namespace at {

struct CUDAGeneratorImpl;

namespace cuda {

// <bojian/Grape> By default, use zero compression for the memory regions.
using CompressedRegion_t = RLECompressedRegion;

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();

struct TORCH_CUDA_CPP_API CUDAGraph {
  // <bojian/Grape>
  // clang-format on
  // CUDAGraph();
  /// @brief Construct a CUDAGraph object.
  /// @param postpone_instantiation Whether to postpone the graph instantiation
  /// later (i.e., not when the capture ends). This could be helpful in the case
  /// of device CUDAGraphs and/or metadata compression.
  /// @param frugal_launch Whether the launch should be frugal, i.e., without
  /// @param instantiate_on_device Whether the instantiation should happen on
  /// the device side.
  /// handling the random number generator.
  explicit CUDAGraph(
      const bool postpone_instantiation = false,
      const bool frugal_launch = true,
      const bool instantiate_on_device = true);
  // clang-format off

  ~CUDAGraph();

  void capture_begin(MempoolId_t pool={0, 0});

  // <bojian/Grape>
  // clang-format on
  // void capture_end();
  /// @brief Instantiate the CUDAGraph only if no exception happened during the
  /// capture.
  /// @param no_exception_in_capture
  void capture_end(const bool no_exception_in_capture);
  // clang-format off

  // <bojian/Grape>
  // clang-format on
  /// @brief Split the epilog into a different method since we might need
  // to postpone it when doing the compression.
  /// @param instantiate_on_device Whether the instantiation should happen on
  /// device.
  void capture_end_epilog(const bool instantiate_on_device = false);

  /// @brief Decompress the metadata region of the CUDAGraph.
  void decompress();
  // clang-format off
  // </bojian/Grape>

  void replay();
  void reset();
  MempoolId_t pool();

  protected:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;

  // <bojian/Grape>
  // clang-format on
  bool _postpone_instantiation;
  bool _frugal_launch;
  bool _instantiate_on_device;

  // Device-Side CUDAGraph
  cudaGraph_t device_graph_ = nullptr;
  cudaGraphExec_t device_graph_exec_ = nullptr;

 public:
  std::vector<CUDAGraph*> subgraphs;
  void addSubgraph(CUDAGraph& subgraph) {
    subgraphs.push_back(&subgraph);
  }

 protected:
  // Compression-related members
  std::shared_ptr<NVMemoryRegion> _orig_yin_main_metadata,
      _orig_yang_main_metadata;
  NVMemoryRegion _orig_yin_residual_metadata;
  CompressedRegion_t _compressed_yin_residual_metadata,
      _compressed_yang_main_metadata;
  // std::vector<std::shared_ptr<NVMemoryRegion>> _orig_list_of_residuals;
  std::vector<NVMemoryRegion> _orig_list_of_residuals;
  std::vector<CompressedRegion_t> _compressed_list_of_residuals;
  // clang-format off
  // </bojian/Grape>

#endif

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if cudaStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by cudaGraphInstantiate
  // to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if cudaGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, retrieved from Cuda
  CaptureId_t id_;

  // uuid used to request a particular private mempool from CUDACachingAllocator.
  // By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's mempool_id_
  // will be set to the other graph's mempool_id_, and therefore share a mempool with the
  // other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from graph_pool_handle(),
  // it will share a mempool with any other captures that used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Stream on which capture began
  at::cuda::CUDAStream capture_stream_;

  // Default generator on device where capture began
  at::CUDAGeneratorImpl* capture_gen_;

  // Device where capture occurred. Right now, for simplicity, we require all ops
  // in a capture to run on the same device, but this is a limitation of CUDAGraph,
  // not CUDA itself.  We can straightforwardly modify CUDAGraph to support multi-device
  // captures if needed.
  int capture_dev_;

  // RNG state trackers
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;

  // <bojian/Grape>
  // clang-format on
  friend void instantiateCUDAGraphsOnCompressedMetadata(
      std::vector<std::reference_wrapper<CUDAGraph>>& graphs,
      const bool debug_mode,
      const bool instantiate_on_device,
      const bool compress_residuals);
  friend void embedDeviceCUDAGraph(CUDAGraph& graph, Tensor sync_barrier);
};

// <bojian/Grape>
/// @brief Instantiate a list of CUDAGraphs.
/// @param graphs The list of CUDAGraphs that are to be instantiated into
/// executors
/// @param debug_mode Whether to use instantiate in debug mode
void instantiateCUDAGraphsOnCompressedMetadata(
    std::vector<std::reference_wrapper<CUDAGraph>>& graphs,
    const bool debug_mode,
    const bool instantiate_on_device,
    const bool compress_residuals);

/// @brief Instantiate a CUDAGraph on the device side (V2). This API only
/// instantiates but does not create the scheduling graph.
/// @param graph The graph that is to be conditionally launched.
void instantiateCUDAGraphOnDeviceV2(CUDAGraph& graph);

/// @brief Embed a CUDAGraph on the device side.
/// @param graph
void embedDeviceCUDAGraph(CUDAGraph& graph, Tensor sync_barrier);

// clang-format off
// </bojian/Grape>

} // namespace cuda
} // namespace at
