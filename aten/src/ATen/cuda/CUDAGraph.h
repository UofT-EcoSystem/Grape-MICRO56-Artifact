#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>


#include "NVPMAAllocCapturer.h"  // <bojian/DynamicCUDAGraph>
#include "NVPMAAllocCompressor.h"


namespace at {

struct CUDAGeneratorImpl;

namespace cuda {

// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_CUDA_CPP_API MempoolId_t graph_pool_handle();


// class CUDAGraphShadowRef;  // <bojian/DynamicCUDAGraph>
class NVPMAAllocCapturer;


struct TORCH_CUDA_CPP_API CUDAGraph {
  
  // <bojian/DynamicCUDAGraph>
  // CUDAGraph();
  explicit CUDAGraph(const bool compress_metadata = false);


  ~CUDAGraph();

  void capture_begin(MempoolId_t pool={0, 0});
  void capture_end();


  // <bojian/DynamicCUDAGraph> Split the epilog into a different method since we
  //                           might need to postpone it when doing the
  //                           compression.
  void capture_end_epilog();
  void decompress();


  void replay();
  void reset();
  MempoolId_t pool();

  protected:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;


  // <bojian/DynamicCUDAGraph>
  // NVPMAAllocCapturer capturer_;
  // bool capturer_materialized_ = false;
  bool _compress_metadata = false;
  NVPMAAllocCapturer _main_capturer, _residual_capturer;
  ZeroCompressedPtr _zero_compressed_main_metadata,
                    _zero_compressed_residual_metadata;


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


  // friend class CUDAGraphShadowRef;  // <bojian/DynamicCUDAGraph>
  friend class NVPMAAllocCapturer;


};

} // namespace cuda
} // namespace at
