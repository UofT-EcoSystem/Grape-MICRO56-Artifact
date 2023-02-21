#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

// <bojian/Grape>
#include <dmlc/logging.h>

namespace at {
namespace cuda {

MempoolId_t graph_pool_handle() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uuid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uuid++};
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
  return {0, 0};
#endif
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

CUDAGraph::CUDAGraph(
    // <bojian/Grape>
    const bool postpone_instantiation,
    const bool frugal_launch,
    const bool instantiate_on_device)
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if (defined(CUDA_VERSION) && CUDA_VERSION < 11000) || defined(USE_ROCM)
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
  _postpone_instantiation = postpone_instantiation;
  _frugal_launch = frugal_launch;
  _instantiate_on_device = instantiate_on_device;
}

void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // For now, a CUDAGraph instance only accommodates the default generator on the device that's
  // current when capture begins. If any op in the captured region uses a non-default generator,
  // or a generator on another device, the offending generator will throw an error.
  // These restrictions simplify CUDAGraph, but could be relaxed in the future:
  // in principle, the underlying Cuda calls do permit cross-device ops to be captured.
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());

  auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
  offset_extragraph_ = at::empty({1}, options);

  gen->capture_prologue(offset_extragraph_.data_ptr<int64_t>());

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = c10::cuda::current_device();

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));

  // Stashes the current capture's uuid.
  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  TORCH_INTERNAL_ASSERT(id_ > 0);
  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our mempool_id_.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // When CUDACachingAllocator allocates while a capture is underway, it calls cudaStreamGetCaptureInfo
  // to get the current stream's capture id, if any. Here we tell CUDACachingAllocator: if the stream
  // has a capture id matching this graph's id_, use the private pool mempool_id_ identifies.
  //
  // There's a small chance of a bad allocation here if another thread launches a kernel on
  // capture_stream_ between the call to cudaStreamBeginCapture above and the call to
  // notifyCaptureBegin below.
  // But I don't think we need to worry about it because that use case makes no sense:
  // The user has no business launching kernels on capture_stream_ from another thread
  // while calling capture_begin. They'll have no idea if their side thread's
  // kernel will end up as part of the capture or not.
  c10::cuda::CUDACachingAllocator::notifyCaptureBegin(capture_dev_, id_, mempool_id_);
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

void CUDAGraph::capture_end(const bool no_exception_in_capture) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  c10::cuda::CUDACachingAllocator::notifyCaptureEnd(capture_dev_, id_);

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  // <bojian/Grape>
  if (_postpone_instantiation) {
    // Directly exit if compressing the metadata, since we will be materializing
    // the executors later.
    // LOG(INFO) << "Postponing the instantiation of graph=" << graph_
    //           << " to later stages";
    return;
  }
  // Split the epilog into a different method since we might need to postpone it
  // when doing the compression.

  // <bojian/Grape> Only instantiate if there is no exception during the capture.
  if (no_exception_in_capture) {
    capture_end_epilog(_instantiate_on_device);
  } else {
    LOG(WARNING) << "Not instantiating due to an exception "
                    "that happened during the capture";
  }
}

// <bojian/Grape>
void CUDAGraph::capture_end_epilog(const bool instantiate_on_device) {
  // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
  // who prefer not to report error message through these arguments moving forward
  // (they prefer return value, or errors on api calls internal to the capture)
  // <bojian/Grape>
  // AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  if (instantiate_on_device) {
    // LOG(INFO) << "Instantiating the CUDAGraph on the device side";
    AT_CUDA_CHECK(cudaGraphInstantiate(
        &graph_exec_, graph_, cudaGraphInstantiateFlagDeviceLaunch));
    AT_CUDA_CHECK(cudaGraphUpload(graph_exec_, ::at::cuda::getCurrentCUDAStream()));
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  }
  // </bojian/Grape>
  has_graph_exec_ = true;

  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  TORCH_CHECK(gen == capture_gen_,
              "Default CUDA RNG generator on current device at capture end "
              "is different from default generator on current device "
              "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  // Now that we've instantiated graph_ into graph_exec_,
  // we don't need graph_ anymore.

  // <bojian/Graph> Comment out the destruction of the original graph data
  // structure since it might be useful for debugging purposes
  // AT_CUDA_CHECK(cudaGraphDestroy(graph_));
  // has_graph_ = false;

#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

// <bojian/Grape>
void CUDAGraph::decompress() {
  for (CUDAGraph* const subgraph : subgraphs) {
    subgraph->decompress();
  }
  if (_orig_yang_main_metadata != nullptr) {
    for (size_t residual_id = 0; residual_id < _orig_list_of_residuals.size();
         ++residual_id) {
      gCompressEngine.decompress(
          _compressed_list_of_residuals[residual_id],
          _orig_list_of_residuals[residual_id]);
    }
    gCompressEngine.decompress(
        _compressed_yin_residual_metadata, _orig_yin_residual_metadata);
    gCompressEngine.decompress(
        _compressed_yang_main_metadata, *_orig_yang_main_metadata);
    gCompressEngine.waitForAllWorkitems();
  }
}

void CUDAGraph::replay() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  // <bojian/Grape> Do not manipulate the RNG engine in the case of frugality.
  if (!_frugal_launch) {

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(wholegraph_increment_);
  }
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif

  // <bojian/Grape>
  } // if (!_frugal_launch)
  else {
    AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));
  }
  // Reset the contiguous counter every time a CUDAGraph is executed.
  ::c10::cuda::CUDACachingAllocator::resetAllocationContextContigCnt();
  // </bojian/Grape>
}

void CUDAGraph::reset() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Juptyer notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::notifyCaptureDestroy(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
  }

  // <bojian/Grape> Add the cleanup for the device CUDAGraph.
  // if (device_graph_) {
  //   C10_CUDA_CHECK_WARN(cudaGraphDestroy(device_graph_));
  // }
  // if (device_graph_exec_) {
  //   C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(device_graph_exec_));
  // }

#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 and not yet supported on ROCM");
#endif
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  reset();
}

} // namespace cuda
} // namespace at
