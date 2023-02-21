#include <fstream>
#include <memory>

#include <helper_cuda.h>

#include "CUDAGraph.h"

#include <dmlc/logging.h>
#include <dmlc/parameter.h>

// #define LZW_IMPLEMENTATION
// #include "lzw.hpp"

namespace at {
namespace cuda {

static std::pair<size_t, size_t> _getLargestGraphAndClearCachedAllocSizes(
    std::vector<NVPMAQueryResult_t>& queried_pma_alloc_sizes) {
  if (queried_pma_alloc_sizes.empty()) {
    return std::make_pair<size_t, size_t>(static_cast<size_t>(-1), 0);
  }
  size_t max_pma_alloc_size = queried_pma_alloc_sizes[0].record_entries[0];
  size_t max_pma_alloc_size_id = 0;

  for (size_t pma_alloc_sizes_id = 1;
       pma_alloc_sizes_id < queried_pma_alloc_sizes.size();
       ++pma_alloc_sizes_id) {
    if (queried_pma_alloc_sizes[pma_alloc_sizes_id].record_entries[0] >
        max_pma_alloc_size) {
      max_pma_alloc_size_id = pma_alloc_sizes_id;
    }
  }
  queried_pma_alloc_sizes.clear();
  LOG(INFO) << "Graph #" << max_pma_alloc_size_id
            << " has the largest allocation size of " << max_pma_alloc_size
            << " bytes";
  return std::make_pair(max_pma_alloc_size_id, max_pma_alloc_size);
}

static size_t _cudaGetGPUMemInUse() {
  size_t free_mem_in_bytes, total_mem_in_bytes;
  checkCudaErrors(cudaMemGetInfo(&free_mem_in_bytes, &total_mem_in_bytes));
  return total_mem_in_bytes - free_mem_in_bytes;
}

void instantiateCUDAGraphsOnCompressedMetadata(
    std::vector<std::reference_wrapper<CUDAGraph>>& graphs,
    const bool debug_mode,
    const bool instantiate_on_device,
    const bool compress_residuals) {
  const bool nv_pma_alloc_capturer_old_verbose = NVPMAAllocCapturer::verbose;
  NVPMAAllocCapturer::verbose = debug_mode;

  // 1. Run through the CUDAGraph instantiations and query the allocation size
  //    of each CUDAGraph.
  cudaGraphExec_t tmp_instance_workspace;
  std::vector<NVPMAQueryResult_t> queried_pma_alloc_sizes;
  for (size_t shape_id = 0; shape_id < graphs.size(); ++shape_id) {
    NVPMAQueryResult_t query_result;
    PROBE_NEXT_IMM_MALLOC_FROM_STMT(
        query_result,
        cudaGraphInstantiate(
            &tmp_instance_workspace,
            graphs[shape_id].get().graph_,
            nullptr,
            nullptr,
            0));
    queried_pma_alloc_sizes.emplace_back(std::move(query_result));
  }
  size_t max_pma_alloc_size_id, max_pma_alloc_size;
  std::tie(max_pma_alloc_size_id, max_pma_alloc_size) =
      _getLargestGraphAndClearCachedAllocSizes(queried_pma_alloc_sizes);
  if (max_pma_alloc_size == 0) {
    return;
  }
  std::vector<CompressedRegion_t> compressed_regions;
  /// @todo(bojian/Grape) The current implementation requires to simultaneously
  /// maintain two buffers (Yin and Yang).
  ///
  /// Only the CUDAGraphs instantiated on the Yang buffer can be modified and
  /// compressed, while manipulating those instantiated on the Yin buffer causes
  /// runtime errors. The exact reason is yet unknown.
  void* curr_graph_ptr;
  NVMemoryRegion* curr_buffer_region;

  NVPMAAllocCapturer::setToRecordMallocs();
  // The Yin and Yang buffer region have to be made persistent and passed on to
  // the CUDAGraph object itself. Their sizes are slightly greater than the
  // maximum the reserved allocation size to avoid alignment issues.
  size_t buffer_reserved_size = max_pma_alloc_size + 2 * 1024 * 1024;
  std::shared_ptr<NVMemoryRegion> yin_buffer_region(
      new NVMemoryRegion(buffer_reserved_size)),
      yang_buffer_region(new NVMemoryRegion(buffer_reserved_size));
  NVPMAAllocCapturer::resetToDefault();

  // std::vector<std::shared_ptr<NVMemoryRegion>> inferred_list_of_residuals;
  // bool yin_is_occupied_by_residual = false,
  //      yang_is_occupied_by_residual = false;

  NVMemoryRegion yin_residual_buffer;
  int num_yin_residauls;
  size_t last_nonzero_residual_idx = 0;
  // NVMemoryRegion yin_residual_residual_buffer;

  size_t gpu_memory_in_use_before_instantiation = _cudaGetGPUMemInUse();

  for (size_t graph_id = 0; graph_id < graphs.size() * 2; ++graph_id) {
    size_t shape_id = graph_id / 2;

    LOG(INFO) << "Current graph_id=" << graph_id;
    if (debug_mode) {
      yin_buffer_region->dump(
          "cuda_graph_yin_" + std::to_string(graph_id) + "-before_inst");
      yang_buffer_region->dump(
          "cuda_graph_yang_" + std::to_string(graph_id) + "-before_inst");
    }
    // if (yin_is_occupied_by_residual && graph_id % 2 == 0) {
    //   inferred_list_of_residuals.push_back(yin_buffer_region);
    //   NVPMAAllocCapturer::setToRecordNextAndOverwrite();
    //   yin_buffer_region = std::shared_ptr<NVMemoryRegion>(
    //       new NVMemoryRegion(buffer_reserved_size));
    //   yin_is_occupied_by_residual = false;
    // }
    // if (yang_is_occupied_by_residual && graph_id % 2 != 0) {
    //   inferred_list_of_residuals.push_back(yang_buffer_region);
    //   NVPMAAllocCapturer::setToRecordNextAndOverwrite();
    //   yang_buffer_region = std::shared_ptr<NVMemoryRegion>(
    //       new NVMemoryRegion(buffer_reserved_size));
    //   yang_is_occupied_by_residual = false;
    // }

    // make sure that the memory region is cleared
    curr_graph_ptr =
        graph_id % 2 == 0 ? yin_buffer_region->dptr : yang_buffer_region->dptr;
    checkCudaErrors(cudaMemset(curr_graph_ptr, 0, buffer_reserved_size));
    /// @todo(bojian/Grape) Add support for the residuals.

    if (compress_residuals) {
      if (graph_id % 2 != 0) {
        NVPMAAllocCapturer::setToShadowNextMallocAndAppendResiduals();
      } else {
        NVPMAAllocCapturer::setToShadowNextMallocAndStashResiduals();
      }
    } else {
      NVPMAAllocCapturer::setToShadowNextMalloc();
    }
    if (graph_id % 2 != 0) {
      // 2. Truly instantiate the CUDAGraph only on the Yang buffer.
      graphs[shape_id].get().capture_end_epilog(instantiate_on_device);
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
    checkCudaErrors(cudaStreamSynchronize(getCurrentCUDAStream().stream()));
    NVPMAAllocCapturer::resetToDefault();

    // @todo(bojian/Grape) Temporarily comment out this section since we are not
    // handling for CUDAGraphs that are too small.

    // NVPMAQueryResult_t query_result =
    //     NVPMAAllocCapturer::queryRecordedPMAAllocSizes();

    // while (std::get<0>(query_result) == kReplayNext) {
    //   LOG(INFO) << "Unable to get the memory allocation at graph_id="
    //             << graph_id
    //             << ", probably due to the small graph size. "
    //                "Attempting to instantiate again ...";

    //   if (instantiate_on_device) {
    //     if (graph_id % 2 != 0) {
    //       checkCudaErrors(cudaGraphInstantiate(
    //           &graphs[shape_id].get().graph_exec_,
    //           graphs[shape_id].get().graph_,
    //           cudaGraphInstantiateFlagDeviceLaunch));
    //       checkCudaErrors(cudaGraphUpload(
    //           graphs[shape_id].get().graph_exec_,
    //           ::at::cuda::getCurrentCUDAStream()));
    //       checkCudaErrors(cudaGraphLaunch(
    //           graphs[shape_id].get().graph_exec_, getCurrentCUDAStream()));
    //     } else {
    //       checkCudaErrors(cudaGraphInstantiate(
    //           &tmp_instance_workspace,
    //           graphs[shape_id].get().graph_,
    //           cudaGraphInstantiateFlagDeviceLaunch));
    //       checkCudaErrors(cudaGraphUpload(
    //           tmp_instance_workspace, ::at::cuda::getCurrentCUDAStream()));
    //       checkCudaErrors(
    //           cudaGraphLaunch(tmp_instance_workspace,
    //           getCurrentCUDAStream()));
    //     }
    //   } else {
    //     if (graph_id % 2 != 0) {
    //       checkCudaErrors(cudaGraphInstantiate(
    //           &graphs[shape_id].get().graph_exec_,
    //           graphs[shape_id].get().graph_,
    //           nullptr,
    //           nullptr,
    //           0));
    //       checkCudaErrors(cudaGraphLaunch(
    //           graphs[shape_id].get().graph_exec_, getCurrentCUDAStream()));
    //     } else {
    //       checkCudaErrors(cudaGraphInstantiate(
    //           &tmp_instance_workspace,
    //           graphs[shape_id].get().graph_,
    //           nullptr,
    //           nullptr,
    //           0));
    //       checkCudaErrors(
    //           cudaGraphLaunch(tmp_instance_workspace,
    //           getCurrentCUDAStream()));
    //     }
    //   }
    //   checkCudaErrors(cudaStreamSynchronize(getCurrentCUDAStream().stream()));
    //   query_result = NVPMAAllocCapturer::queryRecordedPMAAllocSizes();
    // } // while (std::get<0>(query_result) == kReplayNext)

    if (debug_mode) {
      yin_buffer_region->dump(
          "cuda_graph_yin_" + std::to_string(graph_id) + "-after_inst");
      yang_buffer_region->dump(
          "cuda_graph_yang_" + std::to_string(graph_id) + "-after_inst");
      // for (size_t residual_id = 0;
      //      residual_id < inferred_list_of_residuals.size();
      //      ++residual_id) {
      //   inferred_list_of_residuals[residual_id]->dump(
      //       "cuda_graph_residual_" + std::to_string(residual_id) + "_" +
      //       std::to_string(graph_id));
      // }
    }

    // NVPMAAllocCapturer::updateResidualsSnapshot();
    std::vector<NVMemoryRegion> inferred_list_of_residuals;
    NVPMAQueryResult_t pma_query_result =
        NVPMAAllocCapturer::queryRecordedPMAAllocSizes();

    if (compress_residuals) {
      if (graph_id % 2 != 0) {
        LOG(INFO) << "Materializing " << pma_query_result.current_residual_idx
                  << " residuals";
        NVPMAAllocCapturer::setToShadowResiduals();

        // 1. Initialize the first pma_query_result.current_residual_idx
        //    residuals, while dropping the preceding (num_yin_residauls - 2)
        //    ones.
        int yin_residual_idx_upper_bound = num_yin_residauls - 2;
        for (int residual_id = 0; residual_id < yin_residual_idx_upper_bound;
             ++residual_id) {
          void* dptr;
          checkCudaErrors(cudaMalloc(&dptr, 2 * 1024 * 1024));
        }
        int residual_id = yin_residual_idx_upper_bound <= 0
            ? 0
            : yin_residual_idx_upper_bound;
        for (; residual_id < pma_query_result.current_residual_idx;
             ++residual_id) {
          inferred_list_of_residuals.emplace_back(2 * 1024 * 1024);
        }

        // 2. In the case when the current residual index is 0, ask for the last
        //    1-2 residauls from the last non-empty residual stack.
        if (pma_query_result.current_residual_idx == 0 &&
            last_nonzero_residual_idx != 0) {
          if (last_nonzero_residual_idx == 1) {
            inferred_list_of_residuals.emplace_back(2 * 1024 * 1024);
          } else {
            for (size_t residual_id = 0;
                 residual_id < last_nonzero_residual_idx - 2;
                 ++residual_id) {
              void* dptr;
              checkCudaErrors(cudaMalloc(&dptr, 2 * 1024 * 1024));
            }
            inferred_list_of_residuals.emplace_back(2 * 1024 * 1024);
            inferred_list_of_residuals.emplace_back(2 * 1024 * 1024);
          }
        } // if (pma_query_result.current_residual_idx == 0 &&
          //     last_nonzero_residual_idx != 0)

        // 3. Same
        if (num_yin_residauls == 0 &&
            last_nonzero_residual_idx > pma_query_result.current_residual_idx) {
          if (last_nonzero_residual_idx -
                  pma_query_result.current_residual_idx ==
              1) {
            inferred_list_of_residuals.emplace_back(2 * 1024 * 1024);
          } else {
            for (size_t residual_id = pma_query_result.current_residual_idx;
                 residual_id < last_nonzero_residual_idx - 2;
                 ++residual_id) {
              void* dptr;
              checkCudaErrors(cudaMalloc(&dptr, 2 * 1024 * 1024));
            }
            inferred_list_of_residuals.emplace_back(2 * 1024 * 1024);
            inferred_list_of_residuals.emplace_back(2 * 1024 * 1024);
          }
        } // if (pma_query_result.current_residual_idx == 1 &&
          //     num_yin_residauls == 0 && last_nonzero_residual_idx > 1)

        NVPMAAllocCapturer::resetToDefault();
      }

      if (pma_query_result.current_residual_idx != 0) {
        last_nonzero_residual_idx = pma_query_result.current_residual_idx;
      }

      if (debug_mode) {
        LOG(INFO) << "Current number of residuals: "
                  << pma_query_result.current_residual_idx;
        NVPMAAllocCapturer::setToShadowResiduals();
        size_t residual_idx_upper_bound = graph_id % 2 == 0
            ? pma_query_result.current_residual_idx
            : pma_query_result.current_residual_capacity;
        for (size_t residual_id = 0; residual_id < residual_idx_upper_bound;
             ++residual_id) {
          NVMemoryRegion memory_region(2 * 1024 * 1024);
          memory_region.dump(
              "cuda_graph_residual_" + std::to_string(residual_id) + "_" +
              std::to_string(graph_id));
        }
        NVPMAAllocCapturer::resetToDefault();
      }
    } // if (compress_residuals)

    // if (debug_mode) {
    //   if (graph_id == 0) {
    //     yin_buffer_region->dump("cuda_graph_sample");
    //   }
    //   if (graph_id % 2 != 0) {
    //     std::string cuda_graph_metadata_dump_filename =
    //         "cuda_graph_shape" + std::to_string(shape_id);
    //     yang_buffer_region->dump(cuda_graph_metadata_dump_filename);
    //   }
    // } // if (debug_mode)

    if (graph_id % 2 != 0) {
      CompressedRegion_t
          compressed_yin_residual_region =
              gCompressEngine.template compress<CompressedRegion_t>(
                  yin_residual_buffer),
          compressed_yang_main_region =
              gCompressEngine.template compress<CompressedRegion_t>(
                  *yang_buffer_region);
      std::vector<CompressedRegion_t> compressed_list_of_residuals;

      for (size_t residual_id = 0;
           residual_id < inferred_list_of_residuals.size();
           ++residual_id) {
        compressed_list_of_residuals.push_back(
            gCompressEngine.template compress<CompressedRegion_t>(
                inferred_list_of_residuals[residual_id]));
      }
      // if (compressed_yin_residual_region.size == 0) {
      //   LOG(WARNING) << "Compressed size of Yin="
      //                << compressed_yin_residual_region.size
      //                << " has zeroes, look at graph_id=" << graph_id
      //                << ", shape_id=" << shape_id;
      //   yin_residual_buffer.dptr = yin_buffer_region->dptr;
      //   yin_residual_buffer.size = yin_buffer_region->size;
      //   compressed_yin_residual_region =
      //       gCompressEngine.template compress<CompressedRegion_t>(
      //           yin_residual_buffer);
      // }
      if (compressed_yang_main_region.size == 0
          // || graph_id == 1971
      ) {
        LOG(FATAL) << "Compressed size of Yang="
                   << compressed_yang_main_region.size
                   << " has zeroes, look at graph_id=" << graph_id
                   << ", shape_id=" << shape_id;
        // yang_is_occupied_by_residual = true;
      }
      graphs[shape_id].get()._orig_yin_main_metadata = yin_buffer_region;
      graphs[shape_id].get()._orig_yang_main_metadata = yang_buffer_region;
      graphs[shape_id].get()._orig_yin_residual_metadata =
          std::move(yin_residual_buffer);
      graphs[shape_id].get()._orig_list_of_residuals =
          std::move(inferred_list_of_residuals);
      graphs[shape_id].get()._compressed_yin_residual_metadata =
          std::move(compressed_yin_residual_region);
      graphs[shape_id].get()._compressed_yang_main_metadata =
          std::move(compressed_yang_main_region);
      graphs[shape_id].get()._compressed_list_of_residuals =
          std::move(compressed_list_of_residuals);
    } else { // graph_id % 2 == 0
      size_t zero_compressed_yin_buffer_size =
          yin_buffer_region->getZeroCompressedUInt32Size() * sizeof(uint32_t);
      num_yin_residauls = pma_query_result.current_residual_idx;
      // size_t zero_compressed_yin_residual_buffer_size =
      //     inferred_list_of_residuals[inferred_list_of_residuals.size() - 1]
      //         .getZeroCompressedUInt32Size() *
      //     sizeof(uint32_t);
      if (debug_mode) {
        LOG(INFO) << "Zero-compressed Yin buffer size="
                  << zero_compressed_yin_buffer_size
                  << ", num_yin_residauls=" << num_yin_residauls;
      }
      if (zero_compressed_yin_buffer_size == 0
          // || graph_id == 1014
          // || graph_id == 414
      ) {
        LOG(FATAL) << "Compressed size of Yin="
                   << zero_compressed_yin_buffer_size
                   << " has zeroes, look at graph_id=" << graph_id
                   << ", shape_id=" << shape_id;
        // yin_is_occupied_by_residual = true;
      }
      yin_residual_buffer.dptr = static_cast<char*>(yin_buffer_region->dptr) +
          zero_compressed_yin_buffer_size;
      yin_residual_buffer.size =
          yin_buffer_region->size - zero_compressed_yin_buffer_size;
    } // graph_id % 2 != 0
    size_t gpu_memory_used_by_instantiation =
        _cudaGetGPUMemInUse() - gpu_memory_in_use_before_instantiation;
    LOG(INFO) << "Total GPU consumption="
              << (gpu_memory_used_by_instantiation * 1.0 / (1024 * 1024))
              << " (MB), "
              << (gpu_memory_used_by_instantiation * 1.0 /
                  ((shape_id + 1) * 1024 * 1024))
              << " (MB) per graph";
  } // for (grape_id in [0, graphs.size() * 2))

  size_t gpu_memory_used_by_instantiation =
      _cudaGetGPUMemInUse() - gpu_memory_in_use_before_instantiation;
  double gpu_memory_per_graph =
      (gpu_memory_used_by_instantiation * 1.0 / (graphs.size() * 1024 * 1024));
  LOG(INFO) << "Total GPU consumption="
            << (gpu_memory_used_by_instantiation * 1.0 / (1024 * 1024))
            << " (MB), " << gpu_memory_per_graph << " (MB) per graph";

  if (dmlc::GetEnv("LOG_COMPRESSION_RESULTS_TO_CSV", 0)) {
    std::ofstream fout(
        dmlc::GetEnv(
            "METADATA_COMPRESSION_CSV_FILENAME",
            std::string("metadata_compression.csv")),
        std::ios_base::app);
    fout << dmlc::GetEnv("MODEL", std::string("")) << ","
         << buffer_reserved_size << "," << gpu_memory_per_graph << "\n";
  }

  if (debug_mode) {
    LOG(INFO) << "Sanity checking on the compressed data";
    for (size_t shape_id = 0; shape_id < graphs.size(); ++shape_id) {
      LOG(INFO) << "Executing graph #" << shape_id << " ...";
      graphs[shape_id].get().decompress();
      graphs[shape_id].get().replay();
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }
  NVPMAAllocCapturer::verbose = nv_pma_alloc_capturer_old_verbose;
}

} // namespace cuda
} // namespace at
