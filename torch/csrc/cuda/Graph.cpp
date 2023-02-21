// clang-format off
#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>

// <bojian/Grape>
#include <ATen/cuda/NVPMAAllocCapturer.h>
#include <ATen/cuda/CUDAGlobalIndicator.h>
#include <ATen/cuda/CUDAGlobalIndicator.cuh>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer (csrc/Module.cpp)
// I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// <bojian/Grape>
// clang-format on

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

C10_CUDA_API void notifyMempoolBegin(
    const int device,
    const MempoolId_t& mempool_id,
    const int enter_cnt);
C10_CUDA_API void notifyMempoolEnd(
    const int device,
    const MempoolId_t& mempool_id,
    const int enter_cnt,
    const bool force_retire_all_active_blocks);
C10_CUDA_API void retireOutputDataPtrs(const std::vector<size_t>& output_dptrs);

C10_CUDA_API void notifyPrivatePoolBegin(
    const int device,
    const MempoolId_t& mempool_id);
C10_CUDA_API void notifyPrivatePoolEnd(const int device);

C10_CUDA_API void notifyMemtapeBegin(
    const int device,
    const MempoolId_t& mempool_id,
    const int entry_cnt);
C10_CUDA_API void notifyMemtapeEnd(
    const int device,
    const MempoolId_t& mempool_id,
    const int entry_cnt);
C10_CUDA_API size_t
getCurrentMemtapePos(const int device, const MempoolId_t& mempool_id);
C10_CUDA_API size_t getCurrentMemtapeInReplayPos(const int device);
C10_CUDA_API void setCurrentMemtapePos(
    const int device,
    const size_t num_of_mallocs_requested);

C10_CUDA_API void lookupDataPtrInMemtape(
    const int device,
    const MempoolId_t& mempool_id,
    const size_t data_ptr);

C10_CUDA_API void notifyCUDAGraphPlaceholdersBegin(const int device);
C10_CUDA_API void notifyCUDAGraphPlaceholdersEnd(const int device);

C10_CUDA_API void enableAliasPredictionWithVerbosity(const bool verbose);
C10_CUDA_API void disableAliasPrediction();

C10_CUDA_API void linkAllocationCtxToCUDAGraphPlaceholder(
    const AllocationContext& alloc_ctx,
    const CUDAGraphPlaceholderId& cuda_graph_placeholder_id);

/// @brief Check whether the CUDAGraph placeholder has been preemptively
/// allocated.
/// @return
C10_CUDA_API void checkPreemptiveAllocation(
    const CUDAGraphPlaceholderId& cuda_graph_placeholder_id);

/// @brief Clear the preemptive allocation status.
/// @return
C10_CUDA_API void clearPreemptiveAllocation(
    const CUDAGraphPlaceholderId& cuda_graph_placeholder_id);

/// @brief Force aliasing
/// @return
C10_CUDA_API void fwdToCUDAGraphPlaceholdersBegin(
    const std::vector<size_t>& placeholder_dptrs);

C10_CUDA_API void fwdToCUDAGraphPlaceholdersEnd();

/// @brief Track the workspace allocations.
/// @return
C10_CUDA_API void workspaceSizeTrackerBegin(const int tracker_mode);

C10_CUDA_API void workspaceSizeTrackerEnd();

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10

class TensorVisitor {
 private:
  std::function<void(::at::Tensor)> _fvisit;

  virtual void VisitListItem(
      const py::size_t idx,
      const py::handle& py_list_item) {
    operator()(py_list_item);
  }
  virtual void VisitTupleItem(
      const py::size_t idx,
      const py::handle& py_tuple_item) {
    operator()(py_tuple_item);
  }
  virtual void VisitDictItem(
      const py::handle& py_dict_kitem,
      const py::handle& py_dict_vitem) {
    operator()(py_dict_vitem);
  }

 public:
  TensorVisitor(std::function<void(::at::Tensor)> fvisit) : _fvisit(fvisit) {}

  void operator()(const py::handle& py_obj) {
    if (py::isinstance<py::list>(py_obj)) {
      py::list py_list_obj = py::cast<py::list>(py_obj);
      for (py::size_t i = 0; i < py_list_obj.size(); ++i) {
        VisitListItem(i, py_list_obj[i]);
      }
    } else if (py::isinstance<py::tuple>(py_obj)) {
      py::tuple py_tuple_obj = py::cast<py::tuple>(py_obj);
      for (py::size_t i = 0; i < py_tuple_obj.size(); ++i) {
        VisitTupleItem(i, py_tuple_obj[i]);
      }
    } else if (py::isinstance<py::dict>(py_obj)) {
      py::dict py_dict_obj = py::cast<py::dict>(py_obj);
      for (const std::pair<py::handle, py::handle>& py_dict_kv_item :
           py_dict_obj) {
        VisitDictItem(py_dict_kv_item.first, py_dict_kv_item.second);
      }
    } else {
      torch::jit::InferredType arg_type_info =
          torch::jit::tryToInferType(py_obj);
      if (!arg_type_info.success()) {
        return;
      }
      if (arg_type_info.type()->isSubtypeOf(*::c10::TensorType::get())) {
        _fvisit(torch::jit::toIValue(py_obj, arg_type_info.type()).toTensor());
      }
    }
  }
};

struct TensorArgsFlattener : public TensorVisitor {
  std::vector<::at::Tensor> tensor_args;

  TensorArgsFlattener()
      : TensorVisitor([this](::at::Tensor tensor) {
          this->tensor_args.push_back(tensor);
        }) {}
};

std::ostream& operator<<(std::ostream& out, const at::DataPtr& dptr) {
  out << "DataPtr{.alloc_ctx=" << dptr.alloc_ctx;
  if (dptr.is_cuda_graph_placeholder) {
    out << ", .cuda_graph_placeholder_id=" << dptr.cuda_graph_placeholder_id;
  }
  out << "}";
  return out;
}

class C10_HIDDEN TensorArgsCopier : public TensorVisitor {
 private:
  py::handle _src_py_obj;
  int _tensor_id = 0;

  virtual void VisitListItem(
      const py::size_t idx,
      const py::handle& py_list_item) final {
    py::handle src_py_obj_copy = _src_py_obj;

    TORCH_CHECK(py::isinstance<py::list>(_src_py_obj), _src_py_obj);
    py::list src_py_list = py::cast<py::list>(_src_py_obj);
    _src_py_obj = src_py_list[idx];
    operator()(py_list_item);

    _src_py_obj = src_py_obj_copy;
  }
  virtual void VisitTupleItem(
      const py::size_t idx,
      const py::handle& py_tuple_item) final {
    py::handle src_py_obj_copy = _src_py_obj;

    TORCH_CHECK(py::isinstance<py::tuple>(_src_py_obj), _src_py_obj);
    py::tuple src_py_tuple = py::cast<py::tuple>(_src_py_obj);
    _src_py_obj = src_py_tuple[idx];
    operator()(py_tuple_item);

    _src_py_obj = src_py_obj_copy;
  }
  virtual void VisitDictItem(
      const py::handle& py_dict_kitem,
      const py::handle& py_dict_vitem) final {
    py::handle src_py_obj_copy = _src_py_obj;

    TORCH_CHECK(py::isinstance<py::dict>(_src_py_obj), _src_py_obj);
    py::tuple src_py_dict = py::cast<py::dict>(_src_py_obj);
    _src_py_obj = src_py_dict[py_dict_kitem];
    operator()(py_dict_vitem);

    _src_py_obj = src_py_obj_copy;
  }

 public:
  explicit TensorArgsCopier(const py::object& src_py_obj)
      : TensorVisitor([this](at::Tensor dst_tensor) {
          torch::jit::InferredType arg_type_info =
              torch::jit::tryToInferType(_src_py_obj);
          TORCH_CHECK(
              arg_type_info.success() &&
              arg_type_info.type()->isSubtypeOf(*::c10::TensorType::get()));
          at::Tensor src_tensor =
              torch::jit::toIValue(_src_py_obj, arg_type_info.type())
                  .toTensor();

          using namespace ::c10::cuda::CUDACachingAllocator;
          if (src_tensor.data_ptr() != dst_tensor.data_ptr()) {
            const at::DataPtr
                &src_dptr =
                    src_tensor.unsafeGetTensorImpl()->storage_.data_ptr(),
                &dst_dptr =
                    dst_tensor.unsafeGetTensorImpl()->storage_.data_ptr();
            if (dst_dptr.is_cuda_graph_placeholder) {
              // Do not perform alias prediction when the source tensor is a view
              if (!src_tensor.is_view()) {
                checkPreemptiveAllocation(dst_dptr.cuda_graph_placeholder_id);
                linkAllocationCtxToCUDAGraphPlaceholder(
                    src_dptr.alloc_ctx, dst_dptr.cuda_graph_placeholder_id);
              }
            }
            dst_tensor.copy_(src_tensor, true);
          } else { // if (src_tensor.data_ptr() == dst_tensor.data_ptr())
            const at::DataPtr& dst_dptr =
                dst_tensor.unsafeGetTensorImpl()->storage_.data_ptr();
            if (dst_dptr.is_cuda_graph_placeholder) {
              clearPreemptiveAllocation(dst_tensor.unsafeGetTensorImpl()
                                            ->storage_.data_ptr()
                                            .cuda_graph_placeholder_id);
            }
          } // if (src_tensor.data_ptr() != dst_tensor.data_ptr())
          ++_tensor_id;
        }),
        _src_py_obj(src_py_obj) {}
};

static std::vector<::at::Tensor> flattenTensorArgs(py::object tensor_args) {
  TensorArgsFlattener flattener;
  flattener(tensor_args);
  return flattener.tensor_args;
}

struct DependencyGraphNode {
  size_t id;
  std::vector<std::shared_ptr<DependencyGraphNode>> edges_from_this;
  size_t num_edges_to_this = 0;

  explicit DependencyGraphNode(const size_t id) : id(id) {}
};

using DependencyGraphNodePtr = std::shared_ptr<DependencyGraphNode>;

static void copyTensorArgs(
    py::object graph_placeholders,
    py::object runtime_tensor_args,
    const bool anti_aliasing) {
  // Check whether there is any internal dependency in between the tensor
  // arguments.
  if (anti_aliasing) {
    TensorArgsFlattener runtime_args_flattener, placeholder_flattener;
    runtime_args_flattener(runtime_tensor_args);
    placeholder_flattener(graph_placeholders);

    // 1. Construct a dependency graph.
    std::unordered_map<size_t, DependencyGraphNodePtr> dependency_graph;
    dependency_graph.reserve(runtime_args_flattener.tensor_args.size());

    for (size_t arg_id = 0; arg_id < runtime_args_flattener.tensor_args.size();
         ++arg_id) {
      dependency_graph.emplace(
          arg_id, std::make_shared<DependencyGraphNode>(arg_id));
    }

    // 2. Initialize dependency edges
    for (size_t runtime_arg_id = 0;
         runtime_arg_id < runtime_args_flattener.tensor_args.size();
         ++runtime_arg_id) {
      for (size_t placeholder_id = 0;
           placeholder_id < placeholder_flattener.tensor_args.size();
           ++placeholder_id) {
        if (runtime_arg_id == placeholder_id) {
          continue;
        }
        if (runtime_args_flattener.tensor_args[runtime_arg_id].data_ptr() ==
            placeholder_flattener.tensor_args[placeholder_id].data_ptr()) {
          dependency_graph.at(placeholder_id)
              .get()
              ->edges_from_this.emplace_back(
                  dependency_graph.at(runtime_arg_id));
          dependency_graph.at(runtime_arg_id).get()->num_edges_to_this += 1;
        }
      }
    }

    // 3. Obtain a topological order of the dependency graph. Raise an error in
    // the case when there is a cycle.
    std::queue<DependencyGraphNodePtr> worklist;
    std::vector<DependencyGraphNodePtr> topo_sorted_order;
    topo_sorted_order.reserve(runtime_args_flattener.tensor_args.size());

    for (size_t runtime_arg_id = 0; runtime_arg_id < dependency_graph.size();
         ++runtime_arg_id) {
      DependencyGraphNodePtr& node = dependency_graph.at(runtime_arg_id);
      if (node.get()->num_edges_to_this == 0) {
        worklist.push(node);
      }
    }
    while (!worklist.empty()) {
      DependencyGraphNodePtr& workitem = worklist.front();
      topo_sorted_order.push_back(workitem);
      for (DependencyGraphNodePtr& neighbor : workitem->edges_from_this) {
        neighbor->num_edges_to_this -= 1;
        if (neighbor->num_edges_to_this == 0) {
          worklist.push(neighbor);
        }
      }
      worklist.pop();
    }

    if (topo_sorted_order.size() != dependency_graph.size()) {
      LOG(FATAL) << "There exists a cycle in the dependency graph";
    }
    for (const DependencyGraphNodePtr& node : topo_sorted_order) {
      placeholder_flattener.tensor_args[node->id].copy_(
          runtime_args_flattener.tensor_args[node->id], true);
    }
  } else {
    TensorArgsCopier copier(runtime_tensor_args);
    copier(graph_placeholders);
  }
  ::at::cuda::device_synchronize();
}

void ChangeRLECompressionPageSize(const size_t new_page_size) {
  LOG(INFO) << "Changing the RLE compression page size to " << new_page_size;
  ::at::cuda::RLECompressedRegion::page_size = new_page_size;
}
// clang-format off
// </bojian/Grape>

void THCPGraph_init(PyObject *module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m
      .def("_graph_pool_handle",
           &::at::cuda::graph_pool_handle);

  shared_ptr_class_<::at::cuda::CUDAGraph>
      (torch_C_m,
       "_CUDAGraph")

      // <bojian/Grape> Add an extra indicator to indicate whether the metadata
      // region is to be compressed or not.
      // .def(py::init<>())
      .def(py::init<const bool, const bool, const bool>())

      // <bojian/Grape>
      // clang-format on
      .def_readwrite(
          "subgraphs",
          &::at::cuda::CUDAGraph::subgraphs,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "addSubgraph",
          &::at::cuda::CUDAGraph::addSubgraph,
          py::call_guard<py::gil_scoped_release>())
      // clang-format off

      // I'm not sure this is the correct order of all the arguments. Pybind11 docs
      // aren't clear. But it works.
      .def("capture_begin",
           &::at::cuda::CUDAGraph::capture_begin,
           py::call_guard<py::gil_scoped_release>(),
           py::arg("pool") = c10::cuda::MempoolId_t{0, 0})
      .def("capture_end",
           &::at::cuda::CUDAGraph::capture_end,
           py::call_guard<py::gil_scoped_release>())
      .def("replay",
           &::at::cuda::CUDAGraph::replay,
           py::call_guard<py::gil_scoped_release>())

      // <bojian/Grape>
      .def("decompress",
           &::at::cuda::CUDAGraph::decompress,
           py::call_guard<py::gil_scoped_release>())

      .def("reset",
           &::at::cuda::CUDAGraph::reset,
           py::call_guard<py::gil_scoped_release>())
      .def("pool",
           &::at::cuda::CUDAGraph::pool,
           py::call_guard<py::gil_scoped_release>());

  // <bojian/Grape>
  // clang-format on
  // Tensor Arguments
  torch_C_m.def("flattenTensorArgs", &flattenTensorArgs);
  torch_C_m.def("copyTensorArgs", &copyTensorArgs);

  torch_C_m.def(
      "_notifyPrivatePoolBegin",
      &::c10::cuda::CUDACachingAllocator::notifyPrivatePoolBegin);
  torch_C_m.def(
      "_notifyPrivatePoolEnd",
      &::c10::cuda::CUDACachingAllocator::notifyPrivatePoolEnd);

  // Memory Pool
  torch_C_m.def(
      "_notifyMempoolBegin",
      &::c10::cuda::CUDACachingAllocator::notifyMempoolBegin);
  torch_C_m.def(
      "_notifyMempoolEnd",
      &::c10::cuda::CUDACachingAllocator::notifyMempoolEnd);
  torch_C_m.def(
      "_retireOutputDataPtrs",
      &::c10::cuda::CUDACachingAllocator::retireOutputDataPtrs);

  torch_C_m.def(
      "_notifyMemtapeBegin",
      &::c10::cuda::CUDACachingAllocator::notifyMemtapeBegin);
  torch_C_m.def(
      "_notifyMemtapeEnd",
      &::c10::cuda::CUDACachingAllocator::notifyMemtapeEnd);
  torch_C_m.def(
      "_getCurrentMemtapePos",
      &::c10::cuda::CUDACachingAllocator::getCurrentMemtapePos);
  torch_C_m.def(
      "_getCurrentMemtapeInReplayPos",
      &::c10::cuda::CUDACachingAllocator::getCurrentMemtapeInReplayPos);
  torch_C_m.def(
      "_setCurrentMemtapePos",
      &::c10::cuda::CUDACachingAllocator::setCurrentMemtapePos);
  torch_C_m.def(
      "_lookupDataPtrInMemtape",
      &::c10::cuda::CUDACachingAllocator::lookupDataPtrInMemtape);

  torch_C_m.def(
      "_notifyCUDAGraphPlaceholdersBegin",
      &::c10::cuda::CUDACachingAllocator::notifyCUDAGraphPlaceholdersBegin);
  torch_C_m.def(
      "_notifyCUDAGraphPlaceholdersEnd",
      &::c10::cuda::CUDACachingAllocator::notifyCUDAGraphPlaceholdersEnd);

  torch_C_m.def(
      "enableAliasPredictionWithVerbosity",
      &::c10::cuda::CUDACachingAllocator::enableAliasPredictionWithVerbosity);
  torch_C_m.def(
      "disableAliasPrediction",
      &::c10::cuda::CUDACachingAllocator::disableAliasPrediction);

  torch_C_m.def(
      "fwdToCUDAGraphPlaceholdersBegin",
      &::c10::cuda::CUDACachingAllocator::fwdToCUDAGraphPlaceholdersBegin);
  torch_C_m.def(
      "fwdToCUDAGraphPlaceholdersEnd",
      &::c10::cuda::CUDACachingAllocator::fwdToCUDAGraphPlaceholdersEnd);

  torch_C_m.def(
      "_workspaceSizeTrackerBegin",
      &::c10::cuda::CUDACachingAllocator::workspaceSizeTrackerBegin);
  torch_C_m.def(
      "_workspaceSizeTrackerEnd",
      &::c10::cuda::CUDACachingAllocator::workspaceSizeTrackerEnd);

  // Instantiating a list of CUDAGraph's, with compression being applied to each
  // of their metadata.
  torch_C_m.def(
      "_instantiateCUDAGraphsOnCompressedMetadata",
      &::at::cuda::instantiateCUDAGraphsOnCompressedMetadata);

  torch_C_m.def(
      "_setNVPMAAllocCapturerVerbosity",
      &::at::cuda::NVPMAAllocCapturer::setVerbosity);

  // Conditional CUDAGraphs
  shared_ptr_class_<::at::cuda::CUDAGlobalIndicator>(
      torch_C_m, "_CUDAGlobalIndicator")
      .def(py::init<const bool>());

  torch_C_m.def(
      "_instantiateCUDAGraphOnDeviceV2",
      &::at::cuda::instantiateCUDAGraphOnDeviceV2);
  torch_C_m.def("_embedDeviceCUDAGraph", &::at::cuda::embedDeviceCUDAGraph);

  torch_C_m.def("_forceMemcpy", &::at::cuda::forceMemcpy);
  torch_C_m.def("_forceMemset", &::at::cuda::forceMemset);

  torch_C_m.def(
      "_EnterCUDAGlobalIndicatorScope",
      &::at::cuda::EnterCUDAGlobalIndicatorScope);
  torch_C_m.def(
      "_ExitCUDAGlobalIndicatorScope",
      &::at::cuda::ExitCUDAGlobalIndicatorScope);
  torch_C_m.def(
      "_EnterConstTrueCUDAGlobalIndicatorScope",
      &::at::cuda::EnterConstTrueCUDAGlobalIndicatorScope);
  torch_C_m.def(
      "_ExitConstTrueCUDAGlobalIndicatorScope",
      &::at::cuda::ExitConstTrueCUDAGlobalIndicatorScope);
  torch_C_m.def(
      "BeamHypotheses_copyDataPtr", &::at::cuda::BeamHypotheses_copyDataPtr);
  torch_C_m.def("ChangeRLECompressionPageSize", &ChangeRLECompressionPageSize);
  // clang-format off
  // </bojian/Grape>
}
