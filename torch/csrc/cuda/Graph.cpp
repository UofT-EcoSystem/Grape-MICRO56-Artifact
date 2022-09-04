#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>

// <bojian/DynamicCUDAGraph>
#include <ATen/cuda/NVPMAAllocCapturer.h>
#include <ATen/cuda/CUDAGlobalExecMask.h>
#include <ATen/cuda/CUDAGlobalExecMask.cuh>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>

#include "Stream.h"

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer (csrc/Module.cpp)
// I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// <bojian/DynamicCUDAGraph>
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

class C10_HIDDEN TensorArgsCopier : public TensorVisitor {
 private:
  py::handle _src_py_obj;

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
          if (src_tensor.data_ptr() != dst_tensor.data_ptr()) {
            dst_tensor.copy_(src_tensor, true);
          }
        }),
        _src_py_obj(src_py_obj) {}
};

static std::vector<::at::Tensor> flatten_tensor_args(py::object tensor_args) {
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

static void copy_tensor_args(
    py::object runtime_tensor_args,
    py::object graph_placeholders) {
  // Check whether there is any internal dependency in between the tensor
  // arguments.
  if (dmlc::GetEnv("CUDA_GRAPH_ANTI_ALIASING", false)) {
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
    // TensorArgsFlattener runtime_args_flattener, placeholder_flattener;
    // runtime_args_flattener(runtime_tensor_args);
    // placeholder_flattener(graph_placeholders);

    // for (size_t i = 0; i < runtime_args_flattener.tensor_args.size(); ++i) {
    //   if (runtime_args_flattener.tensor_args[i].data_ptr() !=
    //       placeholder_flattener.tensor_args[i].data_ptr()) {
    //     LOG(INFO) << "Runtime arg_id=" << i << " has not been aligned"
    //     placeholder_flattener.tensor_args[i].copy_(
    //         runtime_args_flattener.tensor_args[i], true);
    //   }
    // }
  }
  ::at::cuda::device_synchronize();
}

// </bojian/DynamicCUDAGraph>


void THCPGraph_init(PyObject *module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m
      .def("_graph_pool_handle",
           &::at::cuda::graph_pool_handle);

  torch_C_m.def("_flatten_tensor_args", &flatten_tensor_args);
  torch_C_m.def("_copy_tensor_args", &copy_tensor_args);

  torch_C_m.def("_notifyMempoolBegin", &::c10::cuda::CUDACachingAllocator::notifyMempoolBegin);
  torch_C_m.def("_notifyMempoolEnd",&::c10::cuda::CUDACachingAllocator::notifyMempoolEnd);

  torch_C_m.def("_notifyMemtapeBegin", &::c10::cuda::CUDACachingAllocator::notifyMemtapeBegin);
  torch_C_m.def("_notifyMemtapeEnd",&::c10::cuda::CUDACachingAllocator::notifyMemtapeEnd);

  // torch_C_m.def("_notifyModuleArgsGeneratorBegin", &::c10::cuda::CUDACachingAllocator::notifyModuleArgsGeneratorBegin);
  // torch_C_m.def("_notifyModuleArgsGeneratorEnd",&::c10::cuda::CUDACachingAllocator::notifyModuleArgsGeneratorEnd);

  shared_ptr_class_<::at::cuda::CUDAGraph>
      (torch_C_m,
       "_CUDAGraph")

      // <bojian/DynamicCUDAGraph>
      // .def(py::init<>())
      .def(py::init<const bool>())


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


      // <bojian/DynamicCUDAGraph>
      .def("decompress",
           &::at::cuda::CUDAGraph::decompress,
           py::call_guard<py::gil_scoped_release>())


      .def("reset",
           &::at::cuda::CUDAGraph::reset,
           py::call_guard<py::gil_scoped_release>())
      .def("pool",
           &::at::cuda::CUDAGraph::pool,
           py::call_guard<py::gil_scoped_release>());


  // <bojian/DynamicCUDAGraph>
  shared_ptr_class_<::at::cuda::CUDAGlobalExecMask>
      (torch_C_m,
       "_CUDAGlobalExecMask")
      .def(py::init<>());

  // shared_ptr_class_<::at::cuda::CUDAGraphShadowRef>
  //     (torch_C_m,
  //      "_CUDAGraphShadowRef")
  //     .def(py::init<const ::at::cuda::CUDAGraph&>());

  // shared_ptr_class_<::at::cuda::NVPMAAllocCapturer>
  //     (torch_C_m,
  //      "_NVPMAAllocCapturer")
  //     .def(py::init<>())
  //     .def("materializeCUDAGraphs",
  //          &::at::cuda::NVPMAAllocCapturer::materializeCUDAGraphs,
  //          py::call_guard<py::gil_scoped_release>(),
  //          py::arg("graphs"));
  torch_C_m.def("MaterializeCUDAGraphs",
                &::at::cuda::NVPMAAllocCapturer::MaterializeCUDAGraphs);

  torch_C_m.def("EnterGlobalExecMask", &::at::cuda::EnterGlobalExecMask);
  torch_C_m.def("ExitGlobalExecMask", &::at::cuda::ExitGlobalExecMask);
  // torch_C_m.def("EnterModuleArgsGeneratorStreamContext", &EnterModuleArgsGeneratorStreamContext);
  // torch_C_m.def("ExitModuleArgsGeneratorStreamContext", &ExitModuleArgsGeneratorStreamContext);

  // </bojian/DynamicCUDAGraph>

}
