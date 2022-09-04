#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense_native.h>
#endif

#include <ATen/cuda/CUDAContext.h>


// <bojian/DynamicCUDAGraph> BOOKMARK
// #include <ATen/core/Formatting.h>

// #include <dmlc/parameter.h>
// #include <dmlc/logging.h>


namespace at {
namespace native {

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
    kComplexHalf, kHalf, kBool, kBFloat16, self.scalar_type(), "_local_scalar_dense_cuda", [&] {
        scalar_t value;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();


        // <bojian/DynamicCUDAGraph>
        // if (dmlc::GetEnv("BACKTRACE_VECTORIZED_ELEMENTWISE_KERNEL", false)) {
        //   LOG(INFO) << "Checkpoint 1";
        //   LOG(INFO) << dmlc::StackTrace(1, 25);
        // }


        at::cuda::memcpy_and_sync(&value, self.data_ptr<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost, stream);


        // <bojian/DynamicCUDAGraph>
        // if (dmlc::GetEnv("BACKTRACE_VECTORIZED_ELEMENTWISE_KERNEL", false)) {
        //   LOG(INFO) << "Copied value=" << value;
        //   LOG(INFO) << dmlc::StackTrace(1, 25);
        // }


        r = Scalar(value);
      });


  // <bojian/DynamicCUDAGraph>
  // if (dmlc::GetEnv("BACKTRACE_VECTORIZED_ELEMENTWISE_KERNEL", false)) {
  //   LOG(INFO) << "self=" << self << " into scalar=" << r;
  // }


  return r;
}

}} // at::native
