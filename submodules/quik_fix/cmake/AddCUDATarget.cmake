function(add_cuda_target target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PUBLIC /usr/local/cuda/samples/Common)
  target_compile_options(${target}
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
  target_include_directories(
    ${target} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/usr/local/cuda/include>)
  target_link_directories(${target} PUBLIC /usr/local/cuda/lib64)
endfunction()

function(add_cuda_dylib target)
  add_library(${target} SHARED ${ARGN})
  target_include_directories(${target} PUBLIC /usr/local/cuda/samples/Common)
  target_compile_options(${target}
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
  target_include_directories(
    ${target} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/usr/local/cuda/include>)
  target_link_directories(${target} PUBLIC /usr/local/cuda/lib64)
endfunction()
