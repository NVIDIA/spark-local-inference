// SPDX-FileCopyrightText: Copyright (c) 2026 Fraser Price
// SPDX-License-Identifier: Apache-2.0
// Adapted from BRRR; see NOTICE for the upstream project and contributors.
#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <vector>

template <class T>
T* zeroed(long count) {
  void* ptr = nullptr;
  TORCH_CHECK(cudaMalloc(&ptr, count * sizeof(T)) == cudaSuccess);
  TORCH_CHECK(cudaMemset(ptr, 0, count * sizeof(T)) == cudaSuccess);
  return static_cast<T*>(ptr);
}

template <class T>
T* data_as(const torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <class T>
const T* ptr(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous());
  return data_as<const T>(t);
}

template <class T>
T* mut(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous());
  return data_as<T>(t);
}

template <class T>
const T* typed(const torch::Tensor& t, at::ScalarType type) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == type);
  return data_as<const T>(t);
}

inline std::vector<int64_t> alloc_ipc(int64_t bytes) {
  void* ptr = zeroed<uint8_t>(bytes);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  cudaIpcMemHandle_t handle;
  TORCH_CHECK(cudaIpcGetMemHandle(&handle, ptr) == cudaSuccess);
  std::vector<int64_t> out{reinterpret_cast<int64_t>(ptr)};
  const auto* raw = reinterpret_cast<const unsigned char*>(&handle);
  for (size_t i = 0; i < sizeof(handle); ++i) out.push_back(raw[i]);
  return out;
}

inline int64_t open_ipc(std::vector<int64_t> handle_bytes) {
  cudaIpcMemHandle_t handle;
  auto* raw = reinterpret_cast<unsigned char*>(&handle);
  for (size_t i = 0; i < sizeof(handle); ++i) raw[i] = static_cast<unsigned char>(handle_bytes[i]);
  void* ptr = nullptr;
  TORCH_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess);
  return reinterpret_cast<int64_t>(ptr);
}

template <auto Kernel>
void opt_in_smem(int bytes) {
  static const bool done = [bytes] {
    TORCH_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess);
    return true;
  }();
  (void)done;
}

template <class... Params, class... Args>
void launch_pdl(const char* name, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, Args... args) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = c10::cuda::getCurrentCUDAStream();
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&config, kernel, args...);
  TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
}
