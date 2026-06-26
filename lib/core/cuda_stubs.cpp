// Weak fallback stubs for CUDA API symbols when libtensorcore is built
// without TC_ENABLE_CUDA. The strong definitions live in
// lib/cuda/buffer.cpp and are linked when CUDA is on. On portable-CPU
// builds those symbols are not defined, so CPU paths that call
// tc_cuda_is_active() etc. will fail to link.
//
// This file resolves those references to "CUDA is not active" stubs.
// All symbols are __attribute__((weak)) so a real CUDA build wins when
// both are linked.

#ifndef TC_ENABLE_CUDA

extern "C" {

__attribute__((weak)) int tc_cuda_is_active(void) { return 0; }

}  // extern "C"

#endif  // !TC_ENABLE_CUDA
