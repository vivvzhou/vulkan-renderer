# Résumé entry

A ready-to-paste project entry, plus shorter variants.

---

## Full version (4 bullets)

**Real-Time Vulkan Renderer** — *C++20, Vulkan, GLSL/SPIR-V, Compute Shaders, CMake, NumPy*

- Built a physically based renderer from scratch (no engine) across 10 incrementally-verified
  milestones — ~4,800 lines of C++20 + GLSL — with a **deferred pipeline**: a four-target G-buffer
  geometry pass feeding a fullscreen PBR lighting pass (GGX/Cook-Torrance BRDF, image-based
  lighting, PCF shadow mapping), sustaining a **~1.5 ms GPU frame** measured with `VkQueryPool`
  timestamp queries and per-pass profiling.
- Wrote a **custom `VkDeviceMemory` allocator** that sub-allocates buffers and images from 64 MB
  device-memory blocks via an aligned free-list with coalescing and linear/non-linear
  segregation, **replacing the reference VMA library** and managing 384 MB across 6 heaps with
  **zero validation-layer errors**.
- **Parallelized command-buffer recording** across a 4-thread pool using secondary command buffers
  and per-thread command pools, and enforced GPU/CPU correctness with semaphores, fences, and
  pipeline barriers plus **pipeline-cache serialization** for warm pipeline creation across runs.
- Extended the renderer with **machine learning**: trained a 3-layer MLP offline (NumPy, hand-written
  backprop + Adam) and ran it **per-pixel in a compute shader over the G-buffer** to approximate
  ambient occlusion (~1.2 ms/frame), integrating learned inference directly into the real-time
  graphics pipeline.

---

## Condensed version (2 bullets)

**Real-Time Vulkan Renderer** — *C++20, Vulkan, GLSL/SPIR-V, Compute, NumPy*

- From-scratch deferred PBR renderer (GGX BRDF, IBL, PCF shadows) with a hand-written
  `VkDeviceMemory` allocator (aligned free-list over 64 MB heaps, replacing VMA) and 4-thread
  secondary-command-buffer recording; ~1.5 ms GPU frame, profiled via timestamp queries, zero
  validation errors.
- Added an offline-trained MLP (NumPy backprop + Adam) that runs per-pixel in a compute shader on
  the G-buffer to approximate ambient occlusion — learned inference inside a real-time pipeline.

---

## One-liner

Real-time deferred PBR Vulkan renderer with a custom GPU memory allocator, multithreaded command
recording, and a neural (compute-shader MLP) ambient-occlusion pass — C++20 + GLSL, ~1.5 ms/frame.

---

## Talking points (for interviews)

- **Explicit memory:** `VkMemoryRequirements`, memory-type selection by property flags,
  `bufferImageGranularity`, host-visible vs device-local, persistent mapping.
- **Synchronization:** semaphores vs fences, pipeline barriers, subpass dependencies, image layout
  transitions, per-frame vs per-image sync objects, frames-in-flight hazards.
- **Deferred rendering:** MRT G-buffer layout/packing, fullscreen lighting, reconstructing the
  background from cleared attachments.
- **Multithreading:** command pools are externally synchronized → per-thread pools; secondary
  command buffers with render-pass inheritance; fork/join recording.
- **ML systems:** feature engineering for scale-invariance, tiny-MLP inference as GEMM in a compute
  shader, weights as an SSBO, offline training → runtime weights pipeline.
