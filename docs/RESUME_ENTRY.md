# Resume entry

Ready-to-paste project entry, tuned for graphics, systems, and performance-oriented software
engineering roles.

---

## Full version

**Real-Time Vulkan Renderer** - *C++20, Vulkan, GLSL/SPIR-V, GPU Systems, Compute Shaders, CMake, NumPy*

- Built a from-scratch real-time renderer in C++20/Vulkan across ~4.3K lines of C++/GLSL/Python,
  implementing a deferred PBR pipeline with 4-target G-buffer, GGX/Cook-Torrance shading,
  image-based lighting, HDR environment precomputation, glTF asset loading, PCF shadow mapping,
  and GLSL-to-SPIR-V shader compilation.
- Engineered explicit GPU systems components: custom `VkDeviceMemory` sub-allocator over 64 MB
  heaps, descriptor/pipeline management, image layout transitions, synchronization with fences,
  semaphores, and pipeline barriers, plus validation-layer-clean resource lifetime handling.
- Optimized and profiled a 4-pass graphics/compute frame using `VkQueryPool` GPU timestamps,
  serialized `VkPipelineCache` data for warm starts, and recorded draw work across a 4-thread pool
  with per-thread command pools and secondary command buffers; measured ~1.5 ms GPU frame time at
  1024x576 on an RTX 4050 Laptop GPU.
- Integrated ML inference into the rendering pipeline by training a compact MLP offline in NumPy
  with manual backprop/Adam and executing it per-pixel in a Vulkan compute shader over the
  G-buffer for neural ambient occlusion.

## Condensed version

**Real-Time Vulkan Renderer** - *C++20, Vulkan, GLSL/SPIR-V, GPU Systems, Compute Shaders*

- Built a from-scratch deferred PBR renderer with glTF loading, GGX/Cook-Torrance shading, IBL,
  PCF shadows, a custom `VkDeviceMemory` allocator, explicit synchronization, and 4-thread
  secondary command-buffer recording; profiled ~1.5 ms GPU frame time with `VkQueryPool` on RTX
  4050 Laptop.
- Added a neural ambient-occlusion pass by training a compact MLP in NumPy and running per-pixel
  inference in a Vulkan compute shader over the G-buffer, bridging real-time graphics, GPU
  performance engineering, and ML systems.

## One-liner

From-scratch C++20/Vulkan renderer with deferred PBR, custom GPU memory allocation, multithreaded
command recording, GPU timestamp profiling, and compute-shader MLP ambient occlusion.

## Keyword targets

C++20, Vulkan, GLSL, SPIR-V, real-time rendering, graphics pipeline, compute shader, GPU memory
management, synchronization, command buffers, deferred rendering, PBR, image-based lighting, glTF,
shader compilation, profiling, performance optimization, multithreading, systems programming,
software design, ML inference, NumPy.
