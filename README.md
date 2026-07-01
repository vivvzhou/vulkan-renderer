# vulkan-renderer

A real-time 3D renderer built from scratch with the **Vulkan** API in C++ — no engine, no
abstraction layer hiding the driver. PBR materials, image-based lighting, shadow mapping,
multithreaded command recording, a hand-written GPU memory allocator, and a neural
ambient-occlusion compute pass.

> Built as a deep dive into the systems concepts that underlie GPU drivers: explicit memory
> management, synchronization, queue submission, SPIR-V, and pipeline-cache serialization.

## Status

Built incrementally in phases — each one a runnable milestone:

- [x] **Phase 0** — toolchain, CMake skeleton, `VkInstance` smoke test
- [x] **Phase 1** — swapchain + first triangle
- [x] **Phase 2** — VMA, buffers, depth, camera, textured glTF mesh
- [ ] **Phase 3** — PBR (GGX) + image-based lighting
- [ ] **Phase 4** — shadow mapping
- [ ] **Phase 5** — deferred G-buffer
- [ ] **Phase 6** — multithreaded secondary command buffers
- [ ] **Phase 7** — custom `VkDeviceMemory` allocator
- [ ] **Phase 8** — pipeline cache + GPU timestamp profiling
- [ ] **Phase 9** — neural ambient occlusion (offline-trained MLP, GLSL compute inference)
- [ ] **Phase 10** — portfolio polish

## Build

Requires the [Vulkan SDK](https://vulkan.lunarg.com), CMake ≥ 3.24, and a C++20 compiler.
Third-party dependencies (GLFW, GLM, VMA, stb, tinygltf) are fetched automatically by CMake.

```sh
cmake -S . -B build
cmake --build build --config Debug
# Windows: build\bin\Debug\vkrenderer.exe
```

## License

TBD
