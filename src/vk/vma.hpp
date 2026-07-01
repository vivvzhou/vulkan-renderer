#pragma once

// Single include point for the VulkanMemoryAllocator header (declarations only).
// VMA is a large third-party header that trips our /W4 /permissive- warnings, so we
// silence diagnostics just for it. The implementation lives in VmaImpl.cpp.
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vk_mem_alloc.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
