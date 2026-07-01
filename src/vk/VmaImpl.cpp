// The one translation unit that instantiates VulkanMemoryAllocator's implementation.
// VMA_IMPLEMENTATION must be defined in exactly one .cpp across the whole program.
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
