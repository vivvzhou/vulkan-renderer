// Phase 0 smoke test: create a VkInstance, report the loader's API version, and
// enumerate physical devices. On this machine that should list the NVIDIA RTX 4050
// (discrete) and the Intel Arc (integrated) GPU -- confirming the loader, the SDK,
// and the CMake/MSVC toolchain are all wired up before any real rendering begins.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "Discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "Virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU";
        default:                                     return "Other";
    }
}

int main() {
    uint32_t loaderVersion = 0;
    if (vkEnumerateInstanceVersion(&loaderVersion) != VK_SUCCESS) {
        std::fprintf(stderr, "vkEnumerateInstanceVersion failed\n");
        return EXIT_FAILURE;
    }
    std::printf("Vulkan loader instance version: %u.%u.%u\n",
                VK_API_VERSION_MAJOR(loaderVersion),
                VK_API_VERSION_MINOR(loaderVersion),
                VK_API_VERSION_PATCH(loaderVersion));

    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "vulkan-renderer";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName        = "vkrenderer";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed\n");
        return EXIT_FAILURE;
    }
    std::printf("VkInstance created.\n");

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    std::printf("Physical devices (%u):\n", deviceCount);
    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        std::printf("  - %-40s [%s] API %u.%u.%u\n",
                    props.deviceName,
                    device_type_name(props.deviceType),
                    VK_API_VERSION_MAJOR(props.apiVersion),
                    VK_API_VERSION_MINOR(props.apiVersion),
                    VK_API_VERSION_PATCH(props.apiVersion));
    }

    vkDestroyInstance(instance, nullptr);
    std::printf("OK\n");
    return EXIT_SUCCESS;
}
