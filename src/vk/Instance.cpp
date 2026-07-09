#include "vk/Instance.hpp"
#include "vk/Common.hpp"

#include <GLFW/glfw3.h> // GLFW_INCLUDE_VULKAN is defined by the build

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

bool validationLayerSupported() {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& layer : layers) {
        if (std::strcmp(layer.layerName, kValidationLayer) == 0) {
            return true;
        }
    }
    return false;
}

void populateDebugCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& info) {
    info = {};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
}

} // namespace

Instance::Instance(bool enableValidation) {
    validationEnabled_ = enableValidation && validationLayerSupported();
    if (enableValidation && !validationEnabled_) {
        std::fprintf(stderr, "Validation requested but layer unavailable; continuing without it.\n");
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vulkan-renderer";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.pEngineName = "vkrenderer";
    app.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    // Required extensions: GLFW's surface extensions, plus debug-utils when validating.
    uint32_t glfwCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwCount);
    if (validationEnabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    if (validationEnabled_) {
        ci.enabledLayerCount = 1;
        ci.ppEnabledLayerNames = &kValidationLayer;
        populateDebugCreateInfo(debugInfo);
        ci.pNext = &debugInfo; // also covers instance create/destroy
    }

    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));

    if (validationEnabled_) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT info;
            populateDebugCreateInfo(info);
            VK_CHECK(create(instance_, &info, nullptr, &debugMessenger_));
        }
    }
}

Instance::~Instance() {
    if (debugMessenger_ != VK_NULL_HANDLE) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy != nullptr) {
            destroy(instance_, debugMessenger_, nullptr);
        }
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}
