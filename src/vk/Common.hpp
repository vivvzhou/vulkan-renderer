#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// Throws std::runtime_error on any non-success VkResult, including the failing call text.
#define VK_CHECK(expr)                                                              \
    do {                                                                            \
        const VkResult vk_result_ = (expr);                                         \
        if (vk_result_ != VK_SUCCESS) {                                             \
            throw std::runtime_error(std::string(#expr " -> VkResult ") +           \
                                     std::to_string(static_cast<int>(vk_result_))); \
        }                                                                           \
    } while (false)
