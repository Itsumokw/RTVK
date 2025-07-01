#define GLFW_INCLUDE_VULKAN
#define PROJECT_NAME "rtvk"
#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "samplevk.h"
#include <fstream>
#include <string>
#include <stdexcept>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <vulkan/vulkan_core.h>
#include <limits>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include "tiny_obj_loader.h"
#include "stb_image.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

namespace fs = std::filesystem;

using namespace std;

// 常量定义
const int SAMPLE_WIDTH = 1024;
const int SAMPLE_HEIGHT = 768;

// 前置声明
class RTVKApplication;

// GLFW回调函数
static void onErrorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// 顶点结构
struct Vertex {
    glm::vec3 position;
    float pad0;
    glm::vec3 normal;
    float pad1;
    glm::vec2 texCoord;
    float pad2[2];
};

struct SimpleVertex {
    glm::vec3 position;
};


// 几何体结构
struct Geometry {
    uint32_t materialindex = 0;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

    VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
    VkBuffer blasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory blasMemory = VK_NULL_HANDLE;
};

//场景
struct Scene {
    std::vector<Geometry> geometries;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    VkBuffer tlasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory tlasMemory = VK_NULL_HANDLE;
};


// 相机参数结构
struct CameraParams {
    glm::mat4 view_inverse;
    glm::mat4 proj_inverse;
} camera_params;

// 光源类型
enum LightType {
    LIGHT_DIRECTIONAL = 0,
    LIGHT_POINT = 1,
    LIGHT_SPOT = 2
};

// 光源数据结构 (16字节对齐)
struct Light {
    glm::vec4 position;        // 位置 (点光源和聚光灯光源) 或方向 (方向光源)
    glm::vec4 color;           // RGB 颜色 + 强度 (A)
    float range;               // 光源作用范围
    float spotOuterAngle;      // 聚光灯外角
    float spotInnerAngle;      // 聚光灯内角
    float padding;
    glm::vec4 direction;             // 填充对齐
};

// 确保结构大小为 32 字节 (GLSL 的 std140 对齐要求)
//static_assert(sizeof(Light) == 32, "Light struct must be 32 bytes");

enum MaterialType {
    DIFFUSE = 0,
    METALLIC = 1,
    SPECULAR = 2,
    DIELECTRIC = 3
};

struct Material {
    glm::vec3 albedo;       // 基础颜色
    glm::vec3 emission;     // 自发光
    float roughness;        // 粗糙度 (0.0-1.0)
    float metallic;         // 金属度 (0.0-1.0)
    float ior;              // 折射率
    float transparency;     // 透明度 (0.0-1.0)
    uint32_t materialType;  // 材质类型 (0=漫反射, 1=金属, 2=电介质/玻璃)
    float pad;
};

// 在 samplevk.h 中添加
class ArcballCamera {
public:
    ArcballCamera(glm::vec3 center, float distance, float minDistance = 1.0f, float maxDistance = 50.0f)
        : m_center(center), m_distance(distance),
        m_minDistance(minDistance), m_maxDistance(maxDistance) {
        updateCamera();
    }

    void rotate(glm::vec2 delta) {
        m_rotation.x += delta.y * ROTATION_SPEED;
        m_rotation.y += delta.x * ROTATION_SPEED;

        // 限制俯仰角范围
        m_rotation.x = glm::clamp(m_rotation.x, -MAX_PITCH, MAX_PITCH);
        updateCamera();
    }

    void zoom(float delta) {
        m_distance = glm::clamp(m_distance - delta * ZOOM_SPEED, m_minDistance, m_maxDistance);
        updateCamera();
    }

    void pan(glm::vec2 delta) {
        glm::vec3 right = glm::normalize(glm::cross(m_up, m_direction));
        glm::vec3 up = glm::normalize(glm::cross(m_direction, right));

        m_center -= right * delta.x * PAN_SPEED;
        m_center += up * delta.y * PAN_SPEED;
        updateCamera();
    }

    glm::vec3 getPosition() const { return m_position; }
    glm::vec3 getTarget() const { return m_center; }
    glm::vec3 getDirection() const { return m_direction; }
    glm::vec3 getUp() const { return m_up; }

    void setCenter(glm::vec3 center) {
        m_center = center;
        updateCamera();
    }

    void setDistance(float distance) {
        m_distance = glm::clamp(distance, m_minDistance, m_maxDistance);
        updateCamera();
    }

private:
    void updateCamera() {
        // 计算相机位置
        m_position.x = m_center.x + m_distance * glm::cos(m_rotation.y) * glm::cos(m_rotation.x);
        m_position.y = m_center.y + m_distance * glm::sin(m_rotation.x);
        m_position.z = m_center.z + m_distance * glm::sin(m_rotation.y) * glm::cos(m_rotation.x);

        // 计算方向向量
        m_direction = glm::normalize(m_center - m_position);
        m_up = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // 常量
    static constexpr float ROTATION_SPEED = 0.01f;
    static constexpr float ZOOM_SPEED = 0.1f;
    static constexpr float PAN_SPEED = 0.01f;
    static constexpr float MAX_PITCH = glm::radians(89.0f);

    // 相机参数
    glm::vec3 m_center;
    glm::vec3 m_position;
    glm::vec3 m_direction;
    glm::vec3 m_up;
    glm::vec2 m_rotation = { 0.0f, 0.0f }; // x: pitch, y: yaw
    float m_distance;
    float m_minDistance;
    float m_maxDistance;
};

// 资源管理封装类
class ResourceManager {
public:
    static uint32_t findMemoryType(VkPhysicalDevice device, uint32_t typeFilter, VkMemoryPropertyFlags properties);
    static vector<uint32_t> readSPIRV(const string& filename);
};

// Vulkan渲染器封装类
class VulkanRenderer {
public:
    VulkanRenderer(RTVKApplication* app);
    ~VulkanRenderer();

    void init();
    void createSwapChain();
    void createImageViews();
    void createRayTracingOutputImage();
    void loadShaders();
    void createRayTracingPipeline();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSet();
    void createSBT();
    void createCommandPool();
    void createAccelerationStructures();
    void initSyncObjects();
    void drawFrame();
    void initImGui(); // 初始化ImGui
    void renderImGui(VkCommandBuffer commandBuffer); // 渲染ImGui UI
    void createImGuiRenderPass();

    // 函数指针获取
    void loadRayTracingFunctions();

    // 成员访问器
    VkDevice getDevice() const { return m_device; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }

    // 添加鼠标处理函数
    void handleMouseMove(double x, double y);
    void handleMouseButton(int button, int action);
    void handleScroll(double xoffset, double yoffset);

private:
    void flush_command_buffer(VkCommandBuffer command_buffer, VkQueue queue, bool free, VkSemaphore signalSemaphore) const;
    VkCommandBuffer create_command_buffer(VkCommandBufferLevel level, bool begin) const;
    void image_layout_transition(VkCommandBuffer                command_buffer,
        VkImage                        image,
        VkPipelineStageFlags           src_stage_mask,
        VkPipelineStageFlags           dst_stage_mask,
        VkAccessFlags                  src_access_mask,
        VkAccessFlags                  dst_access_mask,
        VkImageLayout                  old_layout,
        VkImageLayout                  new_layout,
        VkImageSubresourceRange const& subresource_range);

    void pickPhysicalDevice();
    void createDevice();
    void createPipelineLayout();
    void createBuffer(
        void* data,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory
    );
    VkDeviceAddress getBufferDeviceAddress(
        VkBuffer buffer
    );
    VkDeviceAddress getAccelerationStructureDeviceAddress(
        VkAccelerationStructureKHR accelerationStructure
    );
    void createAccelerationStructure(
        VkAccelerationStructureTypeKHR type,
        VkDeviceSize size,
        VkAccelerationStructureKHR& accelerationStructure,
        VkBuffer& buffer,
        VkDeviceMemory& memory
    );
    void loadModel(const std::string& path);
    void createBLAS(Geometry& geom);
    void createTLAS(std::vector<VkAccelerationStructureInstanceKHR>& instances);
    VkTransformMatrixKHR toTransformMatrixKHR(const glm::mat4& matrix);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void createCameraBuffer();
    void updateCameraBuffer();
    void createLightBuffer();
    void setupSceneLights();
    void createMaterialBuffer();
    void setupMaterials();
    void createEnvironmentMap(const std::string& path);
    void createChessboardPlane();

    glm::vec3 m_cameraPosition = glm::vec3(-0.f, -0.f, -4.f);
    glm::vec3 m_cameraTarget = glm::vec3(-0.f, -0.f, 0.f);
    glm::vec3 m_cameraUp = glm::vec3(0.0f, -1.0f, 0.f);
    float m_cameraFOV = 60.0f;

    uint32_t alignUp(uint32_t value, uint32_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    RTVKApplication* m_app;

    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkQueue m_graphicsQueue;
    uint32_t m_queueFamilyIndex;

    // Swap chain
    VkSwapchainKHR m_swapchain;
    VkFormat m_swapchainFormat;
    VkExtent2D m_swapchainExtent;
    vector<VkImage> m_swapchainImages;
    vector<VkImageView> m_swapchainImageViews;

    // Ray tracing resources
    VkImage m_outputImage;
    VkImageView m_outputImageView;
    VkDeviceMemory m_outputImageMemory;

    // Shaders
    VkShaderModule m_rgenModule;
    VkShaderModule m_rmissModule;
    VkShaderModule m_rchitModule;
    VkShaderModule m_rmiss2Module;

    //descriptor
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSet m_descriptorSet;

    // Pipeline
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_rayTracingPipeline;
    VkDescriptorSetLayout m_descriptorSetLayout;

    // Ray tracing function pointers
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetBufferDeviceAddressKHR m_vkGetBufferDeviceAddressKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR m_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR = nullptr;



    // model
    Scene m_scene;
    VkBuffer m_cameraBuffer;
    VkDeviceMemory m_cameraBufferMemory;

    // acceleration structure
    VkAccelerationStructureKHR m_blas;
    VkBuffer m_blasBuffer;
    VkDeviceMemory m_blasMemory;

    VkBuffer m_tlasInstancesBuffer;
    VkDeviceMemory m_tlasInstancesMemory;

    VkCommandPool m_commandPool;
    VkCommandBuffer m_commandBuffer;

    // SBT
    struct SBT {
        VkBuffer buffer;
        VkDeviceMemory memory;
        VkDeviceAddress address;
        VkStridedDeviceAddressRegionKHR rgenRegion;
        VkStridedDeviceAddressRegionKHR missRegion;
        VkStridedDeviceAddressRegionKHR hitRegion;
    } m_sbt;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rayTracingProperties;

    // Semaphore
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;                  //用于commandbuffer同步
    uint32_t m_currentFrame = 0;
    const int MAX_FRAMES_IN_FLIGHT = 2;

    //模型偏移
    glm::vec3 m_modelMin = glm::vec3(FLT_MAX);
    glm::vec3 m_modelMax = glm::vec3(-FLT_MAX);
    glm::vec3 m_modelCenter = glm::vec3(0.0f);

    //light!!!!!!!
    std::vector<Light> m_lights;
    VkBuffer m_lightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_lightBufferMemory = VK_NULL_HANDLE;

    //material
    std::vector<Material> m_materials;
    VkBuffer m_materialBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_materialBufferMemory = VK_NULL_HANDLE;

    //env
    VkImage m_envImage = VK_NULL_HANDLE;
    VkImageView m_envImageView = VK_NULL_HANDLE;
    VkDeviceMemory m_envImageMemory = VK_NULL_HANDLE;
    VkSampler m_envSampler = VK_NULL_HANDLE;

    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;
    VkRenderPass m_imguiRenderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_imguiFramebuffers;

    std::unique_ptr<ArcballCamera> m_arcballCamera;

    // 鼠标状态变量
    bool m_mouseLeftPressed = false;
    bool m_mouseRightPressed = false;
    bool m_mouseMiddlePressed = false;
    glm::vec2 m_lastMousePos = { 0.0f, 0.0f };

    // 添加初始化方法
    void initArcballCamera();


};

// 主应用类
class RTVKApplication {
public:
    GLFWwindow* window = nullptr;
    VulkanRenderer* renderer = nullptr;

    RTVKApplication();
    ~RTVKApplication();

    void run();
    void setProjectRoot(const fs::path& root) { m_projectRoot = root; }
    const fs::path& getProjectRoot() const { return m_projectRoot; }

    // 成员访问器
    VkInstance getInstance() const { return m_instance; }
    VkSurfaceKHR getSurface() const { return m_surface; }
    GLFWwindow* getWindow() const { return window; }

private:
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    fs::path m_projectRoot;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;

    double m_lastTime = 0.0;
    int m_frameCount = 0;
    double m_fps = 0.0;
};

int main() {
    cout << "Hello CMake." << endl;

    RTVKApplication app;
    fs::path currentPath = fs::current_path().parent_path().parent_path().parent_path().parent_path();
    app.setProjectRoot(currentPath);
    std::cout << "当前项目根目录: " << app.getProjectRoot() << std::endl;

    try {
        app.run();
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// ResourceManager实现
uint32_t ResourceManager::findMemoryType(VkPhysicalDevice device, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw runtime_error("Failed to find suitable memory type!");
}

vector<uint32_t> ResourceManager::readSPIRV(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Failed to open SPIR-V file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

// RTVKApplication实现
RTVKApplication::RTVKApplication() {
    initWindow();
}

RTVKApplication::~RTVKApplication() {
    cleanup();
}

void RTVKApplication::run() {
    initVulkan();
    mainLoop();
}

void RTVKApplication::initWindow() {
    glfwSetErrorCallback(onErrorCallback);
    if (!glfwInit()) {
        throw runtime_error("GLFW initialization failed!");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);
    if (!window) {
        throw runtime_error("Failed to create GLFW window!");
    }

    // 设置窗口用户指针为当前应用实例
    glfwSetWindowUserPointer(window, this);
    // 注册鼠标回调
    glfwSetMouseButtonCallback(window, RTVKApplication::mouseButtonCallback);
    glfwSetCursorPosCallback(window, RTVKApplication::cursorPosCallback);
    glfwSetScrollCallback(window, RTVKApplication::scrollCallback);
}

void RTVKApplication::initVulkan() {
    // 检查Vulkan支持
    if (!glfwVulkanSupported()) {
        throw runtime_error("GLFW: Vulkan Not Supported");
    }

    // 创建Vulkan实例
    VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = PROJECT_NAME;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    vector<const char*> instanceExtensions;
    vector<const char*> instanceLayers = { "VK_LAYER_KHRONOS_validation" };

    // 获取GLFW所需的扩展
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    for (uint32_t i = 0; i < glfwExtensionCount; i++) {
        instanceExtensions.push_back(glfwExtensions[i]);
    }
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(instanceLayers.size());
    createInfo.ppEnabledLayerNames = instanceLayers.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        throw runtime_error("Failed to create Vulkan instance!");
    }

    // 创建表面
    if (glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface) != VK_SUCCESS) {
        throw runtime_error("Failed to create window surface!");
    }

    // 创建渲染器
    renderer = new VulkanRenderer(this);
    renderer->init();
}

void RTVKApplication::mainLoop() {
    // 初始化计时器
    double lastTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        // 计算帧率
        double currentTime = glfwGetTime();
        frameCount++;

        // 每1秒更新一次FPS
        if (currentTime - lastTime >= 1.0) {
            m_fps = frameCount / (currentTime - lastTime);
            std::cout << "FPS: " << m_fps << std::endl;

            // 重置计数器
            frameCount = 0;
            lastTime = currentTime;
        }

        glfwPollEvents();
        renderer->drawFrame();
    }

    vkDeviceWaitIdle(renderer->getDevice());
}

void RTVKApplication::cleanup() {
    if (renderer) {
        delete renderer;
        renderer = nullptr;
    }

    if (m_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

// VulkanRenderer实现
VulkanRenderer::VulkanRenderer(RTVKApplication* app) :
    m_app(app),
    m_physicalDevice(VK_NULL_HANDLE),
    m_device(VK_NULL_HANDLE),
    m_graphicsQueue(VK_NULL_HANDLE),
    m_swapchain(VK_NULL_HANDLE),
    m_outputImage(VK_NULL_HANDLE),
    m_outputImageView(VK_NULL_HANDLE),
    m_outputImageMemory(VK_NULL_HANDLE),
    m_rgenModule(VK_NULL_HANDLE),
    m_rmissModule(VK_NULL_HANDLE),
    m_rchitModule(VK_NULL_HANDLE),
    m_descriptorPool(VK_NULL_HANDLE),
    m_descriptorSet(VK_NULL_HANDLE),
    m_pipelineLayout(VK_NULL_HANDLE),
    m_rayTracingPipeline(VK_NULL_HANDLE),
    m_descriptorSetLayout(VK_NULL_HANDLE),
    m_blas(VK_NULL_HANDLE),
    m_blasBuffer(VK_NULL_HANDLE),
    m_blasMemory(VK_NULL_HANDLE),
    m_tlasInstancesBuffer(VK_NULL_HANDLE),
    m_tlasInstancesMemory(VK_NULL_HANDLE),
    m_commandPool(VK_NULL_HANDLE),
    m_commandBuffer(VK_NULL_HANDLE) {
}

VulkanRenderer::~VulkanRenderer() {
    // 清理Vulkan对象
    if (m_device != VK_NULL_HANDLE) {
        if (m_envSampler) vkDestroySampler(m_device, m_envSampler, nullptr);
        if (m_envImageView) vkDestroyImageView(m_device, m_envImageView, nullptr);
        if (m_envImage) vkDestroyImage(m_device, m_envImage, nullptr);
        if (m_envImageMemory) vkFreeMemory(m_device, m_envImageMemory, nullptr);
        // 清理交换链相关资源
        for (auto imageView : m_swapchainImageViews) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }

        if (m_swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        }

        // 清理着色器模块
        if (m_rgenModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, m_rgenModule, nullptr);
        }
        if (m_rmissModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, m_rmissModule, nullptr);
        }
        if (m_rchitModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, m_rchitModule, nullptr);
        }

        // 清理管线
        if (m_rayTracingPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_rayTracingPipeline, nullptr);
        }

        // 清理布局
        if (m_pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        }

        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        }

        // 清理光线追踪输出图像
        if (m_outputImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_outputImageView, nullptr);
        }
        if (m_outputImage != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, m_outputImage, nullptr);
        }
        if (m_outputImageMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_outputImageMemory, nullptr);
        }
        // 添加以下销毁
        if (m_sbt.buffer) {
            vkDestroyBuffer(m_device, m_sbt.buffer, nullptr);
        }
        if (m_sbt.memory) {
            vkFreeMemory(m_device, m_sbt.memory, nullptr);
        }
        if (m_descriptorPool) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        }
        if (m_commandPool) {
            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        }
        // 添加几何体缓冲区清理
        for (auto& geom : m_scene.geometries) {
            if (geom.vertexBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, geom.vertexBuffer, nullptr);
            }
            if (geom.vertexBufferMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, geom.vertexBufferMemory, nullptr);
            }
            if (geom.indexBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, geom.indexBuffer, nullptr);
            }
            if (geom.indexBufferMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, geom.indexBufferMemory, nullptr);
            }

            // 清理BLAS资源
            if (vkDestroyAccelerationStructureKHR && geom.blas != VK_NULL_HANDLE) {
                vkDestroyAccelerationStructureKHR(m_device, geom.blas, nullptr);
            }
            if (geom.blasBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, geom.blasBuffer, nullptr);
            }
            if (geom.blasMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, geom.blasMemory, nullptr);
            }
        }
        auto destroyBufferAndMemory = [&](VkBuffer& buffer, VkDeviceMemory& memory) {
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
            }
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }
            };
        destroyBufferAndMemory(m_scene.tlasBuffer, m_scene.tlasMemory);
        destroyBufferAndMemory(m_blasBuffer, m_blasMemory);
        destroyBufferAndMemory(m_tlasInstancesBuffer, m_tlasInstancesMemory);

        vkDestroyDevice(m_device, nullptr);
    }
}

void VulkanRenderer::init() {

    const fs::path& projectRoot = m_app->getProjectRoot();
    pickPhysicalDevice();
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR
    };
    VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);
    m_rayTracingProperties = rtProps;
    createDevice();
    createCommandPool();
    loadRayTracingFunctions();
    createSwapChain();
    createImageViews();
    createRayTracingOutputImage();
    initArcballCamera();
    loadModel((projectRoot / "samplevk/models/asschercut-mesh.obj").string());
    loadModel((projectRoot / "samplevk/models/venus-mesh.obj").string());
    loadModel((projectRoot / "samplevk/models/bunny-mesh.obj").string());
    loadModel((projectRoot / "samplevk/models/dragon-mesh.obj").string());
    createEnvironmentMap((projectRoot / "samplevk/models/envmap.jpg").string());
    createChessboardPlane();
    loadModel((projectRoot / "samplevk/models/fudanlogo-mesh.obj").string());
    setupMaterials();
    setupSceneLights();
    createCameraBuffer();
    createImGuiRenderPass();
    initImGui();
    createAccelerationStructures();
    loadShaders();
    createDescriptorPool();
    createDescriptorSetLayout();
    createPipelineLayout();
    createDescriptorSet();
    createRayTracingPipeline();
    createSBT();
    initSyncObjects();
}

void VulkanRenderer::pickPhysicalDevice() {
    RTVKApplication* app = m_app;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(app->getInstance(), &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw runtime_error("No Vulkan physical devices found!");
    }

    vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(app->getInstance(), &deviceCount, physicalDevices.data());

    for (const auto& device : physicalDevices) {
        // 检查队列族支持
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        bool graphicsFamilyFound = false;
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                m_queueFamilyIndex = i;
                graphicsFamilyFound = true;
                break;
            }
        }

        if (!graphicsFamilyFound) {
            continue; // 没有图形队列族，继续下一个设备
        }

        // 检查表面支持
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, m_queueFamilyIndex, app->getSurface(), &presentSupport);
        if (!presentSupport) {
            continue;
        }

        // 检查光线追踪支持
        VkPhysicalDeviceFeatures2 features2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
        features2.pNext = &rtFeatures;

        vkGetPhysicalDeviceFeatures2(device, &features2);

        if (rtFeatures.rayTracingPipeline) {
            m_physicalDevice = device;
            break;
        }
    }

    if (m_physicalDevice == VK_NULL_HANDLE) {
        throw runtime_error("No Ray Tracing supported device found!");
    }
}
void VulkanRenderer::createDevice() {
    RTVKApplication* app = m_app;

    vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_MAINTENANCE3_EXTENSION_NAME
    };

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.bufferDeviceAddress = VK_TRUE;
    vulkan12Features.runtimeDescriptorArray = VK_TRUE;

    // 设置光线追踪特性
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{};
    rtPipelineFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeature.rayTracingPipeline = VK_TRUE;
    rtPipelineFeature.pNext = &vulkan12Features;

    // 设置加速结构特性
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{};
    accelFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelFeature.accelerationStructure = VK_TRUE;
    accelFeature.pNext = &rtPipelineFeature;

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueInfo.queueFamilyIndex = m_queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.pNext = &accelFeature;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device) != VK_SUCCESS) {
        throw runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(m_device, m_queueFamilyIndex, 0, &m_graphicsQueue);
}

void VulkanRenderer::loadRayTracingFunctions() {
    // 初始化光线追踪扩展函数指针
    vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(m_device, "vkCreateAccelerationStructureKHR"));
    vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR"));
    vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureBuildSizesKHR"));
    vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(m_device, "vkCmdBuildAccelerationStructuresKHR"));
    vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(m_device, "vkCreateRayTracingPipelinesKHR"));
    m_vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
        vkGetDeviceProcAddr(m_device, "vkGetBufferDeviceAddressKHR"));
    m_vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureDeviceAddressKHR"));
    vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
        vkGetDeviceProcAddr(m_device, "vkGetRayTracingShaderGroupHandlesKHR"));
    vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
        vkGetDeviceProcAddr(m_device, "vkCmdTraceRaysKHR"));


    // 检查所有函数指针是否成功加载
    if (!vkCreateAccelerationStructureKHR ||
        !vkDestroyAccelerationStructureKHR ||
        !vkGetAccelerationStructureBuildSizesKHR ||
        !vkCmdBuildAccelerationStructuresKHR ||
        !vkCreateRayTracingPipelinesKHR ||
        !m_vkGetBufferDeviceAddressKHR ||
        !m_vkGetAccelerationStructureDeviceAddressKHR ||
        !vkGetRayTracingShaderGroupHandlesKHR ||
        !vkCmdTraceRaysKHR) {
        throw runtime_error("Failed to load ray tracing function pointers!");
    }
}

void VulkanRenderer::createSwapChain() {
    RTVKApplication* app = m_app;
    VkSurfaceKHR surface = app->getSurface();

    // 1. 获取表面能力
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, surface, &capabilities);

    // 2. 获取表面格式
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, surface, &formatCount, nullptr);
    vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, surface, &formatCount, surfaceFormats.data());

    // 3. 选择表面格式
    VkSurfaceFormatKHR selectedFormat = surfaceFormats[0];
    for (const auto& format : surfaceFormats) {
        if (format.format == VK_FORMAT_R8G8B8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            selectedFormat = format;
            break;
        }
    }
    m_swapchainFormat = selectedFormat.format;

    // 4. 选择交换链尺寸
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        m_swapchainExtent = capabilities.currentExtent;
    }
    else {
        m_swapchainExtent = {
            static_cast<uint32_t>(SAMPLE_WIDTH),
            static_cast<uint32_t>(SAMPLE_HEIGHT)
        };
    }

    // 5. 创建交换链
    VkSwapchainCreateInfoKHR swapchainInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    swapchainInfo.surface = surface;
    swapchainInfo.minImageCount = capabilities.minImageCount + 1;
    swapchainInfo.imageFormat = m_swapchainFormat;
    swapchainInfo.imageColorSpace = selectedFormat.colorSpace;
    swapchainInfo.imageExtent = m_swapchainExtent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainInfo.preTransform = capabilities.currentTransform;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR; // 标准垂直同步
    swapchainInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(m_device, &swapchainInfo, nullptr, &m_swapchain) != VK_SUCCESS) {
        throw runtime_error("Failed to create swapchain!");
    }

    // 6. 获取交换链图像
    uint32_t imageCount;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());
}

void VulkanRenderer::createImageViews() {
    m_swapchainImageViews.resize(m_swapchainImages.size());

    for (size_t i = 0; i < m_swapchainImages.size(); i++) {
        VkImageViewCreateInfo viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_swapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = m_swapchainFormat;
        viewInfo.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY
        };
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &viewInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
            throw runtime_error("Failed to create swapchain image view!");
        }
    }
}

void VulkanRenderer::createRayTracingOutputImage() {
    // 1. 创建图像
    VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = { SAMPLE_WIDTH, SAMPLE_HEIGHT, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(m_device, &imageInfo, nullptr, &m_outputImage) != VK_SUCCESS) {
        throw runtime_error("Failed to create ray tracing output image!");
    }

    // 2. 分配内存
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, m_outputImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = ResourceManager::findMemoryType(
        m_physicalDevice,
        memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_outputImageMemory) != VK_SUCCESS) {
        throw runtime_error("Failed to allocate image memory!");
    }

    vkBindImageMemory(m_device, m_outputImage, m_outputImageMemory, 0);

    // 3. 创建图像视图
    VkImageViewCreateInfo viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image = m_outputImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_device, &viewInfo, nullptr, &m_outputImageView) != VK_SUCCESS) {
        throw runtime_error("Failed to create output image view!");
    }
    VkCommandBuffer command_buffer = create_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    image_layout_transition(command_buffer,
        m_outputImage,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        {},
        {},
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
    flush_command_buffer(command_buffer, m_graphicsQueue, true, 0Ui64);
}

void VulkanRenderer::loadShaders() {
    try {
        // 加载光线生成着色器
        const fs::path& projectRoot = m_app->getProjectRoot();
        auto rgenCode = ResourceManager::readSPIRV((projectRoot / "samplevk/shaders/raygen.spv").string());
        VkShaderModuleCreateInfo shaderInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        shaderInfo.codeSize = rgenCode.size() * sizeof(uint32_t);
        shaderInfo.pCode = rgenCode.data();
        if (vkCreateShaderModule(m_device, &shaderInfo, nullptr, &m_rgenModule) != VK_SUCCESS) {
            throw runtime_error("Failed to create ray generation shader module!");
        }

        // 加载未命中着色器
        auto rmissCode = ResourceManager::readSPIRV((projectRoot / "samplevk/shaders/rmiss.spv").string());
        shaderInfo.codeSize = rmissCode.size() * sizeof(uint32_t);
        shaderInfo.pCode = rmissCode.data();
        if (vkCreateShaderModule(m_device, &shaderInfo, nullptr, &m_rmissModule) != VK_SUCCESS) {
            throw runtime_error("Failed to create ray miss shader module!");
        }

        // 加载最近命中着色器
        auto rchitCode = ResourceManager::readSPIRV((projectRoot / "samplevk/shaders/closethit.spv").string());
        shaderInfo.codeSize = rchitCode.size() * sizeof(uint32_t);
        shaderInfo.pCode = rchitCode.data();
        if (vkCreateShaderModule(m_device, &shaderInfo, nullptr, &m_rchitModule) != VK_SUCCESS) {
            throw runtime_error("Failed to create closest hit shader module!");
        }

        // 加载rmiss2
        auto rmiss2Code = ResourceManager::readSPIRV((projectRoot / "samplevk/shaders/rmiss2.spv").string());
        shaderInfo.codeSize = rmiss2Code.size() * sizeof(uint32_t);
        shaderInfo.pCode = rmiss2Code.data();
        if (vkCreateShaderModule(m_device, &shaderInfo, nullptr, &m_rmiss2Module) != VK_SUCCESS) {
            throw runtime_error("Failed to create miss2 hit shader module!");
        }
    }
    catch (const exception& e) {
        throw runtime_error("Shader loading failed: " + string(e.what()));
    }
}

void VulkanRenderer::createDescriptorSetLayout() {
    // 绑定定义
    std::vector<VkDescriptorSetLayoutBinding> bindings(8);

    // 加速结构 (set=0, binding=0)
    bindings[0] = {
        0,
        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr
    };

    // 存储图像 (set=0, binding=1)
    bindings[1] = {
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr
    };

    // 顶点缓冲区 (set=0, binding=2)
    bindings[2] = {
        2,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        static_cast<uint32_t>(m_scene.geometries.size()),
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr
    };

    // 相机参数 (set=0, binding=3)
    bindings[3] = {
        3,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr
    };

    // 光照参数 (set=0, binding=4)
    bindings[4] = {
        4,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr
    };

    bindings[5] = {
        5,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        static_cast<uint32_t>(m_scene.geometries.size()),
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr
    };

    bindings[6] = {
        6,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        nullptr
    };

    bindings[7] = {  // 修改：从7改为0-indexed第8个位置
        7,          // binding=7
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        1,
        VK_SHADER_STAGE_MISS_BIT_KHR,  // 只在miss shader中使用
        nullptr
    };
    // 创建描述符集布局
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }
}

void VulkanRenderer::createDescriptorPool() {
    // 池大小定义
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}

void VulkanRenderer::createDescriptorSet() {
    // 分配描述符集
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets!");
    }

    // 更新描述符集
    std::vector<VkWriteDescriptorSet> writes(8);

    // 加速结构
    VkWriteDescriptorSetAccelerationStructureKHR accelInfo{};
    accelInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    accelInfo.accelerationStructureCount = 1;
    accelInfo.pAccelerationStructures = &m_scene.tlas;

    writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[0].pNext = &accelInfo;

    // 存储图像
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = m_outputImageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &imageInfo;

    // 顶点缓冲区（使用第一个几何体的缓冲区）
    std::vector<VkDescriptorBufferInfo> vertexBufferInfos(m_scene.geometries.size());
    for (size_t i = 0; i < m_scene.geometries.size(); i++) {
        vertexBufferInfos[i] = {
            m_scene.geometries[i].vertexBuffer,
            0,
            VK_WHOLE_SIZE
        };
    }

    writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[2].dstSet = m_descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = static_cast<uint32_t>(vertexBufferInfos.size());
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = vertexBufferInfos.data();

    // 相机参数
    VkDescriptorBufferInfo cameraBufferInfo{};
    cameraBufferInfo.buffer = m_cameraBuffer;
    cameraBufferInfo.range = sizeof(CameraParams);

    writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[3].dstSet = m_descriptorSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[3].pBufferInfo = &cameraBufferInfo;

    VkDescriptorBufferInfo lightBufferInfo{};
    lightBufferInfo.buffer = m_lightBuffer;
    lightBufferInfo.offset = 0;
    lightBufferInfo.range = sizeof(Light) * m_lights.size();

    writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[4].dstSet = m_descriptorSet;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &lightBufferInfo;

    // 创建索引缓冲区描述符信息
    std::vector<VkDescriptorBufferInfo> indexBufferInfos(m_scene.geometries.size());
    for (size_t i = 0; i < m_scene.geometries.size(); i++) {
        indexBufferInfos[i] = { 
            m_scene.geometries[i].indexBuffer,
            0,
            VK_WHOLE_SIZE 
        };
    }

    // 更新描述符集
    writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[5].dstSet = m_descriptorSet;
    writes[5].dstBinding = 5; // 绑定点5
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[5].pBufferInfo = indexBufferInfos.data();
    writes[5].descriptorCount = static_cast<uint32_t>(indexBufferInfos.size());

    VkDescriptorBufferInfo materialBufferInfo{};
    materialBufferInfo.buffer = m_materialBuffer;
    materialBufferInfo.range = VK_WHOLE_SIZE;

    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = m_descriptorSet;
    writes[6].dstBinding = 6;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[6].pBufferInfo = &materialBufferInfo;

    VkDescriptorImageInfo envImageInfo{};
    envImageInfo.sampler = m_envSampler;
    envImageInfo.imageView = m_envImageView;
    envImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[7].dstSet = m_descriptorSet;
    writes[7].dstBinding = 7;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &envImageInfo;

    vkUpdateDescriptorSets(
        m_device,
        static_cast<uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr
    );
}

void VulkanRenderer::createPipelineLayout() {
    // 创建管线布局
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw runtime_error("Failed to create pipeline layout!");
    }
}

void VulkanRenderer::createRayTracingPipeline() {
    // 1. 准备着色器阶段
    vector<VkPipelineShaderStageCreateInfo> shaderStages;

    // 光线生成
    VkPipelineShaderStageCreateInfo rgenStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    rgenStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    rgenStage.module = m_rgenModule;
    rgenStage.pName = "main";
    shaderStages.push_back(rgenStage);

    // 未命中
    VkPipelineShaderStageCreateInfo missStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    missStage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    missStage.module = m_rmissModule;
    missStage.pName = "main";
    shaderStages.push_back(missStage);

    // shadowray miss
    VkPipelineShaderStageCreateInfo rmiss2Stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    rmiss2Stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    rmiss2Stage.module = m_rmiss2Module;
    rmiss2Stage.pName = "main";
    shaderStages.push_back(rmiss2Stage);

    // 最近命中
    VkPipelineShaderStageCreateInfo hitStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    hitStage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    hitStage.module = m_rchitModule;
    hitStage.pName = "main";
    shaderStages.push_back(hitStage);

    // 2. 准备着色器组
    vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;

    // 光线生成组
    VkRayTracingShaderGroupCreateInfoKHR rgenGroup{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
    rgenGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    rgenGroup.generalShader = 0; // shaderStages索引
    rgenGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    rgenGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    rgenGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(rgenGroup);

    // 未命中组
    VkRayTracingShaderGroupCreateInfoKHR missGroup{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
    missGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    missGroup.generalShader = 1; // shaderStages索引
    missGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    missGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(missGroup);

    // 阴影未命中组
    VkRayTracingShaderGroupCreateInfoKHR shadowMissGroup{};
    shadowMissGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shadowMissGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shadowMissGroup.generalShader = 2; // shadow rmiss 的索引
    shadowMissGroup.closestHitShader = VK_SHADER_UNUSED_KHR; // shaderStages索引
    shadowMissGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    shadowMissGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(shadowMissGroup);

    // 最近命中组
    VkRayTracingShaderGroupCreateInfoKHR hitGroup{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
    hitGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    hitGroup.generalShader = VK_SHADER_UNUSED_KHR;
    hitGroup.closestHitShader = 3; // shaderStages索引
    hitGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
    hitGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(hitGroup);

    // 3. 创建光线追踪管线
    VkRayTracingPipelineCreateInfoKHR pipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    pipelineInfo.pGroups = shaderGroups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 6; // 最大递归深度
    pipelineInfo.layout = m_pipelineLayout;

    if (vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_rayTracingPipeline) != VK_SUCCESS) {
        throw runtime_error("Failed to create ray tracing pipeline!");
    }
}

void VulkanRenderer::createSBT() {
    // 获取正确的对齐值
    uint32_t baseAlignment = m_rayTracingProperties.shaderGroupBaseAlignment;
    uint32_t handleSize = m_rayTracingProperties.shaderGroupHandleSize;

    // 区域大小必须是对齐值的倍数
    m_sbt.rgenRegion.size = alignUp(handleSize, baseAlignment);
    m_sbt.missRegion.size = alignUp(handleSize, baseAlignment) * 2; // 两个miss shader
    m_sbt.hitRegion.size = alignUp(handleSize, baseAlignment); // 两个hit shader

    // 步长设置为相同大小（每组一个着色器）
    m_sbt.rgenRegion.stride = m_sbt.rgenRegion.size;
    m_sbt.missRegion.stride = m_sbt.missRegion.size / 2;
    m_sbt.hitRegion.stride = m_sbt.hitRegion.size;

    // 计算总大小
    VkDeviceSize sbtSize =
        m_sbt.rgenRegion.size +
        m_sbt.missRegion.size +
        m_sbt.hitRegion.size;

    // 创建SBT缓冲区（确保设备地址对齐）
    VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = sbtSize;
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_sbt.buffer) != VK_SUCCESS) {
        throw runtime_error("Failed to create SBT buffer!");
    }

    // 获取内存需求（考虑对齐）
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(m_device, m_sbt.buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = ResourceManager::findMemoryType(
        m_physicalDevice, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // 添加设备地址标志
    VkMemoryAllocateFlagsInfo flagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocInfo.pNext = &flagsInfo;

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_sbt.memory) != VK_SUCCESS) {
        throw runtime_error("Failed to allocate SBT memory!");
    }
    vkBindBufferMemory(m_device, m_sbt.buffer, m_sbt.memory, 0);

    // 获取设备地址
    m_sbt.address = getBufferDeviceAddress(m_sbt.buffer);

    // 计算各区域地址（确保按baseAlignment对齐）
    m_sbt.rgenRegion.deviceAddress = m_sbt.address;
    m_sbt.missRegion.deviceAddress = m_sbt.address + m_sbt.rgenRegion.size;
    m_sbt.hitRegion.deviceAddress = m_sbt.address + m_sbt.rgenRegion.size + m_sbt.missRegion.size;

    // 验证地址对齐
    if ((m_sbt.rgenRegion.deviceAddress % baseAlignment) != 0 ||
        (m_sbt.missRegion.deviceAddress % baseAlignment) != 0 ||
        (m_sbt.hitRegion.deviceAddress % baseAlignment) != 0) {
        throw runtime_error("SBT region address not aligned!");
    }

    // 获取着色器组句柄
    std::vector<uint8_t> handles(handleSize * 4);
    if (vkGetRayTracingShaderGroupHandlesKHR(
        m_device, m_rayTracingPipeline, 0, 4, handles.size(), handles.data()) != VK_SUCCESS) {
        throw runtime_error("Failed to get shader group handles!");
    }

    // 映射内存并写入句柄
    void* mapped;
    vkMapMemory(m_device, m_sbt.memory, 0, sbtSize, 0, &mapped);
    uint8_t* pData = static_cast<uint8_t*>(mapped);

    // 复制数据到对齐区域
    memcpy(pData, handles.data() + 0 * handleSize, handleSize); // Raygen
    memcpy(pData + m_sbt.rgenRegion.size, handles.data() + 1 * handleSize, handleSize); // Miss
    memcpy(pData + m_sbt.rgenRegion.size + m_sbt.rgenRegion.stride, handles.data() + 2 * handleSize, handleSize); // shadowMiss
    memcpy(pData + m_sbt.rgenRegion.size + m_sbt.missRegion.size, handles.data() + 3 * handleSize, handleSize); // Hit
    vkUnmapMemory(m_device, m_sbt.memory);
}

void VulkanRenderer::createAccelerationStructures() {
    // 创建BLAS列表
    std::vector<VkAccelerationStructureKHR> blasList;
    for (auto& geom : m_scene.geometries) {
        createBLAS(geom);
        if (geom.blas == VK_NULL_HANDLE) {
            throw std::runtime_error("BLAS creation failed!");
        }
        blasList.push_back(geom.blas);
    }
    glm::vec3 corners[6] = {
        glm::vec3(0.4f, -1.5f, 5.5f), 
        glm::vec3(-0.5f, 1.f, 6.f),  
        glm::vec3(2.0f, -1.f, 4.5f),  
        glm::vec3(-0.9f, 0.2f, 1.2f),    
        glm::vec3(0.f, -0.5f, 2.5f),
        glm::vec3(1.5f, 3.05f, 2.5f)
    };

    // 获取模型当前位置 ,钻石 雕像 兔子 龙 棋盘格 logo
    glm::vec3 modelPositions[6] = {
        glm::vec3(0.000000f, 0.000000f, 0.000000f),
        glm::vec3(0.044104f, 0.010394f, -0.082350f),
        glm::vec3(-0.138641f, 0.837148f, 0.069242f),
        glm::vec3(1.612950f, 2.237735f, 0.915691f),
        glm::vec3(0.f, 0.f, 0.f),
        glm::vec3(0.f, 0.f, 0.f)

    };
    glm::vec3 rotations[6] = {
        glm::vec3(-50.0f, -27.0f, -25.0f),    // 
        glm::vec3(0.0f, 180.0f, 0.0f),   // 
        glm::vec3(0.0f, 180.0f, 0.0f),  // 
        glm::vec3(0.0f, -70.0f, 00.0f),   // 
        glm::vec3(0.0f, 0.0f, 0.0f),   // 
        glm::vec3(90.0f, 0.0f, 0.0f)   // 
    };

    // 修改后的缩放值
    glm::vec3 scales[6] = {
        glm::vec3(1.f, 1.f, 1.f),
        glm::vec3(-3.0f, 3.0f, 3.0f),
        glm::vec3(-1.2f, 1.2f, 1.2f),
        glm::vec3(-0.6f, 0.6f, 0.6f),
        glm::vec3(1.0f),
        glm::vec3(0.7f)
    };
    // 创建TLAS实例
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    uint32_t instanceId = 0;
    int i = 0;
    glm::vec3 offset(0);
    for (auto& geom : m_scene.geometries) {
        VkAccelerationStructureInstanceKHR instance{};
        // 计算平移向量（目标角 - 当前位置）
        glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), scales[i]);
        glm::mat4 rotateMat = glm::rotate(glm::mat4(1.0f), glm::radians(rotations[i].x), glm::vec3(1, 0, 0));
        rotateMat = glm::rotate(rotateMat, glm::radians(rotations[i].y), glm::vec3(0, 1, 0));
        rotateMat = glm::rotate(rotateMat, glm::radians(rotations[i].z), glm::vec3(0, 0, 1));
        glm::vec3 translation = corners[i] - modelPositions[i];

        // 创建平移矩阵
        glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) * scaleMat * rotateMat;

        instance.transform = toTransformMatrixKHR(transform); // 单位矩阵
        instance.instanceCustomIndex = instanceId++;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = 0;
        //这里要用m_tlas还是geom.tlas??
        instance.accelerationStructureReference = getAccelerationStructureDeviceAddress(blasList[i]);
        instances.push_back(instance);
        i++;
    }

    createTLAS(instances);
}

void VulkanRenderer::createBLAS(Geometry& geom) {
    /*
    VkTransformMatrixKHR transform_matrix = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f };
    VkBuffer tmBuffer = VK_NULL_HANDLE;
    VkDeviceMemory tmMemory = VK_NULL_HANDLE;
    createBuffer(
        &transform_matrix,
        sizeof(transform_matrix),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        tmBuffer,
        tmMemory
    );
    */
    // 1. 设置几何数据
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = getBufferDeviceAddress(geom.vertexBuffer);
    triangles.vertexStride = sizeof(Vertex);
    triangles.maxVertex = static_cast<uint32_t>(geom.vertices.size() - 1);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = getBufferDeviceAddress(geom.indexBuffer);
    //triangles.transformData.deviceAddress = getBufferDeviceAddress(tmBuffer);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;

    // 2. 创建构建信息
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // 3. 查询构建所需大小
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    uint32_t primitiveCount = static_cast<uint32_t>(geom.indices.size() / 3);
    vkGetAccelerationStructureBuildSizesKHR(
        m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo
    );

    // 4. 创建BLAS
    createAccelerationStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        sizeInfo.accelerationStructureSize,
        geom.blas,
        geom.blasBuffer,
        geom.blasMemory
    );

    // 5. 获取scratch buffer
    VkBuffer scratchBuffer = VK_NULL_HANDLE;
    VkDeviceMemory scratchMemory = VK_NULL_HANDLE;

    VkBufferCreateInfo scratchBufferInfo{};
    scratchBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    scratchBufferInfo.size = sizeInfo.buildScratchSize;
    scratchBufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (vkCreateBuffer(m_device, &scratchBufferInfo, nullptr, &scratchBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create scratch buffer!");
    }

    VkMemoryRequirements scratchMemReqs;
    vkGetBufferMemoryRequirements(m_device, scratchBuffer, &scratchMemReqs);

    VkMemoryAllocateInfo scratchAllocInfo{};
    scratchAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    scratchAllocInfo.allocationSize = scratchMemReqs.size;
    scratchAllocInfo.memoryTypeIndex = ResourceManager::findMemoryType(
        m_physicalDevice, scratchMemReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkMemoryAllocateFlagsInfo allocFlagsInfo{};
    allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    scratchAllocInfo.pNext = &allocFlagsInfo;

    if (vkAllocateMemory(m_device, &scratchAllocInfo, nullptr, &scratchMemory) != VK_SUCCESS) {
        vkDestroyBuffer(m_device, scratchBuffer, nullptr);
        throw std::runtime_error("Failed to allocate scratch memory!");
    }

    vkBindBufferMemory(m_device, scratchBuffer, scratchMemory, 0);
    // actual building ac
    VkAccelerationStructureBuildGeometryInfoKHR acceleration_build_geometry_info{};
    acceleration_build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    acceleration_build_geometry_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    acceleration_build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    acceleration_build_geometry_info.dstAccelerationStructure = geom.blas;
    acceleration_build_geometry_info.geometryCount = 1;
    acceleration_build_geometry_info.pGeometries = &geometry;
    acceleration_build_geometry_info.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);

    // 6. 构建范围信息
    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
    buildRangeInfo.primitiveCount = primitiveCount;
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.firstVertex = 0;
    buildRangeInfo.transformOffset = 0;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRangeInfos = { &buildRangeInfo };

    // 7. 构建加速结构（在命令缓冲区中执行）
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR(
        commandBuffer,
        1, &acceleration_build_geometry_info,
        buildRangeInfos.data()
    );
    endSingleTimeCommands(commandBuffer);

    vkDestroyBuffer(m_device, scratchBuffer, nullptr);
    vkFreeMemory(m_device, scratchMemory, nullptr);
}

void VulkanRenderer::createTLAS(std::vector<VkAccelerationStructureInstanceKHR>& instances) {
    // 1. 创建实例缓冲区
    createBuffer(
        instances.data(),
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_tlasInstancesBuffer,
        m_tlasInstancesMemory
    );

    // 2. 设置几何数据
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.data.deviceAddress = getBufferDeviceAddress(m_tlasInstancesBuffer);
    instancesData.arrayOfPointers = VK_FALSE;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances = instancesData;

    // 3. 创建构建信息
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // 4. 查询构建所需大小
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    uint32_t primitiveCount = static_cast<uint32_t>(instances.size());
    vkGetAccelerationStructureBuildSizesKHR(
        m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo
    );


    // 5. 创建TLAS
    createAccelerationStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        sizeInfo.accelerationStructureSize,
        m_scene.tlas,
        m_scene.tlasBuffer,
        m_scene.tlasMemory
    );

    // 6. 设置构建信息的目标
    VkBuffer scratchBuffer = VK_NULL_HANDLE;
    VkDeviceMemory scratchMemory = VK_NULL_HANDLE;

    VkBufferCreateInfo scratchBufferInfo{};
    scratchBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    scratchBufferInfo.size = sizeInfo.buildScratchSize;
    scratchBufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (vkCreateBuffer(m_device, &scratchBufferInfo, nullptr, &scratchBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create scratch buffer!");
    }

    VkMemoryRequirements scratchMemReqs;
    vkGetBufferMemoryRequirements(m_device, scratchBuffer, &scratchMemReqs);

    VkMemoryAllocateInfo scratchAllocInfo{};
    scratchAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    scratchAllocInfo.allocationSize = scratchMemReqs.size;
    scratchAllocInfo.memoryTypeIndex = ResourceManager::findMemoryType(
        m_physicalDevice, scratchMemReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkMemoryAllocateFlagsInfo allocFlagsInfo{};
    allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    scratchAllocInfo.pNext = &allocFlagsInfo;

    if (vkAllocateMemory(m_device, &scratchAllocInfo, nullptr, &scratchMemory) != VK_SUCCESS) {
        vkDestroyBuffer(m_device, scratchBuffer, nullptr);
        throw std::runtime_error("Failed to allocate scratch memory!");
    }

    vkBindBufferMemory(m_device, scratchBuffer, scratchMemory, 0);
    VkAccelerationStructureBuildGeometryInfoKHR acceleration_build_geometry_info{};
    acceleration_build_geometry_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    acceleration_build_geometry_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    acceleration_build_geometry_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    acceleration_build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    acceleration_build_geometry_info.dstAccelerationStructure = m_scene.tlas;
    acceleration_build_geometry_info.geometryCount = 1;
    acceleration_build_geometry_info.pGeometries = &geometry;
    acceleration_build_geometry_info.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);

    // 7. 构建范围信息
    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
    buildRangeInfo.primitiveCount = primitiveCount;
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.firstVertex = 0;
    buildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRangeInfos = { &buildRangeInfo };

    // 8. 构建加速结构（在命令缓冲区中执行）
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    vkCmdBuildAccelerationStructuresKHR(
        commandBuffer,
        1, &acceleration_build_geometry_info,
        buildRangeInfos.data()
    );
    endSingleTimeCommands(commandBuffer);

    vkDestroyBuffer(m_device, scratchBuffer, nullptr);
    vkFreeMemory(m_device, scratchMemory, nullptr);
}

// 辅助函数：转换矩阵格式
VkTransformMatrixKHR VulkanRenderer::toTransformMatrixKHR(const glm::mat4& m) {
    VkTransformMatrixKHR transform;
    // glm是列主序，Vulkan的变换矩阵需要行主序
    transform.matrix[0][0] = m[0][0];
    transform.matrix[0][1] = m[1][0];
    transform.matrix[0][2] = m[2][0];
    transform.matrix[0][3] = m[3][0];

    transform.matrix[1][0] = m[0][1];
    transform.matrix[1][1] = m[1][1];
    transform.matrix[1][2] = m[2][1];
    transform.matrix[1][3] = m[3][1];

    transform.matrix[2][0] = m[0][2];
    transform.matrix[2][1] = m[1][2];
    transform.matrix[2][2] = m[2][2];
    transform.matrix[2][3] = m[3][2];
    return transform;
}

// 辅助函数：执行单次命令
VkCommandBuffer VulkanRenderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanRenderer::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQueue);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
}

void VulkanRenderer::initSyncObjects() {
    m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; //初始为已发出信号的状态

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]);
        vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]);
        vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]);
    }
}

void VulkanRenderer::drawFrame() {
    // 0. 等待前一帧完成
    vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

    // 1. 获取交换链图像
    uint32_t imageIndex;
    vkAcquireNextImageKHR(
        m_device, m_swapchain, UINT64_MAX,
        m_imageAvailableSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex
    );

    vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);

    // 2. 创建命令缓冲区
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

    // 3. 开始录制命令
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // 4. 转换输出图像布局 UNDEFINED->GENERAL
    VkImageMemoryBarrier barrier1{};
    barrier1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier1.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier1.image = m_outputImage;
    barrier1.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    barrier1.srcAccessMask = 0;
    barrier1.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier1.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier1.subresourceRange.baseMipLevel = 0;
    barrier1.subresourceRange.levelCount = 1;
    barrier1.subresourceRange.baseArrayLayer = 0;
    barrier1.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier1
    );

    // 5. 绑定管线
    vkCmdBindPipeline(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        m_rayTracingPipeline
    );

    // 6. 绑定描述符集
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr
    );

    // 7. 调度光线追踪
    VkStridedDeviceAddressRegionKHR callableRegion = {};
    vkCmdTraceRaysKHR(
        commandBuffer,
        &m_sbt.rgenRegion,
        &m_sbt.missRegion,
        &m_sbt.hitRegion,
        &callableRegion, // callable
        SAMPLE_WIDTH, SAMPLE_HEIGHT, 1
    );

    // 7.5 确保光线追踪完成后再操作其他资源
    VkMemoryBarrier rtCompleteBarrier{};
    rtCompleteBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    rtCompleteBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    rtCompleteBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, // 等待RT完成
        VK_PIPELINE_STAGE_TRANSFER_BIT,               // 后续传输操作
        0,
        1, &rtCompleteBarrier,
        0, nullptr,
        0, nullptr
    );

    //  swapchain布局 UNDEFINED->TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier swapchainToDstBarrier{};
    swapchainToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapchainToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapchainToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapchainToDstBarrier.image = m_swapchainImages[imageIndex];
    swapchainToDstBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    swapchainToDstBarrier.srcAccessMask = 0;
    swapchainToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // 任何早期操作
        VK_PIPELINE_STAGE_TRANSFER_BIT,   // 传输阶段
        0, 0, nullptr, 0, nullptr,
        1, &swapchainToDstBarrier
    );

    // 8. 复制结果到交换链图像
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.extent = { SAMPLE_WIDTH, SAMPLE_HEIGHT, 1 };

    vkCmdCopyImage(
        commandBuffer,
        m_outputImage, VK_IMAGE_LAYOUT_GENERAL,
        m_swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copyRegion
    );

    // 8.5 swapchain布局 TRANSFER_DST_OPTIMAL->PRESENT_SRC_KHR
    VkImageMemoryBarrier barrier2{};
    barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier2.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier2.dstAccessMask = 0;
    barrier2.image = m_swapchainImages[imageIndex];
    barrier2.subresourceRange = barrier1.subresourceRange;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier2
    );

    // 开始ImGui帧
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 创建UI界面
    ImGui::Begin("Camera Control");

    // 显示当前相机位置和目标
    ImGui::Text("Position: %.2f, %.2f, %.2f",
        m_cameraPosition.x, m_cameraPosition.y, m_cameraPosition.z);
    ImGui::Text("Target: %.2f, %.2f, %.2f",
        m_cameraTarget.x, m_cameraTarget.y, m_cameraTarget.z);

    // 添加重置按钮
    if (ImGui::Button("Reset Camera")) {
        // 重置相机到初始位置
        initArcballCamera();
        m_cameraPosition = m_arcballCamera->getPosition();
        m_cameraTarget = m_arcballCamera->getTarget();
        updateCameraBuffer();
    }

    ImGui::End();

    ImGui::Render();

    VkRenderPassBeginInfo uiPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    uiPassInfo.renderPass = m_imguiRenderPass;
    uiPassInfo.framebuffer = m_imguiFramebuffers[imageIndex];
    uiPassInfo.renderArea.offset = { 0, 0 };
    uiPassInfo.renderArea.extent = m_swapchainExtent;
    uiPassInfo.clearValueCount = 0; // loadOp = LOAD，保留已有内容

    vkCmdBeginRenderPass(commandBuffer, &uiPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // 调用你写好的 renderImGui，将 ImGui 绘制命令录入 commandBuffer
    renderImGui(commandBuffer);

    vkCmdEndRenderPass(commandBuffer);


    // 9. 结束命令录制
    vkEndCommandBuffer(commandBuffer);

    // 10. 提交命令
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &m_imageAvailableSemaphores[m_currentFrame];
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_renderFinishedSemaphores[m_currentFrame];

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]);
    vkQueueWaitIdle(m_graphicsQueue);

    // 11. 呈现图像
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[m_currentFrame];

    vkQueuePresentKHR(m_graphicsQueue, &presentInfo);

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    // 12. 清理命令缓冲区
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
}
// 通用缓冲区创建函数
void VulkanRenderer::createBuffer(
    void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory
) {
    // 1. 创建缓冲区
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        throw runtime_error("Failed to create buffer: " + to_string(result));
    }

    // 2. 获取内存需求
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

    // 3. 准备内存分配信息
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    // 4. 为需要设备地址的缓冲区添加特殊标志
    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        allocInfo.pNext = &flagsInfo;
    }

    // 5. 选择内存类型
    allocInfo.memoryTypeIndex = ResourceManager::findMemoryType(
        m_physicalDevice,
        memRequirements.memoryTypeBits,
        properties
    );

    // 6. 分配内存
    result = vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(m_device, buffer, nullptr);
        throw runtime_error("Failed to allocate buffer memory: " + to_string(result));
    }

    // 7. 绑定内存
    result = vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(m_device, buffer, nullptr);
        vkFreeMemory(m_device, bufferMemory, nullptr);
        throw runtime_error("Failed to bind buffer memory: " + to_string(result));
    }

    // 8. 复制数据（如果有）
    if (data != nullptr) {
        void* mappedData;
        result = vkMapMemory(m_device, bufferMemory, 0, size, 0, &mappedData);
        if (result != VK_SUCCESS) {
            vkDestroyBuffer(m_device, buffer, nullptr);
            vkFreeMemory(m_device, bufferMemory, nullptr);
            throw runtime_error("Failed to map buffer memory: " + to_string(result));
        }

        memcpy(mappedData, data, (size_t)size);
        vkUnmapMemory(m_device, bufferMemory);
    }
}

// 辅助函数：获取缓冲区设备地址
VkDeviceAddress VulkanRenderer::getBufferDeviceAddress(VkBuffer buffer) {
    if (buffer == VK_NULL_HANDLE) {
        throw runtime_error("Attempted to get device address for null buffer!");
    }

    if (!m_vkGetBufferDeviceAddressKHR) {
        throw runtime_error("vkGetBufferDeviceAddressKHR function not loaded!");
    }

    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;

    return m_vkGetBufferDeviceAddressKHR(m_device, &info);
}

// 辅助函数：获取加速结构设备地址
VkDeviceAddress VulkanRenderer::getAccelerationStructureDeviceAddress(VkAccelerationStructureKHR accelerationStructure) {
    if (!m_vkGetAccelerationStructureDeviceAddressKHR) {
        throw runtime_error("vkGetAccelerationStructureDeviceAddressKHR function not loaded!");
    }

    VkAccelerationStructureDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    info.accelerationStructure = accelerationStructure;

    return m_vkGetAccelerationStructureDeviceAddressKHR(m_device, &info);
}

// 创建加速结构的通用函数
void VulkanRenderer::createAccelerationStructure(
    VkAccelerationStructureTypeKHR type,
    VkDeviceSize size,
    VkAccelerationStructureKHR& accelerationStructure,
    VkBuffer& buffer,
    VkDeviceMemory& memory
) {
    // 创建缓冲区
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create acceleration structure buffer!");
    }

    // 分配内存
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = ResourceManager::findMemoryType(
        m_physicalDevice,
        memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkMemoryAllocateFlagsInfo allocFlagsInfo{};
    allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocInfo.pNext = &allocFlagsInfo;

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate acceleration structure memory!");
    }

    vkBindBufferMemory(m_device, buffer, memory, 0);

    // 创建加速结构对象
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = buffer;
    createInfo.size = size;
    createInfo.type = type;

    if (vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &accelerationStructure) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create acceleration structure!");
    }
}

// 命令池创建函数
void VulkanRenderer::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_queueFamilyIndex;

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        throw runtime_error("Failed to create command pool!");
    }
}

// 自定义 glm::vec3 的哈希函数
namespace std {
    template <>
    struct hash<glm::vec3> {
        size_t operator()(const glm::vec3& v) const {
            // 组合三个分量的哈希值
            return ((hash<float>()(v.x) ^
                (hash<float>()(v.y) << 1)) >> 1) ^
                (hash<float>()(v.z) << 1);
        }
    };
}

void VulkanRenderer::loadModel(const std::string& path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) {
        throw std::runtime_error(warn + err);
    }

    m_modelMin = glm::vec3(FLT_MAX);
    m_modelMax = glm::vec3(-FLT_MAX);


    // 第一步：收集所有顶点并映射唯一位置
    unordered_map<glm::vec3, uint32_t> uniqueVertices;
    std::vector<Vertex> tempVertices;
    std::vector<uint32_t> indices;

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};
            vertex.position = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            // 更新包围盒
            m_modelMin = glm::min(m_modelMin, vertex.position);
            m_modelMax = glm::max(m_modelMax, vertex.position);

            // 检查是否已存在相同位置的顶点
            if (uniqueVertices.count(vertex.position)) {
                indices.push_back(uniqueVertices[vertex.position]);
            }
            else {
                uint32_t newIndex = static_cast<uint32_t>(tempVertices.size());
                uniqueVertices[vertex.position] = newIndex;
                indices.push_back(newIndex);
                tempVertices.push_back(vertex);
            }
        }
    }

    // 第二步：计算面法向并累加到顶点
    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];

        Vertex& v0 = tempVertices[i0];
        Vertex& v1 = tempVertices[i1];
        Vertex& v2 = tempVertices[i2];

        glm::vec3 edge1 = v1.position - v0.position;
        glm::vec3 edge2 = v2.position - v0.position;
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

        // 累加到顶点法向
        v0.normal += normal;
        v1.normal += normal;
        v2.normal += normal;
    }

    // 第三步：归一化顶点法向
    for (auto& vertex : tempVertices) {
        vertex.normal = glm::normalize(vertex.normal);
    }

    // 第四步：创建几何体
    Geometry geom;
    geom.vertices = std::move(tempVertices);
    geom.indices = std::move(indices);

    // 计算模型中心点
    m_modelCenter = (m_modelMin + m_modelMax) * 0.5f;
    printf("Model center: %.2f, %.2f, %.2f\n", m_modelCenter.x, m_modelCenter.y, m_modelCenter.z);

    // 创建GPU缓冲区...
    createBuffer(
        geom.vertices.data(),
        geom.vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        geom.vertexBuffer,
        geom.vertexBufferMemory
    );

    createBuffer(
        geom.indices.data(),
        geom.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        geom.indexBuffer,
        geom.indexBufferMemory
    );

    m_scene.geometries.push_back(std::move(geom));
}

void VulkanRenderer::createCameraBuffer() {
    // 只创建缓冲区，不设置数据
    createBuffer(
        nullptr, // 数据指针设为nullptr
        sizeof(CameraParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_cameraBuffer,
        m_cameraBufferMemory
    );

    // 初始更新一次
    updateCameraBuffer();
}

void VulkanRenderer::updateCameraBuffer() {
    CameraParams camera;
    // 使用成员变量计算视图矩阵
    glm::mat4 view = glm::lookAt(
        m_cameraPosition,  // 相机位置
        m_cameraTarget,    // 观察目标
        m_cameraUp // 上向量
    );

    // 计算投影矩阵
    float aspect = static_cast<float>(SAMPLE_WIDTH) / SAMPLE_HEIGHT;
    glm::mat4 proj = glm::perspective(
        glm::radians(m_cameraFOV), // FOV
        aspect,               // 宽高比
        0.1f,                // 近平面
        100.0f               // 远平面
    );

    // 使用逆矩阵
    camera.view_inverse = glm::inverse(view);
    camera.proj_inverse = glm::inverse(proj);

    // 映射内存并更新数据
    void* data;
    VkResult result = vkMapMemory(m_device, m_cameraBufferMemory, 0, sizeof(CameraParams), 0, &data);
    if (result == VK_SUCCESS) {
        memcpy(data, &camera, sizeof(CameraParams));
        vkUnmapMemory(m_device, m_cameraBufferMemory);
    }
    else {
        // 处理错误
        std::cerr << "Failed to map camera buffer memory: " << result << std::endl;
    }
}

void VulkanRenderer::image_layout_transition(VkCommandBuffer command_buffer,
    VkImage                        image,
    VkPipelineStageFlags           src_stage_mask,
    VkPipelineStageFlags           dst_stage_mask,
    VkAccessFlags                  src_access_mask,
    VkAccessFlags                  dst_access_mask,
    VkImageLayout                  old_layout,
    VkImageLayout                  new_layout,
    VkImageSubresourceRange const& subresource_range)
{
    // Create an image barrier object
    VkImageMemoryBarrier image_memory_barrier{};
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.srcAccessMask = src_access_mask;
    image_memory_barrier.dstAccessMask = dst_access_mask;
    image_memory_barrier.oldLayout = old_layout;
    image_memory_barrier.newLayout = new_layout;
    image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.image = image;
    image_memory_barrier.subresourceRange = subresource_range;

    // Put barrier inside setup command buffer
    vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask, 0, 0, nullptr, 0, nullptr, 1, &image_memory_barrier);
}


VkCommandBuffer VulkanRenderer::create_command_buffer(VkCommandBufferLevel level, bool begin) const
{
    if (m_commandPool == VK_NULL_HANDLE) {
        throw std::runtime_error("Don't have a commandpool yet!");
    }
    VkCommandBufferAllocateInfo cmd_buf_allocate_info{};
    cmd_buf_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_buf_allocate_info.commandPool = m_commandPool;
    cmd_buf_allocate_info.level = level;
    cmd_buf_allocate_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(m_device, &cmd_buf_allocate_info, &command_buffer);

    // 是否开启record command
    if (begin)
    {
        VkCommandBufferBeginInfo command_buffer_info{};
        command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(command_buffer, &command_buffer_info);
    }

    return command_buffer;
}

void VulkanRenderer::flush_command_buffer(VkCommandBuffer command_buffer, VkQueue queue, bool free, VkSemaphore signalSemaphore) const
{
    if (command_buffer == VK_NULL_HANDLE)
    {
        return;
    }

    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    if (signalSemaphore)
    {
        submit_info.pSignalSemaphores = &signalSemaphore;
        submit_info.signalSemaphoreCount = 1;
    }

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = 0;

    VkFence fence;
    vkCreateFence(m_device, &fence_info, nullptr, &fence);

    // Submit to the queue
    VkResult result = vkQueueSubmit(queue, 1, &submit_info, fence);
    // Wait for the fence to signal that command buffer has finished executing
    vkWaitForFences(m_device, 1, &fence, VK_TRUE, 1000000);

    vkDestroyFence(m_device, fence, nullptr);

    if (m_commandPool && free)
    {
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &command_buffer);
    }
}

void VulkanRenderer::createLightBuffer() {

    // 创建光源缓冲区
    VkDeviceSize bufferSize = sizeof(Light) * m_lights.size();

    createBuffer(
        m_lights.data(),
        bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_lightBuffer,
        m_lightBufferMemory
    );
}

void VulkanRenderer::setupSceneLights() {
    // 清空现有光源
    m_lights.clear();
    
    // 添加主方向光
    Light sunLight{};
    sunLight.position = glm::vec4(-0.5f, -1.f, -.5f, 0.0f); // w=0 表示方向
    sunLight.color = glm::vec4(1.0f, 1.0f, 1.0f, .9f); // 白色, 强度1.0
    m_lights.push_back(sunLight);

    Light sunLight1{};
    sunLight1.position = glm::vec4(.0f, -1.0f, 1.f, 0.0f); // w=0 表示方向
    sunLight1.color = glm::vec4(1.0f, 1.0f, 1.0f, .5f); // 白色, 强度1.0
    m_lights.push_back(sunLight1);

    // 添加点光源
    Light pointLight{};
    pointLight.position = glm::vec4(-4.f, 4.f, 4.f, 1.0f); // w=0 表示方向
    pointLight.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.2f); // 白色, 强度1.0
    m_lights.push_back(pointLight);

    //添加聚光灯
    Light spotLight{};
    spotLight.position = glm::vec4(3.0f, 2.f, 2.0f, 2.0f); // w=1 表示位置
    spotLight.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.4f); // 橙黄色, 强度0.8
    spotLight.spotInnerAngle = 15.5f;
    spotLight.spotOuterAngle = 32.5f;
    spotLight.direction = glm::vec4(-1.f, -1.f, -1.f, 0.f);

    pointLight.range = 15.0f;
    m_lights.push_back(spotLight);
    // 创建光源缓冲区
    createLightBuffer();
}

// 初始化材质
void VulkanRenderer::setupMaterials() {
    m_materials.clear();

    // 0: 漫反射材质（灰色）
    Material diffuseMat0;
    diffuseMat0.albedo = glm::vec3(0.8f, 0.8f, 0.8f);
    diffuseMat0.emission = glm::vec3(0.0f);
    diffuseMat0.roughness = 1.0f;    // 完全漫反射
    diffuseMat0.metallic = 0.0f;
    diffuseMat0.ior = 1.0f;
    diffuseMat0.transparency = 0.0f;
    diffuseMat0.materialType = static_cast<uint32_t>(MaterialType::DIFFUSE);
    m_materials.push_back(diffuseMat0);

    // 1: 另一个漫反射，可以复用或微调颜色
    Material diffuseMat1 = diffuseMat0;
    // 如果希望 ID=1 物体颜色不同，可修改 albedo：
    // diffuseMat1.albedo = glm::vec3(0.7f, 0.2f, 0.2f);
    m_materials.push_back(diffuseMat1);

    // 2: 镜面反射材质（完全镜面/金属）
    Material specMat;
    specMat.albedo = glm::vec3(1.0f);  // 镜面反射一般用白色基底
    specMat.emission = glm::vec3(0.0f);
    specMat.roughness = 0.0f;         // 完全光滑镜面
    specMat.metallic = 1.0f;          // 设为金属以便 PBR 反射
    specMat.ior = 1.0f;               // 对完全镜面来说无折射，IOR=1
    specMat.transparency = 0.0f;
    specMat.materialType = static_cast<uint32_t>(MaterialType::SPECULAR);
    m_materials.push_back(specMat);

    // 3: 折射玻璃材质
    Material glassMat;
    glassMat.albedo = glm::vec3(1.0f);  // 玻璃一般不吸收或轻微吸收，可用白色
    glassMat.emission = glm::vec3(0.0f);
    glassMat.roughness = 0.0f;         // 理想透明可设为 0；若想粗糙可改非 0
    glassMat.metallic = 0.0f;
    glassMat.ior = 1.5f;               // 玻璃典型 IOR
    glassMat.transparency = 0.95f;     // 高透明度
    glassMat.materialType = static_cast<uint32_t>(MaterialType::DIELECTRIC);
    m_materials.push_back(glassMat);

    // 创建或更新 GPU 材质缓冲区
    createMaterialBuffer();
}


void VulkanRenderer::createMaterialBuffer() {
    createBuffer(
        m_materials.data(),
        m_materials.size() * sizeof(Material),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_materialBuffer,
        m_materialBufferMemory
    );
}

void VulkanRenderer::createEnvironmentMap(const std::string& path) {
    // 使用stb_image加载图片
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("Failed to load environment map!");
    }

    // 创建暂存缓冲区并复制像素数据
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    createBuffer(pixels, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingMemory);

    stbi_image_free(pixels);

    // 创建环境贴图图像
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = texWidth;
    imageInfo.extent.height = texHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(m_device, &imageInfo, nullptr, &m_envImage) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create environment map image!");
    }

    // 分配内存并绑定图像
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, m_envImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = ResourceManager::findMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_envImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate environment map memory!");
    }

    vkBindImageMemory(m_device, m_envImage, m_envImageMemory, 0);

    // 转换图像布局 UNDEFINED -> TRANSFER_DST_OPTIMAL
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_envImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // 复制缓冲区数据到图像
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 };

    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, m_envImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // 转换图像布局 TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);

    // 清理暂存资源
    vkDestroyBuffer(m_device, stagingBuffer, nullptr);
    vkFreeMemory(m_device, stagingMemory, nullptr);

    // 创建图像视图
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_envImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_device, &viewInfo, nullptr, &m_envImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create environment map image view!");
    }
    
    // 创建纹理采样器
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_envSampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create environment map sampler!");
    }
}

void VulkanRenderer::createChessboardPlane() {
    Geometry geom;
    geom.materialindex = 4;  // 使用棋盘格材质

    const float planeSize = 10.0f;
    const int gridSize = 10;
    const float cellSize = planeSize / gridSize;

    // 创建顶点数据
    for (int z = 0; z <= gridSize; z++) {
        for (int x = 0; x <= gridSize; x++) {
            Vertex vertex;
            vertex.position = glm::vec3(
                x * cellSize - planeSize / 2.0f,
                -1.7f,  // 放在场景底部作为地面
                (z * cellSize - planeSize / 2.0f) + 4.0f
            );
            vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex.texCoord = glm::vec2(
                static_cast<float>(x) / gridSize,
                static_cast<float>(z) / gridSize
            );
            geom.vertices.push_back(vertex);
        }
    }

    // 创建索引数据
    for (int z = 0; z < gridSize; z++) {
        for (int x = 0; x < gridSize; x++) {
            uint32_t base = z * (gridSize + 1) + x;
            geom.indices.push_back(base);
            geom.indices.push_back(base + 1);
            geom.indices.push_back(base + gridSize + 1);

            geom.indices.push_back(base + 1);
            geom.indices.push_back(base + gridSize + 2);
            geom.indices.push_back(base + gridSize + 1);
        }
    }

    // 创建顶点缓冲区
    createBuffer(
        geom.vertices.data(),
        geom.vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        geom.vertexBuffer,
        geom.vertexBufferMemory
    );

    // 创建索引缓冲区
    createBuffer(
        geom.indices.data(),
        geom.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        geom.indexBuffer,
        geom.indexBufferMemory
    );
    m_scene.geometries.push_back(std::move(geom));
}


void VulkanRenderer::initImGui() {
    // 1. 创建ImGui上下文
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // 2. 设置ImGui样式
    ImGui::StyleColorsDark();

    // 3. 初始化GLFW绑定
    ImGui_ImplGlfw_InitForVulkan(m_app->getWindow(), true);

    // 4. 创建ImGui描述符池
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
    pool_info.pPoolSizes = pool_sizes;

    if (vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_imguiDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui descriptor pool!");
    }

    // 5. 初始化Vulkan绑定
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = m_app->getInstance();
    init_info.PhysicalDevice = m_physicalDevice;
    init_info.Device = m_device;
    init_info.QueueFamily = m_queueFamilyIndex;
    init_info.Queue = m_graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = m_imguiDescriptorPool;
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(m_swapchainImages.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    init_info.CheckVkResultFn = nullptr;

    if (!ImGui_ImplVulkan_Init(&init_info, m_imguiRenderPass)) {
        throw std::runtime_error("Failed to initialize ImGui Vulkan backend!");
    }

    // 6. 上传字体
    VkCommandBuffer command_buffer = beginSingleTimeCommands();
    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
    endSingleTimeCommands(command_buffer);
    ImGui_ImplVulkan_DestroyFontUploadObjects();

    ImGui::GetIO().FontGlobalScale = 1.2f;
}

void VulkanRenderer::createImGuiRenderPass() {
    // 创建ImGui专用的渲染通道
    VkAttachmentDescription attachment = {};
    attachment.format = m_swapchainFormat;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // 保留现有内容
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment = {};
    color_attachment.attachment = 0;
    color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment;

    VkRenderPassCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;

    vkCreateRenderPass(m_device, &info, nullptr, &m_imguiRenderPass);

    // 为每个交换链图像创建帧缓冲区
    m_imguiFramebuffers.resize(m_swapchainImageViews.size());
    for (size_t i = 0; i < m_swapchainImageViews.size(); i++) {
        VkFramebufferCreateInfo fb_info = {};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = m_imguiRenderPass;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &m_swapchainImageViews[i];
        fb_info.width = m_swapchainExtent.width;
        fb_info.height = m_swapchainExtent.height;
        fb_info.layers = 1;

        vkCreateFramebuffer(m_device, &fb_info, nullptr, &m_imguiFramebuffers[i]);
    }
}

void VulkanRenderer::renderImGui(VkCommandBuffer commandBuffer) {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void VulkanRenderer::initArcballCamera() {
    // 使用场景中心作为焦点
    glm::vec3 sceneCenter = (m_modelMin + m_modelMax) * 0.5f;
    float initialDistance = glm::length(sceneCenter - m_cameraPosition);

    m_arcballCamera = std::make_unique<ArcballCamera>(
        sceneCenter,
        initialDistance,
        0.5f,   // min distance
        50.0f   // max distance
    );
}

void VulkanRenderer::handleMouseMove(double x, double y) {
    glm::vec2 currentPos = { static_cast<float>(x), static_cast<float>(y) };

    if (m_mouseLeftPressed) {
        // 旋转操作
        glm::vec2 delta = currentPos - m_lastMousePos;
        m_arcballCamera->rotate(delta);

        // 更新成员变量
        m_cameraPosition = m_arcballCamera->getPosition();
        m_cameraTarget = m_arcballCamera->getTarget();

        // 更新相机缓冲区
        updateCameraBuffer();
    }
    else if (m_mouseRightPressed) {
        // 平移操作
        glm::vec2 delta = currentPos - m_lastMousePos;
        m_arcballCamera->pan(delta);

        // 更新成员变量
        m_cameraPosition = m_arcballCamera->getPosition();
        m_cameraTarget = m_arcballCamera->getTarget();

        // 更新相机缓冲区
        updateCameraBuffer();
    }

    m_lastMousePos = currentPos;
}

void VulkanRenderer::handleMouseButton(int button, int action) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        m_mouseLeftPressed = (action == GLFW_PRESS);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        m_mouseRightPressed = (action == GLFW_PRESS);
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        m_mouseMiddlePressed = (action == GLFW_PRESS);
    }
}

void VulkanRenderer::handleScroll(double xoffset, double yoffset) {
    // 缩放操作
    m_arcballCamera->zoom(static_cast<float>(yoffset));

    // 更新成员变量
    m_cameraPosition = m_arcballCamera->getPosition();
    m_cameraTarget = m_arcballCamera->getTarget();

    // 更新相机缓冲区
    updateCameraBuffer();
}
// 回调函数实现
void RTVKApplication::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    auto app = static_cast<RTVKApplication*>(glfwGetWindowUserPointer(window));
    if (app && app->renderer) {
        app->renderer->handleMouseButton(button, action);
    }
}

void RTVKApplication::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto app = static_cast<RTVKApplication*>(glfwGetWindowUserPointer(window));
    if (app && app->renderer) {
        app->renderer->handleMouseMove(xpos, ypos);
    }
}

void RTVKApplication::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto app = static_cast<RTVKApplication*>(glfwGetWindowUserPointer(window));
    if (app && app->renderer) {
        app->renderer->handleScroll(xoffset, yoffset);
    }
}