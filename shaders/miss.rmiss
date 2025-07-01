#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable  // 启用动态索引支持

layout(location = 0) rayPayloadInEXT vec4 hitValue;

// 环境贴图采样器 - 绑定描述符位置与上面一致
layout(set = 0, binding = 7) uniform sampler2D envMap;

void main() {
    // 获取光线方向并归一化
    vec3 rayDirection = normalize(gl_WorldRayDirectionEXT);
    
    // 将方向转换为球面UV坐标
    // 首先计算方位角（azimuth）和俯仰角（elevation）
    float theta = acos(rayDirection.y); // [0, π]
    float phi = atan(rayDirection.z, rayDirection.x); // [-π, π]
    
    // 标准化到[0, 1]范围
    float u = (phi + 3.1415926535) / (2.0 * 3.1415926535); // φ从[−π, π]映射到[0, 1]
    float v = theta / 3.1415926535;       // θ从[0, π]映射到[0, 1]
    
    // 采样环境贴图
    vec3 envColor = texture(envMap, vec2(u, v)).rgb;
    
    // 输出环境贴图颜色
    hitValue.rgb = envColor;
    hitValue.a = 1.0;
}