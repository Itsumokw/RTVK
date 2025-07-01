#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(location = 0) rayPayloadInEXT vec3 payloadColor;
layout(location = 1) rayPayloadInEXT int payloadDepth;
hitAttributeEXT vec2 attribs;

// 光源结构体
struct Light {
    vec4 position;    // w=0: directional, w=1: point/spot
    vec4 color;        // RGB + intensity (A)
    uint type;         // LightType: 0=directional, 1=point, 2=spot
    float range;       // Point/spot light range
    float spotOuterAngle;
    float spotInnerAngle;
    float padding;
};

layout(set = 0, binding = 4) buffer LightBuffer {
    Light lights[];
} lightBuffer;

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
};

// 材质结构
struct Material {
    vec3 albedo;
    vec3 emission;
    float ior;
    uint materialType; // 0=diffuse, 1=specular, 2=refractive
};

const int MAX_BOUNCES = 3;
const float EPSILON = 0.001;

// 光线结构
struct Ray {
    vec3 origin;
    vec3 direction;
};

// 统一缓冲区绑定
layout(set = 0, binding = 2) buffer VertexBuffers {
    Vertex vertices[];
} vertexBuffer[];

layout(set = 0, binding = 5) buffer IndicesBuffer { 
    uint indices[];
} indicesBuffer[];

// 递归光线追踪函数
vec3 traceRay(Ray ray, int depth);

// 计算镜面反射方向
vec3 reflectDirection(vec3 incident, vec3 normal) {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// 计算折射方向（返回 true 如果发生折射）
bool refractDirection(vec3 incident, vec3 normal, float ior, out vec3 refractedDir) {
    float cosi = clamp(dot(incident, normal), -1.0, 1.0);
    float etai = 1.0, etat = ior;
    vec3 n = normal;
    
    if (cosi < 0.0) { // 从外部进入
        cosi = -cosi;
    } else { // 从内部出来
        float temp = etai;
        etai = etat;
        etat = temp;
        n = -normal;
    }
    
    float eta = etai / etat;
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    
    if (k < 0.0) return false; // 全反射
    
    refractedDir = normalize(eta * incident + (eta * cosi - sqrt(k)) * n);
    return true;
}

// 计算菲涅尔反射系数
float fresnel(vec3 incident, vec3 normal, float ior) {
    float cosi = clamp(dot(incident, normal), -1.0, 1.0);
    float etai = 1.0, etat = ior;
    if (cosi > 0.0) {
        float temp = etai;
        etai = etat;
        etat = temp;
        cosi = -cosi;
    }
    
    float sint = etai / etat * sqrt(max(0.0, 1.0 - cosi * cosi));
    if (sint >= 1.0) return 1.0; // 全反射
    
    float cost = sqrt(max(0.0, 1.0 - sint * sint));
    cosi = abs(cosi);
    float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    return (Rs * Rs + Rp * Rp) / 2.0;
}

// 硬编码材质数据
Material getMaterial(uint instanceIndex) {
    Material mat;
    
    // 根据实例ID分配材质
    if (instanceIndex == 3 || instanceIndex == 2) {
        // 漫反射材质
        mat.albedo = vec3(0.8, 0.8, 0.8);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 0u;
    } else if (instanceIndex == 1) {
        // 镜面反射材质
        mat.albedo = vec3(0.95, 0.95, 0.95);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 1u;
    } else if (instanceIndex == 0) {
        // 折射材质（玻璃）
        mat.albedo = vec3(1.0, 1.0, 1.0); // 白色
        mat.emission = vec3(0.0);
        mat.ior = 1.5; // 玻璃折射率
        mat.materialType = 2u;
    } else {
        // 默认材质（漫反射）
        mat.albedo = vec3(0.5, 0.5, 0.7);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 0u;
    }
    
    return mat;
}

// 计算光源照明贡献
vec3 calculateLightContribution(vec3 worldPos, vec3 N, Material mat) {
    vec3 lighting = vec3(0.0);
    uint lightCount = lightBuffer.lights.length();
    
    for (uint i = 0; i < lightCount; i++) {
        Light light = lightBuffer.lights[i];
        
        // 计算光照方向
        vec3 lightDir;
        if (light.type == 0u) { // 平行光
            lightDir = normalize(-light.position.xyz);
        } else { // 点光源/聚光灯
            vec3 lightVec = light.position.xyz - worldPos;
            lightDir = normalize(lightVec);
        }
        
        // 计算衰减
        float attenuation = 1.0;
        if (light.type == 1u || light.type == 2u) { // 点光源/聚光灯
            float distance = length(light.position.xyz - worldPos);
            attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
            attenuation *= light.color.a; // 乘以光源强度
            
            // 聚光灯衰减因子
            if (light.type == 2u) { 
                float cosAngle = dot(lightDir, normalize(light.position.xyz));
                float innerCone = cos(light.spotInnerAngle);
                float outerCone = cos(light.spotOuterAngle);
                float spotFactor = smoothstep(outerCone, innerCone, cosAngle);
                attenuation *= spotFactor;
            }
        }
        
        // 计算漫反射贡献
        float NdotL = max(dot(N, lightDir), 0.0);
        vec3 diffuse = mat.albedo * light.color.rgb * NdotL * attenuation;
        
        lighting += diffuse;
    }
    
    return lighting;
}

void main() {
    uint instanceIndex = gl_InstanceCustomIndexEXT;
    uint triIndex = gl_PrimitiveID;

    // 获取三角形数据
    uint index0 = indicesBuffer[instanceIndex].indices[triIndex * 3];
    uint index1 = indicesBuffer[instanceIndex].indices[triIndex * 3 + 1];
    uint index2 = indicesBuffer[instanceIndex].indices[triIndex * 3 + 2];
    
    Vertex v0 = vertexBuffer[instanceIndex].vertices[index0];
    Vertex v1 = vertexBuffer[instanceIndex].vertices[index1];
    Vertex v2 = vertexBuffer[instanceIndex].vertices[index2];
    
    // 重心坐标插值
    vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    
    // 插值法线和位置
    vec3 n0 = v0.normal;
    vec3 n1 = v1.normal;
    vec3 n2 = v2.normal;
    vec3 N = normalize(n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
    N = normalize(mat3(gl_ObjectToWorldEXT) * N); // 转换为世界空间
    
    // 世界坐标和视线方向
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 incident = normalize(gl_WorldRayDirectionEXT);
    
    // 获取硬编码材质
    Material mat = getMaterial(instanceIndex);
    vec3 color = vec3(0.0);
    
    // 检查最大递归深度
    if (payloadDepth >= MAX_BOUNCES) {
        payloadColor = mat.emission;
        return;
    }
    
    // 处理不同材质类型
    if (mat.materialType == 0u) { // 漫反射材质
        // 使用lightBuffer中的光源计算光照
        vec3 directLighting = calculateLightContribution(worldPos, N, mat);
        
        // 添加环境光照
        vec3 ambient = mat.albedo * 0.1;
        
        // 组合结果
        color = ambient + directLighting;
    } 
    else if (mat.materialType == 1u) { // 镜面反射材质
        vec3 reflectedDir = reflectDirection(incident, N);
        
        // 创建新的反射射线
        Ray reflectedRay;
        reflectedRay.origin = worldPos + N * EPSILON;
        reflectedRay.direction = reflectedDir;
        
        // 递归追踪
        color = traceRay(reflectedRay, payloadDepth + 1);
    }
    else if (mat.materialType == 2u) { // 折射材质
        // 确保法线方向正确（指向入射方向的反面）
        if (dot(incident, N) > 0.0) {
            N = -N; // 调整法线方向，确保指向光线来源的一侧
        }
        
        // 计算菲涅尔反射率
        float F = fresnel(incident, N, mat.ior);
        
        vec3 reflectedColor = vec3(0.0);
        vec3 refractedColor = vec3(0.0);
        
        // 总是追踪反射光线
        vec3 reflectedDir = reflectDirection(incident, N);
        Ray reflectedRay;
        reflectedRay.origin = worldPos + N * EPSILON;
        reflectedRay.direction = reflectedDir;
        reflectedColor = traceRay(reflectedRay, payloadDepth + 1);
        
        // 处理折射
        vec3 refractedDir;
        bool refracted = refractDirection(incident, N, mat.ior, refractedDir);
        if (refracted) {
            Ray refractedRay;
            // 使用法线的相反方向作为折射光线的起点偏移
            refractedRay.origin = worldPos - N * EPSILON; 
            refractedRay.direction = refractedDir;
            refractedColor = traceRay(refractedRay, payloadDepth + 1);
            
            // 混合反射和折射
            color = mix(refractedColor, reflectedColor, F);
        } else {
            // 全反射情况 - 只使用反射光线
            color = reflectedColor;
        }
    }
    
    // 应用材质自发光
    color += mat.emission;
    
    // Gamma 校正
    color = pow(color, vec3(1.0/2.2));
    
    // 写入 payload
    payloadColor = color;
}

// 递归光线追踪函数
vec3 traceRay(Ray ray, int depth) {
    // 保存当前payload状态
    int currentDepth = payloadDepth;
    vec3 currentColor = payloadColor;
    
    // 设置新的递归深度
    payloadDepth = depth;
    
    // 发起新的光线追踪
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        0, // sbtRecordOffset
        0, // sbtRecordStride
        0, // missIndex
        ray.origin,
        0.001, // Tmin
        ray.direction,
        10000.0, // Tmax
        0   // payload index
    );
    
    // 获取结果并恢复原状态
    vec3 color = payloadColor;
    payloadColor = currentColor;
    payloadDepth = currentDepth;
    
    return color;
}