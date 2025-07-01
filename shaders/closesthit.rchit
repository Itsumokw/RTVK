#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(location = 0) rayPayloadInEXT vec4 payloadColor;
layout(location = 1) rayPayloadEXT int shadowPayload;
hitAttributeEXT vec2 attribs;

// 光源结构体
struct Light {
    vec4 position;    // w=0: directional, w=1: point/spot
    vec4 color;        // RGB + intensity (A)
    float range;       // Point/spot light range
    float spotOuterAngle;
    float spotInnerAngle;
    float padding;
    vec4 direction;
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

const int MAX_BOUNCES = 10;
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
bool refractDirection(vec3 incident, vec3 normal, float ior, 
                     out vec3 refractedDir, out vec3 usedNormal) 
{
    float cosi = clamp(dot(incident, normal), -1.0, 1.0);
    float etai = 1.0, etat = ior;
    usedNormal = normal;
    
    if (cosi < 0.0) { // 从外部进入
        cosi = -cosi;
    } else { // 从内部出来
        float temp = etai;
        etai = etat;
        etat = temp;
        usedNormal = -normal;
    }
    
    float eta = etai / etat;
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    
    if (k < 0.0) return false; // 全反射
    
    refractedDir = normalize(eta * incident + (eta * cosi - sqrt(k)) * usedNormal);
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

// 阴影检测函数
bool isInShadow(vec3 origin, vec3 direction, float maxDist) {
    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | 
                 gl_RayFlagsOpaqueEXT | 
                 gl_RayFlagsSkipClosestHitShaderEXT;
    
    shadowPayload = 1; // 默认可见
    traceRayEXT(
        topLevelAS,
        flags,
        0xFF,
        0,
        0,
        1, // miss着色器索引
        origin,
        0.05,
        direction,
        maxDist,
        1 // payload位置
    );
    
    return (shadowPayload == 1); // 1表示被遮挡
}

// 硬编码材质数据
Material getMaterial(uint instanceIndex) {
    Material mat;
    
    // 根据实例ID分配材质
    if (instanceIndex == 5) {
        mat.albedo = vec3(0.521, 0.423, 0.327);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 0u;
    } else if (instanceIndex == 4) { // 棋盘格平面
        mat.albedo = vec3(1.0);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 4u; // CHECKERBOARD
    } else if (instanceIndex == 3) {
        // 漫反射材质
        mat.albedo = vec3(1.0, 1.0, 1.0);
        mat.emission = vec3(0.0);
        mat.ior = 1.5;
        mat.materialType = 2u;
    } else if (instanceIndex == 2) {
        // 漫反射材质
        mat.albedo = vec3(0.0, 0.26, 0.16) * 0.3;
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
        mat.albedo = vec3(1.0, 1.0, 1.0);
        mat.emission = vec3(0.0);
        mat.ior = 1.5;
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
vec3 calculateLightContribution(vec3 worldPos, vec3 N, vec3 faceNormal, Material mat) {
    vec3 lighting = vec3(0.0);
    uint lightCount = lightBuffer.lights.length();
    if (mat.materialType == 4u) {
        faceNormal = N;
    }
    
    for (uint i = 0; i < lightCount; i++) {
        Light light = lightBuffer.lights[i];
        
        // 计算光照方向和距离
        vec3 lightDir;
        float lightDist = 1000.0; // 默认足够大
        if (light.position.w == 0u) { // 平行光
            lightDir = normalize(-light.position.xyz);
        } else { // 点光源/聚光灯
            vec3 lightVec = light.position.xyz - worldPos;
            lightDist = length(lightVec);
            lightDir = normalize(lightVec);
        }

        // 阴影检测
        vec3 shadowOrigin = worldPos + faceNormal * 0.05;
        if (isInShadow(shadowOrigin, lightDir, lightDist)) {
            continue; // 被遮挡，跳过此光源
        }
        
        // 计算衰减
        float attenuation = 1.0;
        if (light.position.w == 1u ||light.position.w == 2u) { // 点光源/聚光灯
            attenuation = 1.0 / (1.0 + 0.1 * lightDist + 0.01 * lightDist * lightDist);
        }
        attenuation *= light.color.a; // 乘以光源强度

        // 如果是聚光灯，再乘以聚光系数
        if (light.position.w == 2u) {
            // 聚光灯方向向量 S 已归一化
            float cosTheta = dot(lightDir, normalize(light.direction.xyz));
            float cosOuter = cos(light.spotOuterAngle * 0.5);
            float cosInner = cos(light.spotInnerAngle * 0.5);
            float spotFactor = 0.0;
            if (cosTheta >= cosOuter) {
                // 完全在聚光范围内
                spotFactor = 1.0;
            } else if (cosTheta <= cosInner) {
                // 完全在聚光锥外
                spotFactor = 0.0;
            } else {
                // 在内外锥之间，线性过渡
                spotFactor = (cosTheta - cosInner) / (cosOuter - cosInner);
            }
            attenuation *= spotFactor;
        }
        
        // 计算漫反射贡献
        float NdotL = max(dot(N, lightDir), 0.0);
        vec3 diffuse = mat.albedo * light.color.rgb * NdotL * attenuation;

        vec3 V = normalize(-gl_WorldRayDirectionEXT);
        vec3 H = normalize(lightDir + V);
        float NdotH = max(dot(N, H), 0.0);
        vec3 specular = light.color.rgb
                      * pow(NdotH, 64)
                      * attenuation * 0.05;
        
        lighting += diffuse + specular;
    }
    
    return lighting;
}

void main() {
    uint instanceIndex = gl_InstanceCustomIndexEXT;
    uint triIndex = gl_PrimitiveID;

    // 获取三角形数据
    uint index0 = indicesBuffer[nonuniformEXT(instanceIndex)].indices[triIndex * 3];
    uint index1 = indicesBuffer[nonuniformEXT(instanceIndex)].indices[triIndex * 3 + 1];
    uint index2 = indicesBuffer[nonuniformEXT(instanceIndex)].indices[triIndex * 3 + 2];
    
    Vertex v0 = vertexBuffer[nonuniformEXT(instanceIndex)].vertices[index0];
    Vertex v1 = vertexBuffer[nonuniformEXT(instanceIndex)].vertices[index1];
    Vertex v2 = vertexBuffer[nonuniformEXT(instanceIndex)].vertices[index2];
    
    // 重心坐标插值
    vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    
    // 插值法线和位置
    vec3 n0 = v0.normal;
    vec3 n1 = v1.normal;
    vec3 n2 = v2.normal;
    vec3 N = normalize(n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
    N = normalize(mat3(gl_ObjectToWorldEXT) * N); // 转换为世界空间

    vec3 edge1 = v1.position - v0.position;
    vec3 edge2 = v2.position - v0.position;
    vec3 faceNormal = normalize(cross(edge1, edge2));
    mat3 normalMatrix = transpose(inverse(mat3(gl_ObjectToWorldEXT)));
    faceNormal = normalize(normalMatrix * faceNormal);
    if ((instanceIndex == 0) || (instanceIndex == 2) || (instanceIndex == 5)) {
        N = faceNormal;
    }
    // 世界坐标和视线方向
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 incident = normalize(gl_WorldRayDirectionEXT);

    // 获取硬编码材质
    Material mat = getMaterial(instanceIndex);
    vec3 color = vec3(0.0);
    
    // 检查最大递归深度
    if (int(payloadColor.a) >= MAX_BOUNCES) {
        payloadColor.rgb = mat.emission;
        return;
    }
    
    // 处理不同材质类型
    if (mat.materialType == 0u) { // 漫反射材质
        // 使用lightBuffer中的光源计算光照
        vec3 directLighting = calculateLightContribution(worldPos, N, faceNormal, mat);
        
        // 添加环境光照
        vec3 ambient = mat.albedo * 0.01;
        
        // 组合结果
        color = ambient + directLighting;
    } 
    else if (mat.materialType == 1u) { // 镜面反射材质
        vec3 reflectedDir = reflectDirection(incident, N);
        
        // 创建新的反射射线
        Ray reflectedRay;
        reflectedRay.origin = worldPos + faceNormal * EPSILON;
        reflectedRay.direction = reflectedDir;
        
        // 递归追踪
        color = traceRay(reflectedRay, int(payloadColor.a) + 1) * 0.7;
    }
    else if (mat.materialType == 2u) { // 折射材质
        // 计算菲涅尔反射率
        float F = fresnel(incident, N, mat.ior);
        
        // 首先追踪反射光线
        vec3 reflectedDir = reflectDirection(incident, N);
        Ray reflectedRay;
        reflectedRay.origin = worldPos + faceNormal * EPSILON;
        reflectedRay.direction = reflectedDir;
        vec3 reflectedColor = traceRay(reflectedRay, int(payloadColor.a) + 1);
        
        // 处理折射
        vec3 refractedDir, usedNormal;
        if (refractDirection(incident, N, mat.ior, refractedDir, usedNormal)) {
            Ray refractedRay;
            // 使用折射函数返回的法线确定偏移方向
            refractedRay.origin = worldPos - usedNormal * EPSILON;
            refractedRay.direction = refractedDir;
            vec3 refractedColor = traceRay(refractedRay, int(payloadColor.a) + 1);
            
            // 物理正确的菲涅尔混合
            color = mix(refractedColor, reflectedColor, F) * 0.8;
        } else {
            // 全反射情况 - 只使用反射光线
            color = reflectedColor * 0.8;
        }
    }
    else if (mat.materialType == 4u) { // 棋盘格材质
        // 获取UV坐标
        vec2 uv = v0.texCoord * barycentrics.x +
                  v1.texCoord * barycentrics.y +
                  v2.texCoord * barycentrics.z;
        
        // 创建10x10棋盘格
        float scale = 10.0;
        vec2 grid = floor(uv * scale);
        bool isWhite = mod(grid.x + grid.y, 2.0) == 0.0;
        
        // 设置颜色：白色和橙色
        vec3 color1 = vec3(1.0, 1.0, 1.0); // 白色
        vec3 color2 = vec3(1.0, 0.4, 0.); // 橙色
        
        color = isWhite ? color1 : color2;
        
        // 添加简单光照
        vec3 directLighting = calculateLightContribution(worldPos, N, faceNormal, mat);
        // 添加环境光照
        vec3 ambient = mat.albedo * 0.01;
        // 组合结果
        color = (directLighting) * color * 0.55 + ambient;
        // 应用Gamma校正
        //color = pow(color, vec3(1.0/2.2));
    }
    
    // 应用材质自发光
    color += mat.emission;
    
    // Gamma 校正
    //color = pow(color, vec3(1.0/2.2));
    
    // 写入 payload
    payloadColor.rgb = color;
    payloadColor.a = 1.0f;
}

// 递归光线追踪函数
vec3 traceRay(Ray ray, int depth) {
    // 保存当前payload状态
    int currentDepth = int(payloadColor.a);
    vec3 currentColor = payloadColor.rgb;
    
    // 设置新的递归深度
    payloadColor.a = float(depth);
    
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
    vec3 color = payloadColor.rgb;
    payloadColor.rgb = currentColor;
    payloadColor.a = float(currentDepth);
    
    return color;
}