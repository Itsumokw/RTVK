#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(location = 0) rayPayloadInEXT vec3 payloadColor;
layout(location = 1) rayPayloadInEXT int payloadDepth;
hitAttributeEXT vec2 attribs;

// ��Դ�ṹ��
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

// ���ʽṹ
struct Material {
    vec3 albedo;
    vec3 emission;
    float ior;
    uint materialType; // 0=diffuse, 1=specular, 2=refractive
};

const int MAX_BOUNCES = 3;
const float EPSILON = 0.001;

// ���߽ṹ
struct Ray {
    vec3 origin;
    vec3 direction;
};

// ͳһ��������
layout(set = 0, binding = 2) buffer VertexBuffers {
    Vertex vertices[];
} vertexBuffer[];

layout(set = 0, binding = 5) buffer IndicesBuffer { 
    uint indices[];
} indicesBuffer[];

// �ݹ����׷�ٺ���
vec3 traceRay(Ray ray, int depth);

// ���㾵�淴�䷽��
vec3 reflectDirection(vec3 incident, vec3 normal) {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// �������䷽�򣨷��� true ����������䣩
bool refractDirection(vec3 incident, vec3 normal, float ior, out vec3 refractedDir) {
    float cosi = clamp(dot(incident, normal), -1.0, 1.0);
    float etai = 1.0, etat = ior;
    vec3 n = normal;
    
    if (cosi < 0.0) { // ���ⲿ����
        cosi = -cosi;
    } else { // ���ڲ�����
        float temp = etai;
        etai = etat;
        etat = temp;
        n = -normal;
    }
    
    float eta = etai / etat;
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    
    if (k < 0.0) return false; // ȫ����
    
    refractedDir = normalize(eta * incident + (eta * cosi - sqrt(k)) * n);
    return true;
}

// �������������ϵ��
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
    if (sint >= 1.0) return 1.0; // ȫ����
    
    float cost = sqrt(max(0.0, 1.0 - sint * sint));
    cosi = abs(cosi);
    float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    return (Rs * Rs + Rp * Rp) / 2.0;
}

// Ӳ�����������
Material getMaterial(uint instanceIndex) {
    Material mat;
    
    // ����ʵ��ID�������
    if (instanceIndex == 3 || instanceIndex == 2) {
        // ���������
        mat.albedo = vec3(0.8, 0.8, 0.8);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 0u;
    } else if (instanceIndex == 1) {
        // ���淴�����
        mat.albedo = vec3(0.95, 0.95, 0.95);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 1u;
    } else if (instanceIndex == 0) {
        // ������ʣ�������
        mat.albedo = vec3(1.0, 1.0, 1.0); // ��ɫ
        mat.emission = vec3(0.0);
        mat.ior = 1.5; // ����������
        mat.materialType = 2u;
    } else {
        // Ĭ�ϲ��ʣ������䣩
        mat.albedo = vec3(0.5, 0.5, 0.7);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 0u;
    }
    
    return mat;
}

// �����Դ��������
vec3 calculateLightContribution(vec3 worldPos, vec3 N, Material mat) {
    vec3 lighting = vec3(0.0);
    uint lightCount = lightBuffer.lights.length();
    
    for (uint i = 0; i < lightCount; i++) {
        Light light = lightBuffer.lights[i];
        
        // ������շ���
        vec3 lightDir;
        if (light.type == 0u) { // ƽ�й�
            lightDir = normalize(-light.position.xyz);
        } else { // ���Դ/�۹��
            vec3 lightVec = light.position.xyz - worldPos;
            lightDir = normalize(lightVec);
        }
        
        // ����˥��
        float attenuation = 1.0;
        if (light.type == 1u || light.type == 2u) { // ���Դ/�۹��
            float distance = length(light.position.xyz - worldPos);
            attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
            attenuation *= light.color.a; // ���Թ�Դǿ��
            
            // �۹��˥������
            if (light.type == 2u) { 
                float cosAngle = dot(lightDir, normalize(light.position.xyz));
                float innerCone = cos(light.spotInnerAngle);
                float outerCone = cos(light.spotOuterAngle);
                float spotFactor = smoothstep(outerCone, innerCone, cosAngle);
                attenuation *= spotFactor;
            }
        }
        
        // ���������乱��
        float NdotL = max(dot(N, lightDir), 0.0);
        vec3 diffuse = mat.albedo * light.color.rgb * NdotL * attenuation;
        
        lighting += diffuse;
    }
    
    return lighting;
}

void main() {
    uint instanceIndex = gl_InstanceCustomIndexEXT;
    uint triIndex = gl_PrimitiveID;

    // ��ȡ����������
    uint index0 = indicesBuffer[instanceIndex].indices[triIndex * 3];
    uint index1 = indicesBuffer[instanceIndex].indices[triIndex * 3 + 1];
    uint index2 = indicesBuffer[instanceIndex].indices[triIndex * 3 + 2];
    
    Vertex v0 = vertexBuffer[instanceIndex].vertices[index0];
    Vertex v1 = vertexBuffer[instanceIndex].vertices[index1];
    Vertex v2 = vertexBuffer[instanceIndex].vertices[index2];
    
    // ���������ֵ
    vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    
    // ��ֵ���ߺ�λ��
    vec3 n0 = v0.normal;
    vec3 n1 = v1.normal;
    vec3 n2 = v2.normal;
    vec3 N = normalize(n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
    N = normalize(mat3(gl_ObjectToWorldEXT) * N); // ת��Ϊ����ռ�
    
    // ������������߷���
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 incident = normalize(gl_WorldRayDirectionEXT);
    
    // ��ȡӲ�������
    Material mat = getMaterial(instanceIndex);
    vec3 color = vec3(0.0);
    
    // ������ݹ����
    if (payloadDepth >= MAX_BOUNCES) {
        payloadColor = mat.emission;
        return;
    }
    
    // ����ͬ��������
    if (mat.materialType == 0u) { // ���������
        // ʹ��lightBuffer�еĹ�Դ�������
        vec3 directLighting = calculateLightContribution(worldPos, N, mat);
        
        // ��ӻ�������
        vec3 ambient = mat.albedo * 0.1;
        
        // ��Ͻ��
        color = ambient + directLighting;
    } 
    else if (mat.materialType == 1u) { // ���淴�����
        vec3 reflectedDir = reflectDirection(incident, N);
        
        // �����µķ�������
        Ray reflectedRay;
        reflectedRay.origin = worldPos + N * EPSILON;
        reflectedRay.direction = reflectedDir;
        
        // �ݹ�׷��
        color = traceRay(reflectedRay, payloadDepth + 1);
    }
    else if (mat.materialType == 2u) { // �������
        // ȷ�����߷�����ȷ��ָ�����䷽��ķ��棩
        if (dot(incident, N) > 0.0) {
            N = -N; // �������߷���ȷ��ָ�������Դ��һ��
        }
        
        // ���������������
        float F = fresnel(incident, N, mat.ior);
        
        vec3 reflectedColor = vec3(0.0);
        vec3 refractedColor = vec3(0.0);
        
        // ����׷�ٷ������
        vec3 reflectedDir = reflectDirection(incident, N);
        Ray reflectedRay;
        reflectedRay.origin = worldPos + N * EPSILON;
        reflectedRay.direction = reflectedDir;
        reflectedColor = traceRay(reflectedRay, payloadDepth + 1);
        
        // ��������
        vec3 refractedDir;
        bool refracted = refractDirection(incident, N, mat.ior, refractedDir);
        if (refracted) {
            Ray refractedRay;
            // ʹ�÷��ߵ��෴������Ϊ������ߵ����ƫ��
            refractedRay.origin = worldPos - N * EPSILON; 
            refractedRay.direction = refractedDir;
            refractedColor = traceRay(refractedRay, payloadDepth + 1);
            
            // ��Ϸ��������
            color = mix(refractedColor, reflectedColor, F);
        } else {
            // ȫ������� - ֻʹ�÷������
            color = reflectedColor;
        }
    }
    
    // Ӧ�ò����Է���
    color += mat.emission;
    
    // Gamma У��
    color = pow(color, vec3(1.0/2.2));
    
    // д�� payload
    payloadColor = color;
}

// �ݹ����׷�ٺ���
vec3 traceRay(Ray ray, int depth) {
    // ���浱ǰpayload״̬
    int currentDepth = payloadDepth;
    vec3 currentColor = payloadColor;
    
    // �����µĵݹ����
    payloadDepth = depth;
    
    // �����µĹ���׷��
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
    
    // ��ȡ������ָ�ԭ״̬
    vec3 color = payloadColor;
    payloadColor = currentColor;
    payloadDepth = currentDepth;
    
    return color;
}