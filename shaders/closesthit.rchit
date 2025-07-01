#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(location = 0) rayPayloadInEXT vec4 payloadColor;
layout(location = 1) rayPayloadEXT int shadowPayload;
hitAttributeEXT vec2 attribs;

// ��Դ�ṹ��
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

// ���ʽṹ
struct Material {
    vec3 albedo;
    vec3 emission;
    float ior;
    uint materialType; // 0=diffuse, 1=specular, 2=refractive
};

const int MAX_BOUNCES = 10;
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
bool refractDirection(vec3 incident, vec3 normal, float ior, 
                     out vec3 refractedDir, out vec3 usedNormal) 
{
    float cosi = clamp(dot(incident, normal), -1.0, 1.0);
    float etai = 1.0, etat = ior;
    usedNormal = normal;
    
    if (cosi < 0.0) { // ���ⲿ����
        cosi = -cosi;
    } else { // ���ڲ�����
        float temp = etai;
        etai = etat;
        etat = temp;
        usedNormal = -normal;
    }
    
    float eta = etai / etat;
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    
    if (k < 0.0) return false; // ȫ����
    
    refractedDir = normalize(eta * incident + (eta * cosi - sqrt(k)) * usedNormal);
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

// ��Ӱ��⺯��
bool isInShadow(vec3 origin, vec3 direction, float maxDist) {
    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | 
                 gl_RayFlagsOpaqueEXT | 
                 gl_RayFlagsSkipClosestHitShaderEXT;
    
    shadowPayload = 1; // Ĭ�Ͽɼ�
    traceRayEXT(
        topLevelAS,
        flags,
        0xFF,
        0,
        0,
        1, // miss��ɫ������
        origin,
        0.05,
        direction,
        maxDist,
        1 // payloadλ��
    );
    
    return (shadowPayload == 1); // 1��ʾ���ڵ�
}

// Ӳ�����������
Material getMaterial(uint instanceIndex) {
    Material mat;
    
    // ����ʵ��ID�������
    if (instanceIndex == 5) {
        mat.albedo = vec3(0.521, 0.423, 0.327);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 0u;
    } else if (instanceIndex == 4) { // ���̸�ƽ��
        mat.albedo = vec3(1.0);
        mat.emission = vec3(0.0);
        mat.ior = 1.0;
        mat.materialType = 4u; // CHECKERBOARD
    } else if (instanceIndex == 3) {
        // ���������
        mat.albedo = vec3(1.0, 1.0, 1.0);
        mat.emission = vec3(0.0);
        mat.ior = 1.5;
        mat.materialType = 2u;
    } else if (instanceIndex == 2) {
        // ���������
        mat.albedo = vec3(0.0, 0.26, 0.16) * 0.3;
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
        mat.albedo = vec3(1.0, 1.0, 1.0);
        mat.emission = vec3(0.0);
        mat.ior = 1.5;
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
vec3 calculateLightContribution(vec3 worldPos, vec3 N, vec3 faceNormal, Material mat) {
    vec3 lighting = vec3(0.0);
    uint lightCount = lightBuffer.lights.length();
    if (mat.materialType == 4u) {
        faceNormal = N;
    }
    
    for (uint i = 0; i < lightCount; i++) {
        Light light = lightBuffer.lights[i];
        
        // ������շ���;���
        vec3 lightDir;
        float lightDist = 1000.0; // Ĭ���㹻��
        if (light.position.w == 0u) { // ƽ�й�
            lightDir = normalize(-light.position.xyz);
        } else { // ���Դ/�۹��
            vec3 lightVec = light.position.xyz - worldPos;
            lightDist = length(lightVec);
            lightDir = normalize(lightVec);
        }

        // ��Ӱ���
        vec3 shadowOrigin = worldPos + faceNormal * 0.05;
        if (isInShadow(shadowOrigin, lightDir, lightDist)) {
            continue; // ���ڵ��������˹�Դ
        }
        
        // ����˥��
        float attenuation = 1.0;
        if (light.position.w == 1u ||light.position.w == 2u) { // ���Դ/�۹��
            attenuation = 1.0 / (1.0 + 0.1 * lightDist + 0.01 * lightDist * lightDist);
        }
        attenuation *= light.color.a; // ���Թ�Դǿ��

        // ����Ǿ۹�ƣ��ٳ��Ծ۹�ϵ��
        if (light.position.w == 2u) {
            // �۹�Ʒ������� S �ѹ�һ��
            float cosTheta = dot(lightDir, normalize(light.direction.xyz));
            float cosOuter = cos(light.spotOuterAngle * 0.5);
            float cosInner = cos(light.spotInnerAngle * 0.5);
            float spotFactor = 0.0;
            if (cosTheta >= cosOuter) {
                // ��ȫ�ھ۹ⷶΧ��
                spotFactor = 1.0;
            } else if (cosTheta <= cosInner) {
                // ��ȫ�ھ۹�׶��
                spotFactor = 0.0;
            } else {
                // ������׶֮�䣬���Թ���
                spotFactor = (cosTheta - cosInner) / (cosOuter - cosInner);
            }
            attenuation *= spotFactor;
        }
        
        // ���������乱��
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

    // ��ȡ����������
    uint index0 = indicesBuffer[nonuniformEXT(instanceIndex)].indices[triIndex * 3];
    uint index1 = indicesBuffer[nonuniformEXT(instanceIndex)].indices[triIndex * 3 + 1];
    uint index2 = indicesBuffer[nonuniformEXT(instanceIndex)].indices[triIndex * 3 + 2];
    
    Vertex v0 = vertexBuffer[nonuniformEXT(instanceIndex)].vertices[index0];
    Vertex v1 = vertexBuffer[nonuniformEXT(instanceIndex)].vertices[index1];
    Vertex v2 = vertexBuffer[nonuniformEXT(instanceIndex)].vertices[index2];
    
    // ���������ֵ
    vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    
    // ��ֵ���ߺ�λ��
    vec3 n0 = v0.normal;
    vec3 n1 = v1.normal;
    vec3 n2 = v2.normal;
    vec3 N = normalize(n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
    N = normalize(mat3(gl_ObjectToWorldEXT) * N); // ת��Ϊ����ռ�

    vec3 edge1 = v1.position - v0.position;
    vec3 edge2 = v2.position - v0.position;
    vec3 faceNormal = normalize(cross(edge1, edge2));
    mat3 normalMatrix = transpose(inverse(mat3(gl_ObjectToWorldEXT)));
    faceNormal = normalize(normalMatrix * faceNormal);
    if ((instanceIndex == 0) || (instanceIndex == 2) || (instanceIndex == 5)) {
        N = faceNormal;
    }
    // ������������߷���
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 incident = normalize(gl_WorldRayDirectionEXT);

    // ��ȡӲ�������
    Material mat = getMaterial(instanceIndex);
    vec3 color = vec3(0.0);
    
    // ������ݹ����
    if (int(payloadColor.a) >= MAX_BOUNCES) {
        payloadColor.rgb = mat.emission;
        return;
    }
    
    // ����ͬ��������
    if (mat.materialType == 0u) { // ���������
        // ʹ��lightBuffer�еĹ�Դ�������
        vec3 directLighting = calculateLightContribution(worldPos, N, faceNormal, mat);
        
        // ��ӻ�������
        vec3 ambient = mat.albedo * 0.01;
        
        // ��Ͻ��
        color = ambient + directLighting;
    } 
    else if (mat.materialType == 1u) { // ���淴�����
        vec3 reflectedDir = reflectDirection(incident, N);
        
        // �����µķ�������
        Ray reflectedRay;
        reflectedRay.origin = worldPos + faceNormal * EPSILON;
        reflectedRay.direction = reflectedDir;
        
        // �ݹ�׷��
        color = traceRay(reflectedRay, int(payloadColor.a) + 1) * 0.7;
    }
    else if (mat.materialType == 2u) { // �������
        // ���������������
        float F = fresnel(incident, N, mat.ior);
        
        // ����׷�ٷ������
        vec3 reflectedDir = reflectDirection(incident, N);
        Ray reflectedRay;
        reflectedRay.origin = worldPos + faceNormal * EPSILON;
        reflectedRay.direction = reflectedDir;
        vec3 reflectedColor = traceRay(reflectedRay, int(payloadColor.a) + 1);
        
        // ��������
        vec3 refractedDir, usedNormal;
        if (refractDirection(incident, N, mat.ior, refractedDir, usedNormal)) {
            Ray refractedRay;
            // ʹ�����亯�����صķ���ȷ��ƫ�Ʒ���
            refractedRay.origin = worldPos - usedNormal * EPSILON;
            refractedRay.direction = refractedDir;
            vec3 refractedColor = traceRay(refractedRay, int(payloadColor.a) + 1);
            
            // ������ȷ�ķ��������
            color = mix(refractedColor, reflectedColor, F) * 0.8;
        } else {
            // ȫ������� - ֻʹ�÷������
            color = reflectedColor * 0.8;
        }
    }
    else if (mat.materialType == 4u) { // ���̸����
        // ��ȡUV����
        vec2 uv = v0.texCoord * barycentrics.x +
                  v1.texCoord * barycentrics.y +
                  v2.texCoord * barycentrics.z;
        
        // ����10x10���̸�
        float scale = 10.0;
        vec2 grid = floor(uv * scale);
        bool isWhite = mod(grid.x + grid.y, 2.0) == 0.0;
        
        // ������ɫ����ɫ�ͳ�ɫ
        vec3 color1 = vec3(1.0, 1.0, 1.0); // ��ɫ
        vec3 color2 = vec3(1.0, 0.4, 0.); // ��ɫ
        
        color = isWhite ? color1 : color2;
        
        // ��Ӽ򵥹���
        vec3 directLighting = calculateLightContribution(worldPos, N, faceNormal, mat);
        // ��ӻ�������
        vec3 ambient = mat.albedo * 0.01;
        // ��Ͻ��
        color = (directLighting) * color * 0.55 + ambient;
        // Ӧ��GammaУ��
        //color = pow(color, vec3(1.0/2.2));
    }
    
    // Ӧ�ò����Է���
    color += mat.emission;
    
    // Gamma У��
    //color = pow(color, vec3(1.0/2.2));
    
    // д�� payload
    payloadColor.rgb = color;
    payloadColor.a = 1.0f;
}

// �ݹ����׷�ٺ���
vec3 traceRay(Ray ray, int depth) {
    // ���浱ǰpayload״̬
    int currentDepth = int(payloadColor.a);
    vec3 currentColor = payloadColor.rgb;
    
    // �����µĵݹ����
    payloadColor.a = float(depth);
    
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
    vec3 color = payloadColor.rgb;
    payloadColor.rgb = currentColor;
    payloadColor.a = float(currentDepth);
    
    return color;
}