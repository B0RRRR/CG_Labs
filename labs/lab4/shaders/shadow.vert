#version 450

layout(location = 0) in vec3 in_position;

// Набор 0, привязка 0 — общие данные сцены (для карты теней нужна матрица солнца)
layout(set = 0, binding = 0) uniform SceneData {
	mat4 view_projection;
	mat4 light_view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_color;
	uint point_light_count;
	uint spot_light_count;
	float shadow_bias;
	uint shadow_pcf;
} scene;

// Набор 1, привязка 0 — динамический uniform-буфер с матрицей текущей модели
layout(set = 1, binding = 0) uniform ModelData {
	mat4 transform;
	vec4 specular;
} object;

// Единственный шейдер конвейера генерации карты теней: проецирует вершину
// в пространство направленного источника света (солнца)
void main() {
	gl_Position = scene.light_view_projection * object.transform * vec4(in_position, 1.0);
}
