#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;
layout(location = 2) out vec2 uv;

// Набор 0 — данные сцены, общие для всех объектов
layout(set = 0, binding = 0) uniform SceneData {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;           // rgb — цвет, умноженный на интенсивность
	vec4 light_direction;   // направленный источник: xyz — направление лучей
	vec4 light_color;       // rgb — цвет, a — интенсивность
	uint point_light_count;
	uint spot_light_count;
} scene;

// Набор 1, привязка 0 — динамический uniform-буфер: массив свойств всех моделей.
// Какой элемент массива читать, задаёт сдвиг при привязке набора
layout(set = 1, binding = 0) uniform ModelData {
	mat4 transform;
	vec4 specular;          // rgb — цвет блика, a — коэффициент блеска
} object;

// Выполняется для каждой вершины: переводит её и нормаль в мировые координаты,
// текстурные координаты передаёт дальше без изменений.
void main() {
	vec4 world = object.transform * vec4(in_position, 1.0);
	world_position = world.xyz;

	// Нормаль преобразуется обратной транспонированной матрицей, иначе при
	// неравномерном масштабе она перестала бы быть перпендикулярной грани
	world_normal = transpose(inverse(mat3(object.transform))) * in_normal;

	uv = in_uv;

	gl_Position = scene.view_projection * world;
}
