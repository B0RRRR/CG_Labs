#version 450

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform SceneData {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_color;
	uint point_light_count;
	uint spot_light_count;
} scene;

struct PointLight {
	vec4 position;          // xyz
	vec4 color;             // rgb — цвет, a — интенсивность
};

struct SpotLight {
	vec4 position;          // xyz
	vec4 direction;         // xyz — куда светит
	vec4 color;             // rgb — цвет, a — интенсивность
	vec4 cutoff;            // x — косинус внутреннего угла, y — внешнего
};

// Массивы без указания размера: сколько элементов, знает только буфер
// (доп. задания 2 и 3)
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 1, binding = 0) uniform ModelData {
	mat4 transform;
	vec4 albedo;
	vec4 specular;
} object;

// Модель Блинна-Фонга для одного источника.
// N — нормаль, V — направление на камеру, L — направление на источник,
// radiance — цвет источника с учётом интенсивности и затухания.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
	float n_dot_l = max(dot(N, L), 0.0);
	if (n_dot_l == 0.0) {
		return vec3(0.0);
	}

	vec3 H = normalize(L + V);
	float n_dot_h = max(dot(N, H), 0.0);

	vec3 diffuse = object.albedo.rgb * n_dot_l;
	vec3 specular = object.specular.rgb * pow(n_dot_h, object.specular.a);

	return (diffuse + specular) * radiance;
}

// Выполняется для каждого пикселя: складывает вклад всех источников света.
void main() {
	vec3 N = normalize(world_normal);
	vec3 V = normalize(scene.camera_position.xyz - world_position);

	// Рассеянный свет
	vec3 color = scene.ambient.rgb * object.albedo.rgb;

	// Направленный источник: лучи параллельны, затухания нет
	vec3 sun_direction = normalize(-scene.light_direction.xyz);
	color += blinnPhong(N, V, sun_direction, scene.light_color.rgb * scene.light_color.a);

	// Доп. задание 2: точечные источники, затухание по закону обратных квадратов
	for (uint i = 0; i < scene.point_light_count; ++i) {
		PointLight light = point_lights[i];

		vec3 to_light = light.position.xyz - world_position;
		float distance_squared = max(dot(to_light, to_light), 0.0001);
		vec3 L = to_light / sqrt(distance_squared);

		vec3 radiance = light.color.rgb * light.color.a / distance_squared;
		color += blinnPhong(N, V, L, radiance);
	}

	// Доп. задание 3: прожекторы — затухание и мягкий край пятна
	for (uint i = 0; i < scene.spot_light_count; ++i) {
		SpotLight light = spot_lights[i];

		vec3 to_light = light.position.xyz - world_position;
		float distance_squared = max(dot(to_light, to_light), 0.0001);
		vec3 L = to_light / sqrt(distance_squared);

		// Косинус угла между осью прожектора и направлением на точку.
		// Внутри внутреннего конуса — 1, за внешним — 0, между ними плавный переход
		float cos_angle = dot(-L, normalize(light.direction.xyz));
		float cos_inner = light.cutoff.x;
		float cos_outer = light.cutoff.y;
		float edge = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
						   0.0, 1.0);

		vec3 radiance = light.color.rgb * light.color.a * edge / distance_squared;
		color += blinnPhong(N, V, L, radiance);
	}

	out_color = vec4(color, 1.0);
}
