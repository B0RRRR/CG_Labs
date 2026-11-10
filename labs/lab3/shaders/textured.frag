#version 450

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 uv;

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

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 1, binding = 0) uniform ModelData {
	mat4 transform;
	vec4 specular;
} object;

// Набор 1, привязка 1 — текстура альбедо вместе с сэмплером.
// У каждой текстуры свой набор (доп. задание 1)
layout(set = 1, binding = 1) uniform sampler2D albedo_texture;

// Доп. задание 2: параметры нетривиального сэмплирования приходят в
// push-константах, а не в uniform-буфере модели
layout(push_constant) uniform Effect {
	uint enabled;           // 0 — обычная выборка, 1 — анимированная
	float time;             // время в секундах
	float speed;            // множитель скорости анимации
	float distortion;       // сила искажения текстурных координат
} effect;

const float TAU = 6.28318530718;

// Поворачивает точку на плоскости на угол a вокруг начала координат.
vec2 rotate(vec2 p, float a) {
	float s = sin(a);
	float c = cos(a);
	return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

// Доп. задание 2: нетривиальное сэмплирование, зависящее от времени.
vec3 animatedAlbedo(vec2 coords) {
	float t = effect.time * effect.speed;

	// 1. Искажение координат: две бегущие синусоидальные волны
	vec2 wave = effect.distortion * vec2(
		sin(coords.y * 3.0 * TAU + 2.0 * t),
		sin(coords.x * 3.0 * TAU + 1.6 * t));

	// 2. Две выборки одной и той же текстуры разными координатами:
	//    первый слой сдвигается по диагонали, второй увеличен вдвое
	//    и медленно вращается вокруг центра развёртки
	vec2 uv_a = coords + wave + vec2(0.05, 0.03) * t;
	vec2 uv_b = rotate(coords - 0.5, 0.2 * t) * 2.0 + 0.5 - wave;

	vec3 a = texture(albedo_texture, uv_a).rgb;
	vec3 b = texture(albedo_texture, uv_b).rgb;

	// Слои смешиваются: яркие участки обоих слоёв остаются видны
	vec3 layers = max(a, b);

	// 3. Модуляция цвета: оттенок плавно обходит цветовой круг,
	//    а яркость пульсирует волной, бегущей по поверхности
	vec3 tint = 0.6 + 0.4 * cos(TAU * (0.08 * t + vec3(0.0, 0.33, 0.67)));
	float pulse = 0.85 + 0.15 * sin(3.0 * t - 8.0 * (coords.x + coords.y));

	return layers * tint * pulse * 1.4;
}

// Цвет поверхности в точке: из текстуры, обычной или анимированной выборкой.
vec3 albedo() {
	if (effect.enabled != 0) {
		return animatedAlbedo(uv);
	}

	return texture(albedo_texture, uv).rgb;
}

// Модель Блинна-Фонга для одного источника.
// N — нормаль, V — направление на камеру, L — направление на источник,
// kd — альбедо из текстуры, radiance — свет, пришедший от источника.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 kd, vec3 radiance) {
	float n_dot_l = max(dot(N, L), 0.0);
	if (n_dot_l == 0.0) {
		return vec3(0.0);
	}

	vec3 H = normalize(L + V);
	float n_dot_h = max(dot(N, H), 0.0);

	vec3 diffuse = kd * n_dot_l;
	vec3 specular = object.specular.rgb * pow(n_dot_h, object.specular.a);

	return (diffuse + specular) * radiance;
}

// Выполняется для каждого пикселя: берёт альбедо из текстуры и складывает
// вклад всех источников света.
void main() {
	vec3 N = normalize(world_normal);
	vec3 V = normalize(scene.camera_position.xyz - world_position);
	vec3 kd = albedo();

	// Рассеянный свет
	vec3 color = scene.ambient.rgb * kd;

	// Направленный источник
	vec3 sun_direction = normalize(-scene.light_direction.xyz);
	color += blinnPhong(N, V, sun_direction, kd, scene.light_color.rgb * scene.light_color.a);

	// Точечные источники, затухание по закону обратных квадратов
	for (uint i = 0; i < scene.point_light_count; ++i) {
		PointLight light = point_lights[i];

		vec3 to_light = light.position.xyz - world_position;
		float distance_squared = max(dot(to_light, to_light), 0.0001);
		vec3 L = to_light / sqrt(distance_squared);

		vec3 radiance = light.color.rgb * light.color.a / distance_squared;
		color += blinnPhong(N, V, L, kd, radiance);
	}

	// Прожекторы: затухание и мягкий край пятна
	for (uint i = 0; i < scene.spot_light_count; ++i) {
		SpotLight light = spot_lights[i];

		vec3 to_light = light.position.xyz - world_position;
		float distance_squared = max(dot(to_light, to_light), 0.0001);
		vec3 L = to_light / sqrt(distance_squared);

		float cos_angle = dot(-L, normalize(light.direction.xyz));
		float cos_inner = light.cutoff.x;
		float cos_outer = light.cutoff.y;
		float edge = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001),
						   0.0, 1.0);

		vec3 radiance = light.color.rgb * light.color.a * edge / distance_squared;
		color += blinnPhong(N, V, L, kd, radiance);
	}

	out_color = vec4(color, 1.0);
}
