#version 450

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 light_space_position;

layout(location = 0) out vec4 out_color;

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

// Набор 0, привязка 3 — карта теней (текстура глубины + сэмплер со сравнением '<')
layout(set = 0, binding = 3) uniform sampler2DShadow shadow_map;

layout(set = 1, binding = 0) uniform ModelData {
	mat4 transform;
	vec4 specular;
} object;

// Набор 1, привязка 1 — текстура альбедо вместе с обычным сэмплером
layout(set = 1, binding = 1) uniform sampler2D albedo_texture;

// Push-константы для анимированного сэмплирования из 3-й лабораторной
layout(push_constant) uniform Effect {
	uint enabled;           // 0 — обычная выборка, 1 — анимированная
	float time;             // время в секундах
	float speed;            // множитель скорости анимации
	float distortion;       // сила искажения текстурных координат
} effect;

const float TAU = 6.28318530718;

vec2 rotate(vec2 p, float a) {
	float s = sin(a);
	float c = cos(a);
	return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

vec3 animatedAlbedo(vec2 coords) {
	float t = effect.time * effect.speed;

	vec2 wave = effect.distortion * vec2(
		sin(coords.y * 3.0 * TAU + 2.0 * t),
		sin(coords.x * 3.0 * TAU + 1.6 * t));

	vec2 uv_a = coords + wave + vec2(0.05, 0.03) * t;
	vec2 uv_b = rotate(coords - 0.5, 0.2 * t) * 2.0 + 0.5 - wave;

	vec3 a = texture(albedo_texture, uv_a).rgb;
	vec3 b = texture(albedo_texture, uv_b).rgb;

	vec3 layers = max(a, b);

	vec3 tint = 0.6 + 0.4 * cos(TAU * (0.08 * t + vec3(0.0, 0.33, 0.67)));
	float pulse = 0.85 + 0.15 * sin(3.0 * t - 8.0 * (coords.x + coords.y));

	return layers * tint * pulse * 1.4;
}

vec3 albedo() {
	if (effect.enabled != 0) {
		return animatedAlbedo(uv);
	}

	return texture(albedo_texture, uv).rgb;
}

// Вычисляет коэффициент освещённости солнцем: 1.0 — на свету, 0.0 — в тени.
// Использует sampler2DShadow с аппаратным сравнением глубины ('<'):
// сравнение проходит (возвращает 1.0), если глубина фрагмента меньше значения в карте теней.
float calculateShadow(vec4 light_space_pos) {
	vec3 proj = light_space_pos.xyz / light_space_pos.w;
	vec2 shadow_uv = proj.xy * 0.5 + 0.5;

	// За пределами ортографической области солнца тень не накладывается
	if (proj.z <= 0.0 || proj.z >= 1.0 ||
		shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
		shadow_uv.y < 0.0 || shadow_uv.y > 1.0) {
		return 1.0;
	}

	float depth = proj.z - scene.shadow_bias;

	if (scene.shadow_pcf == 0) {
		return texture(shadow_map, vec3(shadow_uv, depth));
	}

	vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
	float sum = 0.0;

	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			sum += texture(shadow_map, vec3(shadow_uv + vec2(x, y) * texel_size, depth));
		}
	}

	return sum / 9.0;
}

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

void main() {
	vec3 N = normalize(world_normal);
	vec3 V = normalize(scene.camera_position.xyz - world_position);
	vec3 kd = albedo();

	// Рассеянный свет не затеняется
	vec3 color = scene.ambient.rgb * kd;

	// Направленный источник (солнце) с картой теней
	vec3 sun_direction = normalize(-scene.light_direction.xyz);
	float shadow = calculateShadow(light_space_position);
	color += shadow * blinnPhong(N, V, sun_direction, kd, scene.light_color.rgb * scene.light_color.a);

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
