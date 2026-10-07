#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

layout(push_constant) uniform PushConstants {
	mat4 mvp;   // модель * вид * проекция
	vec4 tint;  // rgb — цвет из интерфейса, w — использовать ли цвета вершин
} pc;

// Выполняется для каждой вершины: переводит её в координаты экрана
// и вычисляет цвет, который дальше интерполируется вдоль граней.
void main() {
	gl_Position = pc.mvp * vec4(in_position, 1.0);

	// Доп. задания 4 и 5: цвет вершины задан процедурно от её позиции
	// в локальной системе координат, а цвет из интерфейса умножается на него.
	// Если tint.w равен нулю, цвета вершин заменяются белым
	vec3 base = mix(vec3(1.0), in_color, pc.tint.w);
	frag_color = base * pc.tint.rgb;
}
