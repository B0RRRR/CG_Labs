#pragma once

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

constexpr float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

struct Vec3 {
	float x, y, z;
};

constexpr Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
constexpr Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
constexpr Vec3 operator*(Vec3 v, float s) { return { v.x * s, v.y * s, v.z * s }; }

constexpr float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr Vec3 cross(Vec3 a, Vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline Vec3 normalize(Vec3 v) {
	const float length = std::sqrt(dot(v, v));
	return length > 0.0f ? v * (1.0f / length) : v;
}

struct Mat4 {
	float m[4][4];
};

constexpr Mat4 identity() {
	return { {
		{ 1.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f },
	} };
}

constexpr Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 result = {};

	for (int c = 0; c < 4; ++c) {
		for (int r = 0; r < 4; ++r) {
			float sum = 0.0f;

			for (int k = 0; k < 4; ++k) {
				sum += a.m[k][r] * b.m[c][k];
			}

			result.m[c][r] = sum;
		}
	}

	return result;
}

constexpr Mat4 translation(Vec3 t) {
	Mat4 result = identity();

	result.m[3][0] = t.x;
	result.m[3][1] = t.y;
	result.m[3][2] = t.z;

	return result;
}

constexpr Mat4 scaling(Vec3 s) {
	Mat4 result = identity();

	result.m[0][0] = s.x;
	result.m[1][1] = s.y;
	result.m[2][2] = s.z;

	return result;
}

inline Mat4 rotationX(float angle) {
	const float s = std::sin(angle), c = std::cos(angle);

	Mat4 result = identity();

	result.m[1][1] = c;
	result.m[1][2] = s;
	result.m[2][1] = -s;
	result.m[2][2] = c;

	return result;
}

inline Mat4 rotationY(float angle) {
	const float s = std::sin(angle), c = std::cos(angle);

	Mat4 result = identity();

	result.m[0][0] = c;
	result.m[0][2] = -s;
	result.m[2][0] = s;
	result.m[2][2] = c;

	return result;
}

inline Mat4 rotationZ(float angle) {
	const float s = std::sin(angle), c = std::cos(angle);

	Mat4 result = identity();

	result.m[0][0] = c;
	result.m[0][1] = s;
	result.m[1][0] = -s;
	result.m[1][1] = c;

	return result;
}

inline Mat4 rotation(Vec3 angles) {
	return rotationZ(angles.z) * rotationX(angles.x) * rotationY(angles.y);
}


inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
	const Vec3 forward = normalize(target - eye);
	const Vec3 right = normalize(cross(forward, up));
	const Vec3 u = cross(right, forward);

	Mat4 result = identity();

	result.m[0][0] = right.x;
	result.m[1][0] = right.y;
	result.m[2][0] = right.z;

	result.m[0][1] = u.x;
	result.m[1][1] = u.y;
	result.m[2][1] = u.z;

	result.m[0][2] = -forward.x;
	result.m[1][2] = -forward.y;
	result.m[2][2] = -forward.z;

	result.m[3][0] = -dot(right, eye);
	result.m[3][1] = -dot(u, eye);
	result.m[3][2] = dot(forward, eye);

	return result;
}

inline Mat4 perspective(float fov_y, float aspect, float z_near, float z_far) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);

	Mat4 result = {};

	result.m[0][0] = f / aspect;
	result.m[1][1] = -f;
	result.m[2][2] = z_far / (z_near - z_far);
	result.m[2][3] = -1.0f;
	result.m[3][2] = (z_near * z_far) / (z_near - z_far);

	return result;
}


inline Mat4 orthographic(float height, float aspect, float z_near, float z_far) {
	const float top = height * 0.5f;
	const float right = top * aspect;

	Mat4 result = identity();

	result.m[0][0] = 1.0f / right;
	result.m[1][1] = -1.0f / top;
	result.m[2][2] = 1.0f / (z_near - z_far);
	result.m[3][2] = z_near / (z_near - z_far);

	return result;
}

}
