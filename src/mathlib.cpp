#include "mathlib.hpp"

#include <cassert>
#include <cmath>
#include <numbers>

namespace math {
// vec2
bool operator==(const vec2& a, const vec2& b) {
    return a.x == b.x && a.y == b.y;
}
// vec3
float& vec3::operator[](int idx) {
    assert(idx >= 0 && idx <= 2);
    return (&x)[idx];
}
const float& vec3::operator[](int idx) const {
    assert(idx >= 0 && idx <= 2);
    return (&x)[idx];
}
vec3 operator-(const vec3& a, const vec3& b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
vec3 operator*(const vec3& a, float b) {
    return vec3(a.x * b, a.y * b, a.z * b);
}
vec3 operator*(float a, const vec3& b) { return b * a; }
bool operator==(const vec3& a, const vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// vec4
float& vec4::operator[](int idx) {
    assert(idx >= 0 && idx <= 3);
    return (&x)[idx];
}
const float& vec4::operator[](int idx) const {
    assert(idx >= 0 && idx <= 3);
    return (&x)[idx];
}
vec4 operator+(const vec4& a, const vec4& b) {
    return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
vec4 operator*(const vec4& a, float b) {
    return vec4(a.x * b, a.y * b, a.z * b, a.w * b);
}
vec4 operator*(float a, const vec4& b) { return b * a; }

// mat4
mat4::mat4() {}

mat4::mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3) : data(v0, v1, v2, v3) {}

mat4::mat4(float i)
    : data({i, 0.0f, 0.0f, 0.0f},
           {0.0f, i, 0.0f, 0.0f},
           {0.0f, 0.0f, i, 0.0f},
           {0.0f, 0.0f, 0.0f, i}) {}

vec4& mat4::operator[](int idx) {
    assert(idx >= 0 && idx <= 3);
    return data[idx];
}

const vec4& mat4::operator[](int idx) const {
    assert(idx >= 0 && idx <= 3);
    return data[idx];
}

mat4 mat4::identity() { return mat4(1.0f); }

// Functions

float radians(float degrees) { return degrees * std::numbers::pi / 180; }

vec3 normalize(const vec3& vector) {
    const auto isqrt = 1 / std::hypot(vector.x, vector.y, vector.z);
    return vec3(vector.x * isqrt, vector.y * isqrt, vector.z * isqrt);
}

vec3 cross(const vec3& a, const vec3& b) {
    return vec3({a.y * b.z - a.z * b.y},
                {a.z * b.x - a.x * b.z},
                {a.x * b.y - a.y * b.x});
}

float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

mat4 look_at(const vec3& position, const vec3& target, const vec3& up) {
    const auto f = normalize(target - position);
    const auto s = normalize(cross(f, up));
    const auto u = cross(s, f);

    mat4 result(1.0f);
    result[0][0] = s.x;
    result[1][0] = s.y;
    result[2][0] = s.z;
    result[0][1] = u.x;
    result[1][1] = u.y;
    result[2][1] = u.z;
    result[0][2] = -f.x;
    result[1][2] = -f.y;
    result[2][2] = -f.z;
    result[3][0] = -dot(s, position);
    result[3][1] = -dot(u, position);
    result[3][2] = dot(f, position);
    return result;
};

mat4 perspesctive(float fov,
                  float aspect,
                  float near_clipping,
                  float far_clipping) {
    assert(aspect != 0.0f);
    assert(near_clipping != far_clipping);
    const float half_fov_tan = std::tan(fov / 2);

    mat4 result{};
    result[0][0] = 1 / (aspect * half_fov_tan);
    result[1][1] = 1 / half_fov_tan;
    result[2][2] =
        -(far_clipping + near_clipping) / (far_clipping - near_clipping);
    result[2][3] = -1;
    result[3][2] =
        -(2 * near_clipping * far_clipping) / (far_clipping - near_clipping);

    return result;
}

mat4 rotate(const mat4& matrix, float angle, const vec3& vector) {
    const auto cos = std::cos(angle);
    const auto sin = std::sin(angle);

    const auto axis = normalize(vector);
    const auto temp = (1.0f - cos) * axis;

    mat4 rotate{};
    rotate[0][0] = cos + temp[0] * axis[0];
    rotate[0][1] = 0 + temp[0] * axis[1] + sin * axis[2];
    rotate[0][2] = 0 + temp[0] * axis[2] - sin * axis[1];

    rotate[1][0] = 0 + temp[1] * axis[0] - sin * axis[2];
    rotate[1][1] = cos + temp[1] * axis[1];
    rotate[1][2] = 0 + temp[1] * axis[2] + sin * axis[0];

    rotate[2][0] = 0 + temp[2] * axis[0] + sin * axis[1];
    rotate[2][1] = 0 + temp[2] * axis[1] - sin * axis[0];
    rotate[2][2] = cos + temp[2] * axis[2];

    mat4 result{};
    result[0] = matrix[0] * rotate[0][0] + matrix[1] * rotate[0][1] +
                matrix[2] * rotate[0][2];
    result[1] = matrix[0] * rotate[1][0] + matrix[1] * rotate[1][1] +
                matrix[2] * rotate[1][2];
    result[2] = matrix[0] * rotate[2][0] + matrix[1] * rotate[2][1] +
                matrix[2] * rotate[2][2];
    result[3] = matrix[3];
    return result;
}

} // namespace math
