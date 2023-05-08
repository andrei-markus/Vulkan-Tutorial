#ifndef MATHLIB_HPP
#define MATHLIB_HPP

namespace math {

struct vec2 {
    float x;
    float y;
};

struct vec3 {
    float x;
    float y;
    float z;

    float& operator[](int idx);
    const float& operator[](int idx) const;
};
vec3 operator-(const vec3& a, const vec3& b);
vec3 operator*(const vec3& a, float b);
vec3 operator*(float a, const vec3& b);

struct vec4 {
    float x;
    float y;
    float z;
    float w;

    float& operator[](int idx);
    const float& operator[](int idx) const;
};
vec4 operator+(const vec4& a, const vec4& b);
vec4 operator*(const vec4& a, float b);
vec4 operator*(float a, const vec4& b);

struct mat4 {
    vec4 data[4]{};
    mat4();
    mat4(vec4 v0, vec4 v1, vec4 v2, vec4 v3);
    mat4(float i);

    vec4& operator[](int idx);
    const vec4& operator[](int idx) const;

    static mat4 identity();
};

// Functions

float radians(float degrees);
vec3 normalize(const vec3& vector);
vec3 cross(const vec3& a, const vec3& b);
float dot(const vec3& a, const vec3& b);
mat4 look_at(const vec3& position, const vec3& target, const vec3& up);

mat4 perspesctive(float fov,
                  float aspect,
                  float near_clipping,
                  float far_clipping);

mat4 rotate(const mat4& matrix, float angle, const vec3& vector);

} // namespace math

#endif
