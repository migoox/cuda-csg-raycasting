#ifndef CSG_RAY_TRACING_CUDA_RAY_TRACER_CUH
#define CSG_RAY_TRACING_CUDA_RAY_TRACER_CUH

#include <glm/glm.hpp>
#include <cstdint>
uint32_t per_pixel(int x, int y, glm::vec2 canvas, glm::vec3 eye, glm::mat4 inv_proj, glm::mat4 inv_view);


#endif //CSG_RAY_TRACING_CUDA_RAY_TRACER_CUH
