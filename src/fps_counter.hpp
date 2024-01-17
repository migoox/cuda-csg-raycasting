#ifndef CUDA_CSG_RAYCASTING_FPS_COUNTER_H
#define CUDA_CSG_RAYCASTING_FPS_COUNTER_H

#include <cstdint>

namespace utils {
    class FPSCounter {
    public:
        explicit FPSCounter(double interval = 0.5);

        void update(double delta_time);

        double get_curr_fps() const { return m_curr_fps; }

    private:
        double m_interval;
        uint32_t m_frames_count = 0;

        double m_deltas_sum = 0.0;
        double m_curr_fps = 0.0;
    };

}

#endif //CUDA_CSG_RAYCASTING_FPS_COUNTER_H
