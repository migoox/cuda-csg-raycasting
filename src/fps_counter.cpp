#include "fps_counter.hpp"

utils::FPSCounter::FPSCounter(double interval)
: m_interval(interval) { }

void utils::FPSCounter::update(double delta_time) {
    m_frames_count++;
    m_deltas_sum += delta_time;
    if (m_deltas_sum >= m_interval) {
        m_curr_fps =  static_cast<double>(m_frames_count) / m_deltas_sum;
        m_deltas_sum = 0.0;
        m_frames_count = 0;
    }
}


