#ifndef CSG_RAY_TRACING_BILLBOARD_H
#define CSG_RAY_TRACING_BILLBOARD_H

#include <GL/glew.h>

namespace app {
    class Billboard {
    public:
        Billboard();
        ~Billboard();

        void draw();

    private:
        GLuint m_vao, m_vbo, m_ebo;
    };
}


#endif //CSG_RAY_TRACING_BILLBOARD_H
