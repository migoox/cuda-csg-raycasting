#define GLM_FORCE_CUDA
#include <vector>

#include "imgui_internal.h"
#include "imgui.h"
#include "vendor/imgui/backend/imgui_impl_glfw.h"
#include "vendor/imgui/backend/imgui_impl_opengl3.h"
#include <stdio.h>
#include "gl_debug.h"
#include "image.hpp"

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

// This example can also compile and run with Emscripten! See 'Makefile.emscripten' for details.
#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif

const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;

static int curr_win_pos_x = 0;
static int curr_win_pos_y = 0;

static int curr_scr_width = SCREEN_WIDTH;
static int curr_scr_height = SCREEN_HEIGHT;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void on_pos(GLFWwindow* window, int x, int y) {
    curr_win_pos_x = x;
    curr_win_pos_y = y;
}

static void on_resize(GLFWwindow* window, int width, int height) {
    curr_scr_height = height;
    curr_scr_width = width;
}
// Function to retrieve supported resolutions
std::vector<std::pair<int, int>> get_supported_resolutions() {
    std::vector<std::pair<int, int>> resolutions;
    // Add supported resolutions (You can use GLFW functions to get available modes)
    resolutions.emplace_back(1920, 1080);
    resolutions.emplace_back(1280, 720);
    resolutions.emplace_back(1024, 768);

    return resolutions;
}

// Main code
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "CSG Ray Tracing", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable vsync
    glfwSetWindowSizeCallback(window, on_resize);
    glfwSetWindowPosCallback(window, on_pos);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);

#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCanvasResizeCallback("#canvas");
#endif
    ImGui_ImplOpenGL3_Init(glsl_version);

    // INIT
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    img::Image canvas(640, 360);
    canvas.clear(0x0000FFFF);

    GLuint canvas_texture;
    GLCall(glGenTextures(1, &canvas_texture));
    GLCall(glBindTexture(GL_TEXTURE_2D, canvas_texture));
    GLCall(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        canvas.get_width(), canvas.get_height(), 0,
                        GL_RGBA, GL_UNSIGNED_INT_8_8_8_8,
                        canvas.raw()));

    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
    bool is_fullscreen = false;
    ImVec2 old_size = ImVec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    ImVec2 old_pos = ImVec2(0, 0);

    // Get supported resolutions
    std::vector<std::pair<int, int>> supp_res = get_supported_resolutions();
    char res_items[1024];
    {
        size_t curr_offset = 0;
        for (int i = 0; i < supp_res.size(); i++) {
            std::string curr_res = std::to_string(supp_res[i].first) + "x" + std::to_string(supp_res[i].second);
            strcpy(res_items + curr_offset, curr_res.c_str());
            curr_offset += curr_res.size() + 1;
        }
        res_items[curr_offset] = '\0';
    }
    int selected_res = 0;

    // Main loop
#ifdef __EMSCRIPTEN__
    // For an Emscripten build we are disabling file-system access, so let's not attempt to do a fopen() of the imgui.ini file.
    // You may manually call LoadIniSettingsFromMemory() to load settings from your own storage.
    io.IniFilename = nullptr;
    EMSCRIPTEN_MAINLOOP_BEGIN
#else
    while (!glfwWindowShouldClose(window))
#endif
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        auto height = static_cast<float>(curr_scr_height);
        auto width_win1 = static_cast<float>(curr_scr_width) / 4.f * 3.f;
        auto width_win2 = static_cast<float>(curr_scr_width) / 4.f;
        ImVec2 win1_size = ImVec2(width_win1, height);
        ImVec2 win2_size = ImVec2(width_win2, height);

        if (is_fullscreen) {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(curr_scr_width, curr_scr_height));
            win1_size = ImVec2(curr_scr_width, curr_scr_height);
        } else {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(win1_size);
        }

        ImGui::Begin("Window 1", nullptr, windowFlags);

        if (ImGui::Button("Toggle Fullscreen")) {
            is_fullscreen = !is_fullscreen;

            if (is_fullscreen) {
                old_size = ImVec2(curr_scr_width, curr_scr_height);
                old_pos = ImVec2(curr_win_pos_x, curr_win_pos_y);
                GLFWmonitor* monitor = glfwGetPrimaryMonitor();
                const GLFWvidmode* mode = glfwGetVideoMode(monitor);
                glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
            } else {
                glfwSetWindowMonitor(window, nullptr, old_pos.x, old_pos.y, old_size.x, old_size.y, GLFW_DONT_CARE);
            }
        }

        // Display supported resolutions in a combo box
        ImGui::SameLine();
        if (ImGui::Combo("##Resolution", &selected_res, res_items, std::max(6, (int)supp_res.size()))) {

        }

        // Calculate the position to center the image
        ImVec2 img_size(canvas.get_width(), canvas.get_height());
        ImVec2 img_pos = ImGui::GetCursorScreenPos();
        img_pos.x += (win1_size.x - img_size.x) * 0.5f;
        img_pos.y += (win1_size.y - img_size.y) * 0.5f;

        // Display the image at the calculated position
        ImGui::SetCursorScreenPos(img_pos);
        ImGui::Image((void*)(intptr_t)canvas_texture, img_size);
        ImGui::End();

        if (!is_fullscreen) {
            ImGui::SetNextWindowPos(ImVec2(width_win1, 0));
            ImGui::SetNextWindowSize(win2_size);
            ImGui::Begin("Window 2", nullptr, windowFlags);

            ImGui::End();
        }


        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
