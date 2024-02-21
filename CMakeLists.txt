cmake_minimum_required(VERSION 3.25)

project("Vulkan_Tutorial")

add_executable(${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

add_compile_definitions(
                    $<$<CONFIG:Debug>:_DEBUG>
                    $<$<CONFIG:RelWithDebInfo>:_REL_DEBUG>
                    $<$<CONFIG:Release>:_RELEASE>
                    $<$<CONFIG:Release>:NDEBUG>
                    $<$<CONFIG:MinSizeRel>:_RELEASE>
                    $<$<CONFIG:MinSizeRel>:NDEBUG>)

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/models/ $<TARGET_FILE_DIR:${PROJECT_NAME}>/models/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/textures/ $<TARGET_FILE_DIR:${PROJECT_NAME}>/textures/
)

find_package(Vulkan COMPONENTS glslangValidator)

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD

    COMMAND Vulkan::glslangValidator -V --target-env vulkan1.3 --quiet
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag -o $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/shader.frag.spv
    COMMAND Vulkan::glslangValidator -V --target-env vulkan1.3 --quiet
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert -o $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/shader.vert.spv
)

add_subdirectory(src)
add_subdirectory(third_party)

target_link_libraries(${PROJECT_NAME} SDL2::SDL2main SDL2::SDL2 Vulkan::Vulkan stb tinyobj)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND_EXPAND_LISTS
)

install(TARGETS ${PROJECT_NAME})
install(FILES $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> TYPE BIN)
install(DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/ DESTINATION ${CMAKE_INSTALL_BINDIR}/shaders/)
install(DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>/models/ DESTINATION ${CMAKE_INSTALL_BINDIR}/models/)
install(DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>/textures/ DESTINATION ${CMAKE_INSTALL_BINDIR}/textures/)
