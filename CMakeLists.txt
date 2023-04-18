cmake_minimum_required(VERSION 3.25)

project("Game_Engine" CXX)

add_executable(${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)


set( ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SHADERS ${SHADER_DIR}/*.vert
                  ${SHADER_DIR}/*.tesc
                  ${SHADER_DIR}/*.tese
                  ${SHADER_DIR}/*.geom
                  ${SHADER_DIR}/*.frag
                  ${SHADER_DIR}/*.comp
                  ${SHADER_DIR}/*.mesh
                  ${SHADER_DIR}/*.task
                  ${SHADER_DIR}/*.rgen
                  ${SHADER_DIR}/*.rint
                  ${SHADER_DIR}/*.rahit
                  ${SHADER_DIR}/*.rchit
                  ${SHADER_DIR}/*.rmiss
                  ${SHADER_DIR}/*.rcall)

find_package(Vulkan COMPONENTS glslangValidator)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders/)
set(SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
foreach(SHADER IN LISTS SHADERS)
    get_filename_component(FILENAME ${SHADER} NAME)
        add_custom_command(OUTPUT ${SPV_DIR}/${FILENAME}.spv
            COMMAND Vulkan::glslangValidator -V --target-env vulkan1.3 --quiet ${SHADER} -o ${SPV_DIR}/${FILENAME}.spv
            DEPENDS ${SHADER}
            COMMENT "Compiling ${FILENAME}")
            list(APPEND SPV_SHADERS ${SPV_DIR}/${FILENAME}.spv)
endForeach()
add_custom_target(shaders ALL DEPENDS ${SPV_SHADERS})


add_subdirectory(src)
add_subdirectory(third_party)

add_dependencies(${PROJECT_NAME} shaders)
target_link_libraries(${PROJECT_NAME} sdl2 Vulkan::Vulkan)
