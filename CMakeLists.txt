cmake_minimum_required(VERSION 3.25)

project("Game_Engine")

add_executable(${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

add_compile_definitions(
                    $<$<CONFIG:Debug>:_DEBUG>
                    $<$<CONFIG:RelWithDebInfo>:_REL_DEBUG>
                    $<$<CONFIG:Release>:_RELEASE>
                    $<$<CONFIG:Release>:NDEBUG>
                    $<$<CONFIG:MinSizeRel>:_RELEASE>
                    $<$<CONFIG:MinSizeRel>:NDEBUG>)

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

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/models/)
set(MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/models)
file(GLOB MODELS ${CMAKE_CURRENT_SOURCE_DIR}/models/*)
foreach(MODEL IN LISTS MODELS)
    get_filename_component(FILENAME ${MODEL} NAME)
    configure_file(${MODEL} ${MODEL_DIR}/${FILENAME} COPYONLY)
endForeach()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/textures/)
set(TEX_DIR ${CMAKE_CURRENT_BINARY_DIR}/textures)
file(GLOB TEXTURES ${CMAKE_CURRENT_SOURCE_DIR}/textures/*)
foreach(TEXTURE IN LISTS TEXTURES)
    get_filename_component(FILENAME ${TEXTURE} NAME)
    configure_file(${TEXTURE} ${TEX_DIR}/${FILENAME} COPYONLY)
endForeach()

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
target_link_libraries(${PROJECT_NAME} sdl2 Vulkan::Vulkan stb tinyobj)
