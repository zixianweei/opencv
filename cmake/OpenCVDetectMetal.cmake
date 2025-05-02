if (NOT APPLE)
  set(HAVE_METAL 0)
  return()
endif()

set(METAL_LIBRARIES "-framework Foundation;-framework Metal;-framework QuartzCore" CACHE STRING "Metal library")

try_compile(VALID_METAL
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/metal.mm"
    CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=${METAL_LIBRARIES}"
)

if(VALID_METAL)
  set(HAVE_METAL 1)
else()
  set(HAVE_METAL 0)
endif()

set(METALLIB_CFLAGS -Wall -Wextra -fno-fast-math)
if (WERROR)
    string(APPEND METALLIB_CFLAGS -Werror)
endif()

function(ocv_metallib_metal_to_air SRC TGT FLAGS)
  add_custom_command(
    COMMAND xcrun metal -c ${SRC} -I ${CMAKE_SOURCE_DIR} -o ${TGT} ${FLAGS} ${METALLIB_CFLAGS}
    DEPENDS ${SRC}
    OUTPUT ${TGT}
    COMMENT "Compiling ${SRC} to ${TGT}"
    VERBATIM
  )
endfunction()

function(ocv_metallib_air_to_metallib TGT OBJS)
  set(_OBJECTS ${OBJS} ${ARGN})
  add_custom_command(
    COMMAND xcrun metallib -o ${CMAKE_BINARY_DIR}/${TGT} ${_OBJECTS}
    DEPENDS ${_OBJECTS}
    OUTPUT ${TGT}
    COMMENT "Linking ${TGT}"
    VERBATIM
  )
endfunction()

function(ocv_metallib_compile_shaders SHADERS)
  foreach(SHADER ${SHADERS})
    cmake_path(GET SHADER STEM TGT_STEM)
    string(CONCAT SHADER_AIR ${TGT_STEM} ".air")
    list(APPEND SHADERS_AIR ${SHADER_AIR})
    ocv_metallib_metal_to_air(${SHADER} ${SHADER_AIR} "")
  endforeach()
  ocv_metallib_air_to_metallib(OpenCV.metallib ${SHADERS_AIR})
endfunction()
