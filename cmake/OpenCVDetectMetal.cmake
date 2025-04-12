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
