function(set_default_value var default_value)
  if(NOT ${var} OR ${var} STREQUAL "")
    set(${var}
        ${default_value}
        CACHE STRING "" FORCE)
    message(STATUS "${var}: ${default_value}")
  endif()
endfunction()
