# ctest driver: pipe two requests through the JSON-lines shim and check
# that the first succeeds and the second reports an error.
execute_process(COMMAND ${SHIM} INPUT_FILE ${INPUT}
                OUTPUT_VARIABLE out RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "shim exited with ${rc}")
endif()
if(NOT out MATCHES "\"id\":1,\"ok\":true,\"out\":{\"price\":10.45058357218")
  message(FATAL_ERROR "unexpected success line: ${out}")
endif()
if(NOT out MATCHES "\"id\":2,\"ok\":false,\"err\":\"S must be finite|err\":\"Spot S must")
  message(FATAL_ERROR "unexpected error line: ${out}")
endif()
