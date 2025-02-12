set(CPACK_VERBATIM_VARIABLES YES)

set(CPACK_PACKAGE_NAME "acpp")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Implementation of SYCL and C++ standard parallelism for CPUs and GPUs from all vendors.")
set(CPACK_PACKAGE_VENDOR "AdaptiveCpp")
set(CPACK_PACKAGE_CONTACT "Sanchi Vaishnavi sanchi.vaishnavi@stud.uni-heidelberg.de")

set(CPACK_PACKAGE_VERSION_MAJOR ${ACPP_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${ACPP_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${ACPP_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION_EXTRA "${ACPP_GIT_COMMIT_HASH}+${ACPP_GIT_DATE}")
string(CONCAT CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}"
                                    ".${CPACK_PACKAGE_VERSION_MINOR}"
                                    ".${CPACK_PACKAGE_VERSION_PATCH}"
                                    "~${CPACK_PACKAGE_VERSION_EXTRA}")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

SET(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_SOURCE_DIR}/_packages")
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/AdaptiveCpp")

set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
set(CPACK_BINARY_DEB ON)
set(CPACK_DEB_COMPONENT_INSTALL YES)
set(CPACK_DEBIAN_PACKAGE_NAME acpp)


include(CPack)

message(STATUS "Components to pack: ${CPACK_COMPONENTS_ALL}")