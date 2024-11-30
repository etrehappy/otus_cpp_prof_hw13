cmake_minimum_required(VERSION 3.26)

project(fashio_mnist_prj)

add_executable(fashio_mnist 
    main.cpp
    logreg_classifier.h logreg_classifier.cpp
    helpers.h helpers.cpp
    classifier.h

)


target_compile_options(fashio_mnist PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set_target_properties(fashio_mnist PROPERTIES
CXX_STANDARD 20
CXX_STANDARD_REQUIRED ON
) 


