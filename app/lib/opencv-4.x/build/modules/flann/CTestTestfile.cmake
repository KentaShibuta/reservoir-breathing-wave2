# CMake generated Testfile for 
# Source directory: /root/app/lib/opencv-4.x/include/modules/flann
# Build directory: /root/app/lib/opencv-4.x/build/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "/root/app/lib/opencv-4.x/build/bin/opencv_test_flann" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "/root/app/lib/opencv-4.x/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/root/app/lib/opencv-4.x/include/cmake/OpenCVUtils.cmake;1799;add_test;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1365;ocv_add_test_from_target;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1123;ocv_add_accuracy_tests;/root/app/lib/opencv-4.x/include/modules/flann/CMakeLists.txt;2;ocv_define_module;/root/app/lib/opencv-4.x/include/modules/flann/CMakeLists.txt;0;")
