# CMake generated Testfile for 
# Source directory: /root/app/lib/opencv-4.x/include/modules/imgcodecs
# Build directory: /root/app/lib/opencv-4.x/build/modules/imgcodecs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_imgcodecs "/root/app/lib/opencv-4.x/build/bin/opencv_test_imgcodecs" "--gtest_output=xml:opencv_test_imgcodecs.xml")
set_tests_properties(opencv_test_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Accuracy" WORKING_DIRECTORY "/root/app/lib/opencv-4.x/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/root/app/lib/opencv-4.x/include/cmake/OpenCVUtils.cmake;1799;add_test;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1365;ocv_add_test_from_target;/root/app/lib/opencv-4.x/include/modules/imgcodecs/CMakeLists.txt;201;ocv_add_accuracy_tests;/root/app/lib/opencv-4.x/include/modules/imgcodecs/CMakeLists.txt;0;")
add_test(opencv_perf_imgcodecs "/root/app/lib/opencv-4.x/build/bin/opencv_perf_imgcodecs" "--gtest_output=xml:opencv_perf_imgcodecs.xml")
set_tests_properties(opencv_perf_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Performance" WORKING_DIRECTORY "/root/app/lib/opencv-4.x/build/test-reports/performance" _BACKTRACE_TRIPLES "/root/app/lib/opencv-4.x/include/cmake/OpenCVUtils.cmake;1799;add_test;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1264;ocv_add_test_from_target;/root/app/lib/opencv-4.x/include/modules/imgcodecs/CMakeLists.txt;212;ocv_add_perf_tests;/root/app/lib/opencv-4.x/include/modules/imgcodecs/CMakeLists.txt;0;")
add_test(opencv_sanity_imgcodecs "/root/app/lib/opencv-4.x/build/bin/opencv_perf_imgcodecs" "--gtest_output=xml:opencv_perf_imgcodecs.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Sanity" WORKING_DIRECTORY "/root/app/lib/opencv-4.x/build/test-reports/sanity" _BACKTRACE_TRIPLES "/root/app/lib/opencv-4.x/include/cmake/OpenCVUtils.cmake;1799;add_test;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1265;ocv_add_test_from_target;/root/app/lib/opencv-4.x/include/modules/imgcodecs/CMakeLists.txt;212;ocv_add_perf_tests;/root/app/lib/opencv-4.x/include/modules/imgcodecs/CMakeLists.txt;0;")
