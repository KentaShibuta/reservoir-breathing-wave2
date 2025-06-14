# CMake generated Testfile for 
# Source directory: /root/app/lib/opencv-4.x/include/modules/videoio
# Build directory: /root/app/lib/opencv-4.x/build/modules/videoio
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_videoio "/root/app/lib/opencv-4.x/build/bin/opencv_test_videoio" "--gtest_output=xml:opencv_test_videoio.xml")
set_tests_properties(opencv_test_videoio PROPERTIES  LABELS "Main;opencv_videoio;Accuracy" WORKING_DIRECTORY "/root/app/lib/opencv-4.x/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/root/app/lib/opencv-4.x/include/cmake/OpenCVUtils.cmake;1799;add_test;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1365;ocv_add_test_from_target;/root/app/lib/opencv-4.x/include/modules/videoio/CMakeLists.txt;281;ocv_add_accuracy_tests;/root/app/lib/opencv-4.x/include/modules/videoio/CMakeLists.txt;0;")
add_test(opencv_perf_videoio "/root/app/lib/opencv-4.x/build/bin/opencv_perf_videoio" "--gtest_output=xml:opencv_perf_videoio.xml")
set_tests_properties(opencv_perf_videoio PROPERTIES  LABELS "Main;opencv_videoio;Performance" WORKING_DIRECTORY "/root/app/lib/opencv-4.x/build/test-reports/performance" _BACKTRACE_TRIPLES "/root/app/lib/opencv-4.x/include/cmake/OpenCVUtils.cmake;1799;add_test;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1264;ocv_add_test_from_target;/root/app/lib/opencv-4.x/include/modules/videoio/CMakeLists.txt;282;ocv_add_perf_tests;/root/app/lib/opencv-4.x/include/modules/videoio/CMakeLists.txt;0;")
add_test(opencv_sanity_videoio "/root/app/lib/opencv-4.x/build/bin/opencv_perf_videoio" "--gtest_output=xml:opencv_perf_videoio.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_videoio PROPERTIES  LABELS "Main;opencv_videoio;Sanity" WORKING_DIRECTORY "/root/app/lib/opencv-4.x/build/test-reports/sanity" _BACKTRACE_TRIPLES "/root/app/lib/opencv-4.x/include/cmake/OpenCVUtils.cmake;1799;add_test;/root/app/lib/opencv-4.x/include/cmake/OpenCVModule.cmake;1265;ocv_add_test_from_target;/root/app/lib/opencv-4.x/include/modules/videoio/CMakeLists.txt;282;ocv_add_perf_tests;/root/app/lib/opencv-4.x/include/modules/videoio/CMakeLists.txt;0;")
