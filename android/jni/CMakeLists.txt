# libqpack_jni.so: JNI bridge for the Kotlin API in android/qpack (built by the top-level CMakeLists.txt when ANDROID is set)
add_library(qpack_jni SHARED qpack_jni.cpp)
target_link_libraries(qpack_jni PRIVATE qpack log)
# smaller .so: drop unused sections, keep only the JNI entry points exported
target_compile_options(qpack_jni PRIVATE -fvisibility=hidden)
target_link_options(qpack_jni PRIVATE -Wl,-s -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384)
