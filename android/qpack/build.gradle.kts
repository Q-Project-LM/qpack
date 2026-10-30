plugins { id("com.android.library"); id("org.jetbrains.kotlin.android") }

android {
    namespace = "ai.qproject.qpack"
    compileSdk = 36
    ndkVersion = "27.1.12297006"
    defaultConfig {
        minSdk = 26
        ndk { abiFilters += "arm64-v8a" }   // phones and watches (Galaxy Watch 8: Exynos W1000, ARMv8.2 + dotprod)
        externalNativeBuild { cmake { arguments += listOf("-DQPACK_BUILD_TOOLS=OFF", "-DQPACK_SIZE_OPT=ON", "-DANDROID_STL=c++_static") } }
    }
    externalNativeBuild { cmake { path = file("../../CMakeLists.txt"); version = "3.22.1" } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
