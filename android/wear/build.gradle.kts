plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

android {
    namespace = "ai.qproject.wear"
    compileSdk = 36
    defaultConfig { applicationId = "ai.qproject.wear"; minSdk = 30; targetSdk = 34; versionCode = 1; versionName = "0.1" }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    packaging { jniLibs { useLegacyPackaging = false } }   // uncompressed, page-aligned .so
}

dependencies {
    implementation(project(":qpack"))
}
