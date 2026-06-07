# llama.cpp Android Release Binaries

This folder contains pre-built release binaries and integration guidelines for executing GGUF models and LoRA adapters on Android devices with optimized performance.

## Folder Contents

*   **[llama-android.aar](file:///d:/Suren/Projects/llama.cpp/examples/llama.android/release/llama-android.aar)**: Android Archive (AAR) library providing Java/JNI bindings for `llama.cpp`.
*   **[llama-android.apk](file:///d:/Suren/Projects/llama.cpp/examples/llama.android/release/llama-android.apk)**: A ready-to-install, optimized debug-signed Release version of the sample chat/embeddings app.
*   **[README.md](file:///d:/Suren/Projects/llama.cpp/examples/llama.android/release/README.md)**: This integration guide.

---

## 1. Installing & Running the Sample App (`llama-android.apk`)

The sample application lets you load GGUF models directly from external storage, configure optional LoRA adapters, run sequence decoding/chat, and compute both normalized and raw embeddings on-device.

### Installation Steps

1.  Connect your Android device (ensure Developer Options and USB Debugging are active).
2.  Install the APK via ADB:
    ```bash
    adb install llama-android.apk
    ```
3.  Grant **All Files Access** (`MANAGE_EXTERNAL_STORAGE`) so the app can load GGUF model files directly from local storage:
    ```bash
    adb shell appops set com.example.llama.aichat MANAGE_EXTERNAL_STORAGE allow
    ```

### Model Setup Directory
By default, the sample app auto-detects models placed at the following location:
*   **Base Model**: `/storage/emulated/0/Download/sse/ranker-gguf/base_model.gguf`
*   **LoRA Adapter**: `/storage/emulated/0/Download/sse/ranker-gguf/ranker_lora_adapter.gguf`

*Tip: If you do not have files at these paths, you can manually select the GGUF model files using the in-app file chooser.*

---

## 2. Integrating `llama-android.aar` in Your Android Project

Follow these steps to import the prebuilt `llama-android.aar` into your own Android studio projects.

### Step A: Place the AAR File
Copy `llama-android.aar` into your application module's `libs` directory (e.g. `app/libs/`).

### Step B: Configure Build Files
Add the binary dependency to your module-level `build.gradle.kts` (or `build.gradle`):

```kotlin
dependencies {
    // Reference the local AAR file
    implementation(files("libs/llama-android.aar"))
    
    // Add required dependencies utilized by the library:
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.datastore:datastore-preferences:1.1.2")
}
```

Ensure your project configures target CPU ABIs. The native libraries inside the AAR support `arm64-v8a` and `x86_64`:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }
}
```

---

## 3. Java/Kotlin API Usage Reference

Once imported, instantiate and use the wrapper `LlamaAndroid` to perform inference and retrieve embeddings.

### Core API Methods

#### Class Initialization
```java
import com.arm.aichat.LlamaAndroid;

// Initialize the native environment
LlamaAndroid llama = new LlamaAndroid(context);
```

#### Loading Models
```java
// Load a base model
llama.loadModel("/path/to/model.gguf");

// Or load a base model with a LoRA adapter
llama.loadModel("/path/to/model.gguf", "/path/to/lora.gguf");
```

#### Fetching Normalized Embeddings
Computes the pooled and L2-normalized embedding vector (dimension of vector based on model specification, e.g. 384 floats):
```java
float[] embeddings = llama.getEmbeddings("Your prompt text here");
```

#### Fetching Raw, Unnormalized Token Embeddings
Computes raw token-level embeddings (returns a float array of size `number_of_tokens * vector_dimension`):
```java
float[] rawEmbeddings = llama.getRawEmbeddings("Your prompt text here");
```

---

## 4. Releasing & Closing Resources
Always close or release the model session when finished to free up native memory buffers:

```java
// Release native resources
llama.close(); 
```
