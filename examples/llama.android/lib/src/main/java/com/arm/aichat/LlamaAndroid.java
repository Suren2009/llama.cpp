package com.arm.aichat;

import android.content.Context;

import java.io.File;
import java.io.IOException;

/**
 * Java API for loading GGUF models and running llama.cpp inference on Android.
 *
 * <p>This class owns one native model session. Calls are synchronized because the
 * native llama.cpp context is stateful.</p>
 */
public final class LlamaAndroid implements AutoCloseable {
    public static final int DEFAULT_PREDICT_LENGTH = 1024;

    private static boolean nativeInitialized;

    /**
     * Loads the native library and initializes llama.cpp backends.
     *
     * @param context Android context used to locate packaged native libraries.
     */
    public LlamaAndroid(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("Context cannot be null");
        }

        synchronized (LlamaAndroid.class) {
            if (!nativeInitialized) {
                System.loadLibrary("ai-chat");
                nativeInit(context.getApplicationInfo().nativeLibraryDir);
                nativeInitialized = true;
            }
        }
    }

    /**
     * Loads a base GGUF model.
     *
     * @param modelPath readable filesystem path to the GGUF model.
     */
    public synchronized void loadModel(String modelPath) throws IOException {
        loadModel(modelPath, null);
    }

    /**
     * Loads a base GGUF model and optionally applies a LoRA adapter.
     *
     * @param modelPath readable filesystem path to the GGUF model.
     * @param loraPath readable filesystem path to a LoRA adapter GGUF, or null.
     */
    public synchronized void loadModel(String modelPath, String loraPath) throws IOException {
        requireReadableFile(modelPath, "model");
        if (loraPath != null && !loraPath.isEmpty()) {
            requireReadableFile(loraPath, "LoRA adapter");
        }

        nativeLoadModel(modelPath, loraPath);
    }

    /**
     * Returns a normalized embedding vector for {@code input}.
     *
     * @throws UnsupportedOperationException when the loaded model is not encoder-only.
     */
    public synchronized float[] getEmbeddings(String input) {
        requireInput(input);
        return nativeGetEmbeddings(input);
    }

    /**
     * Decodes from {@code input} using {@link #DEFAULT_PREDICT_LENGTH}.
     *
     * @throws UnsupportedOperationException when the loaded model does not support decoding.
     */
    public synchronized String decode(String input) {
        return decode(input, DEFAULT_PREDICT_LENGTH);
    }

    /**
     * Decodes from {@code input}.
     *
     * @param predictLength maximum number of tokens to generate.
     * @throws UnsupportedOperationException when the loaded model does not support decoding.
     */
    public synchronized String decode(String input, int predictLength) {
        requireInput(input);
        if (predictLength <= 0) {
            throw new IllegalArgumentException("Predict length must be positive");
        }

        return nativeDecode(input, predictLength);
    }

    /**
     * Releases the loaded model and native resources owned by this Java API.
     */
    public synchronized void release() {
        nativeRelease();
    }

    @Override
    public void close() {
        release();
    }

    private static void requireInput(String input) {
        if (input == null || input.isEmpty()) {
            throw new IllegalArgumentException("Input cannot be empty");
        }
    }

    private static void requireReadableFile(String path, String label) {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException("Path to " + label + " cannot be empty");
        }

        File file = new File(path);
        if (!file.exists()) {
            throw new IllegalArgumentException("Path to " + label + " does not exist: " + path);
        }
        if (!file.isFile()) {
            throw new IllegalArgumentException("Path to " + label + " is not a file: " + path);
        }
        if (!file.canRead()) {
            throw new IllegalArgumentException("Path to " + label + " is not readable: " + path);
        }
    }

    private static native void nativeInit(String nativeLibDir);

    private native void nativeLoadModel(String modelPath, String loraPath) throws IOException;

    private native float[] nativeGetEmbeddings(String input);

    private native String nativeDecode(String input, int predictLength);

    private native void nativeRelease();
}
