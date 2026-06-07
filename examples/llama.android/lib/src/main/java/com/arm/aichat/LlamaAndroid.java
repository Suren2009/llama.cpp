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
        float[] raw = getRawEmbeddings(input);
        if (raw == null || raw.length == 0) {
            return null;
        }

        int dim = nativeGetEmbeddingDimension();
        if (dim <= 0) {
            return null;
        }

        int numTokens = raw.length / dim;
        int poolingType = nativeGetPoolingType();

        float[] pooled = new float[dim];
        if (poolingType == 1) { // LLAMA_POOLING_TYPE_MEAN
            for (int i = 0; i < numTokens; ++i) {
                for (int d = 0; d < dim; ++d) {
                    pooled[d] += raw[i * dim + d];
                }
            }
            for (int d = 0; d < dim; ++d) {
                pooled[d] /= numTokens;
            }
        } else if (poolingType == 2) { // LLAMA_POOLING_TYPE_CLS
            System.arraycopy(raw, 0, pooled, 0, dim);
        } else if (poolingType == 3) { // LLAMA_POOLING_TYPE_LAST
            System.arraycopy(raw, (numTokens - 1) * dim, pooled, 0, dim);
        } else { // default fallback
            System.arraycopy(raw, (numTokens - 1) * dim, pooled, 0, dim);
        }

        // L2 Normalization (embd_norm = 2)
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            sum += pooled[d] * pooled[d];
        }
        double norm = Math.sqrt(sum);
        float normFactor = norm > 0.0 ? (float) (1.0 / norm) : 0.0f;
        for (int d = 0; d < dim; ++d) {
            pooled[d] *= normFactor;
        }

        return pooled;
    }

    /**
     * Returns raw, unnormalized embeddings for {@code input}.
     */
    public synchronized float[] getRawEmbeddings(String input) {
        requireInput(input);
        return nativeGetRawEmbeddings(input);
    }

    /**
     * Returns raw, unnormalized embeddings for {@code tokens}.
     */
    public synchronized float[] getRawEmbeddings(int[] tokens) {
        if (tokens == null || tokens.length == 0) {
            throw new IllegalArgumentException("Tokens cannot be empty");
        }
        return nativeGetRawEmbeddingsFromTokens(tokens);
    }

    public synchronized int[] tokenize(String text, boolean addSpecial) {
        requireInput(text);
        return nativeTokenize(text, addSpecial);
    }

    public synchronized int getBosToken() {
        return nativeGetTokenBos();
    }

    public synchronized int getEosToken() {
        return nativeGetTokenEos();
    }

    /**
     * Supporting backward-compatible names and spelling variations.
     */
    public synchronized float[] getEmbeddingWithoutNormalized(String input) {
        return getRawEmbeddings(input);
    }

    public synchronized float[] getEmbeedingWithoutNormalized(String input) {
        return getRawEmbeddings(input);
    }

    public synchronized float[] getEmbeddingsWithoutNormalized(String input) {
        return getRawEmbeddings(input);
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
     * Returns true when llama.cpp has a GPU or integrated-GPU backend device available.
     */
    public synchronized boolean supportsGpu() {
        return nativeSupportsGpu();
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

    private native int nativeGetEmbeddingDimension();

    private native int nativeGetPoolingType();

    private native float[] nativeGetEmbeddings(String input);

    private native float[] nativeGetRawEmbeddings(String input);

    private native float[] nativeGetRawEmbeddingsFromTokens(int[] tokens);

    private native int[] nativeTokenize(String text, boolean addSpecial);

    private native int nativeGetTokenBos();

    private native int nativeGetTokenEos();

    private native String nativeDecode(String input, int predictLength);

    private native boolean nativeSupportsGpu();

    private native void nativeRelease();
}
