package com.example.llama;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.OpenableColumns;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.arm.aichat.LlamaAndroid;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Minimal Java sample that verifies the LlamaAndroid API.
 */
public final class JavaApiSampleActivity extends AppCompatActivity {
    private static final int DEFAULT_DECODE_TOKENS = 128;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private LlamaAndroid llama;
    private File modelFile;
    private File loraFile;
    private boolean modelLoaded;

    private TextView statusView;
    private TextView modelPathView;
    private TextView loraPathView;
    private EditText promptInput;
    private TextView outputView;
    private Button loadButton;
    private Button embeddingsButton;
    private Button decodeButton;
    private Button releaseButton;

    private final ActivityResultLauncher<String[]> pickModel =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> {
                if (uri != null) {
                    copySelectedFile(uri, "model", file -> {
                        modelFile = file;
                        modelLoaded = false;
                        modelPathView.setText(file.getAbsolutePath());
                        updateButtons();
                    });
                }
            });

    private final ActivityResultLauncher<String[]> pickLora =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> {
                if (uri != null) {
                    copySelectedFile(uri, "lora", file -> {
                        loraFile = file;
                        modelLoaded = false;
                        loraPathView.setText(file.getAbsolutePath());
                        updateButtons();
                    });
                }
            });

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_java_api_sample);

        statusView = findViewById(R.id.status);
        modelPathView = findViewById(R.id.model_path);
        loraPathView = findViewById(R.id.lora_path);
        promptInput = findViewById(R.id.prompt_input);
        outputView = findViewById(R.id.output);
        loadButton = findViewById(R.id.load_model);
        embeddingsButton = findViewById(R.id.get_embeddings);
        decodeButton = findViewById(R.id.decode);
        releaseButton = findViewById(R.id.release);

        llama = new LlamaAndroid(getApplicationContext());
        setStatus("Engine initialized. GPU backend available: " + llama.supportsGpu());

        findViewById(R.id.pick_model).setOnClickListener(v -> pickModel.launch(new String[]{"*/*"}));
        findViewById(R.id.pick_lora).setOnClickListener(v -> pickLora.launch(new String[]{"*/*"}));
        findViewById(R.id.clear_lora).setOnClickListener(v -> {
            loraFile = null;
            modelLoaded = false;
            loraPathView.setText(R.string.no_lora_selected);
            updateButtons();
        });

        loadButton.setOnClickListener(v -> loadModel());
        embeddingsButton.setOnClickListener(v -> getEmbeddings());
        decodeButton.setOnClickListener(v -> decode());
        releaseButton.setOnClickListener(v -> releaseModel());

        setupDefaultFiles();
        updateButtons();
    }

    @Override
    protected void onResume() {
        super.onResume();
        setupDefaultFiles();
    }

    private void setupDefaultFiles() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            if (!android.os.Environment.isExternalStorageManager()) {
                return;
            }
        }
        File externalStorage = android.os.Environment.getExternalStorageDirectory();
        File defaultModel = new File(externalStorage, "Download/sse/ranker-gguf/base_model.gguf");
        File defaultLora = new File(externalStorage, "Download/sse/ranker-gguf/ranker_lora_adapter.gguf");

        if (defaultModel.exists()) {
            modelFile = defaultModel;
            modelPathView.setText(modelFile.getAbsolutePath());
        }
        if (defaultLora.exists()) {
            loraFile = defaultLora;
            loraPathView.setText(loraFile.getAbsolutePath());
        }
    }

    private void loadModel() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            if (!android.os.Environment.isExternalStorageManager()) {
                setStatus("Requesting All Files Access permission...");
                android.content.Intent intent = new android.content.Intent(
                        android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName())
                );
                startActivity(intent);
                return;
            }
        }

        setupDefaultFiles();

        if (modelFile == null || !modelFile.exists()) {
            setStatus("Select a GGUF model first or place it in Download/sse/ranker-gguf/base_model.gguf.");
            return;
        }

        runEngineTask("Loading model...", () -> {
            if (loraFile == null) {
                llama.loadModel(modelFile.getAbsolutePath());
            } else {
                llama.loadModel(modelFile.getAbsolutePath(), loraFile.getAbsolutePath());
            }
            mainHandler.post(() -> {
                modelLoaded = true;
                setStatus("Model loaded. Engine selected GPU automatically when usable.");
                updateButtons();
            });
        });
    }

    private void getEmbeddings() {
        String prompt = promptInput.getText().toString();
        if (prompt.isEmpty()) {
            setStatus("Enter input text first.");
            return;
        }

        runEngineTask("Getting embeddings...", () -> {
            float[] embeddings = llama.getEmbeddings(prompt);
            float[] rawEmbeddings = llama.getRawEmbeddings(prompt);

            // Fetch special tokens & tokenize "hi"
            int bosToken = llama.getBosToken();
            int eosToken = llama.getEosToken();
            int[] hiTokens = llama.tokenize("hi", false);
            int[] customTokens = new int[hiTokens.length + 2];
            customTokens[0] = bosToken;
            System.arraycopy(hiTokens, 0, customTokens, 1, hiTokens.length);
            customTokens[customTokens.length - 1] = eosToken;

            // Retrieve raw embeddings for the custom tokens array
            float[] customRawEmbeddings = llama.getRawEmbeddings(customTokens);

            StringBuilder preview = new StringBuilder();
            int previewCount = Math.min(8, embeddings.length);
            for (int i = 0; i < previewCount; ++i) {
                if (i > 0) {
                    preview.append(", ");
                }
                preview.append(String.format(Locale.US, "%.5f", embeddings[i]));
            }

            StringBuilder rawPreview = new StringBuilder();
            int rawPreviewCount = Math.min(8, rawEmbeddings.length);
            for (int i = 0; i < rawPreviewCount; ++i) {
                if (i > 0) {
                    rawPreview.append(", ");
                }
                rawPreview.append(String.format(Locale.US, "%.5f", rawEmbeddings[i]));
            }

            StringBuilder customTokensStr = new StringBuilder();
            for (int t : customTokens) {
                customTokensStr.append(t).append(" ");
            }

            int dimension = embeddings.length;
            int numTokens = rawEmbeddings.length / dimension;
            int customNumTokens = customRawEmbeddings.length / dimension;

            postOutput("Normalized embedding size: " + embeddings.length + "\nFirst values: [" + preview + "]\n\n" +
                       "Raw (Unnormalized) embedding size: " + rawEmbeddings.length + " (" + numTokens + " tokens * " + dimension + " dims)\nFirst values: [" + rawPreview + "]\n\n" +
                       "Custom Tokens ([BOS] + 'hi' + [EOS]): [" + customTokensStr.toString().trim() + "]\n" +
                       "Custom Raw embedding size: " + customRawEmbeddings.length + " (" + customNumTokens + " tokens * " + dimension + " dims)");
            postStatus("Embeddings complete.");
        });
    }

    private void decode() {
        String prompt = promptInput.getText().toString();
        if (prompt.isEmpty()) {
            setStatus("Enter a prompt first.");
            return;
        }

        runEngineTask("Decoding...", () -> {
            String response = llama.decode(prompt, DEFAULT_DECODE_TOKENS);
            postOutput(response);
            postStatus("Decode complete.");
        });
    }

    private void releaseModel() {
        runEngineTask("Releasing model...", () -> {
            llama.release();
            mainHandler.post(() -> {
                modelLoaded = false;
                outputView.setText("");
                setStatus("Model released.");
                updateButtons();
            });
        });
    }

    private void copySelectedFile(Uri uri, String prefix, FileReadyCallback callback) {
        runEngineTask("Copying selected " + prefix + " file...", () -> {
            File outDir = new File(getFilesDir(), "java-api-sample");
            if (!outDir.exists() && !outDir.mkdirs()) {
                throw new IllegalStateException("Failed to create sample storage directory");
            }

            File outFile = new File(outDir, prefix + "-" + safeDisplayName(this, uri));
            try (InputStream input = getContentResolver().openInputStream(uri);
                 FileOutputStream output = new FileOutputStream(outFile)) {
                if (input == null) {
                    throw new IllegalStateException("Unable to open selected file");
                }

                byte[] buffer = new byte[1024 * 1024];
                int read;
                while ((read = input.read(buffer)) != -1) {
                    output.write(buffer, 0, read);
                }
            }

            mainHandler.post(() -> {
                callback.onFileReady(outFile);
                setStatus("Copied " + prefix + " file.");
            });
        });
    }

    private void runEngineTask(String status, ThrowingRunnable task) {
        setBusy(true);
        setStatus(status);
        executor.execute(() -> {
            try {
                task.run();
            } catch (Throwable t) {
                postStatus(t.getClass().getSimpleName() + ": " + t.getMessage());
            } finally {
                mainHandler.post(() -> setBusy(false));
            }
        });
    }

    private void setBusy(boolean busy) {
        loadButton.setEnabled(!busy && modelFile != null);
        embeddingsButton.setEnabled(!busy && modelLoaded);
        decodeButton.setEnabled(!busy && modelLoaded);
        releaseButton.setEnabled(!busy && modelLoaded);
    }

    private void updateButtons() {
        setBusy(false);
    }

    private void setStatus(String status) {
        statusView.setText(status);
    }

    private void postStatus(String status) {
        mainHandler.post(() -> setStatus(status));
    }

    private void postOutput(String output) {
        mainHandler.post(() -> outputView.setText(output));
    }

    private static String safeDisplayName(Context context, Uri uri) {
        String displayName = null;
        try (Cursor cursor = context.getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (nameIndex >= 0) {
                    displayName = cursor.getString(nameIndex);
                }
            }
        }

        if (displayName == null || displayName.isEmpty()) {
            displayName = "selected.gguf";
        }
        return displayName.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    @Override
    protected void onDestroy() {
        llama.release();
        executor.shutdownNow();
        super.onDestroy();
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    private interface FileReadyCallback {
        void onFileReady(File file);
    }
}
