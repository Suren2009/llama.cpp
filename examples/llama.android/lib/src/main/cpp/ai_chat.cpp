#include <android/log.h>
#include <jni.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 4;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 8192;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 512;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.3f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    // Set llama log handler to Android
    llama_log_set(aichat_android_log_callback, nullptr);

    // Loading all CPU backend variants
    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOGi("Loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    // Initialize backends
    llama_backend_init();
    LOGi("Backend initiated; Log handler set.");
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_load(JNIEnv *env, jobject, jstring jmodel_path) {
    llama_model_params model_params = llama_model_default_params();

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    LOGd("%s: Loading model from: \n%s\n", __func__, model_path);

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }
    g_model = model;
    return 0;
}

static llama_context *init_context(llama_model *model, const int n_ctx = DEFAULT_CONTEXT_SIZE) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    // Multi-threading setup
    const int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                                     (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                                     N_THREADS_HEADROOM));
    LOGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, n_ctx);
    }
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static common_sampler *new_sampler(float temp) {
    common_params_sampling sparams;
    sparams.temp = temp;
    return common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepare(JNIEnv * /*env*/, jobject /*unused*/) {
    auto *context = init_context(g_model);
    if (!context) { return 1; }
    g_context = context;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler(DEFAULT_SAMPLER_TEMP);
    return 0;
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg,
                                                      jint pl, jint nr) {
    auto *context = init_context(g_model, pp);
    if (!context) {
        const auto *const err_msg = "Fail to init_context! Bench aborted.";
        LOGe(err_msg);
        return env->NewStringUTF(err_msg);
    }

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(context);
    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }

            if (llama_decode(context, g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_free(context);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto backend = get_backend();
    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return env->NewStringUTF(result.str().c_str());
}


/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;

    if (clear_kv_cache)
        llama_memory_clear(llama_get_memory(g_context), false);
}

/**
 * TODO-hyin: implement sliding-window version as a better alternative
 *
 * Context shifting by discarding the older half of the tokens appended after system prompt:
 * - take the [system_prompt_position] first tokens from the original prompt
 * - take half of the last (system_prompt_position - system_prompt_position) tokens
 * - recompute the logits in batches
 */
static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    auto formatted = common_chat_format_single(
            g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ false);
    chat_msgs.push_back(new_msg);
    LOGi("%s: Formatted and added %s message: \n%s\n", __func__, role.c_str(), formatted.c_str());
    return formatted;
}

/**
 * Completion loop's short-term states:
 * - stop generation position
 * - token chars caching
 * - current assistant message being generated
 */
static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    // Process tokens in batches using the global batch
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        // Shift context if current batch cannot fit into the context
        if (start_pos + i + cur_batch_size >= DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        // Decode this batch
        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Obtain system prompt from JEnv
    const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOGd("%s: System prompt received: \n%s", __func__, system_prompt);
    std::string formatted_system_prompt(system_prompt);

    // Format system prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, system_prompt);
    }
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);

    // Tokenize system prompt
    const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                               has_chat_template, has_chat_template);
    for (auto id: system_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Handle context overflow
    const int max_batch_size = DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOGe("%s: System prompt too long for context! %d tokens, max: %d",
             __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    // Obtain and tokenize user prompt
    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("%s: User prompt received: \n%s", __func__, user_prompt);
    std::string formatted_user_prompt(user_prompt);

    // Format user prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_user_prompt = chat_add_and_format(ROLE_USER, user_prompt);
    }
    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    // Decode formatted user prompts
    auto user_tokens = common_tokenize(g_context, formatted_user_prompt, has_chat_template, has_chat_template);
    for (auto id: user_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Ensure user prompt doesn't exceed the context size by truncating if necessary.
    const int user_prompt_size = (int) user_tokens.size();
    const int max_batch_size = DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM;
    if (user_prompt_size > max_batch_size) {
        const int skipped_tokens = user_prompt_size - max_batch_size;
        user_tokens.resize(max_batch_size);
        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    current_position += user_prompt_size;
    stop_generation_position = current_position + user_prompt_size + n_predict;
    return 0;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    // Infinite text generation via context shifting
    if (current_position >= DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM) {
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    // Stop if reaching the marked position
    if (current_position >= stop_generation_position) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token", __func__);
        return nullptr;
    }

    // Update position
    current_position++;

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());
        return nullptr;
    }

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Create and return a valid UTF-8 Java string
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Free up resources
    common_sampler_free(g_sampler);
    g_chat_templates.reset();
    llama_batch_free(g_batch);
    llama_free(g_context);
    llama_model_free(g_model);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_shutdown(JNIEnv *, jobject /*unused*/) {
    llama_backend_free();
}

// --------------------------------------------------------------------------
// Java API bridge
// --------------------------------------------------------------------------

static std::mutex            g_java_mutex;
static bool                  g_java_backend_initialized;
static llama_model         * g_java_model;
static llama_context       * g_java_context;
static llama_batch           g_java_batch;
static bool                  g_java_batch_initialized;
static common_sampler      * g_java_sampler;
static llama_adapter_lora  * g_java_lora;
static std::string           g_java_model_path;
static std::string           g_java_lora_path;
static bool                  g_java_use_gpu;

static void java_throw(JNIEnv *env, const char *clazz, const std::string &message) {
    jclass exception_class = env->FindClass(clazz);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
    }
}

static int java_n_threads() {
    return std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                            (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                            N_THREADS_HEADROOM));
}

static bool java_supports_gpu_locked() {
    if (!llama_supports_gpu_offload()) {
        return false;
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }

    return false;
}

static void java_release_model_locked() {
    if (g_java_sampler != nullptr) {
        common_sampler_free(g_java_sampler);
        g_java_sampler = nullptr;
    }

    if (g_java_context != nullptr) {
        llama_free(g_java_context);
        g_java_context = nullptr;
    }

    if (g_java_batch_initialized) {
        llama_batch_free(g_java_batch);
        g_java_batch = {};
        g_java_batch_initialized = false;
    }

    if (g_java_lora != nullptr) {
        llama_adapter_lora_free(g_java_lora);
        g_java_lora = nullptr;
    }

    if (g_java_model != nullptr) {
        llama_model_free(g_java_model);
        g_java_model = nullptr;
    }
}

static bool java_model_loaded() {
    return g_java_model != nullptr && g_java_context != nullptr && g_java_batch_initialized;
}

static std::string java_string(JNIEnv *env, jstring value) {
    if (value == nullptr) {
        return {};
    }

    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

static int java_decode_tokens(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit) {
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);

        for (int j = 0; j < cur_batch_size; ++j) {
            const bool want_logit = compute_last_logit && (i + j == (int) tokens.size() - 1);
            common_batch_add(batch, tokens[i + j], start_pos + i + j, {0}, want_logit);
        }

        const int result = llama_decode(context, batch);
        if (result != 0) {
            LOGe("%s: llama_decode failed with %d", __func__, result);
            return result;
        }
    }

    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_LlamaAndroid_nativeInit(JNIEnv *env, jclass, jstring nativeLibDir) {
    std::lock_guard<std::mutex> lock(g_java_mutex);
    if (g_java_backend_initialized) {
        return;
    }

    llama_log_set(aichat_android_log_callback, nullptr);

    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, nullptr);
    if (path_to_backend == nullptr) {
        java_throw(env, "java/lang/IllegalArgumentException", "Native library directory cannot be null");
        return;
    }

    LOGi("Java API loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    llama_backend_init();
    g_java_backend_initialized = true;
}

static bool java_load_model_locked(
        JNIEnv *env,
        const std::string &model_path,
        const std::string &lora_path,
        const bool use_gpu) {
    if (model_path.empty()) {
        java_throw(env, "java/lang/IllegalArgumentException", "Model path cannot be empty");
        return false;
    }
    if (use_gpu && !java_supports_gpu_locked()) {
        java_throw(env, "java/lang/UnsupportedOperationException", "GPU backend is not available");
        return false;
    }

    java_release_model_locked();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? -1 : 0;

    g_java_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (g_java_model == nullptr) {
        java_throw(env, "java/io/IOException", "Failed to load GGUF model");
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = DEFAULT_CONTEXT_SIZE;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = java_n_threads();
    ctx_params.n_threads_batch = ctx_params.n_threads;
    ctx_params.embeddings = true;

    g_java_context = llama_init_from_model(g_java_model, ctx_params);
    if (g_java_context == nullptr) {
        java_release_model_locked();
        java_throw(env, "java/io/IOException", "Failed to create llama context");
        return false;
    }

    g_java_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    g_java_batch_initialized = true;

    common_params_sampling sparams;
    sparams.temp = DEFAULT_SAMPLER_TEMP;
    g_java_sampler = common_sampler_init(g_java_model, sparams);
    if (g_java_sampler == nullptr) {
        java_release_model_locked();
        java_throw(env, "java/io/IOException", "Failed to create decoder sampler");
        return false;
    }

    if (!lora_path.empty()) {
        g_java_lora = llama_adapter_lora_init(g_java_model, lora_path.c_str());
        if (g_java_lora == nullptr) {
            java_release_model_locked();
            java_throw(env, "java/io/IOException", "Failed to load LoRA adapter");
            return false;
        }

        llama_adapter_lora *adapters[] = {g_java_lora};
        float scales[] = {1.0f};
        if (llama_set_adapters_lora(g_java_context, adapters, 1, scales) != 0) {
            java_release_model_locked();
            java_throw(env, "java/io/IOException", "Failed to apply LoRA adapter");
            return false;
        }
    }

    g_java_model_path = model_path;
    g_java_lora_path = lora_path;
    g_java_use_gpu = use_gpu;
    return true;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_LlamaAndroid_nativeLoadModel(
        JNIEnv *env,
        jobject,
        jstring jmodel_path,
        jstring jlora_path) {
    std::lock_guard<std::mutex> lock(g_java_mutex);

    const std::string model_path = java_string(env, jmodel_path);
    const std::string lora_path = java_string(env, jlora_path);
    java_load_model_locked(env, model_path, lora_path, false);
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_arm_aichat_LlamaAndroid_nativeGetEmbeddings(
        JNIEnv *env,
        jobject,
        jstring jinput,
        jboolean jis_with_gpu) {
    std::lock_guard<std::mutex> lock(g_java_mutex);
    if (!java_model_loaded()) {
        java_throw(env, "java/lang/IllegalStateException", "No model is loaded");
        return nullptr;
    }

    const bool use_gpu = jis_with_gpu == JNI_TRUE;
    if (use_gpu != g_java_use_gpu) {
        if (!java_load_model_locked(env, g_java_model_path, g_java_lora_path, use_gpu)) {
            return nullptr;
        }
    }

    if (!llama_model_has_encoder(g_java_model) || llama_model_has_decoder(g_java_model)) {
        java_throw(env, "java/lang/UnsupportedOperationException",
                   "Loaded model must be encoder-only to return embeddings");
        return nullptr;
    }

    const std::string input = java_string(env, jinput);
    if (input.empty()) {
        java_throw(env, "java/lang/IllegalArgumentException", "Input cannot be empty");
        return nullptr;
    }

    llama_set_embeddings(g_java_context, true);
    llama_set_causal_attn(g_java_context, false);
    llama_memory_clear(llama_get_memory(g_java_context), true);

    llama_tokens tokens = common_tokenize(g_java_context, input, true, true);
    if (tokens.empty()) {
        java_throw(env, "java/lang/IllegalArgumentException", "Input did not produce tokens");
        return nullptr;
    }
    if ((int) tokens.size() > BATCH_SIZE) {
        java_throw(env, "java/lang/IllegalArgumentException", "Input exceeds Android embedding batch size");
        return nullptr;
    }

    common_batch_clear(g_java_batch);
    for (int i = 0; i < (int) tokens.size(); ++i) {
        common_batch_add(g_java_batch, tokens[i], i, {0}, true);
    }

    const int result = llama_encode(g_java_context, g_java_batch);
    if (result < 0) {
        java_throw(env, "java/lang/IllegalStateException", "Failed to encode input");
        return nullptr;
    }

    const enum llama_pooling_type pooling_type = llama_pooling_type(g_java_context);
    const float *embedding = pooling_type == LLAMA_POOLING_TYPE_NONE
                             ? llama_get_embeddings_ith(g_java_context, g_java_batch.n_tokens - 1)
                             : llama_get_embeddings_seq(g_java_context, 0);
    if (embedding == nullptr) {
        java_throw(env, "java/lang/IllegalStateException", "Model did not return embeddings");
        return nullptr;
    }

    int output_size = llama_model_n_embd_out(g_java_model);
    if (pooling_type == LLAMA_POOLING_TYPE_RANK) {
        output_size = std::min(output_size, (int) llama_model_n_cls_out(g_java_model));
    }

    std::vector<float> normalized(output_size);
    common_embd_normalize(embedding, normalized.data(), output_size, 2);

    jfloatArray result_array = env->NewFloatArray(output_size);
    if (result_array == nullptr) {
        return nullptr;
    }

    env->SetFloatArrayRegion(result_array, 0, output_size, normalized.data());
    return result_array;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_LlamaAndroid_nativeDecode(
        JNIEnv *env,
        jobject,
        jstring jinput,
        jint predict_length,
        jboolean jinference_with_gpu) {
    std::lock_guard<std::mutex> lock(g_java_mutex);
    if (!java_model_loaded()) {
        java_throw(env, "java/lang/IllegalStateException", "No model is loaded");
        return nullptr;
    }

    const bool use_gpu = jinference_with_gpu == JNI_TRUE;
    if (use_gpu != g_java_use_gpu) {
        if (!java_load_model_locked(env, g_java_model_path, g_java_lora_path, use_gpu)) {
            return nullptr;
        }
    }

    if (!llama_model_has_decoder(g_java_model)) {
        java_throw(env, "java/lang/UnsupportedOperationException", "Loaded model does not support decoder execution");
        return nullptr;
    }
    if (predict_length <= 0) {
        java_throw(env, "java/lang/IllegalArgumentException", "Predict length must be positive");
        return nullptr;
    }

    const std::string input = java_string(env, jinput);
    if (input.empty()) {
        java_throw(env, "java/lang/IllegalArgumentException", "Input cannot be empty");
        return nullptr;
    }

    llama_set_embeddings(g_java_context, false);
    llama_set_causal_attn(g_java_context, true);
    llama_memory_clear(llama_get_memory(g_java_context), false);
    common_sampler_reset(g_java_sampler);

    llama_tokens tokens = common_tokenize(g_java_context, input, true, true);
    if (tokens.empty()) {
        java_throw(env, "java/lang/IllegalArgumentException", "Input did not produce tokens");
        return nullptr;
    }
    if ((int) tokens.size() >= DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM) {
        java_throw(env, "java/lang/IllegalArgumentException", "Input exceeds Android context size");
        return nullptr;
    }

    if (java_decode_tokens(g_java_context, g_java_batch, tokens, 0, true) != 0) {
        java_throw(env, "java/lang/IllegalStateException", "Failed to decode input");
        return nullptr;
    }

    llama_pos current_pos = (llama_pos) tokens.size();
    std::string output;
    const llama_vocab *vocab = llama_model_get_vocab(g_java_model);

    for (int i = 0; i < predict_length; ++i) {
        if (current_pos >= DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM) {
            break;
        }

        const llama_token token = common_sampler_sample(g_java_sampler, g_java_context, -1);
        common_sampler_accept(g_java_sampler, token, true);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        output += common_token_to_piece(g_java_context, token);

        common_batch_clear(g_java_batch);
        common_batch_add(g_java_batch, token, current_pos, {0}, true);
        const int result = llama_decode(g_java_context, g_java_batch);
        if (result != 0) {
            java_throw(env, "java/lang/IllegalStateException", "Failed to decode generated token");
            return nullptr;
        }
        current_pos++;
    }

    if (!is_valid_utf8(output.c_str())) {
        java_throw(env, "java/lang/IllegalStateException", "Decoder returned invalid UTF-8");
        return nullptr;
    }

    return env->NewStringUTF(output.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_LlamaAndroid_nativeRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_java_mutex);
    java_release_model_locked();
    g_java_model_path.clear();
    g_java_lora_path.clear();
    g_java_use_gpu = false;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_LlamaAndroid_nativeSupportsGpu(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_java_mutex);
    return java_supports_gpu_locked() ? JNI_TRUE : JNI_FALSE;
}
