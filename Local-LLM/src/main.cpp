#include "http_server.h"
#include "transformer.h"

#include <csignal>
#include <cstring>
#include <iostream>
#include <string>

using namespace localllm;

static HttpServer* g_server = nullptr;

void signal_handler(int sig) {
    std::cout << "\n[Main] Received signal " << sig << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void print_usage(const char* prog) {
    std::cout << "Local-LLM: Local Large Language Model Inference Engine\n"
              << "\n"
              << "Usage:\n"
              << "  " << prog << " --model <path>  [options]\n"
              << "\n"
              << "Options:\n"
              << "  --model, -m <path>     Path to GGUF model file (required)\n"
              << "  --port, -p <port>      HTTP server port (default: 8080)\n"
              << "  --prompt <text>        Run in CLI mode with given prompt\n"
              << "  --max-tokens <n>       Maximum tokens to generate (default: 256)\n"
              << "  --temperature <f>      Sampling temperature (default: 0.7)\n"
              << "  --top-p <f>            Top-p sampling (default: 0.9)\n"
              << "  --interactive, -i      Interactive chat mode\n"
              << "  --server, -s           Start HTTP API server (default)\n"
              << "  --help, -h             Show this help\n"
              << "\n"
              << "Examples:\n"
              << "  # Start HTTP API server\n"
              << "  " << prog << " -m model.gguf -p 8080\n"
              << "\n"
              << "  # Single prompt\n"
              << "  " << prog << " -m model.gguf --prompt \"Hello, world!\"\n"
              << "\n"
              << "  # Interactive chat\n"
              << "  " << prog << " -m model.gguf -i\n"
              << "\n"
              << "API Usage (OpenAI compatible):\n"
              << "  curl http://localhost:8080/v1/chat/completions \\\n"
              << "    -H 'Content-Type: application/json' \\\n"
              << "    -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'\n"
              << std::endl;
}

int run_interactive(Transformer& model, float temperature, float top_p, int max_tokens) {
    std::cout << "\n=== Interactive Mode ===\n"
              << "Type your message and press Enter. Type 'quit' or 'exit' to stop.\n"
              << "Type '/reset' to clear conversation history.\n\n";

    while (true) {
        std::cout << "You> " << std::flush;
        std::string input;
        std::getline(std::cin, input);

        if (input.empty()) continue;
        if (input == "quit" || input == "exit") break;
        if (input == "/reset") {
            model.reset();
            std::cout << "[System] Conversation reset.\n\n";
            continue;
        }

        std::cout << "Assistant> " << std::flush;
        model.reset();

        model.generate(input, max_tokens, temperature, top_p,
            [](int32_t /*token*/, const std::string& text) -> bool {
                std::cout << text << std::flush;
                return true;
            });

        std::cout << "\n\n";
    }

    std::cout << "Goodbye!\n";
    return 0;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string prompt;
    int port = 8080;
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    bool interactive = false;
    [[maybe_unused]] bool server_mode = false;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
            server_mode = true;
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--interactive" || arg == "-i") {
            interactive = true;
        } else if (arg == "--server" || arg == "-s") {
            server_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path is required. Use --model <path>\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 加载模型
    Transformer model;
    if (!model.load_model(model_path)) {
        std::cerr << "Error: Failed to load model from: " << model_path << std::endl;
        return 1;
    }

    // 模式选择
    if (interactive) {
        return run_interactive(model, temperature, top_p, max_tokens);
    }

    if (!prompt.empty()) {
        // 单次推理模式
        std::cout << "\n--- Generation ---\n" << std::flush;
        model.generate(prompt, max_tokens, temperature, top_p,
            [](int32_t /*token*/, const std::string& text) -> bool {
                std::cout << text << std::flush;
                return true;
            });
        std::cout << "\n--- End ---\n";
        return 0;
    }

    // 默认: HTTP 服务器模式
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    HttpServer server(model, port);
    g_server = &server;

    if (!server.start()) {
        std::cerr << "Error: Failed to start HTTP server on port " << port << std::endl;
        return 1;
    }

    server.wait();
    return 0;
}
