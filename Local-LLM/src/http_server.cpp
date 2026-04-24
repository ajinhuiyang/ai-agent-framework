#include "http_server.h"
#include "platform.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace localllm {

// ======================== SimpleJSON 实现 ========================

void SimpleJSON::skip_whitespace(const std::string& str, size_t& pos) {
    while (pos < str.size() && (str[pos] == ' ' || str[pos] == '\t' || str[pos] == '\n' || str[pos] == '\r')) {
        pos++;
    }
}

std::string SimpleJSON::escape_string(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:   result += c; break;
        }
    }
    return result;
}

std::string SimpleJSON::parse_string_literal(const std::string& str, size_t& pos) {
    if (str[pos] != '"') throw std::runtime_error("Expected '\"'");
    pos++; // skip opening quote

    std::string result;
    while (pos < str.size() && str[pos] != '"') {
        if (str[pos] == '\\') {
            pos++;
            if (pos >= str.size()) break;
            switch (str[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case '/':  result += '/'; break;
                case 'u': {
                    // 简单处理 unicode escape
                    if (pos + 4 < str.size()) {
                        pos += 4;
                    }
                    result += '?';
                    break;
                }
                default: result += str[pos]; break;
            }
        } else {
            result += str[pos];
        }
        pos++;
    }
    if (pos < str.size()) pos++; // skip closing quote
    return result;
}

SimpleJSON SimpleJSON::parse_value(const std::string& str, size_t& pos) {
    skip_whitespace(str, pos);
    if (pos >= str.size()) return SimpleJSON();

    char c = str[pos];

    if (c == '"') {
        return SimpleJSON(parse_string_literal(str, pos));
    }

    if (c == '{') {
        SimpleJSON obj = SimpleJSON::object();
        pos++; // skip '{'
        skip_whitespace(str, pos);

        if (pos < str.size() && str[pos] == '}') {
            pos++;
            return obj;
        }

        while (pos < str.size()) {
            skip_whitespace(str, pos);
            std::string key = parse_string_literal(str, pos);
            skip_whitespace(str, pos);
            if (pos < str.size() && str[pos] == ':') pos++;
            obj[key] = parse_value(str, pos);
            skip_whitespace(str, pos);
            if (pos < str.size() && str[pos] == ',') {
                pos++;
            } else {
                break;
            }
        }

        skip_whitespace(str, pos);
        if (pos < str.size() && str[pos] == '}') pos++;
        return obj;
    }

    if (c == '[') {
        SimpleJSON arr = SimpleJSON::array();
        pos++; // skip '['
        skip_whitespace(str, pos);

        if (pos < str.size() && str[pos] == ']') {
            pos++;
            return arr;
        }

        while (pos < str.size()) {
            arr.push_back(parse_value(str, pos));
            skip_whitespace(str, pos);
            if (pos < str.size() && str[pos] == ',') {
                pos++;
            } else {
                break;
            }
        }

        skip_whitespace(str, pos);
        if (pos < str.size() && str[pos] == ']') pos++;
        return arr;
    }

    if (c == 't') {
        pos += 4; // "true"
        return SimpleJSON(true);
    }
    if (c == 'f') {
        pos += 5; // "false"
        return SimpleJSON(false);
    }
    if (c == 'n') {
        pos += 4; // "null"
        return SimpleJSON();
    }

    // Number
    size_t start = pos;
    bool is_float = false;
    if (str[pos] == '-') pos++;
    while (pos < str.size() && (isdigit(str[pos]) || str[pos] == '.' || str[pos] == 'e' || str[pos] == 'E' || str[pos] == '+' || str[pos] == '-')) {
        if (str[pos] == '.' || str[pos] == 'e' || str[pos] == 'E') is_float = true;
        pos++;
    }
    std::string num_str = str.substr(start, pos - start);
    if (is_float) {
        return SimpleJSON(std::stod(num_str));
    } else {
        return SimpleJSON(static_cast<int64_t>(std::stoll(num_str)));
    }
}

SimpleJSON SimpleJSON::parse(const std::string& str) {
    size_t pos = 0;
    return parse_value(str, pos);
}

std::string SimpleJSON::dump() const {
    switch (type_) {
        case NUL: return "null";
        case BOOL: return bool_val_ ? "true" : "false";
        case INT: return std::to_string(int_val_);
        case FLOAT: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.6g", float_val_);
            return buf;
        }
        case STRING: return "\"" + escape_string(str_val_) + "\"";
        case ARRAY: {
            std::string s = "[";
            for (size_t i = 0; i < arr_val_.size(); ++i) {
                if (i > 0) s += ",";
                s += arr_val_[i].dump();
            }
            s += "]";
            return s;
        }
        case OBJECT: {
            std::string s = "{";
            bool first = true;
            for (const auto& kv : obj_val_) {
                if (!first) s += ",";
                s += "\"" + escape_string(kv.first) + "\":" + kv.second.dump();
                first = false;
            }
            s += "}";
            return s;
        }
    }
    return "null";
}

std::string SimpleJSON::get_string(const std::string& key, const std::string& def) const {
    auto it = obj_val_.find(key);
    if (it == obj_val_.end() || it->second.type_ != STRING) return def;
    return it->second.str_val_;
}

int64_t SimpleJSON::get_int(const std::string& key, int64_t def) const {
    auto it = obj_val_.find(key);
    if (it == obj_val_.end()) return def;
    if (it->second.type_ == INT) return it->second.int_val_;
    if (it->second.type_ == FLOAT) return static_cast<int64_t>(it->second.float_val_);
    return def;
}

double SimpleJSON::get_float(const std::string& key, double def) const {
    auto it = obj_val_.find(key);
    if (it == obj_val_.end()) return def;
    if (it->second.type_ == FLOAT) return it->second.float_val_;
    if (it->second.type_ == INT) return static_cast<double>(it->second.int_val_);
    return def;
}

bool SimpleJSON::get_bool(const std::string& key, bool def) const {
    auto it = obj_val_.find(key);
    if (it == obj_val_.end() || it->second.type_ != BOOL) return def;
    return it->second.bool_val_;
}

const std::vector<SimpleJSON>& SimpleJSON::get_array(const std::string& key) const {
    static std::vector<SimpleJSON> empty;
    auto it = obj_val_.find(key);
    if (it == obj_val_.end() || it->second.type_ != ARRAY) return empty;
    return it->second.arr_val_;
}

// ======================== HttpResponse ========================

std::string HttpResponse::to_http_response() const {
    std::ostringstream ss;
    ss << "HTTP/1.1 " << status_code << " ";
    switch (status_code) {
        case 200: ss << "OK"; break;
        case 400: ss << "Bad Request"; break;
        case 404: ss << "Not Found"; break;
        case 500: ss << "Internal Server Error"; break;
        default:  ss << "Unknown"; break;
    }
    ss << "\r\n";
    ss << "Content-Type: " << content_type << "\r\n";
    ss << "Content-Length: " << body.size() << "\r\n";
    ss << "Access-Control-Allow-Origin: *\r\n";
    ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    ss << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

    for (const auto& h : headers) {
        ss << h.first << ": " << h.second << "\r\n";
    }

    ss << "\r\n";
    ss << body;
    return ss.str();
}

// ======================== HttpServer ========================

HttpServer::HttpServer(Transformer& model, int port)
    : model_(model), port_(port) {}

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start() {
    platform::socket_init();
    server_fd_ = platform::socket_listen(port_);
    if (server_fd_ == platform::INVALID_SOCK) {
        std::cerr << "[HTTP] Failed to bind to port " << port_ << std::endl;
        return false;
    }

    running_ = true;
    server_thread_ = std::thread(&HttpServer::server_loop, this);

    std::cout << "[HTTP] Server started on http://0.0.0.0:" << port_ << std::endl;
    std::cout << "[HTTP] API Endpoints:" << std::endl;
    std::cout << "  POST /v1/chat/completions  - Chat completions (OpenAI compatible)" << std::endl;
    std::cout << "  POST /v1/completions       - Text completions" << std::endl;
    std::cout << "  GET  /v1/models            - List models" << std::endl;
    std::cout << "  GET  /health               - Health check" << std::endl;

    return true;
}

void HttpServer::stop() {
    running_ = false;
    if (server_fd_ != platform::INVALID_SOCK) {
        platform::socket_close(server_fd_);
        server_fd_ = platform::INVALID_SOCK;
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    platform::socket_cleanup();
}

void HttpServer::wait() {
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void HttpServer::server_loop() {
    while (running_) {
        platform::socket_t client_fd = platform::socket_accept(server_fd_);
        if (client_fd == platform::INVALID_SOCK) {
            continue; // timeout 或错误, 继续循环
        }

        // 在新线程中处理请求 (允许并发接收, 但推理是串行的)
        std::thread(&HttpServer::handle_client, this, client_fd).detach();
    }
}

HttpRequest HttpServer::parse_request(platform::socket_t client_fd) {
    HttpRequest req;

    // 读取请求数据
    std::string raw;
    char buf[4096];
    int n;

    // 读取头部
    while ((n = platform::socket_recv(client_fd, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        raw += buf;

        // 检查是否已读完头部
        auto header_end = raw.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            // 解析 Content-Length
            size_t content_length = 0;
            auto cl_pos = raw.find("Content-Length: ");
            if (cl_pos == std::string::npos) cl_pos = raw.find("content-length: ");
            if (cl_pos != std::string::npos) {
                content_length = std::stoul(raw.substr(cl_pos + 16));
            }

            size_t body_start = header_end + 4;
            size_t body_received = raw.size() - body_start;

            // 继续读取 body
            while (body_received < content_length) {
                n = platform::socket_recv(client_fd, buf, std::min((int)(sizeof(buf) - 1), (int)(content_length - body_received)));
                if (n <= 0) break;
                buf[n] = '\0';
                raw += buf;
                body_received += n;
            }
            break;
        }
    }

    if (raw.empty()) return req;

    // 解析请求行
    auto first_line_end = raw.find("\r\n");
    if (first_line_end == std::string::npos) return req;

    std::string first_line = raw.substr(0, first_line_end);
    auto space1 = first_line.find(' ');
    auto space2 = first_line.find(' ', space1 + 1);
    if (space1 == std::string::npos || space2 == std::string::npos) return req;

    req.method = first_line.substr(0, space1);
    std::string full_path = first_line.substr(space1 + 1, space2 - space1 - 1);

    // 解析 path 和 query parameters
    auto qmark = full_path.find('?');
    if (qmark != std::string::npos) {
        req.path = full_path.substr(0, qmark);
        // 简单解析 query params
        std::string query = full_path.substr(qmark + 1);
        // ... (可扩展)
    } else {
        req.path = full_path;
    }

    // 解析 headers
    auto header_end = raw.find("\r\n\r\n");
    std::string header_section = raw.substr(first_line_end + 2, header_end - first_line_end - 2);
    std::istringstream header_stream(header_section);
    std::string header_line;
    while (std::getline(header_stream, header_line)) {
        if (!header_line.empty() && header_line.back() == '\r') header_line.pop_back();
        auto colon = header_line.find(": ");
        if (colon != std::string::npos) {
            req.headers[header_line.substr(0, colon)] = header_line.substr(colon + 2);
        }
    }

    // Body
    if (header_end != std::string::npos && header_end + 4 < raw.size()) {
        req.body = raw.substr(header_end + 4);
    }

    return req;
}

void HttpServer::handle_client(platform::socket_t client_fd) {
    auto req = parse_request(client_fd);

    if (req.method.empty()) {
        platform::socket_close(client_fd);
        return;
    }

    std::cout << "[HTTP] " << req.method << " " << req.path << std::endl;

    // CORS preflight
    if (req.method == "OPTIONS") {
        HttpResponse resp;
        resp.status_code = 204;
        resp.body = "";
        auto raw = resp.to_http_response();
        platform::socket_send(client_fd, raw.c_str(), static_cast<int>(raw.size()));
        platform::socket_close(client_fd);
        return;
    }

    // 检查是否是流式请求
    if (req.path == "/v1/chat/completions" && req.method == "POST") {
        auto json = SimpleJSON::parse(req.body);
        if (json.get_bool("stream", false)) {
            handle_chat_completions_stream(client_fd, req);
            platform::socket_close(client_fd);
            return;
        }
    }

    auto resp = handle_request(req);
    auto raw = resp.to_http_response();
    platform::socket_send(client_fd, raw.c_str(), static_cast<int>(raw.size()));
    platform::socket_close(client_fd);
}

HttpResponse HttpServer::handle_request(const HttpRequest& req) {
    if (req.path == "/v1/chat/completions" && req.method == "POST") {
        return handle_chat_completions(req);
    }
    if (req.path == "/v1/completions" && req.method == "POST") {
        return handle_completions(req);
    }
    if (req.path == "/v1/models" && req.method == "GET") {
        return handle_models(req);
    }
    if (req.path == "/health" && req.method == "GET") {
        return handle_health(req);
    }

    HttpResponse resp;
    resp.set_error(404, "Not Found");
    return resp;
}

HttpResponse HttpServer::handle_chat_completions(const HttpRequest& req) {
    HttpResponse resp;

    try {
        auto json = SimpleJSON::parse(req.body);

        // 解析 messages
        const auto& messages = json.get_array("messages");
        // 提取消息内容, 让 generate() 内部处理 chat template
        std::string system_msg;
        std::string user_msg;
        for (const auto& msg : messages) {
            std::string role = msg.get_string("role", "user");
            std::string content = msg.get_string("content", "");

            if (role == "system") {
                system_msg = content;
            } else if (role == "user") {
                user_msg += content + "\n";
            }
        }
        // 将 system prompt 和 user message 合并为一个 prompt
        // generate() 会自动包装 Qwen chat template
        std::string prompt = user_msg;
        if (!system_msg.empty()) {
            prompt = system_msg + "\n\n" + user_msg;
        }

        float temperature = json.get_float("temperature", 0.7);
        float top_p = json.get_float("top_p", 0.9);
        int max_tokens = static_cast<int>(json.get_int("max_tokens", 2048));
        float repetition_penalty = json.get_float("repetition_penalty", 1.3);

        // 推理 — system 和 user 分开传给 generate()
        std::lock_guard<std::mutex> lock(model_mutex_);
        model_.reset();

        auto gen_result = model_.generate(user_msg, max_tokens, temperature, top_p, nullptr, system_msg, repetition_penalty);

        // 构建 OpenAI 兼容响应
        auto now = std::chrono::system_clock::now();
        auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();

        SimpleJSON result = SimpleJSON::object();
        result["id"] = SimpleJSON("chatcmpl-localllm");
        result["object"] = SimpleJSON("chat.completion");
        result["created"] = SimpleJSON(static_cast<int64_t>(epoch));
        result["model"] = SimpleJSON(model_.config().name);

        SimpleJSON choice = SimpleJSON::object();
        choice["index"] = SimpleJSON(0);

        SimpleJSON message = SimpleJSON::object();
        message["role"] = SimpleJSON("assistant");
        message["content"] = SimpleJSON(gen_result.text);
        choice["message"] = message;
        choice["finish_reason"] = SimpleJSON("stop");

        SimpleJSON choices = SimpleJSON::array();
        choices.push_back(choice);
        result["choices"] = choices;

        SimpleJSON usage = SimpleJSON::object();
        usage["prompt_tokens"] = SimpleJSON(static_cast<int64_t>(gen_result.prompt_tokens));
        usage["completion_tokens"] = SimpleJSON(static_cast<int64_t>(gen_result.completion_tokens));
        usage["total_tokens"] = SimpleJSON(static_cast<int64_t>(gen_result.prompt_tokens + gen_result.completion_tokens));
        result["usage"] = usage;

        resp.set_json(result.dump());

    } catch (const std::exception& e) {
        resp.set_error(500, std::string("Internal error: ") + e.what());
    }

    return resp;
}

void HttpServer::handle_chat_completions_stream(int client_fd, const HttpRequest& req) {
    try {
        auto json = SimpleJSON::parse(req.body);

        const auto& messages = json.get_array("messages");
        std::string system_msg;
        std::string user_msg;
        for (const auto& msg : messages) {
            std::string role = msg.get_string("role", "user");
            std::string content = msg.get_string("content", "");
            if (role == "system") {
                system_msg = content;
            } else if (role == "user") {
                user_msg += content + "\n";
            }
        }
        std::string prompt = user_msg;
        if (!system_msg.empty()) {
            prompt = system_msg + "\n\n" + user_msg;
        }

        float temperature = json.get_float("temperature", 0.7);
        float top_p = json.get_float("top_p", 0.9);
        int max_tokens = static_cast<int>(json.get_int("max_tokens", 2048));
        float repetition_penalty = json.get_float("repetition_penalty", 1.3);

        // 发送 SSE 头
        std::string header = "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/event-stream\r\n"
                              "Cache-Control: no-cache\r\n"
                              "Connection: keep-alive\r\n"
                              "Access-Control-Allow-Origin: *\r\n"
                              "\r\n";
        platform::socket_send(client_fd, header.c_str(), static_cast<int>(header.size()));

        std::lock_guard<std::mutex> lock(model_mutex_);
        model_.reset();

        auto now = std::chrono::system_clock::now();
        auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();

        model_.generate(user_msg, max_tokens, temperature, top_p,
            [&](int32_t /*token*/, const std::string& text) -> bool {
                SimpleJSON chunk = SimpleJSON::object();
                chunk["id"] = SimpleJSON("chatcmpl-localllm");
                chunk["object"] = SimpleJSON("chat.completion.chunk");
                chunk["created"] = SimpleJSON(static_cast<int64_t>(epoch));
                chunk["model"] = SimpleJSON(model_.config().name);

                SimpleJSON delta = SimpleJSON::object();
                delta["content"] = SimpleJSON(text);

                SimpleJSON choice = SimpleJSON::object();
                choice["index"] = SimpleJSON(0);
                choice["delta"] = delta;
                choice["finish_reason"] = SimpleJSON();

                SimpleJSON choices = SimpleJSON::array();
                choices.push_back(choice);
                chunk["choices"] = choices;

                std::string sse = "data: " + chunk.dump() + "\n\n";
                int sent = platform::socket_send(client_fd, sse.c_str(), static_cast<int>(sse.size()));
                return sent >= 0;
            }, system_msg, repetition_penalty);

        // 发送结束标志
        std::string done = "data: [DONE]\n\n";
        platform::socket_send(client_fd, done.c_str(), static_cast<int>(done.size()));

    } catch (const std::exception& e) {
        std::cerr << "[HTTP] Stream error: " << e.what() << std::endl;
    }
}

HttpResponse HttpServer::handle_completions(const HttpRequest& req) {
    HttpResponse resp;

    try {
        auto json = SimpleJSON::parse(req.body);
        std::string prompt = json.get_string("prompt", "");
        float temperature = json.get_float("temperature", 0.7);
        float top_p = json.get_float("top_p", 0.9);
        int max_tokens = static_cast<int>(json.get_int("max_tokens", 2048));
        float repetition_penalty = json.get_float("repetition_penalty", 1.3);

        std::lock_guard<std::mutex> lock(model_mutex_);
        model_.reset();

        auto gen_result = model_.generate(prompt, max_tokens, temperature, top_p, nullptr, "", repetition_penalty);

        SimpleJSON result = SimpleJSON::object();
        result["id"] = SimpleJSON("cmpl-localllm");
        result["object"] = SimpleJSON("text_completion");

        SimpleJSON choice = SimpleJSON::object();
        choice["text"] = SimpleJSON(gen_result.text);
        choice["index"] = SimpleJSON(0);
        choice["finish_reason"] = SimpleJSON("stop");

        SimpleJSON choices = SimpleJSON::array();
        choices.push_back(choice);
        result["choices"] = choices;

        resp.set_json(result.dump());

    } catch (const std::exception& e) {
        resp.set_error(500, std::string("Internal error: ") + e.what());
    }

    return resp;
}

HttpResponse HttpServer::handle_models(const HttpRequest& /*req*/) {
    HttpResponse resp;

    SimpleJSON model_info = SimpleJSON::object();
    model_info["id"] = SimpleJSON(model_.config().name);
    model_info["object"] = SimpleJSON("model");
    model_info["owned_by"] = SimpleJSON("local");

    SimpleJSON data = SimpleJSON::array();
    data.push_back(model_info);

    SimpleJSON result = SimpleJSON::object();
    result["object"] = SimpleJSON("list");
    result["data"] = data;

    resp.set_json(result.dump());
    return resp;
}

HttpResponse HttpServer::handle_health(const HttpRequest& /*req*/) {
    HttpResponse resp;
    SimpleJSON result = SimpleJSON::object();
    result["status"] = SimpleJSON("ok");
    result["model"] = SimpleJSON(model_.config().name);
    result["vocab_size"] = SimpleJSON(static_cast<int64_t>(model_.config().vocab_size));
    result["num_layers"] = SimpleJSON(static_cast<int64_t>(model_.config().num_layers));
    resp.set_json(result.dump());
    return resp;
}

} // namespace localllm
