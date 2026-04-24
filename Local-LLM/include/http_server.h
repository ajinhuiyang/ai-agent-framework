#pragma once

#include "transformer.h"
#include "platform.h"

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace localllm {

// 简单的 HTTP 请求
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
    std::map<std::string, std::string> query_params;
};

// 简单的 HTTP 响应
struct HttpResponse {
    int status_code = 200;
    std::string body;
    std::string content_type = "application/json";
    std::map<std::string, std::string> headers;

    void set_json(const std::string& json) {
        body = json;
        content_type = "application/json";
    }

    void set_error(int code, const std::string& message) {
        status_code = code;
        body = "{\"error\": \"" + message + "\"}";
        content_type = "application/json";
    }

    std::string to_http_response() const;
};

// 简单 JSON 辅助 (不依赖第三方库)
class SimpleJSON {
public:
    enum Type { NUL, BOOL, INT, FLOAT, STRING, ARRAY, OBJECT };

    SimpleJSON() : type_(NUL) {}
    SimpleJSON(bool v) : type_(BOOL), bool_val_(v) {}
    SimpleJSON(int v) : type_(INT), int_val_(v) {}
    SimpleJSON(int64_t v) : type_(INT), int_val_(v) {}
    SimpleJSON(double v) : type_(FLOAT), float_val_(v) {}
    SimpleJSON(const std::string& v) : type_(STRING), str_val_(v) {}
    SimpleJSON(const char* v) : type_(STRING), str_val_(v) {}

    static SimpleJSON object() { SimpleJSON j; j.type_ = OBJECT; return j; }
    static SimpleJSON array() { SimpleJSON j; j.type_ = ARRAY; return j; }

    // 对象操作
    SimpleJSON& operator[](const std::string& key) {
        type_ = OBJECT;
        return obj_val_[key];
    }

    // 数组操作
    void push_back(const SimpleJSON& val) {
        type_ = ARRAY;
        arr_val_.push_back(val);
    }

    // 解析 JSON 字符串 (简化版, 够用)
    static SimpleJSON parse(const std::string& str);

    // 序列化
    std::string dump() const;

    // 获取值
    Type type() const { return type_; }
    bool is_null() const { return type_ == NUL; }
    std::string get_string(const std::string& key, const std::string& def = "") const;
    int64_t get_int(const std::string& key, int64_t def = 0) const;
    double get_float(const std::string& key, double def = 0.0) const;
    bool get_bool(const std::string& key, bool def = false) const;
    const std::vector<SimpleJSON>& get_array(const std::string& key) const;

    std::string as_string() const { return str_val_; }
    int64_t as_int() const { return int_val_; }
    double as_float() const { return float_val_; }
    bool as_bool() const { return bool_val_; }

    bool has(const std::string& key) const {
        return obj_val_.find(key) != obj_val_.end();
    }

private:
    static SimpleJSON parse_value(const std::string& str, size_t& pos);
    static std::string parse_string_literal(const std::string& str, size_t& pos);
    static void skip_whitespace(const std::string& str, size_t& pos);
    static std::string escape_string(const std::string& s);

    Type type_;
    bool bool_val_ = false;
    int64_t int_val_ = 0;
    double float_val_ = 0.0;
    std::string str_val_;
    std::vector<SimpleJSON> arr_val_;
    std::map<std::string, SimpleJSON> obj_val_;
};

// HTTP Server (OpenAI 兼容 API)
class HttpServer {
public:
    HttpServer(Transformer& model, int port = 8080);
    ~HttpServer();

    // 启动服务器
    bool start();

    // 停止服务器
    void stop();

    // 等待服务器结束
    void wait();

private:
    void server_loop();
    void handle_client(platform::socket_t client_fd);

    // 解析 HTTP 请求
    HttpRequest parse_request(platform::socket_t client_fd);

    // API 路由
    HttpResponse handle_request(const HttpRequest& req);
    HttpResponse handle_chat_completions(const HttpRequest& req);
    HttpResponse handle_completions(const HttpRequest& req);
    HttpResponse handle_models(const HttpRequest& req);
    HttpResponse handle_health(const HttpRequest& req);

    // 处理 SSE 流式响应
    void handle_chat_completions_stream(platform::socket_t client_fd, const HttpRequest& req);

    Transformer& model_;
    int port_;
    platform::socket_t server_fd_ = platform::INVALID_SOCK;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    std::mutex model_mutex_; // 保护模型推理 (单线程推理)
};

} // namespace localllm
