// Package orchestrator coordinates the LLM generation pipeline:
// prompt building → provider selection → generation → conversation management.
package orchestrator

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/your-org/llm-generation/internal/domain"
	"github.com/your-org/llm-generation/internal/llm"
	"github.com/your-org/llm-generation/internal/prompt"
)

// Orchestrator is the main generation engine.
type Orchestrator struct {
	providers   map[string]llm.Provider
	defaultProv string
	promptMgr   *prompt.Manager
	logger      *zap.Logger

	// Conversation store (in-memory).
	convMu    sync.RWMutex
	convStore map[string]*domain.Conversation

	// Default generation config.
	defaultConfig domain.GenerateConfig

	maxTurns int
}

// New creates a new Orchestrator.
func New(
	providers map[string]llm.Provider,
	defaultProv string,
	promptMgr *prompt.Manager,
	logger *zap.Logger,
	defaultConfig domain.GenerateConfig,
	maxTurns int,
) *Orchestrator {
	return &Orchestrator{
		providers:     providers,
		defaultProv:   defaultProv,
		promptMgr:     promptMgr,
		logger:        logger,
		convStore:     make(map[string]*domain.Conversation),
		defaultConfig: defaultConfig,
		maxTurns:      maxTurns,
	}
}

// Generate performs a non-streaming generation.
func (o *Orchestrator) Generate(ctx context.Context, req domain.GenerateRequest) (*domain.GenerateResponse, error) {
	// Select provider.
	provider, err := o.getProvider(req.Provider)
	if err != nil {
		return nil, err
	}

	// Merge config.
	config := o.mergeConfig(req.Config)

	// Load conversation history if applicable.
	if req.ConversationID != "" {
		o.loadConversationHistory(&req)
	}

	// Build messages.
	messages := o.promptMgr.BuildMessages(req)

	o.logger.Info("generating response",
		zap.String("provider", provider.Name()),
		zap.Int("messages", len(messages)),
	)

	// Call LLM.
	result, err := provider.Complete(ctx, messages, config)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	// 输出完整性校验 — 仅修复未闭合的代码块，不自动重试
	// 自动重试对本地模型来说代价太大（推理时间翻倍），改为只做修复
	truncated := result.FinishReason == "length"

	if truncated || detectIncompleteContent(result.Content) {
		if truncated {
			o.logger.Warn("LLM output truncated (finish_reason=length)",
				zap.Int("completion_tokens", result.Usage.CompletionTokens),
			)
		}
		result.Content = repairTruncatedContent(result.Content)
	}

	// 代码块后处理修复 — 修复常见的代码语法问题
	// (无效端口号、未闭合括号/引号、缺少关键调用等)
	result.Content = repairCodeBlocks(result.Content)

	// Update conversation.
	convID := req.ConversationID
	if convID != "" {
		o.updateConversation(convID, req.Prompt, result.Content)
	}

	return &domain.GenerateResponse{
		Content:        result.Content,
		ConversationID: convID,
		Provider:       provider.Name(),
		Model:          result.Model,
		Usage:          result.Usage,
		FinishReason:   result.FinishReason,
		CreatedAt:      time.Now(),
	}, nil
}

// GenerateStream performs a streaming generation.
func (o *Orchestrator) GenerateStream(ctx context.Context, req domain.GenerateRequest) (<-chan domain.StreamChunk, string, error) {
	provider, err := o.getProvider(req.Provider)
	if err != nil {
		return nil, "", err
	}

	config := o.mergeConfig(req.Config)

	if req.ConversationID != "" {
		o.loadConversationHistory(&req)
	}

	messages := o.promptMgr.BuildMessages(req)

	streamCh, err := provider.CompleteStream(ctx, messages, config)
	if err != nil {
		return nil, "", fmt.Errorf("stream generation failed: %w", err)
	}

	// Transform llm.StreamEvent to domain.StreamChunk and accumulate full content.
	outCh := make(chan domain.StreamChunk, 64)
	convID := req.ConversationID

	go func() {
		defer close(outCh)
		var fullContent string

		for event := range streamCh {
			if event.Err != nil {
				outCh <- domain.StreamChunk{
					Content:      fmt.Sprintf("[error: %v]", event.Err),
					Done:         true,
					FinishReason: "error",
				}
				return
			}

			fullContent += event.Content

			if event.Done {
				// 流结束：检测并修复截断内容
				finishReason := event.FinishReason
				truncated := finishReason == "length"

				// 只在明确截断 (finish_reason=length) 时才修复。
				// finish_reason=stop 表示模型自主结束（EOS），即使内容看起来
				// "不完整" 也不应强行追加修复补丁，否则会产生碎片文字。
				if truncated {
					o.logger.Warn("stream output truncated (finish_reason=length)",
						zap.Int("content_len", len(fullContent)),
					)
					repairedContent := repairTruncatedContent(fullContent)
					// 发送修复补丁（仅追加部分）
					if len(repairedContent) > len(fullContent) {
						patch := repairedContent[len(fullContent):]
						outCh <- domain.StreamChunk{
							Content:      patch,
							Done:         false,
							FinishReason: "",
						}
						fullContent = repairedContent
					}
				}

				// 代码块后处理修复
				repairedFull := repairCodeBlocks(fullContent)

				// 发送最终 done 事件（含修复后的完整内容）
				outCh <- domain.StreamChunk{
					Content:      "",
					Done:         true,
					FinishReason: finishReason,
					FullContent:  repairedFull,
				}

				// Update conversation after streaming completes.
				if convID != "" {
					o.updateConversation(convID, req.Prompt, repairedFull)
				}
				return
			}

			outCh <- domain.StreamChunk{
				Content:      event.Content,
				Done:         event.Done,
				FinishReason: event.FinishReason,
			}
		}

		// channel 关闭但未收到 Done 事件（异常断开）
		// 仍然尝试修复并发送兜底 done
		if fullContent != "" {
			if detectIncompleteContent(fullContent) {
				repairedContent := repairTruncatedContent(fullContent)
				if len(repairedContent) > len(fullContent) {
					patch := repairedContent[len(fullContent):]
					outCh <- domain.StreamChunk{
						Content:      patch,
						Done:         false,
						FinishReason: "",
					}
				}
			}
		}
		outCh <- domain.StreamChunk{
			Content:      "",
			Done:         true,
			FinishReason: "disconnect",
		}
	}()

	return outCh, provider.Name(), nil
}

// CreateConversation creates a new conversation session.
func (o *Orchestrator) CreateConversation() string {
	id := uuid.New().String()
	o.convMu.Lock()
	defer o.convMu.Unlock()
	o.convStore[id] = &domain.Conversation{
		ID:        id,
		Messages:  []domain.Message{},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	return id
}

// GetConversation retrieves a conversation by ID.
func (o *Orchestrator) GetConversation(id string) (*domain.Conversation, bool) {
	o.convMu.RLock()
	defer o.convMu.RUnlock()
	conv, ok := o.convStore[id]
	return conv, ok
}

// DeleteConversation removes a conversation.
func (o *Orchestrator) DeleteConversation(id string) {
	o.convMu.Lock()
	defer o.convMu.Unlock()
	delete(o.convStore, id)
}

// ListProviders returns info about available providers.
func (o *Orchestrator) ListProviders(ctx context.Context) []domain.ProviderInfo {
	var infos []domain.ProviderInfo
	for name, p := range o.providers {
		status := "healthy"
		if err := p.HealthCheck(ctx); err != nil {
			status = "unhealthy"
		}
		infos = append(infos, domain.ProviderInfo{
			Name:      name,
			Models:    p.Models(),
			IsDefault: name == o.defaultProv,
			Status:    status,
		})
	}
	return infos
}

func (o *Orchestrator) getProvider(name string) (llm.Provider, error) {
	if name == "" {
		name = o.defaultProv
	}
	p, ok := o.providers[name]
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s (available: %v)", name, o.providerNames())
	}
	return p, nil
}

func (o *Orchestrator) providerNames() []string {
	names := make([]string, 0, len(o.providers))
	for name := range o.providers {
		names = append(names, name)
	}
	return names
}

func (o *Orchestrator) mergeConfig(reqConfig *domain.GenerateConfig) *domain.GenerateConfig {
	config := o.defaultConfig
	if reqConfig != nil {
		if reqConfig.Temperature > 0 {
			config.Temperature = reqConfig.Temperature
		}
		if reqConfig.MaxTokens > 0 {
			config.MaxTokens = reqConfig.MaxTokens
		}
		if reqConfig.TopP > 0 {
			config.TopP = reqConfig.TopP
		}
		if reqConfig.TopK > 0 {
			config.TopK = reqConfig.TopK
		}
		if reqConfig.RepetitionPenalty > 0 {
			config.RepetitionPenalty = reqConfig.RepetitionPenalty
		}
		if len(reqConfig.StopWords) > 0 {
			config.StopWords = reqConfig.StopWords
		}
		if reqConfig.Model != "" {
			config.Model = reqConfig.Model
		}
	}
	return &config
}

func (o *Orchestrator) loadConversationHistory(req *domain.GenerateRequest) {
	o.convMu.RLock()
	defer o.convMu.RUnlock()

	conv, ok := o.convStore[req.ConversationID]
	if !ok {
		return
	}

	// Prepend conversation history before any explicit history.
	if len(conv.Messages) > 0 {
		history := make([]domain.Message, len(conv.Messages))
		copy(history, conv.Messages)
		req.History = append(history, req.History...)
	}
}

func (o *Orchestrator) updateConversation(convID, userMsg, assistantMsg string) {
	o.convMu.Lock()
	defer o.convMu.Unlock()

	conv, ok := o.convStore[convID]
	if !ok {
		conv = &domain.Conversation{
			ID:        convID,
			CreatedAt: time.Now(),
		}
		o.convStore[convID] = conv
	}

	conv.Messages = append(conv.Messages,
		domain.Message{Role: "user", Content: userMsg},
		domain.Message{Role: "assistant", Content: assistantMsg},
	)
	conv.UpdatedAt = time.Now()

	// Enforce max turns (sliding window).
	if o.maxTurns > 0 && len(conv.Messages) > o.maxTurns*2 {
		conv.Messages = conv.Messages[len(conv.Messages)-o.maxTurns*2:]
	}
}

// detectIncompleteContent 检测 LLM 输出的内容是否结构不完整。
// 即使 finish_reason 是 "stop"，小模型也可能提前 EOS 导致输出不完整。
func detectIncompleteContent(content string) bool {
	// 1. 代码块 ``` 未闭合
	if strings.Count(content, "```")%2 != 0 {
		return true
	}

	// 2. 包含代码块但内容在代码块结束后立即终止（没有任何收尾文字）
	// 这种情况通常正常，跳过

	// 3. 内容以明显的半句话结尾（句末没有标点、代码块内最后一行不完整）
	trimmed := strings.TrimRight(content, " \t\n\r")
	if len(trimmed) == 0 {
		return false
	}

	// 如果内容以 ``` 结尾（代码块正常闭合），不认为不完整
	if strings.HasSuffix(trimmed, "```") {
		return false
	}

	// 检查是否以常见的中断模式结尾
	incompleteEndings := []string{
		"的", "到", "在", "为", "和", "与", "或", "而", "但", // 中文虚词结尾
		"the", "a", "an", "to", "of", "in", "for", "and", "or", // 英文虚词结尾
		"(", "[", "{", ",", ":", "=", "+", "-", "*", "/", // 运算符/括号未闭合
	}
	for _, ending := range incompleteEndings {
		if strings.HasSuffix(trimmed, ending) {
			return true
		}
	}

	// 检查最后一行是否像被截断的代码注释（以 // 开头但很短）
	lastNL := strings.LastIndex(trimmed, "\n")
	if lastNL >= 0 {
		lastLine := strings.TrimSpace(trimmed[lastNL+1:])
		if strings.HasPrefix(lastLine, "//") && len(lastLine) < 10 {
			return true
		}
	}

	return false
}

// repairTruncatedContent 尝试修复因截断或不完整而有结构缺陷的内容。
// 主要处理：
// 1. 代码块 ``` 未闭合 — 补上闭合标记
// 2. 行内代码 ` 未闭合 — 补上闭合标记
// 3. 在末尾追加截断提示
func repairTruncatedContent(content string) string {
	// 统计 ``` 出现次数，奇数表示有未闭合的代码块
	tripleBacktickCount := strings.Count(content, "```")
	if tripleBacktickCount%2 != 0 {
		// 未闭合的代码块: 找到最后一个完整行，然后闭合
		if lastNL := strings.LastIndex(content, "\n"); lastNL > 0 {
			content = content[:lastNL+1]
		}
		content += "\n```\n"
	}

	// 检查行内代码 ` 是否闭合（在最后一个 ``` 之后的部分）
	lastTriple := strings.LastIndex(content, "```")
	tail := content
	if lastTriple >= 0 {
		tail = content[lastTriple+3:]
	}
	if strings.Count(tail, "`")%2 != 0 {
		content += "`"
	}

	content += "\n\n> **注意**: 以上内容可能不完整，请尝试缩小问题范围或增加 max_tokens 配置。"
	return content
}

// repairCodeBlocks 对生成内容中的代码块进行后处理修复。
// 修复以下常见问题：
// 1. 未闭合的字符串字面量
// 2. 未闭合的括号 ()、{}
// 3. Go 代码缺少 http.ListenAndServe 等关键调用
// 4. 无效的端口号重复 (如 808080)
func repairCodeBlocks(content string) string {
	// 正则匹配 ```lang\n...``` 代码块
	re := regexp.MustCompile("(?s)```(\\w*)\\n(.*?)```")
	return re.ReplaceAllStringFunc(content, func(block string) string {
		// 提取语言和代码体
		matches := re.FindStringSubmatch(block)
		if len(matches) < 3 {
			return block
		}
		lang := matches[1]
		code := matches[2]

		code = repairCode(code, lang)
		return "```" + lang + "\n" + code + "```"
	})
}

// repairCode 修复单个代码片段中的语法问题。
func repairCode(code string, lang string) string {
	// ===== 1. 修复重复端口号 =====
	// 直接替换已知的退化模式
	for _, bad := range []string{"80808080", "808080", "80808"} {
		if strings.Contains(code, bad) {
			code = strings.ReplaceAll(code, bad, "8080")
		}
	}

	// ===== 2. 修复不完整的 if err := ... 语句 =====
	// 模型经常生成 `if err := someFunc(...)` 后直接换行，缺少 `; err != nil { log.Fatal(err) }`
	lines := strings.Split(code, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		// 匹配: if err := xxx(...) 但没有 ; err != nil {
		if strings.Contains(trimmed, "if err :=") &&
			strings.HasSuffix(trimmed, ")") &&
			!strings.Contains(trimmed, "err != nil") {
			lines[i] = line + "; err != nil {\n\t\tlog.Fatal(err)\n\t}"
		}
		// 匹配: if err := xxx(...)\n 后面直接是 } (缺少 ; err != nil { ... })
		if strings.Contains(trimmed, "if err :=") &&
			!strings.HasSuffix(trimmed, ")") &&
			!strings.Contains(trimmed, "err != nil") &&
			strings.Contains(trimmed, ")") {
			// 在最后一个 ) 后插入 ; err != nil { ... }
			lastParen := strings.LastIndex(line, ")")
			if lastParen > 0 {
				lines[i] = line[:lastParen+1] + "; err != nil {\n\t\tlog.Fatal(err)\n\t}"
			}
		}
	}
	code = strings.Join(lines, "\n")

	// ===== 3. 修复未闭合的字符串字面量 =====
	lines = strings.Split(code, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "//") {
			continue
		}
		// 计算行内双引号数量（忽略转义的 \"）
		unescaped := strings.ReplaceAll(line, `\"`, "")
		quoteCount := strings.Count(unescaped, `"`)
		if quoteCount%2 != 0 {
			// 奇数个引号 — 找合适位置补闭合引号
			line = strings.TrimRight(line, " \t\n\r")
			if strings.HasSuffix(line, ")") {
				line = line[:len(line)-1] + "\")"
			} else {
				line += "\""
			}
			lines[i] = line
		}
	}
	code = strings.Join(lines, "\n")

	// ===== 3. 修复未闭合的括号 =====
	openParens := strings.Count(code, "(") - strings.Count(code, ")")
	openBraces := strings.Count(code, "{") - strings.Count(code, "}")

	suffix := ""
	for i := 0; i < openParens; i++ {
		suffix += ")\n"
	}
	for i := 0; i < openBraces; i++ {
		suffix += "}\n"
	}
	if suffix != "" {
		code = strings.TrimRight(code, "\n") + "\n" + suffix
	}

	// ===== 4. Go: 缺少 http.ListenAndServe 时插入 =====
	if (lang == "go" || lang == "Go") && strings.Contains(code, "func main()") {
		if strings.Contains(code, "http.HandleFunc") && !strings.Contains(code, "ListenAndServe") {
			lastBrace := strings.LastIndex(code, "}")
			if lastBrace > 0 {
				insertion := "\tif err := http.ListenAndServe(\":8080\", nil); err != nil {\n\t\tlog.Fatal(err)\n\t}\n"
				code = code[:lastBrace] + insertion + code[lastBrace:]
			}
		}
	}

	return code
}
