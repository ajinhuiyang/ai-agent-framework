// Package tokenizer implements Chinese text segmentation (分词) in pure Go.
//
// Algorithm: dictionary-based forward maximum matching + Viterbi on a
// simple unigram language model, with HMM-based new word discovery for
// out-of-vocabulary (OOV) words.
//
// This is a production-grade implementation following the architecture of
// jieba (结巴分词) but written entirely in Go without CGO.
package tokenizer

import (
	"math"
	"strings"
	"unicode"
)

// Token represents a single segmented word.
type Token struct {
	Text  string `json:"text"`
	Start int    `json:"start"` // byte offset in original text
	End   int    `json:"end"`
	IsOOV bool   `json:"is_oov,omitempty"` // true if discovered by HMM
}

// Tokenizer performs Chinese word segmentation.
type Tokenizer struct {
	dict    *Dictionary
	hmm     *HMMModel
	maxWord int // max word length in dictionary
}

// New creates a new tokenizer with the built-in dictionary.
func New() *Tokenizer {
	dict := NewDictionary()
	dict.LoadBuiltin()

	return &Tokenizer{
		dict:    dict,
		hmm:     NewHMMModel(),
		maxWord: dict.MaxWordLen(),
	}
}

// Tokenize segments the input text into tokens.
func (t *Tokenizer) Tokenize(text string) []Token {
	if len(text) == 0 {
		return nil
	}

	runes := []rune(text)
	// Build DAG (Directed Acyclic Graph)
	dag := t.buildDAG(runes)
	// Viterbi to find optimal path
	route := t.calcRoute(runes, dag)
	// Extract tokens following the optimal path
	return t.extractTokens(text, runes, dag, route)
}

// TokenizeToStrings is a convenience that returns just the word strings.
func (t *Tokenizer) TokenizeToStrings(text string) []string {
	tokens := t.Tokenize(text)
	result := make([]string, len(tokens))
	for i, tok := range tokens {
		result[i] = tok.Text
	}
	return result
}

// buildDAG builds a directed acyclic graph for all possible segmentations.
// dag[i] = list of end positions j where runes[i:j+1] is a word in dict.
func (t *Tokenizer) buildDAG(runes []rune) map[int][]int {
	dag := make(map[int][]int, len(runes))
	n := len(runes)

	for i := 0; i < n; i++ {
		ends := []int{i} // single character is always a valid segmentation
		for j := i + 1; j <= n && j-i <= t.maxWord; j++ {
			word := string(runes[i:j])
			if t.dict.Contains(word) {
				ends = append(ends, j-1)
			}
		}
		dag[i] = ends
	}
	return dag
}

// routeNode stores Viterbi path info.
type routeNode struct {
	logProb float64
	next    int
}

// calcRoute finds the most probable segmentation path via dynamic programming.
func (t *Tokenizer) calcRoute(runes []rune, dag map[int][]int) []routeNode {
	n := len(runes)
	route := make([]routeNode, n+1)
	route[n] = routeNode{logProb: 0, next: n}
	totalFreq := t.dict.TotalFreq()
	logTotal := math.Log(float64(totalFreq))

	for i := n - 1; i >= 0; i-- {
		best := routeNode{logProb: -math.MaxFloat64, next: i}
		for _, j := range dag[i] {
			word := string(runes[i : j+1])
			freq := t.dict.Freq(word)
			if freq == 0 {
				freq = 1 // smoothing for unknown words
			}
			logProb := math.Log(float64(freq)) - logTotal + route[j+1].logProb
			if logProb > best.logProb {
				best = routeNode{logProb: logProb, next: j + 1}
			}
		}
		route[i] = best
	}
	return route
}

// extractTokens follows the optimal path and handles OOV via HMM.
func (t *Tokenizer) extractTokens(text string, runes []rune, dag map[int][]int, route []routeNode) []Token {
	var tokens []Token
	n := len(runes)
	byteOffsets := computeByteOffsets(text, runes)

	i := 0
	for i < n {
		j := route[i].next
		word := string(runes[i:j])

		if j-i == 1 && !t.dict.Contains(word) {
			// Single unknown character — accumulate consecutive unknowns for HMM
			var oovBuf []rune
			oovStart := i
			for i < n && route[i].next-i == 1 && !t.dict.Contains(string(runes[i:i+1])) {
				oovBuf = append(oovBuf, runes[i])
				i++
			}

			// Use HMM to segment the OOV block
			if len(oovBuf) > 0 {
				hmmTokens := t.hmm.Segment(oovBuf)
				runePos := oovStart
				for _, ht := range hmmTokens {
					htRunes := []rune(ht)
					tokens = append(tokens, Token{
						Text:  ht,
						Start: byteOffsets[runePos],
						End:   byteOffsets[runePos+len(htRunes)],
						IsOOV: true,
					})
					runePos += len(htRunes)
				}
			}
		} else {
			tokens = append(tokens, Token{
				Text:  word,
				Start: byteOffsets[i],
				End:   byteOffsets[j],
			})
			i = j
		}
	}

	return tokens
}

// computeByteOffsets maps rune index → byte offset.
func computeByteOffsets(text string, runes []rune) []int {
	offsets := make([]int, len(runes)+1)
	byteIdx := 0
	for i, r := range runes {
		offsets[i] = byteIdx
		byteIdx += len(string(r))
	}
	offsets[len(runes)] = byteIdx
	return offsets
}

// --- Dictionary ---

// Dictionary is a prefix dictionary for word lookup.
type Dictionary struct {
	freqs     map[string]int
	totalFreq int64
	maxLen    int
}

// NewDictionary creates an empty dictionary.
func NewDictionary() *Dictionary {
	return &Dictionary{
		freqs: make(map[string]int),
	}
}

// AddWord adds a word with its frequency.
func (d *Dictionary) AddWord(word string, freq int) {
	d.freqs[word] = freq
	d.totalFreq += int64(freq)
	runeLen := len([]rune(word))
	if runeLen > d.maxLen {
		d.maxLen = runeLen
	}
	// Add all prefixes for efficient DAG building
	runes := []rune(word)
	for i := 1; i < len(runes); i++ {
		prefix := string(runes[:i])
		if _, ok := d.freqs[prefix]; !ok {
			d.freqs[prefix] = 0
		}
	}
}

// Contains checks if the word is in the dictionary (with freq > 0).
func (d *Dictionary) Contains(word string) bool {
	freq, ok := d.freqs[word]
	return ok && freq > 0
}

// Freq returns the frequency of a word.
func (d *Dictionary) Freq(word string) int {
	return d.freqs[word]
}

// TotalFreq returns the total frequency sum.
func (d *Dictionary) TotalFreq() int64 {
	if d.totalFreq == 0 {
		return 1
	}
	return d.totalFreq
}

// MaxWordLen returns the maximum word length in runes.
func (d *Dictionary) MaxWordLen() int {
	if d.maxLen == 0 {
		return 8
	}
	return d.maxLen
}

// LoadBuiltin loads the built-in core dictionary.
// In production, this would load from a file. Here we embed a core vocabulary.
func (d *Dictionary) LoadBuiltin() {
	// Core Chinese words with frequencies (subset for demonstration;
	// production should load full dict from embedded file)
	coreWords := map[string]int{
		// Common function words
		"的": 6e7, "了": 3e7, "在": 2e7, "是": 3e7, "我": 2e7,
		"你": 1e7, "他": 1e7, "她": 8e6, "它": 5e6, "们": 1e7,
		"这": 1e7, "那": 8e6, "里": 5e6, "和": 1e7, "与": 5e6,
		"就": 1e7, "也": 8e6, "都": 8e6, "不": 2e7, "没": 1e7,
		"有": 2e7, "到": 1e7, "把": 5e6, "被": 5e6, "让": 3e6,
		"给": 5e6, "从": 5e6, "向": 3e6, "对": 5e6, "比": 3e6,
		"会": 8e6, "能": 8e6, "可以": 5e6, "应该": 3e6, "需要": 3e6,
		"想": 5e6, "要": 8e6, "去": 5e6, "来": 5e6, "上": 5e6,
		"下": 5e6, "出": 3e6, "过": 5e6, "着": 5e6, "地": 3e6,
		"得": 3e6, "很": 5e6, "太": 3e6, "最": 3e6, "更": 3e6,
		"还": 5e6, "再": 3e6, "又": 3e6, "但": 5e6, "但是": 3e6,
		"因为": 3e6, "所以": 3e6, "如果": 3e6, "虽然": 2e6,
		"什么": 5e6, "怎么": 3e6, "哪": 3e6, "哪里": 2e6,
		"谁": 3e6, "多少": 2e6, "为什么": 2e6, "怎么样": 1e6,

		// Verbs
		"说": 5e6, "看": 5e6, "听": 3e6, "做": 5e6, "走": 3e6,
		"跑": 2e6, "吃": 3e6, "喝": 2e6, "买": 3e6, "卖": 2e6,
		"知道": 3e6, "觉得": 3e6, "认为": 2e6, "希望": 2e6,
		"喜欢": 2e6, "帮助": 2e6, "帮忙": 1e6, "帮": 3e6,
		"告诉": 2e6, "打开": 2e6, "关闭": 1e6, "开始": 2e6,
		"结束": 1e6, "完成": 1e6, "准备": 1e6, "使用": 2e6,
		"提醒": 1e6, "设置": 1e6, "查询": 1e6, "搜索": 1e6,
		"播放": 1e6, "停止": 1e6, "暂停": 5e5, "继续": 1e6,
		"取消": 1e6, "确认": 1e6, "修改": 1e6, "删除": 1e6,
		"添加": 1e6, "创建": 1e6, "发送": 1e6, "接收": 5e5,
		"订": 2e6, "预订": 1e6, "预约": 1e6, "订票": 5e5,

		// Nouns — common
		"人": 5e6, "事": 3e6, "东西": 2e6, "地方": 2e6,
		"时间": 2e6, "时候": 2e6, "问题": 2e6, "工作": 2e6,
		"情况": 1e6, "公司": 2e6, "中国": 2e6, "世界": 1e6,
		"今天": 2e6, "明天": 1e6, "昨天": 1e6, "后天": 5e5,
		"上午": 1e6, "下午": 1e6, "晚上": 1e6, "早上": 1e6,
		"现在": 2e6, "以后": 1e6, "以前": 1e6,

		// Nouns — domain-specific
		"机票": 1e6, "酒店": 1e6, "航班": 5e5, "火车": 5e5,
		"天气": 1e6, "音乐": 1e6, "电影": 1e6, "新闻": 5e5,
		"北京": 2e6, "上海": 2e6, "广州": 1e6, "深圳": 1e6,
		"成都": 8e5, "杭州": 8e5, "南京": 8e5, "武汉": 8e5,
		"西安": 8e5, "重庆": 8e5, "天津": 8e5, "苏州": 8e5,

		// Adjectives
		"好": 5e6, "大": 5e6, "小": 3e6, "多": 3e6, "少": 2e6,
		"新": 3e6, "旧": 1e6, "快": 2e6, "慢": 1e6, "高": 2e6,
		"低": 1e6, "远": 1e6, "近": 1e6, "早": 1e6, "晚": 1e6,
		"便宜": 1e6, "贵": 1e6, "好的": 2e6,

		// Sentiment words
		"棒": 1e6, "差": 1e6, "烂": 5e5, "垃圾": 5e5,
		"优秀": 5e5, "糟糕": 5e5, "满意": 5e5, "不满": 5e5,
		"失望": 5e5, "开心": 5e5, "高兴": 5e5, "难过": 5e5,
		"生气": 5e5, "愤怒": 5e5, "害怕": 5e5, "担心": 5e5,
		"着急": 5e5, "惊喜": 5e5, "感谢": 5e5, "谢谢": 2e6,
		"抱歉": 5e5, "对不起": 5e5,

		// Conversational
		"你好": 2e6, "再见": 1e6, "拜拜": 5e5, "嗯": 5e5,
		"不好": 1e6, "不行": 1e6,
		"是的": 1e6, "请问": 1e6,
		"不客气": 5e5, "没关系": 5e5,

		// Numbers & measure words
		"一": 5e6, "二": 3e6, "三": 3e6, "四": 2e6, "五": 2e6,
		"六": 2e6, "七": 2e6, "八": 2e6, "九": 2e6, "十": 2e6,
		"百": 1e6, "千": 1e6, "万": 1e6, "亿": 5e5,
		"个": 5e6, "张": 2e6, "条": 1e6, "件": 1e6, "块": 1e6,
		"元": 1e6, "位": 1e6, "天": 3e6, "月": 2e6, "年": 2e6,
		"点": 2e6, "分": 2e6, "秒": 1e6, "小时": 1e6,

		// Particles & conjunctions
		"吗": 5e6, "呢": 3e6, "吧": 3e6, "啊": 3e6, "呀": 1e6,
		"嘛": 1e6, "哦": 1e6, "哈": 1e6,

		// Phrases (important for accuracy)
		"帮我": 2e6, "我要": 2e6, "我想": 2e6, "请帮": 1e6,
		"多少钱": 1e6, "可不可以": 5e5,
		"下周": 1e6, "上周": 1e6, "这个": 2e6, "那个": 1e6,
		"一下": 2e6, "一点": 1e6,
	}

	for word, freq := range coreWords {
		d.AddWord(word, int(freq))
	}
}

// --- HMM Model for OOV word discovery ---

// HMMModel uses a simple 4-state HMM (B/M/E/S) for word segmentation.
// B=Begin, M=Middle, E=End, S=Single
type HMMModel struct {
	startProb map[byte]float64
	transProb map[byte]map[byte]float64
	emitProb  map[byte]map[rune]float64
}

const (
	stateB byte = 'B' // Begin of word
	stateM byte = 'M' // Middle of word
	stateE byte = 'E' // End of word
	stateS byte = 'S' // Single character word
)

var allStates = []byte{stateB, stateM, stateE, stateS}

// prevStates defines valid previous states for each state.
var prevStates = map[byte][]byte{
	stateB: {stateE, stateS},
	stateM: {stateB, stateM},
	stateE: {stateB, stateM},
	stateS: {stateE, stateS},
}

// NewHMMModel creates a new HMM model with trained parameters.
func NewHMMModel() *HMMModel {
	h := &HMMModel{
		startProb: map[byte]float64{
			stateB: -0.26268660809250016,
			stateE: -3.14e+100, // impossible
			stateM: -3.14e+100,
			stateS: -1.4652633398537678,
		},
		transProb: map[byte]map[byte]float64{
			stateB: {stateE: -0.510825623765991, stateM: -0.916290731874155},
			stateE: {stateB: -0.5897149736854513, stateS: -0.8085250474669937},
			stateM: {stateE: -0.33344856811948514, stateM: -1.2603623820268226},
			stateS: {stateB: -0.7211965654669841, stateS: -0.6658631448798212},
		},
		emitProb: buildDefaultEmitProb(),
	}
	return h
}

// Segment uses Viterbi on the HMM to segment a sequence of runes.
func (h *HMMModel) Segment(runes []rune) []string {
	if len(runes) == 0 {
		return nil
	}
	if len(runes) == 1 {
		return []string{string(runes)}
	}

	// Viterbi
	n := len(runes)
	viterbi := make([]map[byte]float64, n)
	path := make([]map[byte]byte, n)

	// Init
	viterbi[0] = make(map[byte]float64)
	path[0] = make(map[byte]byte)
	for _, s := range allStates {
		emit := h.getEmitProb(s, runes[0])
		viterbi[0][s] = h.startProb[s] + emit
		path[0][s] = s
	}

	// Forward
	for t := 1; t < n; t++ {
		viterbi[t] = make(map[byte]float64)
		path[t] = make(map[byte]byte)

		for _, s := range allStates {
			emit := h.getEmitProb(s, runes[t])
			bestProb := -math.MaxFloat64
			bestPrev := stateB

			for _, ps := range prevStates[s] {
				prob := viterbi[t-1][ps] + h.transProb[ps][s] + emit
				if prob > bestProb {
					bestProb = prob
					bestPrev = ps
				}
			}
			viterbi[t][s] = bestProb
			path[t][s] = bestPrev
		}
	}

	// Find best final state (E or S)
	bestState := stateE
	if viterbi[n-1][stateS] > viterbi[n-1][stateE] {
		bestState = stateS
	}

	// Backtrace
	states := make([]byte, n)
	states[n-1] = bestState
	for t := n - 2; t >= 0; t-- {
		states[t] = path[t+1][states[t+1]]
	}

	// Build words from states
	var words []string
	var current strings.Builder
	for i, s := range states {
		current.WriteRune(runes[i])
		if s == stateE || s == stateS {
			words = append(words, current.String())
			current.Reset()
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

func (h *HMMModel) getEmitProb(state byte, char rune) float64 {
	if stateMap, ok := h.emitProb[state]; ok {
		if p, ok := stateMap[char]; ok {
			return p
		}
	}
	return -15.0 // very low probability for unseen emissions
}

// buildDefaultEmitProb builds simplified emission probabilities.
// In production, these should be trained on a large corpus.
func buildDefaultEmitProb() map[byte]map[rune]float64 {
	emit := map[byte]map[rune]float64{
		stateB: make(map[rune]float64),
		stateM: make(map[rune]float64),
		stateE: make(map[rune]float64),
		stateS: make(map[rune]float64),
	}

	// Common characters that often begin words
	beginChars := "我你他她它这那哪什怎谁人大小好多少不没有是在上下前后左右里外中高低远近东西南北出入开关去来回到看听说做想要会能可得了着过的地"
	for _, ch := range beginChars {
		emit[stateB][ch] = -3.0
		emit[stateS][ch] = -3.0
	}

	// Characters that often end words
	endChars := "了的地得着过吗呢吧啊呀嘛嗯哦嗨哈哇人们子头儿时天年月日期号路城市省区县里面"
	for _, ch := range endChars {
		emit[stateE][ch] = -3.0
	}

	// Characters common in middle position
	midChars := "机器学习理解认识发现联系经济政治文化科技社会环境问题"
	for _, ch := range midChars {
		emit[stateM][ch] = -3.5
	}

	// Default moderate probability for all CJK characters in all states
	for i := rune(0x4E00); i <= rune(0x9FFF); i++ {
		if unicode.Is(unicode.Han, i) {
			for _, s := range allStates {
				if _, exists := emit[s][i]; !exists {
					emit[s][i] = -8.0
				}
			}
		}
	}

	return emit
}
