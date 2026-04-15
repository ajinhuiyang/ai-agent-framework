// Package postagger implements part-of-speech tagging for Chinese.
//
// Uses a rule-based + dictionary approach:
// 1. Dictionary lookup for known words → candidate POS tags
// 2. Contextual rules to disambiguate (similar to Brill tagger)
// 3. Default heuristics for unknown words
package postagger

import (
	"strings"
	"unicode"

	"github.com/your-org/nlu/internal/understand/tokenizer"
)

// Tag represents a part-of-speech tag.
type Tag string

// POS tag constants following the PKU/ICTCLAS tagset.
const (
	TagN   Tag = "n"   // 名词 noun
	TagNR  Tag = "nr"  // 人名 personal name
	TagNS  Tag = "ns"  // 地名 place name
	TagNT  Tag = "nt"  // 机构名 organization
	TagNZ  Tag = "nz"  // 其他专名 other proper noun
	TagV   Tag = "v"   // 动词 verb
	TagVD  Tag = "vd"  // 副动词 adverb+verb
	TagVN  Tag = "vn"  // 名动词 noun+verb
	TagA   Tag = "a"   // 形容词 adjective
	TagAD  Tag = "ad"  // 副形词 adverb+adjective
	TagAN  Tag = "an"  // 名形词 noun+adjective
	TagD   Tag = "d"   // 副词 adverb
	TagM   Tag = "m"   // 数词 numeral
	TagQ   Tag = "q"   // 量词 measure word
	TagR   Tag = "r"   // 代词 pronoun
	TagP   Tag = "p"   // 介词 preposition
	TagC   Tag = "c"   // 连词 conjunction
	TagU   Tag = "u"   // 助词 auxiliary
	TagUDE Tag = "ude" // 的
	TagF   Tag = "f"   // 方位词 direction/location
	TagS   Tag = "s"   // 处所词 place word
	TagT   Tag = "t"   // 时间词 time word
	TagW   Tag = "w"   // 标点 punctuation
	TagX   Tag = "x"   // 其他 unknown/other
	TagE   Tag = "e"   // 叹词 interjection
	TagY   Tag = "y"   // 语气词 modal particle
	TagO   Tag = "o"   // 拟声词 onomatopoeia
)

// TaggedWord pairs a token with its POS tag.
type TaggedWord struct {
	Token tokenizer.Token `json:"token"`
	Tag   Tag             `json:"tag"`
}

// Tagger performs part-of-speech tagging.
type Tagger struct {
	wordTags map[string]Tag // known word → primary POS
}

// New creates a new POS tagger with built-in dictionary.
func New() *Tagger {
	t := &Tagger{
		wordTags: make(map[string]Tag),
	}
	t.loadBuiltinTags()
	return t
}

// TagTokens assigns POS tags to a sequence of tokens.
func (t *Tagger) TagTokens(tokens []tokenizer.Token) []TaggedWord {
	result := make([]TaggedWord, len(tokens))

	for i, tok := range tokens {
		tag := t.tagSingle(tok.Text)
		result[i] = TaggedWord{Token: tok, Tag: tag}
	}

	// Apply contextual rules for disambiguation
	t.applyContextRules(result)

	return result
}

// tagSingle determines the POS tag for a single word.
func (t *Tagger) tagSingle(word string) Tag {
	// 1. Dictionary lookup
	if tag, ok := t.wordTags[word]; ok {
		return tag
	}

	// 2. Heuristics for unknown words
	runes := []rune(word)

	// Pure punctuation
	if isPunct(word) {
		return TagW
	}

	// Pure digits or numeric pattern
	if isNumeric(word) {
		return TagM
	}

	// English words
	if isAllAlpha(word) {
		return TagX
	}

	// Suffix-based heuristics for Chinese
	if len(runes) >= 2 {
		lastChar := string(runes[len(runes)-1])
		secondLast := ""
		if len(runes) >= 3 {
			secondLast = string(runes[len(runes)-2])
		}

		// Common noun suffixes
		nounSuffixes := map[string]bool{
			"人": true, "者": true, "家": true, "师": true, "员": true,
			"机": true, "器": true, "物": true, "品": true, "件": true,
			"性": true, "化": true, "度": true, "率": true, "量": true,
			"学": true, "法": true, "论": true, "力": true, "式": true,
		}
		if nounSuffixes[lastChar] {
			return TagN
		}

		// Common verb suffixes
		verbSuffixes := map[string]bool{
			"到": true, "完": true, "好": true, "住": true, "掉": true,
		}
		if verbSuffixes[lastChar] {
			return TagV
		}

		// Place name patterns
		placeSuffixes := map[string]bool{
			"省": true, "市": true, "县": true, "区": true, "镇": true,
			"村": true, "路": true, "街": true, "山": true, "河": true,
			"湖": true, "海": true, "洋": true, "岛": true,
		}
		if placeSuffixes[lastChar] {
			return TagNS
		}

		// Organization suffixes
		orgSuffixes := map[string]bool{
			"局": true, "部": true, "厅": true, "院": true, "所": true,
			"社": true, "司": true, "厂": true, "校": true, "馆": true,
		}
		if orgSuffixes[lastChar] {
			return TagNT
		}

		// Time words
		timeSuffixes := map[string]bool{
			"日": true, "号": true,
		}
		if timeSuffixes[lastChar] {
			return TagT
		}
		_ = secondLast // reserved for multi-char suffix rules
	}

	// Default: noun (most common tag for unknowns)
	return TagN
}

// applyContextRules applies contextual disambiguation rules.
func (t *Tagger) applyContextRules(words []TaggedWord) {
	n := len(words)
	for i := 0; i < n; i++ {
		text := words[i].Token.Text

		// Rule 1: "的" → ude (structural auxiliary)
		if text == "的" {
			words[i].Tag = TagUDE
		}

		// Rule 2: "了/着/过" after verb → auxiliary
		if (text == "了" || text == "着" || text == "过") && i > 0 {
			if words[i-1].Tag == TagV {
				words[i].Tag = TagU
			}
		}

		// Rule 3: word before "的" + noun → adjective modifier
		if i+2 < n && words[i+1].Token.Text == "的" && isNounTag(words[i+2].Tag) {
			if words[i].Tag == TagV {
				words[i].Tag = TagA // treat as adjective in this context
			}
		}

		// Rule 4: "在" before verb → adverb; "在" before noun → preposition
		if text == "在" {
			if i+1 < n {
				if words[i+1].Tag == TagV {
					words[i].Tag = TagD
				} else if isNounTag(words[i+1].Tag) {
					words[i].Tag = TagP
				}
			}
		}

		// Rule 5: Sentence-final particles
		if i == n-1 && (text == "吗" || text == "呢" || text == "吧" || text == "啊" || text == "呀") {
			words[i].Tag = TagY
		}

		// Rule 6: "没/不" → adverb (negation)
		if text == "没" || text == "不" {
			words[i].Tag = TagD
		}
	}
}

func isNounTag(tag Tag) bool {
	return tag == TagN || tag == TagNR || tag == TagNS || tag == TagNT || tag == TagNZ
}

func isPunct(s string) bool {
	for _, r := range s {
		if !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			return false
		}
	}
	return len(s) > 0
}

func isNumeric(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) && r != '.' && r != ',' && r != '%' {
			return false
		}
	}
	return len(s) > 0
}

func isAllAlpha(s string) bool {
	for _, r := range s {
		if !unicode.IsLetter(r) || unicode.Is(unicode.Han, r) {
			return false
		}
	}
	return len(s) > 0
}

// loadBuiltinTags loads the built-in word→POS dictionary.
func (t *Tagger) loadBuiltinTags() {
	// Pronouns
	for _, w := range strings.Fields("我 你 他 她 它 我们 你们 他们 她们 这 那 这个 那个 哪 哪个 什么 谁 自己 大家 每") {
		t.wordTags[w] = TagR
	}

	// Verbs
	for _, w := range strings.Fields("是 有 说 做 看 听 去 来 走 跑 吃 喝 买 卖 想 要 会 能 知道 觉得 认为 希望 喜欢 帮助 帮忙 帮 告诉 打开 关闭 开始 结束 完成 准备 使用 提醒 设置 查询 搜索 播放 停止 暂停 继续 取消 确认 修改 删除 添加 创建 发送 接收 预订 预约 订 给 让 请 叫 找 带 拿 放 坐 站 躺 睡 起 醒 学 教 读 写 算 变 变成 成为 保持 感觉 相信 担心 怀疑 决定 选择 参加 加入 离开 回 回来 回去 到达 出发 飞 开 关 拍 拍照") {
		t.wordTags[w] = TagV
	}

	// Nouns
	for _, w := range strings.Fields("人 事 东西 地方 时间 时候 问题 工作 情况 公司 学校 医院 家 国家 世界 社会 经济 文化 科技 历史 音乐 电影 天气 新闻 机票 酒店 航班 火车 飞机 汽车 手机 电脑 电话 网络 信息 数据 系统 产品 服务 市场 价格 质量 功能 效果 方法 方式 结果 原因 目标 计划 项目 活动 内容 名字 地址 密码 账号") {
		t.wordTags[w] = TagN
	}

	// Place names
	for _, w := range strings.Fields("北京 上海 广州 深圳 成都 杭州 南京 武汉 西安 重庆 天津 苏州 长沙 青岛 大连 厦门 中国 美国 日本 韩国 英国 法国 德国 亚洲 欧洲") {
		t.wordTags[w] = TagNS
	}

	// Time words
	for _, w := range strings.Fields("今天 明天 昨天 后天 前天 现在 刚才 以后 以前 上午 下午 晚上 早上 中午 白天 傍晚 年 月 日 天 周 星期 小时 分钟 秒 下周 上周 今年 明年 去年 春天 夏天 秋天 冬天") {
		t.wordTags[w] = TagT
	}

	// Adjectives
	for _, w := range strings.Fields("好 大 小 多 少 新 旧 快 慢 高 低 远 近 早 晚 长 短 宽 窄 深 浅 厚 薄 轻 重 冷 热 暖 凉 干 湿 便宜 贵 美 丑 胖 瘦 忙 闲 难 易 简单 复杂 重要 安全 危险 干净 脏 漂亮 聪明 笨 开心 高兴 难过 生气 愤怒 害怕 棒 差 烂 优秀 糟糕 满意 失望") {
		t.wordTags[w] = TagA
	}

	// Adverbs
	for _, w := range strings.Fields("很 太 最 更 非常 十分 特别 相当 比较 稍微 有点 有些 略 极 极其 越 越来越 已经 正在 马上 立刻 马上 刚刚 刚 常常 经常 总是 往往 偶尔 从来 一直 一定 必须 应该 大概 可能 也许 或许 简直 居然 竟然 终于 终究 其实 确实 当然 显然 明显 不 没 别 甭 千万 再 又 还 也 都 就 才 只 仅 仅仅") {
		t.wordTags[w] = TagD
	}

	// Prepositions
	for _, w := range strings.Fields("在 从 到 向 往 对 把 被 让 给 比 跟 和 与 按 按照 根据 通过 经过 为 为了 关于 除了 除 沿 沿着") {
		t.wordTags[w] = TagP
	}

	// Conjunctions
	for _, w := range strings.Fields("和 与 或 或者 还是 而 而且 并且 但 但是 然而 虽然 尽管 如果 假如 要是 只要 无论 不管 因为 由于 所以 因此 于是 那么 那 可是 不过") {
		t.wordTags[w] = TagC
	}

	// Auxiliary/particles
	for _, w := range strings.Fields("的 地 得 了 着 过 吗 呢 吧 啊 呀 嘛 哦 啦") {
		t.wordTags[w] = TagU
	}

	// Measure words
	for _, w := range strings.Fields("个 张 条 件 块 把 只 双 对 套 台 部 辆 架 艘 本 封 篇 首 场 次 遍 趟 顿 位 名 元 角 分 米 公里 斤 公斤 克 升 瓶 杯 碗 盘 包") {
		t.wordTags[w] = TagQ
	}

	// Numerals
	for _, w := range strings.Fields("一 二 三 四 五 六 七 八 九 十 百 千 万 亿 零 两 几 第一 第二 第三") {
		t.wordTags[w] = TagM
	}

	// Direction/location words
	for _, w := range strings.Fields("上 下 前 后 左 右 里 外 中 内 东 西 南 北 旁边 附近 对面 周围 中间 之间 上面 下面 前面 后面 里面 外面") {
		t.wordTags[w] = TagF
	}

	// Interjections
	for _, w := range strings.Fields("哈 嗯 哦 啊 呀 哇 唉 嗨 喂") {
		t.wordTags[w] = TagE
	}
}
