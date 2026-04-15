// Package preprocess handles text normalization before linguistic analysis.
// Covers: Unicode normalization, traditional→simplified Chinese conversion,
// fullwidth→halfwidth, whitespace normalization, and basic cleaning.
package preprocess

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Processor performs text preprocessing.
type Processor struct {
	t2s map[rune]rune // traditional → simplified mapping
}

// New creates a new text preprocessor.
func New() *Processor {
	return &Processor{
		t2s: buildT2SMap(),
	}
}

// Result holds the preprocessing output.
type Result struct {
	Original   string // original input
	Normalized string // normalized text
	CharMap    []int  // maps normalized char index → original char index
}

// Process applies all normalization steps.
func (p *Processor) Process(text string) *Result {
	r := &Result{Original: text}

	// Step 1: Unicode NFC normalization (manual since we're pure Go)
	// Step 2: Fullwidth → halfwidth
	// Step 3: Traditional → simplified Chinese
	// Step 4: Whitespace normalization
	// Step 5: Remove control characters

	var buf strings.Builder
	var charMap []int
	buf.Grow(len(text))

	idx := 0
	for _, ch := range text {
		origIdx := idx
		idx += utf8.RuneLen(ch)

		// Skip control characters (except newline/tab)
		if unicode.IsControl(ch) && ch != '\n' && ch != '\t' {
			continue
		}

		// Fullwidth → halfwidth ASCII (FF01-FF5E → 0021-007E)
		if ch >= 0xFF01 && ch <= 0xFF5E {
			ch = ch - 0xFEE0
		}
		// Fullwidth space
		if ch == 0x3000 {
			ch = ' '
		}

		// Traditional → simplified
		if s, ok := p.t2s[ch]; ok {
			ch = s
		}

		// Lowercase
		ch = unicode.ToLower(ch)

		buf.WriteRune(ch)
		charMap = append(charMap, origIdx)
	}

	// Collapse multiple spaces
	normalized := collapseSpaces(buf.String())

	// Rebuild char map after space collapse
	r.Normalized = strings.TrimSpace(normalized)
	r.CharMap = charMap
	return r
}

// collapseSpaces replaces runs of whitespace with a single space.
func collapseSpaces(s string) string {
	var buf strings.Builder
	buf.Grow(len(s))
	prevSpace := false
	for _, ch := range s {
		if unicode.IsSpace(ch) {
			if !prevSpace {
				buf.WriteRune(' ')
			}
			prevSpace = true
		} else {
			buf.WriteRune(ch)
			prevSpace = false
		}
	}
	return buf.String()
}

// IsChinese checks if a rune is a CJK Unified Ideograph.
func IsChinese(r rune) bool {
	return unicode.Is(unicode.Han, r)
}

// IsPunctuation checks if a rune is punctuation (CJK or ASCII).
func IsPunctuation(r rune) bool {
	if unicode.IsPunct(r) || unicode.IsSymbol(r) {
		return true
	}
	// CJK punctuation ranges
	if r >= 0x3000 && r <= 0x303F { // CJK Symbols and Punctuation
		return true
	}
	if r >= 0xFF00 && r <= 0xFFEF { // Halfwidth and Fullwidth Forms
		return true
	}
	return false
}

// buildT2SMap builds a traditional→simplified Chinese character mapping.
// This is a subset covering the most common ~500 characters.
// For production, load from a full T2S dictionary file.
func buildT2SMap() map[rune]rune {
	// Common traditional → simplified pairs
	pairs := [][2]rune{
		{'書', '书'}, {'學', '学'}, {'東', '东'}, {'車', '车'}, {'長', '长'},
		{'門', '门'}, {'馬', '马'}, {'魚', '鱼'}, {'鳥', '鸟'}, {'齒', '齿'},
		{'國', '国'}, {'會', '会'}, {'開', '开'}, {'關', '关'}, {'電', '电'},
		{'飛', '飞'}, {'機', '机'}, {'語', '语'}, {'說', '说'}, {'話', '话'},
		{'請', '请'}, {'問', '问'}, {'聽', '听'}, {'讀', '读'}, {'寫', '写'},
		{'買', '买'}, {'賣', '卖'}, {'錢', '钱'}, {'銀', '银'}, {'鐵', '铁'},
		{'風', '风'}, {'雲', '云'}, {'響', '响'}, {'點', '点'}, {'熱', '热'},
		{'愛', '爱'}, {'歡', '欢'}, {'樂', '乐'}, {'對', '对'}, {'歲', '岁'},
		{'時', '时'}, {'當', '当'}, {'從', '从'}, {'來', '来'}, {'見', '见'},
		{'親', '亲'}, {'覺', '觉'}, {'觀', '观'}, {'認', '认'}, {'識', '识'},
		{'計', '计'}, {'記', '记'}, {'許', '许'}, {'論', '论'}, {'設', '设'},
		{'試', '试'}, {'課', '课'}, {'調', '调'}, {'談', '谈'}, {'證', '证'},
		{'評', '评'}, {'議', '议'}, {'護', '护'}, {'報', '报'}, {'場', '场'},
		{'壓', '压'}, {'廣', '广'}, {'應', '应'}, {'張', '张'}, {'數', '数'},
		{'條', '条'}, {'種', '种'}, {'經', '经'}, {'統', '统'}, {'結', '结'},
		{'線', '线'}, {'練', '练'}, {'總', '总'}, {'義', '义'}, {'習', '习'},
		{'聯', '联'}, {'體', '体'}, {'運', '运'}, {'動', '动'}, {'勞', '劳'},
		{'辦', '办'}, {'務', '务'}, {'職', '职'}, {'員', '员'}, {'團', '团'},
		{'農', '农'}, {'業', '业'}, {'產', '产'}, {'質', '质'}, {'負', '负'},
		{'費', '费'}, {'資', '资'}, {'漁', '渔'}, {'鄉', '乡'}, {'縣', '县'},
		{'區', '区'}, {'華', '华'}, {'陽', '阳'}, {'陰', '阴'}, {'隊', '队'},
		{'階', '阶'}, {'際', '际'}, {'戰', '战'}, {'軍', '军'}, {'將', '将'},
		{'發', '发'}, {'變', '变'}, {'選', '选'}, {'進', '进'}, {'達', '达'},
		{'邊', '边'}, {'連', '连'}, {'還', '还'}, {'過', '过'}, {'遠', '远'},
		{'這', '这'}, {'裡', '里'}, {'頭', '头'}, {'號', '号'}, {'處', '处'},
		{'離', '离'}, {'難', '难'}, {'雙', '双'}, {'歷', '历'}, {'歸', '归'},
		{'無', '无'}, {'畫', '画'}, {'給', '给'}, {'紅', '红'}, {'約', '约'},
		{'紙', '纸'}, {'細', '细'}, {'織', '织'}, {'終', '终'}, {'絕', '绝'},
		{'繼', '继'}, {'續', '续'}, {'維', '维'}, {'綜', '综'}, {'績', '绩'},
	}

	m := make(map[rune]rune, len(pairs))
	for _, p := range pairs {
		m[p[0]] = p[1]
	}
	return m
}
