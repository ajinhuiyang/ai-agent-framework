// Package depparser implements a rule-based Chinese dependency parser.
//
// Since we can't use CGO/ML models, this uses a shift-reduce style parser
// with hand-crafted rules based on POS tag patterns. This covers the most
// common Chinese sentence structures:
//   - SVO (主谓宾)
//   - Topic-comment
//   - Serial verb constructions
//   - 把/被 constructions
//   - Modifier-head relationships (的/地/得)
package depparser

import (
	"github.com/your-org/nlu/internal/understand/postagger"
)

// Relation represents a dependency relation type.
type Relation string

const (
	RelROOT Relation = "ROOT" // root of sentence
	RelSBV  Relation = "SBV"  // subject-verb 主谓关系
	RelVOB  Relation = "VOB"  // verb-object 动宾关系
	RelATT  Relation = "ATT"  // attribute 定中关系
	RelADV  Relation = "ADV"  // adverbial 状中关系
	RelCMP  Relation = "CMP"  // complement 动补关系
	RelCOO  Relation = "COO"  // coordinate 并列关系
	RelPOB  Relation = "POB"  // preposition-object 介宾关系
	RelBA   Relation = "BA"   // 把字句
	RelBEI  Relation = "BEI"  // 被字句
	RelDE   Relation = "DE"   // 的/地/得 structural
	RelMOD  Relation = "MOD"  // modifier 修饰
	RelQUN  Relation = "QUN"  // quantity 数量
	RelAPP  Relation = "APP"  // apposition 同位
	RelWP   Relation = "WP"   // punctuation
	RelIC   Relation = "IC"   // independent clause
	RelHED  Relation = "HED"  // head (default)
	RelUNK  Relation = "UNK"  // unknown
)

// DepNode represents a node in the dependency tree.
type DepNode struct {
	ID       int           `json:"id"` // 1-indexed position
	Word     string        `json:"word"`
	POS      postagger.Tag `json:"pos"`
	Head     int           `json:"head"` // 0 = root
	Relation Relation      `json:"relation"`
	Children []int         `json:"children,omitempty"`
}

// DepTree is a full dependency parse tree.
type DepTree struct {
	Nodes    []DepNode `json:"nodes"`
	RootID   int       `json:"root_id"`
	Sentence string    `json:"sentence"`
}

// Parser performs dependency parsing.
type Parser struct{}

// New creates a new dependency parser.
func New() *Parser {
	return &Parser{}
}

// Parse builds a dependency tree from POS-tagged words.
func (p *Parser) Parse(tagged []postagger.TaggedWord) *DepTree {
	n := len(tagged)
	if n == 0 {
		return &DepTree{}
	}

	// Initialize nodes
	nodes := make([]DepNode, n)
	for i, tw := range tagged {
		nodes[i] = DepNode{
			ID:       i + 1,
			Word:     tw.Token.Text,
			POS:      tw.Tag,
			Head:     0,
			Relation: RelUNK,
		}
	}

	// Step 1: Find the main verb (ROOT)
	rootIdx := p.findRoot(nodes)
	nodes[rootIdx].Head = 0
	nodes[rootIdx].Relation = RelROOT

	// Step 2: Link structural markers (的/地/得)
	p.linkStructural(nodes)

	// Step 3: Link 把/被 constructions
	p.linkBaBei(nodes, rootIdx)

	// Step 4: Link subjects and objects to the main verb
	p.linkSVO(nodes, rootIdx)

	// Step 5: Link adverbials
	p.linkAdverbials(nodes, rootIdx)

	// Step 6: Link quantity-measure constructions
	p.linkQuantity(nodes)

	// Step 7: Link preposition phrases
	p.linkPrepPhrases(nodes, rootIdx)

	// Step 8: Link remaining unlinked words
	p.linkRemaining(nodes, rootIdx)

	// Step 9: Build children lists
	for i := range nodes {
		for j := range nodes {
			if nodes[j].Head == nodes[i].ID {
				nodes[i].Children = append(nodes[i].Children, nodes[j].ID)
			}
		}
	}

	tree := &DepTree{
		Nodes:  nodes,
		RootID: nodes[rootIdx].ID,
	}

	// Build sentence string
	for _, tw := range tagged {
		tree.Sentence += tw.Token.Text
	}

	return tree
}

// findRoot finds the main predicate (verb) of the sentence.
func (p *Parser) findRoot(nodes []DepNode) int {
	// Prefer verbs; among multiple verbs, prefer the first non-auxiliary one
	bestIdx := 0
	bestScore := -1

	for i, node := range nodes {
		score := 0
		switch node.POS {
		case postagger.TagV:
			score = 10
			// Penalize common auxiliary verbs
			if node.Word == "是" || node.Word == "有" {
				score = 8
			}
			// Penalty for verbs that are more likely subordinate
			if node.Word == "要" || node.Word == "想" || node.Word == "会" || node.Word == "能" || node.Word == "可以" {
				score = 7
			}
		case postagger.TagA:
			score = 5 // adjective as predicate
		case postagger.TagN:
			score = 2 // nominal predicate (rare)
		default:
			score = 1
		}

		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	return bestIdx
}

// linkStructural handles 的/地/得 constructions.
func (p *Parser) linkStructural(nodes []DepNode) {
	for i, node := range nodes {
		if node.Word == "的" || node.Word == "地" || node.Word == "得" {
			// Link modifier to 的 (left side)
			if i > 0 && nodes[i-1].Head == 0 && nodes[i-1].Relation == RelUNK {
				nodes[i-1].Head = nodes[i].ID
				nodes[i-1].Relation = RelDE
			}
			// Link 的 to the head word (right side)
			if i+1 < len(nodes) && nodes[i].Head == 0 {
				nodes[i].Head = nodes[i+1].ID
				nodes[i].Relation = RelDE
			}
		}
	}
}

// linkBaBei handles 把/被 constructions.
func (p *Parser) linkBaBei(nodes []DepNode, rootIdx int) {
	for i, node := range nodes {
		if node.Word == "把" {
			nodes[i].Head = nodes[rootIdx].ID
			nodes[i].Relation = RelBA
			// Object of 把 is the next noun
			if i+1 < len(nodes) && isNounPOS(nodes[i+1].POS) {
				nodes[i+1].Head = nodes[rootIdx].ID
				nodes[i+1].Relation = RelVOB
			}
		}
		if node.Word == "被" {
			nodes[i].Head = nodes[rootIdx].ID
			nodes[i].Relation = RelBEI
			// Agent after 被 is a noun
			if i+1 < len(nodes) && isNounPOS(nodes[i+1].POS) {
				nodes[i+1].Head = nodes[rootIdx].ID
				nodes[i+1].Relation = RelSBV
			}
		}
	}
}

// linkSVO links subjects and objects to the main verb.
func (p *Parser) linkSVO(nodes []DepNode, rootIdx int) {
	// Subject: first unlinked noun/pronoun before root
	subjectFound := false
	for i := rootIdx - 1; i >= 0; i-- {
		if nodes[i].Head != 0 || nodes[i].Relation != RelUNK {
			continue
		}
		if isNounPOS(nodes[i].POS) || nodes[i].POS == postagger.TagR {
			if !subjectFound {
				nodes[i].Head = nodes[rootIdx].ID
				nodes[i].Relation = RelSBV
				subjectFound = true
			}
		}
	}

	// Object: first unlinked noun/pronoun after root
	objectFound := false
	for i := rootIdx + 1; i < len(nodes); i++ {
		if nodes[i].Head != 0 || nodes[i].Relation != RelUNK {
			continue
		}
		if isNounPOS(nodes[i].POS) || nodes[i].POS == postagger.TagR {
			if !objectFound {
				nodes[i].Head = nodes[rootIdx].ID
				nodes[i].Relation = RelVOB
				objectFound = true
			}
		}
	}
}

// linkAdverbials links adverbs and auxiliary verbs to the main verb.
func (p *Parser) linkAdverbials(nodes []DepNode, rootIdx int) {
	for i, node := range nodes {
		if node.Head != 0 || node.Relation != RelUNK {
			continue
		}
		if node.POS == postagger.TagD {
			nodes[i].Head = nodes[rootIdx].ID
			nodes[i].Relation = RelADV
		}
	}
}

// linkQuantity links numeral-measure word constructions.
func (p *Parser) linkQuantity(nodes []DepNode) {
	for i := 0; i < len(nodes)-1; i++ {
		if nodes[i].POS == postagger.TagM && nodes[i+1].POS == postagger.TagQ {
			// Link numeral to measure word
			nodes[i].Head = nodes[i+1].ID
			nodes[i].Relation = RelQUN
			// If followed by a noun, link measure to noun
			if i+2 < len(nodes) && isNounPOS(nodes[i+2].POS) {
				nodes[i+1].Head = nodes[i+2].ID
				nodes[i+1].Relation = RelATT
			}
		}
	}
}

// linkPrepPhrases links prepositional phrases.
func (p *Parser) linkPrepPhrases(nodes []DepNode, rootIdx int) {
	for i, node := range nodes {
		if node.Head != 0 || node.Relation != RelUNK {
			continue
		}
		if node.POS == postagger.TagP {
			// Link preposition to the verb
			nodes[i].Head = nodes[rootIdx].ID
			nodes[i].Relation = RelADV
			// Link object of preposition
			if i+1 < len(nodes) && isNounPOS(nodes[i+1].POS) && nodes[i+1].Head == 0 {
				nodes[i+1].Head = nodes[i].ID
				nodes[i+1].Relation = RelPOB
			}
		}
	}
}

// linkRemaining assigns all unlinked words to the nearest head.
func (p *Parser) linkRemaining(nodes []DepNode, rootIdx int) {
	for i := range nodes {
		if nodes[i].Head != 0 || nodes[i].Relation != RelUNK {
			continue
		}
		if i == rootIdx {
			continue
		}
		// Default: attach to root
		nodes[i].Head = nodes[rootIdx].ID

		switch nodes[i].POS {
		case postagger.TagW:
			nodes[i].Relation = RelWP
		case postagger.TagY:
			nodes[i].Relation = RelWP
		case postagger.TagA:
			if i < rootIdx {
				nodes[i].Relation = RelADV // adjective before verb → adverbial
			} else {
				nodes[i].Relation = RelCMP // after verb → complement
			}
		case postagger.TagU:
			nodes[i].Relation = RelDE
		case postagger.TagV:
			nodes[i].Relation = RelCOO // coordinate verb
		default:
			nodes[i].Relation = RelMOD
		}
	}
}

func isNounPOS(tag postagger.Tag) bool {
	return tag == postagger.TagN || tag == postagger.TagNR ||
		tag == postagger.TagNS || tag == postagger.TagNT ||
		tag == postagger.TagNZ || tag == postagger.TagT ||
		tag == postagger.TagS
}
