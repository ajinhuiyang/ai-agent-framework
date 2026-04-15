// Package srl implements semantic role labeling (语义角色标注) for Chinese.
//
// SRL answers "who did what to whom, when, where, how, and why" by
// assigning semantic roles to constituents relative to a predicate.
//
// This is a rule-based implementation that uses dependency parse results
// and POS tags to assign PropBank-style roles (ARG0, ARG1, etc.).
package srl

import (
	"github.com/your-org/nlu/internal/understand/depparser"
	"github.com/your-org/nlu/internal/understand/postagger"
)

// Role represents a semantic role.
type Role string

const (
	RoleAgent       Role = "ARG0"     // Agent / Experiencer (施事)
	RolePatient     Role = "ARG1"     // Patient / Theme (受事)
	RoleBeneficiary Role = "ARG2"     // Beneficiary / Goal (与事)
	RoleStartPoint  Role = "ARG3"     // Start point (起点)
	RoleEndPoint    Role = "ARG4"     // End point (终点)
	RoleTime        Role = "ARGM-TMP" // Temporal (时间)
	RoleLocation    Role = "ARGM-LOC" // Location (地点)
	RoleManner      Role = "ARGM-MNR" // Manner (方式)
	RoleCause       Role = "ARGM-CAU" // Cause (原因)
	RolePurpose     Role = "ARGM-PRP" // Purpose (目的)
	RoleNegation    Role = "ARGM-NEG" // Negation (否定)
	RoleDegree      Role = "ARGM-EXT" // Extent/Degree (程度)
	RoleModal       Role = "ARGM-MOD" // Modal (情态)
	RoleDirection   Role = "ARGM-DIR" // Direction (方向)
	RoleTool        Role = "ARGM-MNR" // Instrument (工具)
	RoleDiscourse   Role = "ARGM-DIS" // Discourse (话语)
)

// Frame represents a predicate-argument structure.
type Frame struct {
	Predicate   string     `json:"predicate"`
	PredicateID int        `json:"predicate_id"` // node ID
	Arguments   []Argument `json:"arguments"`
}

// Argument is a semantic role assignment.
type Argument struct {
	Role    Role   `json:"role"`
	Text    string `json:"text"`
	NodeIDs []int  `json:"node_ids"` // dependency node IDs that form this argument
}

// Labeler performs semantic role labeling.
type Labeler struct{}

// New creates a new SRL labeler.
func New() *Labeler {
	return &Labeler{}
}

// Label assigns semantic roles given a dependency tree.
func (l *Labeler) Label(tree *depparser.DepTree) []Frame {
	if tree == nil || len(tree.Nodes) == 0 {
		return nil
	}

	var frames []Frame

	// Find all predicates (verbs)
	for _, node := range tree.Nodes {
		if node.POS != postagger.TagV {
			continue
		}
		if node.Relation == depparser.RelROOT || node.Relation == depparser.RelCOO {
			frame := l.buildFrame(tree, node)
			if frame != nil {
				frames = append(frames, *frame)
			}
		}
	}

	// If no frames found, try the root regardless of POS
	if len(frames) == 0 {
		for _, node := range tree.Nodes {
			if node.Relation == depparser.RelROOT {
				frame := l.buildFrame(tree, node)
				if frame != nil {
					frames = append(frames, *frame)
				}
				break
			}
		}
	}

	return frames
}

// buildFrame constructs a predicate-argument frame for a given predicate node.
func (l *Labeler) buildFrame(tree *depparser.DepTree, pred depparser.DepNode) *Frame {
	frame := &Frame{
		Predicate:   pred.Word,
		PredicateID: pred.ID,
	}

	for _, node := range tree.Nodes {
		if node.Head != pred.ID {
			continue
		}

		role := l.assignRole(node, pred)
		if role == "" {
			continue
		}

		// Collect the full span text (node + its subtree)
		text := l.collectSpanText(tree, node)
		frame.Arguments = append(frame.Arguments, Argument{
			Role:    role,
			Text:    text,
			NodeIDs: l.collectSubtreeIDs(tree, node),
		})
	}

	return frame
}

// assignRole determines the semantic role based on dependency relation and POS.
func (l *Labeler) assignRole(node depparser.DepNode, pred depparser.DepNode) Role {
	switch node.Relation {
	case depparser.RelSBV:
		return RoleAgent // 主语 → Agent

	case depparser.RelVOB:
		return RolePatient // 宾语 → Patient

	case depparser.RelADV:
		return l.classifyAdverbial(node)

	case depparser.RelCMP:
		return RoleManner // 补语 → Manner (simplified)

	case depparser.RelBA:
		return "" // 把 itself is structural, skip

	case depparser.RelBEI:
		return "" // 被 itself is structural, skip

	case depparser.RelPOB:
		return l.classifyPOB(node)

	case depparser.RelWP:
		return "" // punctuation, skip

	case depparser.RelDE:
		return "" // structural, skip

	default:
		// Try POS-based assignment
		switch node.POS {
		case postagger.TagT:
			return RoleTime
		case postagger.TagNS, postagger.TagS:
			return RoleLocation
		case postagger.TagD:
			return l.classifyAdverb(node)
		}
	}

	return ""
}

// classifyAdverbial classifies adverbial modifiers into semantic roles.
func (l *Labeler) classifyAdverbial(node depparser.DepNode) Role {
	switch node.POS {
	case postagger.TagT:
		return RoleTime
	case postagger.TagNS, postagger.TagS, postagger.TagF:
		return RoleLocation
	case postagger.TagD:
		return l.classifyAdverb(node)
	case postagger.TagP:
		return l.classifyPreposition(node)
	default:
		return RoleManner
	}
}

// classifyAdverb assigns a role based on the specific adverb.
func (l *Labeler) classifyAdverb(node depparser.DepNode) Role {
	negations := map[string]bool{"不": true, "没": true, "没有": true, "别": true, "未": true, "甭": true}
	if negations[node.Word] {
		return RoleNegation
	}

	modals := map[string]bool{"会": true, "能": true, "可以": true, "应该": true, "必须": true, "要": true, "想": true, "得": true}
	if modals[node.Word] {
		return RoleModal
	}

	degrees := map[string]bool{"很": true, "非常": true, "太": true, "特别": true, "极": true, "十分": true, "相当": true, "最": true}
	if degrees[node.Word] {
		return RoleDegree
	}

	return RoleManner
}

// classifyPreposition assigns a role based on the preposition.
func (l *Labeler) classifyPreposition(node depparser.DepNode) Role {
	switch node.Word {
	case "从":
		return RoleStartPoint
	case "到", "向", "往":
		return RoleEndPoint
	case "在", "于":
		return RoleLocation
	case "用", "以", "通过":
		return RoleTool
	case "为", "为了":
		return RolePurpose
	case "因", "因为", "由于":
		return RoleCause
	case "对", "跟", "和", "与", "给":
		return RoleBeneficiary
	default:
		return RoleManner
	}
}

// classifyPOB classifies prepositional object based on its POS.
func (l *Labeler) classifyPOB(node depparser.DepNode) Role {
	switch node.POS {
	case postagger.TagT:
		return RoleTime
	case postagger.TagNS, postagger.TagS:
		return RoleLocation
	default:
		return RolePatient
	}
}

// collectSpanText collects the text span of a node and all its descendants.
func (l *Labeler) collectSpanText(tree *depparser.DepTree, node depparser.DepNode) string {
	ids := l.collectSubtreeIDs(tree, node)
	// Sort by ID and concatenate
	text := ""
	for _, tnode := range tree.Nodes {
		for _, id := range ids {
			if tnode.ID == id {
				text += tnode.Word
			}
		}
	}
	return text
}

// collectSubtreeIDs collects all node IDs in the subtree rooted at the given node.
func (l *Labeler) collectSubtreeIDs(tree *depparser.DepTree, node depparser.DepNode) []int {
	ids := []int{node.ID}
	for _, child := range tree.Nodes {
		if child.Head == node.ID {
			ids = append(ids, l.collectSubtreeIDs(tree, child)...)
		}
	}
	return ids
}
