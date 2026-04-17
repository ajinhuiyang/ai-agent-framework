// Package text implements a local file parser that reads TXT and Markdown files.
package text

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/your-org/rag/internal/loader"
)

// Parser reads local text files (TXT, Markdown).
type Parser struct{}

// New creates a new text file Parser.
func New() *Parser { return &Parser{} }

// Load reads a single file and returns its content.
func (p *Parser) Load(_ context.Context, path string) ([]loader.Result, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	if info.IsDir() {
		return p.loadDir(path)
	}

	return p.loadFile(path)
}

// LoadDir reads all supported files in a directory.
func (p *Parser) loadDir(dir string) ([]loader.Result, error) {
	var results []loader.Result

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read dir %s: %w", dir, err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if ext != ".txt" && ext != ".md" && ext != ".markdown" && ext != ".text" && ext != ".csv" {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		res, err := p.loadFile(path)
		if err != nil {
			continue
		}
		results = append(results, res...)
	}

	return results, nil
}

func (p *Parser) loadFile(path string) ([]loader.Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file %s: %w", path, err)
	}

	content := strings.TrimSpace(string(data))
	if content == "" {
		return nil, nil
	}

	// Extract title from filename or first heading.
	title := titleFromPath(path)
	if strings.HasSuffix(path, ".md") || strings.HasSuffix(path, ".markdown") {
		if mdTitle := extractMarkdownTitle(content); mdTitle != "" {
			title = mdTitle
		}
	}

	return []loader.Result{
		{
			Title:   title,
			Content: content,
			Source:  path,
		},
	}, nil
}

func (p *Parser) Name() string { return "text" }

func titleFromPath(path string) string {
	base := filepath.Base(path)
	ext := filepath.Ext(base)
	return strings.TrimSuffix(base, ext)
}

func extractMarkdownTitle(content string) string {
	for _, line := range strings.SplitN(content, "\n", 10) {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimPrefix(line, "# ")
		}
	}
	return ""
}
