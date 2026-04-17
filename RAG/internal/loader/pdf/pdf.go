// Package pdf implements a basic PDF text extractor.
// Uses github.com/ledongthuc/pdf (BSD License) for pure Go PDF parsing.
package pdf

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"strings"

	pdfReader "github.com/ledongthuc/pdf"

	"github.com/your-org/rag/internal/loader"
)

// Parser reads and extracts text from PDF files.
type Parser struct{}

// New creates a new PDF Parser.
func New() *Parser { return &Parser{} }

// Load reads a PDF file and extracts text content.
func (p *Parser) Load(_ context.Context, path string) ([]loader.Result, error) {
	f, reader, err := pdfReader.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open PDF %s: %w", path, err)
	}
	defer f.Close()

	var buf bytes.Buffer
	totalPages := reader.NumPage()

	for i := 1; i <= totalPages; i++ {
		page := reader.Page(i)
		if page.V.IsNull() {
			continue
		}
		text, err := page.GetPlainText(nil)
		if err != nil {
			continue
		}
		buf.WriteString(text)
		buf.WriteString("\n\n")
	}

	content := strings.TrimSpace(buf.String())
	if content == "" {
		return nil, fmt.Errorf("no text content extracted from %s", path)
	}

	title := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))

	return []loader.Result{
		{
			Title:   title,
			Content: content,
			Source:  path,
		},
	}, nil
}

func (p *Parser) Name() string { return "pdf" }
