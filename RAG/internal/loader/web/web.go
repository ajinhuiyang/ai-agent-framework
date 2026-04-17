// Package web implements a web page scraper that fetches URLs,
// parses HTML, and extracts the main text content.
//
// Uses: net/http (stdlib) + github.com/PuerkitoBio/goquery (MIT License)
package web

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"

	"github.com/your-org/rag/internal/loader"
)

// Scraper fetches and parses web pages.
type Scraper struct {
	client    *http.Client
	userAgent string
}

// New creates a new web Scraper.
func New() *Scraper {
	return &Scraper{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		userAgent: "GoInfer-RAG/1.0 (compatible; bot)",
	}
}

// Load fetches a URL and extracts the main text content.
func (s *Scraper) Load(ctx context.Context, url string) ([]loader.Result, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("User-Agent", s.userAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch %s returned status %d", url, resp.StatusCode)
	}

	// Limit response body to 5MB.
	body := io.LimitReader(resp.Body, 5*1024*1024)

	doc, err := goquery.NewDocumentFromReader(body)
	if err != nil {
		return nil, fmt.Errorf("parse HTML: %w", err)
	}

	// Remove noise elements.
	doc.Find("script, style, nav, footer, header, aside, iframe, noscript, .ad, .ads, .sidebar").Remove()

	// Extract title.
	title := strings.TrimSpace(doc.Find("title").First().Text())

	// Try to find main content area.
	var content string
	for _, selector := range []string{"article", "main", ".content", ".post", ".article-body", "#content"} {
		sel := doc.Find(selector).First()
		if sel.Length() > 0 {
			content = extractText(sel)
			break
		}
	}

	// Fallback: use body text.
	if content == "" {
		content = extractText(doc.Find("body"))
	}

	// Clean up whitespace.
	content = cleanText(content)

	if content == "" {
		return nil, fmt.Errorf("no text content found at %s", url)
	}

	return []loader.Result{
		{
			Title:   title,
			Content: content,
			Source:  url,
		},
	}, nil
}

// LoadMultiple fetches multiple URLs.
func (s *Scraper) LoadMultiple(ctx context.Context, urls []string) ([]loader.Result, error) {
	var results []loader.Result
	for _, url := range urls {
		res, err := s.Load(ctx, url)
		if err != nil {
			continue // Skip failed URLs.
		}
		results = append(results, res...)
	}
	return results, nil
}

func (s *Scraper) Name() string { return "web" }

// extractText gets cleaned text from a goquery selection.
func extractText(sel *goquery.Selection) string {
	var parts []string
	sel.Find("p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, pre").Each(func(_ int, s *goquery.Selection) {
		text := strings.TrimSpace(s.Text())
		if len(text) > 10 { // Skip very short fragments.
			parts = append(parts, text)
		}
	})
	return strings.Join(parts, "\n\n")
}

// cleanText removes excessive whitespace.
func cleanText(s string) string {
	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			cleaned = append(cleaned, line)
		}
	}
	return strings.Join(cleaned, "\n")
}
