// Package search implements search engine scraping without API keys.
// Scrapes Google/Bing search result pages to get links and snippets,
// then optionally fetches and parses the result pages for full content.
package search

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"

	"github.com/your-org/rag/internal/loader"
	"github.com/your-org/rag/internal/loader/web"
)

// Engine specifies which search engine to scrape.
type Engine string

const (
	Google Engine = "google"
	Bing   Engine = "bing"
)

// Searcher scrapes search engine results.
type Searcher struct {
	engine    Engine
	client    *http.Client
	userAgent string
	scraper   *web.Scraper // For fetching full page content.
}

// New creates a new search engine Searcher.
func New(engine Engine) *Searcher {
	return &Searcher{
		engine: engine,
		client: &http.Client{
			Timeout: 15 * time.Second,
		},
		userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
		scraper:   web.New(),
	}
}

// SearchResult is a single search engine result.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// Load searches for the query and returns results with snippets.
// If fetchContent is needed, use LoadWithContent instead.
func (s *Searcher) Load(ctx context.Context, query string) ([]loader.Result, error) {
	results, err := s.scrapeResults(ctx, query)
	if err != nil {
		return nil, err
	}

	var loaderResults []loader.Result
	for _, r := range results {
		loaderResults = append(loaderResults, loader.Result{
			Title:   r.Title,
			Content: r.Snippet,
			Source:  r.URL,
		})
	}
	return loaderResults, nil
}

// LoadWithContent searches and fetches full content of the top N result pages.
func (s *Searcher) LoadWithContent(ctx context.Context, query string, topN int) ([]loader.Result, error) {
	results, err := s.scrapeResults(ctx, query)
	if err != nil {
		return nil, err
	}

	if topN <= 0 {
		topN = 3
	}
	if topN > len(results) {
		topN = len(results)
	}

	var urls []string
	for i := 0; i < topN; i++ {
		urls = append(urls, results[i].URL)
	}

	return s.scraper.LoadMultiple(ctx, urls)
}

func (s *Searcher) Name() string { return "search-" + string(s.engine) }

// scrapeResults scrapes the search engine result page.
func (s *Searcher) scrapeResults(ctx context.Context, query string) ([]SearchResult, error) {
	switch s.engine {
	case Google:
		return s.scrapeGoogle(ctx, query)
	case Bing:
		return s.scrapeBing(ctx, query)
	default:
		return s.scrapeGoogle(ctx, query)
	}
}

func (s *Searcher) scrapeGoogle(ctx context.Context, query string) ([]SearchResult, error) {
	searchURL := fmt.Sprintf("https://www.google.com/search?q=%s&num=10&hl=en", url.QueryEscape(query))

	doc, err := s.fetchAndParse(ctx, searchURL)
	if err != nil {
		return nil, fmt.Errorf("google search: %w", err)
	}

	var results []SearchResult
	doc.Find("div.g").Each(func(_ int, sel *goquery.Selection) {
		link := sel.Find("a").First()
		href, exists := link.Attr("href")
		if !exists || !strings.HasPrefix(href, "http") {
			return
		}

		title := strings.TrimSpace(link.Find("h3").Text())
		if title == "" {
			title = strings.TrimSpace(link.Text())
		}

		snippet := ""
		sel.Find("div.VwiC3b, span.aCOpRe, div[data-sncf]").Each(func(_ int, s *goquery.Selection) {
			if text := strings.TrimSpace(s.Text()); text != "" {
				snippet = text
			}
		})

		if title != "" && href != "" {
			results = append(results, SearchResult{
				Title:   title,
				URL:     href,
				Snippet: snippet,
			})
		}
	})

	return results, nil
}

func (s *Searcher) scrapeBing(ctx context.Context, query string) ([]SearchResult, error) {
	searchURL := fmt.Sprintf("https://www.bing.com/search?q=%s&count=10", url.QueryEscape(query))

	doc, err := s.fetchAndParse(ctx, searchURL)
	if err != nil {
		return nil, fmt.Errorf("bing search: %w", err)
	}

	var results []SearchResult
	doc.Find("li.b_algo").Each(func(_ int, sel *goquery.Selection) {
		link := sel.Find("h2 a").First()
		href, exists := link.Attr("href")
		if !exists {
			return
		}

		title := strings.TrimSpace(link.Text())
		snippet := strings.TrimSpace(sel.Find("p, .b_caption p").First().Text())

		if title != "" && href != "" {
			results = append(results, SearchResult{
				Title:   title,
				URL:     href,
				Snippet: snippet,
			})
		}
	})

	return results, nil
}

func (s *Searcher) fetchAndParse(ctx context.Context, targetURL string) (*goquery.Document, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", s.userAgent)
	req.Header.Set("Accept", "text/html")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9,zh-CN;q=0.8")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}

	return goquery.NewDocumentFromReader(resp.Body)
}
