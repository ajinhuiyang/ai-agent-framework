// Package loader defines the interface for loading content from various sources.
package loader

import "context"

// Result represents content loaded from an external source.
type Result struct {
	Title   string `json:"title,omitempty"`
	Content string `json:"content"`
	Source  string `json:"source"` // URL, file path, etc.
}

// Loader loads content from an external source.
type Loader interface {
	// Load retrieves content and returns parsed results.
	Load(ctx context.Context, input string) ([]Result, error)

	// Name returns the loader type name.
	Name() string
}
