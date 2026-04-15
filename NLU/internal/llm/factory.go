// Package llm provides a factory for creating LLM providers.
package llm

import "fmt"

// ProviderFactory creates LLM providers based on configuration.
type ProviderFactory struct {
	providers map[string]Provider
}

// NewProviderFactory creates a new provider factory.
func NewProviderFactory() *ProviderFactory {
	return &ProviderFactory{
		providers: make(map[string]Provider),
	}
}

// Register adds a provider to the factory.
func (f *ProviderFactory) Register(name string, provider Provider) {
	f.providers[name] = provider
}

// Get returns a provider by name.
func (f *ProviderFactory) Get(name string) (Provider, error) {
	p, ok := f.providers[name]
	if !ok {
		return nil, fmt.Errorf("unknown LLM provider: %s", name)
	}
	return p, nil
}

// List returns all registered provider names.
func (f *ProviderFactory) List() []string {
	names := make([]string, 0, len(f.providers))
	for name := range f.providers {
		names = append(names, name)
	}
	return names
}
