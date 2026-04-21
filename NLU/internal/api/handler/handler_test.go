package handler_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/your-org/nlu/internal/api/handler"
	"github.com/your-org/nlu/internal/nlu/dialog"
	"github.com/your-org/nlu/internal/pipeline"
	"github.com/your-org/nlu/internal/prompt"
)

func setupTestRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	logger, _ := zap.NewDevelopment()
	pm := prompt.NewManager()

	dm := dialog.New(nil, pm, logger, 10)

	engine := pipeline.NewEngine(pipeline.Config{
		Logger:      logger,
		DefaultCaps: []string{"intent", "ner", "sentiment"},
	})

	h := handler.NewNLUHandler(engine, dm, nil, nil, logger)

	r := gin.New()
	r.GET("/health", h.HealthCheck)

	v1 := r.Group("/api/v1")
	{
		nlu := v1.Group("/nlu")
		{
			nlu.POST("/process", h.Process)
			nlu.POST("/intent", h.IntentOnly)
			nlu.POST("/ner", h.NEROnly)
			nlu.POST("/sentiment", h.SentimentOnly)
			nlu.POST("/classify", h.ClassifyOnly)
			nlu.POST("/slot", h.SlotFill)
		}
		dlg := v1.Group("/dialog")
		{
			dlg.GET("/:session_id", h.GetDialogState)
			dlg.DELETE("/:session_id", h.DeleteDialog)
		}
	}

	return r
}

func TestHealthCheck(t *testing.T) {
	router := setupTestRouter()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp["status"] != "ok" {
		t.Errorf("expected status 'ok', got %v", resp["status"])
	}
}

func TestProcess_EmptyText(t *testing.T) {
	router := setupTestRouter()

	body := `{"text": ""}`
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/v1/nlu/process", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestProcess_InvalidJSON(t *testing.T) {
	router := setupTestRouter()

	body := `{invalid json}`
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/v1/nlu/process", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

func TestDialogEndpoints(t *testing.T) {
	router := setupTestRouter()

	// Get dialog state
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/v1/dialog/test-session-123", nil)
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	// Delete dialog
	w = httptest.NewRecorder()
	req, _ = http.NewRequest("DELETE", "/api/v1/dialog/test-session-123", nil)
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}
}

func TestProcess_MissingTextField(t *testing.T) {
	router := setupTestRouter()

	body := `{"language": "zh"}`
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/v1/nlu/process", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}
