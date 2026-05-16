// Package example demonstrates grouped const and var block declarations.
package example

// StatusCodes contains HTTP status codes used by the API.
const (
	StatusOK       = 200
	StatusNotFound = 404
	StatusError    = 500
)

// DefaultLimits holds default resource limits for the pipeline.
var (
	MaxConnections = 100
	MaxRetries     = 3
	Timeout        = 30
)
