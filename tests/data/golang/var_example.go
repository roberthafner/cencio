// Package users provides user management operations.
package users

// ErrNotFound is returned when a user cannot be located.
var ErrNotFound = errors.New("user not found")

// defaultPageSize controls the number of results per page.
var defaultPageSize = 20

var debugMode = false
