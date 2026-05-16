// Package users provides user management operations.
package users

// User represents a user account in the system.
// It contains core identity and profile information.
type User struct {
	ID    string
	Email string
	Age   int
}

type Profile struct {
	Bio     string
	Website string
}
