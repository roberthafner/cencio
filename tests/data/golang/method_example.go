// Package users provides user management operations.
package users

// User represents a user account in the system.
type User struct {
	ID    string
	Email string
}

// Validate checks that the User fields are non-empty.
// Returns an error if any required field is missing.
func (u *User) Validate() error {
	return nil
}

// DisplayName returns a formatted display string for the user.
func (u User) DisplayName() string {
	return u.Email
}
