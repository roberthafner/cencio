// Package users provides user management operations.
package users

// Repository defines the persistence operations for user records.
// All implementations must be safe for concurrent use.
type Repository interface {
	FindByID(ctx context.Context, id string) (*User, error)
	Save(ctx context.Context, u *User) error
	Delete(ctx context.Context, id string) error
}

type Validator interface {
	Validate() error
}
