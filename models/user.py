from flask_login import UserMixin
from datetime import datetime

class User(UserMixin):
    def __init__(self, id_, name, email, profile_pic=None):
        self.id = id_  # This is the Google ID
        self.name = name
        self.email = email
        self.profile_pic = profile_pic or "https://via.placeholder.com/30/007bff/ffffff?text=U"
        self.created_at = datetime.utcnow()
        self.last_login = datetime.utcnow()

    def get_id(self):
        """Return the user ID as a string."""
        return str(self.id)

    def is_active(self):
        """Return True if the user is active."""
        return True

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return True

    def is_anonymous(self):
        """Return True if the user is anonymous."""
        return False

    def update_last_login(self):
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()

    def to_dict(self):
        """Convert user object to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'profile_pic': self.profile_pic,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat()
        }

    @staticmethod
    def from_dict(data):
        """Create a User object from a dictionary."""
        user = User(
            id_=data['id'],
            name=data['name'],
            email=data['email'],
            profile_pic=data.get('profile_pic')
        )
        if 'created_at' in data:
            user.created_at = datetime.fromisoformat(data['created_at'])
        if 'last_login' in data:
            user.last_login = datetime.fromisoformat(data['last_login'])
        return user 