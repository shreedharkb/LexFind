#!/usr/bin/env python3
"""
LexFind — CLI Password Reset Tool
==================================
Easily reset a user's password directly from the server terminal.

Usage:
    python scripts/reset_password.py <email> <new_password>

Example:
    python scripts/reset_password.py admin@lexfind.ai Secret123!
"""

import sys
import os

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import DatabaseSession
from app.db.models import User
from app.services.auth_service import auth_service

def reset_password(email: str, new_password: str):
    email = email.strip()
    if not email or not new_password:
        print("❌ Error: Both email and new_password are required.")
        print("Usage: python scripts/reset_password.py <email> <new_password>")
        sys.exit(1)

    with DatabaseSession() as db:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"❌ User with email '{email}' not found.")
            # List existing users to help out
            users = db.query(User.email).limit(10).all()
            if users:
                print("\nRegistered users (up to 10 shown):")
                for (u_email,) in users:
                    print(f"  - {u_email}")
            sys.exit(1)

        user.hashed_password = auth_service.hash_password(new_password)
        db.commit()
        print(f"✅ Successfully updated password for user: {email}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/reset_password.py <email> <new_password>")
        sys.exit(1)
    
    reset_password(sys.argv[1], sys.argv[2])
