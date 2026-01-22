from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Union
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import os
from .config import settings

# Password Context
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Fernet Encryption for User Data
def initialize_encryption_key() -> Fernet:
    KEY_FILE = settings.SECURE_DATA_DIR / "secret.key"
    
    if KEY_FILE.exists() and KEY_FILE.stat().st_size > 0:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
            try:
                return Fernet(key)
            except ValueError:
                pass
    
    # Generate new key
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    return Fernet(key)

_cipher_suite = initialize_encryption_key()

def encrypt_data(data: str) -> bytes:
    return _cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data: bytes) -> str:
    return _cipher_suite.decrypt(encrypted_data).decode()

# Password Hashing
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
