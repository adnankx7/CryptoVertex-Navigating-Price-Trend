from pydantic import BaseModel, EmailStr
from typing import Optional

# Token
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# User
class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str
    confirm_password: str

class UserInDB(UserBase):
    hashed_password: str

class User(UserBase):
    pass
