from typing import Optional
from ..db.file_handler import db
from ..core.security import get_password_hash, verify_password
from ..schemas.user import UserCreate

class AuthService:
    def authenticate_user(self, email: str, password: str):
        users_df = db.load_users()
        if users_df.empty:
            return None
        
        user = users_df[users_df['email'] == email]
        if user.empty:
            return None
        
        hashed_pw = user.iloc[0]['password']
        if verify_password(password, hashed_pw):
            return user.iloc[0].to_dict()
        return None

    def create_user(self, user_in: UserCreate):
        users_df = db.load_users()
        
        if not users_df.empty and user_in.email in users_df['email'].values:
            return None # Already exists
            
        hashed_password = get_password_hash(user_in.password)
        
        new_user = {
            'username': user_in.username,
            'email': user_in.email,
            'password': hashed_password
        }
        
        # Add to DF
        import pandas as pd
        new_df = pd.DataFrame([new_user])
        
        if users_df.empty:
            users_df = new_df
        else:
            users_df = pd.concat([users_df, new_df], ignore_index=True)
            
        db.save_users(users_df)
        return new_user

auth_service = AuthService()
