import pandas as pd
from pathlib import Path
from io import StringIO
from ..core.config import settings
from ..core.security import encrypt_data, decrypt_data
from cryptography.fernet import InvalidToken

class FileDB:
    def __init__(self):
        self.user_file = settings.SECURE_DATA_DIR / "users.enc"
        if not self.user_file.exists():
            self._init_empty_db()

    def _init_empty_db(self):
        df = pd.DataFrame(columns=['username', 'email', 'password'])
        self.save_users(df)

    def load_users(self) -> pd.DataFrame:
        if not self.user_file.exists() or self.user_file.stat().st_size == 0:
            return pd.DataFrame(columns=['username', 'email', 'password'])
        
        try:
            with open(self.user_file, "rb") as f:
                encrypted_data = f.read()
            
            if len(encrypted_data) == 0:
                return pd.DataFrame(columns=['username', 'email', 'password'])
            
            decrypted_data = decrypt_data(encrypted_data)
            
            if not decrypted_data.strip():
                return pd.DataFrame(columns=['username', 'email', 'password'])
                
            return pd.read_csv(StringIO(decrypted_data))
        except (InvalidToken, Exception):
            return pd.DataFrame(columns=['username', 'email', 'password'])

    def save_users(self, df: pd.DataFrame):
        if not df.empty:
            csv_data = df.to_csv(index=False)
            encrypted_data = encrypt_data(csv_data)
            with open(self.user_file, "wb") as f:
                f.write(encrypted_data)
        else:
            # Handle empty DF if needed, or just write headers encrypted
            pass

db = FileDB()
