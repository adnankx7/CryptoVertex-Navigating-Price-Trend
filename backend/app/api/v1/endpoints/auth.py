from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from backend.app.services.auth_service import auth_service
from backend.app.schemas.user import UserCreate, Token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Ideally we generate a JWT here
    # For now returning a mock token to keep it simple but working as per existing app logic
    return {"access_token": user['email'], "token_type": "bearer"}

@router.post("/signup", response_model=Token)
async def signup(user_in: UserCreate):
    if user_in.password != user_in.confirm_password:
         raise HTTPException(status_code=400, detail="Passwords do not match")

    user = auth_service.create_user(user_in)
    if not user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    return {"access_token": user['email'], "token_type": "bearer"}
