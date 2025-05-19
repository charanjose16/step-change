from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.config.dbConfig import SessionLocal
from app.models.user_model import User
from app.config.settings import settings
import bcrypt
import jwt
import datetime

router = APIRouter(prefix="/auth", tags=["Authentication"])

class UserAuth(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

HARDCODED_USERNAME = "admin"
HARDCODED_PASSWORD_HASH = bcrypt.hashpw(
    "secret".encode("utf-8"),
    bcrypt.gensalt()
).decode("utf-8")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(username: str) -> str:
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
    payload = {"sub": username, "exp": expire}
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token

@router.post("/signup", response_model=TokenResponse)
def signup(user_auth: UserAuth, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user_auth.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    hashed_password = bcrypt.hashpw(
        user_auth.password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")
    new_user = User(username=user_auth.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    access_token = create_access_token(new_user.username)
    return TokenResponse(access_token=access_token)

@router.post("/login", response_model=TokenResponse)
def login(user_auth: UserAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == user_auth.username).first()
    if not user or not bcrypt.checkpw(
        user_auth.password.encode("utf-8"),
        user.hashed_password.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    access_token = create_access_token(user.username)
    return TokenResponse(access_token=access_token)