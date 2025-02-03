# Placeholder content for schemas.py
from pydantic import BaseModel

# Pydantic schema for creating a user
class UserCreate(BaseModel):
    name: str
    email: str

    class Config:
        orm_mode = True

# Pydantic schema for reading a user
class UserRead(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        orm_mode = True
