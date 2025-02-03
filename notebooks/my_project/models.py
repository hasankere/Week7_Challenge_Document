# Placeholder content for models.py
from sqlalchemy import Column, Integer, String
from database import Base

# Define a User model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
