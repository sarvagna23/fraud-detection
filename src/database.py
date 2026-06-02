import os
from sqlalchemy import create_engine, Column, Float, Boolean, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_detection.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class TransactionLog(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    fraud_probability = Column(Float)
    is_fraud = Column(Boolean)
    risk_level = Column(String)

def init_db():
    Base.metadata.create_all(bind=engine)

def log_transaction(transaction_id: str, prob: float, is_fraud: bool, risk: str):
    session = SessionLocal()
    try:
        record = TransactionLog(
            id=transaction_id,
            fraud_probability=prob,
            is_fraud=is_fraud,
            risk_level=risk
        )
        session.add(record)
        session.commit()
    finally:
        session.close()