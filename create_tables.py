from db import engine, Base
from postgres_models import HistoricalFigure, GameSession, DBConversation

Base.metadata.create_all(bind=engine)
print("Tables created")
