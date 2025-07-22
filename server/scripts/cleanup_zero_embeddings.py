import os
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Get DB URL from env or default
pg_url = os.getenv('POSTGRES_URL')
if not pg_url:
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'postgres')
    db = os.getenv('POSTGRES_DB', 'stepChange')
    pg_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

engine = create_engine(pg_url)
Session = sessionmaker(bind=engine)
session = Session()

# Count and delete all-zero embeddings
def is_all_zero(embedding_str):
    try:
        emb = json.loads(embedding_str)
        return isinstance(emb, list) and all(v == 0.0 for v in emb)
    except Exception:
        return False

print("Scanning for all-zero embeddings in vector_documents table...")

# Fetch all ids and embeddings
result = session.execute(text("SELECT id, embedding FROM vector_documents")).fetchall()
zero_ids = [row[0] for row in result if is_all_zero(row[1])]

print(f"Found {len(zero_ids)} all-zero embeddings.")

if zero_ids:
    # Delete in chunks for safety
    chunk_size = 100
    for i in range(0, len(zero_ids), chunk_size):
        chunk = zero_ids[i:i+chunk_size]
        session.execute(text(f"DELETE FROM vector_documents WHERE id = ANY(:ids)"), {"ids": chunk})
        print(f"Deleted {len(chunk)} rows...")
    session.commit()
    print("All-zero embeddings cleanup complete.")
else:
    print("No all-zero embeddings found.")

session.close()
