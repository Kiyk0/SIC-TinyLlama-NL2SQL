import os
import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import json


from engine import NL2SQLEngine
from embedding_pipeline import EmbeddingPipeline, SchemaDocument
import sys

# Fix for pickle loading if vector store was saved from __main__
sys.modules['__main__'].SchemaDocument = SchemaDocument

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
try:
    # Load schema metadata
    with open("elearning_schema_metadata.json", "r") as f:
        schema_metadata = json.load(f)
    
    # Initialize pipeline (load existing vector store)
    pipeline = EmbeddingPipeline()
    pipeline.vector_store = pipeline.vector_store.load("elearning_vector_store")
    
    # Initialize engine
    engine = NL2SQLEngine(
        pipeline=pipeline,
        schema_metadata=schema_metadata,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    print("✓ NL2SQL Engine initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize engine: {e}")
    engine = None

class QueryRequest(BaseModel):
    question: str

def execute_sql(sql: str):
    """Execute SQL query against the database"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "elearning")
        )
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        results = cursor.fetchall()
        
        # Get column names
        columns = [i[0] for i in cursor.description] if cursor.description else []
        
        cursor.close()
        conn.close()
        
        return results, columns
    except Exception as e:
        raise Exception(f"Database error: {str(e)}")

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Generate SQL
        result = engine.generate(request.question)
        
        if not result.success:
            return {
                "success": False,
                "error": result.error or "Failed to generate SQL",
                "explanation": result.explanation
            }
        
        # Execute SQL
        try:
            data, columns = execute_sql(result.sql)
            
            return {
                "success": True,
                "generated_sql": result.sql,
                "data": data,
                "columns": columns,
                "row_count": len(data),
                "explanation": result.explanation,
                "confidence": float(result.confidence)
            }
        except Exception as db_err:
            return {
                "success": False,
                "error": f"SQL Execution Failed: {str(db_err)}",
                "generated_sql": result.sql,
                "explanation": result.explanation
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
