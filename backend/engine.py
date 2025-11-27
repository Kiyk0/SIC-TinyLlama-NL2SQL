"""
Natural Language to SQL Generation Engine
Combines retrieval, prompt engineering, and LLM (Hugging Face) to generate SQL
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

from embedding_pipeline import EmbeddingPipeline, SchemaDocument

# Load environment variables
load_dotenv()


@dataclass
class RetrievalResult:
    """Results from schema retrieval"""
    tables: List[Dict[str, Any]]
    columns: Dict[str, List[Dict[str, Any]]]
    relationships: List[Dict[str, Any]]
    relevance_scores: Dict[str, float]


@dataclass
class SQLGenerationResult:
    """Result of SQL generation"""
    success: bool
    sql: Optional[str]
    explanation: Optional[str]
    retrieved_tables: List[str]
    error: Optional[str] = None
    confidence: float = 0.0


class SchemaRetriever:
    """Retrieves relevant schema elements for a question"""
    
    def __init__(self, pipeline: EmbeddingPipeline, schema_metadata: Dict):
        self.pipeline = pipeline
        self.vector_store = pipeline.vector_store
        self.embedder = pipeline.embedder
        self.schema_metadata = schema_metadata
        
        # Build table lookup
        self.tables_dict = {
            table['table_name']: table 
            for table in schema_metadata['tables']
        }
    
    def retrieve(
        self, 
        question: str, 
        top_k_tables: int = 3,
        top_k_columns: int = 10
    ) -> RetrievalResult:
        """
        Multi-stage retrieval of relevant schema elements
        
        Stage 1: Retrieve relevant tables
        Stage 2: Retrieve relevant columns for each table
        Stage 3: Extract relationships between tables
        """
        # Embed question
        question_embedding = self.embedder.embed_text(question)
        
        # Stage 1: Retrieve tables
        table_docs = self.vector_store.search(
            question_embedding,
            top_k=top_k_tables * 2,  # Get more, then filter
            filter_type="table"
        )
        
        # Extract unique tables with scores
        table_scores = {}
        seen_tables = set()
        
        for doc, score in table_docs:
            table_name = doc.table_name
            if table_name not in seen_tables:
                seen_tables.add(table_name)
                table_scores[table_name] = score
                
                if len(seen_tables) >= top_k_tables:
                    break
        
        # Get full table metadata
        tables = [
            self.tables_dict[name] 
            for name in list(table_scores.keys())[:top_k_tables]
        ]
        
        # Stage 2: Retrieve relevant columns for each table
        columns_by_table = {}
        for table in tables:
            table_name = table['table_name']
            
            # Search for relevant columns in this table
            column_docs = self.vector_store.search(
                question_embedding,
                top_k=top_k_columns,
                filter_type="column",
                filter_table=table_name
            )
            
            columns_by_table[table_name] = [
                doc.metadata for doc, _ in column_docs
            ]
        
        # Stage 3: Extract relationships
        relationships = self._extract_relationships(tables)
        
        return RetrievalResult(
            tables=tables,
            columns=columns_by_table,
            relationships=relationships,
            relevance_scores=table_scores
        )
    
    def _extract_relationships(self, tables: List[Dict]) -> List[Dict]:
        """Extract relationships between retrieved tables"""
        table_names = {t['table_name'] for t in tables}
        relationships = []
        
        for table in tables:
            for rel in table.get('relationships', []):
                # Include only if both tables are in retrieval set
                if rel['to_table'] in table_names:
                    relationships.append(rel)
        
        return relationships


class PromptBuilder:
    """Builds structured prompts for SQL generation"""
    
    def __init__(self, sql_dialect: str = "MySQL"):
        self.sql_dialect = sql_dialect
        self.few_shot_examples = self._load_few_shot_examples()
    
    def build_prompt(
        self, 
        question: str, 
        retrieval_result: RetrievalResult
    ) -> str:
        """Build complete prompt with schema context"""
        
        # Build schema section
        schema_section = self._build_schema_section(retrieval_result)
        
        # Build examples section
        examples_section = self._build_examples_section()
        
        # Build full prompt
        prompt = f"""You are an expert SQL generator for a {self.sql_dialect} database.

Your task is to convert the natural language question into a valid SQL query.

CRITICAL RULES:
1. ONLY use the tables and columns provided in the schema below
2. ONLY generate SELECT queries (no INSERT, UPDATE, DELETE, DROP)
3. Always add a LIMIT clause (max 1000 rows)
4. Use proper JOINs based on the relationships provided
5. Return ONLY the SQL query without explanation
6. If the question cannot be answered with the given schema, respond with: "INSUFFICIENT_SCHEMA"

{schema_section}

{examples_section}

USER QUESTION:
{question}

SQL QUERY:"""
        
        return prompt
    
    def _build_schema_section(self, retrieval: RetrievalResult) -> str:
        """Build the schema context section"""
        sections = ["DATABASE SCHEMA:", ""]
        
        # Add each table
        for table in retrieval.tables:
            table_name = table['table_name']
            sections.append(f"Table: {table_name}")
            sections.append(f"Description: {table['description']}")
            sections.append("Columns:")
            
            # Add columns
            for col in table['columns']:
                col_line = f"  - {col['name']} ({col['data_type']})"
                
                if col['is_primary_key']:
                    col_line += " [PRIMARY KEY]"
                if col['is_foreign_key']:
                    col_line += f" [FOREIGN KEY -> {col['foreign_key_ref']}]"
                if col.get('description'):
                    col_line += f": {col['description']}"
                
                sections.append(col_line)
            
            sections.append("")  # Blank line
        
        # Add relationships
        if retrieval.relationships:
            sections.append("RELATIONSHIPS:")
            for rel in retrieval.relationships:
                sections.append(f"  - {rel['join_condition']}")
            sections.append("")
        
        return "\n".join(sections)
    
    def _build_examples_section(self) -> str:
        """Build few-shot examples section"""
        sections = ["EXAMPLE QUERIES:", ""]
        
        for example in self.few_shot_examples[:3]:  # Use top 3 examples
            sections.append(f"Question: {example['question']}")
            sections.append(f"SQL: {example['sql']}")
            sections.append("")
        
        return "\n".join(sections)
    
    def _load_few_shot_examples(self) -> List[Dict[str, str]]:
        """Load few-shot examples for the database"""
        return [
            {
                "question": "Show me all active students",
                "sql": "SELECT student_id, first_name, last_name, email FROM students WHERE account_status = 'active' LIMIT 100;"
            },
            {
                "question": "Find the top 5 most popular courses",
                "sql": "SELECT course_id, title, enrollment_count, rating FROM courses ORDER BY enrollment_count DESC LIMIT 5;"
            },
            {
                "question": "Get students enrolled in Python courses",
                "sql": """SELECT DISTINCT s.student_id, s.first_name, s.last_name, c.title
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
JOIN courses c ON e.course_id = c.course_id
WHERE c.title LIKE '%Python%'
LIMIT 100;"""
            },
            {
                "question": "Show instructors with ratings above 4.5",
                "sql": "SELECT instructor_id, first_name, last_name, rating, specialization FROM instructors WHERE rating > 4.5 ORDER BY rating DESC LIMIT 50;"
            },
            {
                "question": "Find completed courses with certificates",
                "sql": """SELECT s.first_name, s.last_name, c.title, e.completion_date
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
JOIN courses c ON e.course_id = c.course_id
WHERE e.status = 'completed' AND e.certificate_issued = TRUE
LIMIT 100;"""
            },
            {
                "question": "Calculate average quiz scores per course",
                "sql": """SELECT c.title, AVG(qa.score) as avg_score, COUNT(qa.attempt_id) as total_attempts
FROM courses c
JOIN quizzes q ON c.course_id = q.course_id
JOIN quiz_attempts qa ON q.quiz_id = qa.quiz_id
GROUP BY c.course_id, c.title
ORDER BY avg_score DESC
LIMIT 50;"""
            }
        ]


class SQLGenerator:
    """Generates SQL using local LLM"""
    
    def __init__(self, api_key: str, model: str = "Kiyk0/TinyLlama-NL2SQL"):
        """
        Initialize SQL generator with local model
        
        Args:
            api_key: Hugging Face API key (unused for local, but kept for compatibility)
            model: Model to use
        """
        print(f"Loading model locally: {model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(
                model, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def generate_sql(self, prompt: str) -> Tuple[str, str]:
        """
        Generate SQL from prompt using local model
        
        Returns:
            (sql_query, explanation)
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=False
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Robust prompt removal
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            elif prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            # Extract SQL
            sql = self._extract_sql(generated_text)
            
            # Since we ask for NO explanation in the prompt, we shouldn't return the raw text as explanation
            # The raw text might contain hallucinations or garbage
            explanation = None
            
            return sql, explanation
            
        except Exception as e:
            import traceback
            print(f"❌ Error details: {str(e)}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            raise Exception(f"SQL generation failed: {str(e)}")
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response"""
        # Remove markdown code blocks
        if "```sql" in response:
            parts = response.split("```sql")
            if len(parts) > 1:
                sql = parts[1].split("```")[0].strip()
                return sql
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                sql = parts[1].strip()
                return sql
        
        # Return as-is if no code blocks, but truncate at first semicolon
        sql = response.strip()
        if ";" in sql:
            sql = sql.split(";")[0] + ";"
        
        return sql


class NL2SQLEngine:
    """Complete Natural Language to SQL Engine"""
    
    def __init__(
        self,
        pipeline: EmbeddingPipeline,
        schema_metadata: Dict,
        api_key: str,
        model: str = "Kiyk0/TinyLlama-NL2SQL"
    ):
        if model is None:
            model = "Kiyk0/TinyLlama-NL2SQL"
        self.retriever = SchemaRetriever(pipeline, schema_metadata)
        self.prompt_builder = PromptBuilder()
        self.sql_generator = SQLGenerator(api_key, model)
    
    def generate(
        self, 
        question: str,
        top_k_tables: int = 3
    ) -> SQLGenerationResult:
        """
        End-to-end SQL generation from natural language
        
        Args:
            question: Natural language question
            top_k_tables: Number of tables to retrieve
        
        Returns:
            SQLGenerationResult with generated SQL and metadata
        """
        try:
            # Step 1: Retrieve relevant schema
            print(f"🔍 Retrieving schema for: '{question}'")
            retrieval_result = self.retriever.retrieve(question, top_k_tables=top_k_tables)
            
            retrieved_tables = [t['table_name'] for t in retrieval_result.tables]
            print(f"✓ Retrieved tables: {', '.join(retrieved_tables)}")
            
            # Step 2: Build prompt
            prompt = self.prompt_builder.build_prompt(question, retrieval_result)
            
            # Step 3: Generate SQL
            print("🤖 Generating SQL...")
            sql, explanation = self.sql_generator.generate_sql(prompt)
            
            # Check for special responses
            if sql == "INSUFFICIENT_SCHEMA":
                return SQLGenerationResult(
                    success=False,
                    sql=None,
                    explanation="The question cannot be answered with available schema",
                    retrieved_tables=retrieved_tables,
                    error="Insufficient schema"
                )
            
            print(f"✓ Generated SQL:\n{sql}\n")
            
            # Calculate confidence based on relevance scores
            avg_score = sum(retrieval_result.relevance_scores.values()) / len(retrieval_result.relevance_scores)
            
            return SQLGenerationResult(
                success=True,
                sql=sql,
                explanation=explanation,
                retrieved_tables=retrieved_tables,
                confidence=avg_score
            )
            
        except Exception as e:
            return SQLGenerationResult(
                success=False,
                sql=None,
                explanation=None,
                retrieved_tables=[],
                error=str(e)
            )


# Example usage
if __name__ == "__main__":
    # Load schema metadata
    with open("elearning_schema_metadata.json", "r") as f:
        schema_metadata = json.load(f)
    
    # Initialize pipeline (load existing vector store)
    pipeline = EmbeddingPipeline()
    pipeline.vector_store = pipeline.vector_store.load("elearning_vector_store")
    
    # Get API key from environment variable (optional for local model)
    hf_token = os.getenv("HF_TOKEN")
    
    # Initialize engine
    engine = NL2SQLEngine(
        pipeline=pipeline,
        schema_metadata=schema_metadata,
        api_key=hf_token
    )
    
    # Test queries
    test_questions = [
        "Show me all students from USA who completed more than 2 courses",
        "What are the top 5 highest-rated courses?",
        "Find instructors who specialize in web development",
        "Get the average quiz score for each course",
        "Show students enrolled in beginner-level courses",
        "List courses with more than 200 enrollments"
    ]
    
    print("="*60)
    print("Testing NL-to-SQL Engine")
    print("="*60 + "\n")
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        result = engine.generate(question)
        
        if result.success:
            print(f"✓ Success!")
            print(f"SQL: {result.sql}")
            print(f"Tables used: {', '.join(result.retrieved_tables)}")
            print(f"Confidence: {result.confidence:.2f}")
        else:
            print(f"✗ Failed: {result.error}")
        
        print()