"""
Demo NLP to Knowledge Graph
Interactive demo connecting the Question-to-Cypher model with Neo4j
"""
from neo4j import GraphDatabase
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import warnings
warnings.filterwarnings('ignore')

class QuestionToCypherModel:
    """Trained T5 model for question-to-Cypher translation"""
    
    def __init__(self, model_path='models/question_to_cypher'):
        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded on {self.device.upper()}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("  Run: python scripts/train_model.py first")
            raise
    
    def translate(self, question, max_length=256, num_beams=4):
        """Translate natural language question to Cypher query"""
        input_text = f"translate to cypher: {question}"
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=128,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        cypher = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return cypher


class KnowledgeGraphDemo:
    """Interactive demo with Neo4j knowledge graph"""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_path='models/question_to_cypher'):
        print("\n" + "="*70)
        print("KNOWLEDGE GRAPH DEMO - NLP to Cypher")
        print("="*70 + "\n")
        
        # Load NLP model
        self.nlp_model = QuestionToCypherModel(model_path)
        
        # Connect to Neo4j
        print(f"\nConnecting to Neo4j at {neo4j_uri}...")
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Connected to Neo4j")
        except Exception as e:
            print(f"✗ Error connecting to Neo4j: {e}")
            raise
        
        print("\n" + "="*70)
        print("Ready! Ask questions about the movie database.")
        print("="*70 + "\n")
    
    def process_question(self, question, verbose=True):
        """
        Process a natural language question end-to-end
        
        Steps:
        1. Translate question to Cypher
        2. Execute query against Neo4j
        3. Format and return results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Question: {question}")
            print('='*70)
        
        # Step 1: Translate to Cypher
        if verbose:
            print("\n[1] Translating to Cypher...")
        
        cypher = self.nlp_model.translate(question)
        
        if verbose:
            print(f"    Generated: {cypher}")
        
        # Step 2: Execute query
        if verbose:
            print("\n[2] Executing query...")
        
        try:
            results = self.execute_query(cypher)
            if verbose:
                print(f"    ✓ Retrieved {len(results)} results")
        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            return {'error': str(e), 'cypher': cypher}
        
        # Step 3: Format results
        if verbose:
            print("\n[3] Formatting results...")
        
        formatted = self.format_results(results, question)
        
        if verbose:
            print(f"\n{'='*70}")
            print("RESULTS:")
            print('='*70)
            print(formatted)
            print('='*70 + "\n")
        
        return {
            'question': question,
            'cypher': cypher,
            'results': results,
            'formatted': formatted
        }
    
    def execute_query(self, cypher):
        """Execute Cypher query against Neo4j"""
        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]
    
    def format_results(self, results, question):
        """Format query results for display"""
        if not results:
            return "No results found."
        
        # Detect result type
        if len(results) == 1 and len(results[0]) == 1:
            # Single value (count, name, etc.)
            key = list(results[0].keys())[0]
            value = results[0][key]
            return f"{value}"
        
        # Multiple results - format as table
        output = []
        keys = list(results[0].keys())
        
        # Header
        header = " | ".join(keys)
        output.append(header)
        output.append("-" * len(header))
        
        # Rows
        for row in results[:20]:  # Limit to 20 results
            values = [str(row.get(k, '')) for k in keys]
            output.append(" | ".join(values))
        
        if len(results) > 20:
            output.append(f"... and {len(results) - 20} more results")
        
        return "\n".join(output)
    
    def run_interactive(self):
        """Run interactive demo"""
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("\nAsk questions in natural language!")
        print("Examples:")
        print("  - Who acted in The Matrix?")
        print("  - What movies did Tom Hanks star in?")
        print("  - Who directed Inception?")
        print("  - Movies from 1999")
        print("\nType 'quit' to exit\n")
        
        while True:
            try:
                question = input("❓ Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋\n")
                    break
                
                if not question:
                    continue
                
                self.process_question(question, verbose=True)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋\n")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
    
    def run_batch(self, questions):
        """Run batch of questions"""
        print("\n" + "="*70)
        print(f"BATCH MODE - Processing {len(questions)} questions")
        print("="*70 + "\n")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question}")
            result = self.process_question(question, verbose=False)
            print(f"Cypher: {result.get('cypher', 'N/A')}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Results: {result.get('formatted', 'N/A')}")
            print("-" * 70)
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Knowledge Graph Demo - NLP to Cypher')
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', default='password123', help='Neo4j password')
    parser.add_argument('--model', default='models/question_to_cypher', help='Model path')
    parser.add_argument('--question', help='Single question to ask')
    parser.add_argument('--batch', help='File with questions (one per line)')
    
    args = parser.parse_args()
    
    try:
        # Initialize demo
        demo = KnowledgeGraphDemo(
            neo4j_uri=args.uri,
            neo4j_user=args.user,
            neo4j_password=args.password,
            model_path=args.model
        )
        
        # Run mode
        if args.question:
            # Single question mode
            demo.process_question(args.question, verbose=True)
        elif args.batch:
            # Batch mode
            with open(args.batch, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
            demo.run_batch(questions)
        else:
            # Interactive mode
            demo.run_interactive()
        
        # Cleanup
        demo.close()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
