#!/usr/bin/env python3
"""
Main evaluation orchestrator script.
Coordinates the entire evaluation pipeline from Wikipedia scraping to accuracy calculation.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import argparse

class EvaluationOrchestrator:
    def __init__(self, config_file: str = "evaluation_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.setup_directories()
        
    def load_config(self) -> Dict:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
            "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", "sk-your-deepseek-key-here"),
            "deepinfra_api_key": os.getenv("DEEPINFRA_API_KEY", "WJkNzU3cHwGGC6d5nGrGGCoFF9qIW8li"),
            "deepseek_model": "deepseek-chat",
            "deepinfra_model": "meta-llama/Llama-2-70b-chat-hf",
            "authors_limit": 10,  # Limit for testing
                "confidence_threshold": 0.5,
                "batch_size": 5
            }
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            "inputs/authors",
            "evaluation_outputs",
            "evaluation_outputs/srl",
            "evaluation_outputs/rel", 
            "evaluation_outputs/rdf",
            "evaluation_outputs/graphs",
            "evaluation_outputs/evaluations"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def step1_scrape_wikipedia(self) -> bool:
        """Step 1: Scrape Wikipedia for author biographies."""
        print("\n" + "="*60)
        print("STEP 1: WIKIPEDIA SCRAPING")
        print("="*60)
        
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_collection'))
            from wikipedia_scraper import WikipediaScraper
            scraper = WikipediaScraper("authors_data")
            
            # Get limited authors for testing
            authors = scraper.extract_author_names_from_imdb("")
            authors = authors[:self.config.get("authors_limit", 10)]
            
            print(f"Scraping {len(authors)} authors...")
            results = scraper.scrape_all_authors(authors)
            
            successful = sum(results.values())
            print(f"Successfully scraped {successful}/{len(authors)} authors")
            
            return successful > 0
            
        except Exception as e:
            print(f"Error in Wikipedia scraping: {e}")
            return False
    
    def step2_process_pipeline(self) -> bool:
        """Step 2: Process authors through SRL-to-RDF pipeline."""
        print("\n" + "="*60)
        print("STEP 2: SRL-TO-RDF PIPELINE PROCESSING")
        print("="*60)
        
        try:
            from batch_pipeline import BatchPipeline
            pipeline = BatchPipeline("authors_data", "evaluation_outputs")
            
            # Check if virtual environments exist
            venvs = [".srl_env", ".orchestrator_env", ".rel_env"]
            for venv in venvs:
                if not os.path.exists(venv):
                    print(f"Error: Virtual environment {venv} not found!")
                    print("Please run: python setup_simple.py first")
                    return False
            
            results = pipeline.process_all_authors()
            pipeline.generate_summary()
            
            successful = sum(1 for r in results.values() if r['success'])
            print(f"Successfully processed {successful}/{len(results)} authors")
            
            return successful > 0
            
        except Exception as e:
            print(f"Error in pipeline processing: {e}")
            return False
    
    def step3_evaluate_triples(self) -> bool:
        """Step 3: Evaluate triples using Deepseek v3."""
        print("\n" + "="*60)
        print("STEP 3: TRIPLE EXTRACTION EVALUATION")
        print("="*60)
        
        try:
            from triple_evaluator import TripleEvaluator, extract_triples_from_rdf
            
            if not self.config.get("deepseek_api_key"):
                print("Error: Deepseek API key not configured!")
                return False
            
            evaluator = TripleEvaluator(
                self.config["deepseek_api_key"], 
                self.config["deepseek_model"]
            )
            
            # Find all RDF files
            rdf_dir = Path("evaluation_outputs/rdf")
            rdf_files = list(rdf_dir.glob("*_rdf.ttl"))
            
            if not rdf_files:
                print("No RDF files found for evaluation!")
                return False
            
            print(f"Evaluating triples for {len(rdf_files)} authors...")
            
            for rdf_file in rdf_files:
                author_name = rdf_file.stem.replace("_rdf", "")
                print(f"Evaluating {author_name}...")
                
                # Find corresponding paragraph file
                paragraph_file = Path("authors_data") / f"{author_name}.txt"
                if not paragraph_file.exists():
                    print(f"  Warning: No paragraph file found for {author_name}")
                    continue
                
                # Load paragraph
                with open(paragraph_file, 'r', encoding='utf-8') as f:
                    paragraph = f.read()
                
                # Extract triples
                triples = extract_triples_from_rdf(str(rdf_file))
                if not triples:
                    print(f"  No triples found for {author_name}")
                    continue
                
                # Evaluate triples
                triples_with_paragraphs = [
                    {'triple': triple, 'paragraph': paragraph} 
                    for triple in triples
                ]
                
                results = evaluator.evaluate_triples_batch(triples_with_paragraphs)
                
                # Save evaluation results
                eval_output = Path("evaluation_outputs/evaluations") / f"{author_name}_evaluation.json"
                with open(eval_output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"  Saved evaluation results: {eval_output}")
            
            return True
            
        except Exception as e:
            print(f"Error in triple evaluation: {e}")
            return False
    
    def step4_calculate_accuracy(self) -> bool:
        """Step 4: Calculate accuracy metrics."""
        print("\n" + "="*60)
        print("STEP 4: ACCURACY CALCULATION")
        print("="*60)
        
        try:
            from accuracy_calculator import AccuracyCalculator
            
            calculator = AccuracyCalculator("evaluation_outputs/evaluations")
            author_metrics, overall_metrics = calculator.process_all_authors()
            
            # Save metrics
            calculator.save_metrics(author_metrics, overall_metrics, "evaluation_outputs/accuracy_metrics.json")
            
            # Generate report
            report = calculator.generate_detailed_report(author_metrics, overall_metrics)
            with open("evaluation_outputs/accuracy_report.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            
            print("Accuracy calculation completed!")
            print(f"Overall extraction rate: {overall_metrics.get('overall_extraction_rate', 0)*100:.1f}%")
            print(f"Average confidence: {overall_metrics.get('overall_confidence', 0):.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error in accuracy calculation: {e}")
            return False
    
    def run_full_evaluation(self) -> bool:
        """Run the complete evaluation pipeline."""
        print("🚀 STARTING RDF TRIPLE EXTRACTION EVALUATION")
        print("="*60)
        
        start_time = time.time()
        
        # Check API keys
        if not self.config.get("deepseek_api_key"):
            print("❌ Error: Deepseek API key not configured!")
            print("Set DEEPSEEK_API_KEY environment variable or update config file")
            return False
        
        if not self.config.get("deepinfra_api_key"):
            print("❌ Error: DeepInfra API key not configured!")
            print("Set DEEPINFRA_API_KEY environment variable or update config file")
            return False
        
        # Run steps
        steps = [
            ("Wikipedia Scraping", self.step1_scrape_wikipedia),
            ("Pipeline Processing", self.step2_process_pipeline),
            ("Triple Evaluation", self.step3_evaluate_triples),
            ("Accuracy Calculation", self.step4_calculate_accuracy)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 Running: {step_name}")
            if not step_func():
                print(f"❌ Failed: {step_name}")
                return False
            print(f"✅ Completed: {step_name}")
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total time: {duration/60:.1f} minutes")
        print(f"Results saved in: evaluation_outputs/")
        print(f"Accuracy report: evaluation_outputs/accuracy_report.txt")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="RDF Triple Extraction Evaluation Orchestrator")
    parser.add_argument("--config", default="evaluation_config.json", help="Configuration file")
    parser.add_argument("--step", choices=["1", "2", "3", "4", "all"], default="all", help="Run specific step or all")
    parser.add_argument("--deepseek-key", help="Deepseek API key")
    parser.add_argument("--deepinfra-key", help="DeepInfra API key")
    parser.add_argument("--authors-limit", type=int, help="Limit number of authors to process")
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_key
    if args.deepinfra_key:
        os.environ["DEEPINFRA_API_KEY"] = args.deepinfra_key
    
    orchestrator = EvaluationOrchestrator(args.config)
    
    # Update config if provided
    if args.authors_limit:
        orchestrator.config["authors_limit"] = args.authors_limit
        orchestrator.save_config(orchestrator.config)
    
    # Run specific step or all
    if args.step == "1":
        success = orchestrator.step1_scrape_wikipedia()
    elif args.step == "2":
        success = orchestrator.step2_process_pipeline()
    elif args.step == "3":
        success = orchestrator.step3_evaluate_triples()
    elif args.step == "4":
        success = orchestrator.step4_calculate_accuracy()
    else:  # all
        success = orchestrator.run_full_evaluation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
