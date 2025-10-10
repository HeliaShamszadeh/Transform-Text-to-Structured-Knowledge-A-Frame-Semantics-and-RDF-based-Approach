#!/usr/bin/env python3
"""
Batch processing pipeline to run each author's text through the SRL-to-RDF pipeline.
Processes all author files and generates RDF triples and visualizations.
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import re

class BatchPipeline:
    def __init__(self, authors_dir: str = "data_collection/authors_data/authors_reduced_more", output_dir: str = "evaluation_outputs"):
        self.authors_dir = Path(authors_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create main subdirectories only for backward compatibility (if needed)
        # Note: Each author now has their own folder structure
        pass
        
        self.results = {}
    
    def _create_author_folders(self, author_name: str) -> Dict[str, Path]:
        """Create author-specific folder structure."""
        author_dir = self.output_dir / author_name
        author_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for this author
        folders = {
            'srl': author_dir / "srl",
            'rel': author_dir / "rel", 
            'rdf': author_dir / "rdf",
            'graph': author_dir / "graph",
            'evaluations': author_dir / "evaluations"
        }
        
        for folder in folders.values():
            folder.mkdir(exist_ok=True)
        
        return folders
        
    def run_in_venv(self, venv_name: str, script_path: str, args: List[str]) -> Tuple[bool, str]:
        """Run a script in a virtual environment."""
        if platform.system() == "Windows":
            python_cmd = f"{venv_name}\\Scripts\\python"
        else:
            python_cmd = f"{venv_name}/bin/python"
        
        cmd = [python_cmd, script_path] + args
        
        # Set environment variables for API keys
        env = os.environ.copy()
        env['DEEPINFRA_API_KEY'] = env.get('DEEPINFRA_API_KEY', 'WJkNzU3cHwGGC6d5nGrGGCoFF9qIW8li')
        env['DEEPSEEK_API_KEY'] = env.get('DEEPSEEK_API_KEY', 'sk-3dbf2f6094d744eea9f335e5e504b87d')
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600, env=env)  # 1 hour
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except subprocess.TimeoutExpired:
            return False, "Process timed out after 30 minutes"
    
    def process_author(self, author_file: Path) -> Dict:
        """Process a single author file through the entire pipeline."""
        author_name = author_file.stem
        print(f"\n{'='*60}")
        print(f"Processing: {author_name}")
        print(f"{'='*60}")
        
        # Create author-specific folders
        author_folders = self._create_author_folders(author_name)
        
        result = {
            'author': author_name,
            'file': str(author_file),
            'success': False,
            'stages': {},
            'error': None,
            'triples_count': 0,
            'entities_count': 0,
            'frames_count': 0,
            'author_folders': {k: str(v) for k, v in author_folders.items()}
        }
        
        try:
            # Stage 1: SRL (Semantic Role Labeling)
            print(f"🔍 Stage 1/4: Semantic Role Labeling...")
            srl_output = author_folders['srl'] / f"{author_name}_frames.json"
            
            success, output = self.run_in_venv(
                ".srl_env", 
                "modules/framesrl/framesrl_runner.py",
                ["--infile", str(author_file), "--outfile", str(srl_output), "--model", "base"]
            )
            
            result['stages']['srl'] = {'success': success, 'output': output}
            if not success:
                result['error'] = f"SRL failed: {output}"
                return result
            
            # Count frames
            with open(srl_output, 'r', encoding='utf-8') as f:
                srl_data = json.load(f)
                result['frames_count'] = sum(len(sent.get('frames', [])) for sent in srl_data.get('sentences', []))
            
            # Stage 2: Entity Linking
            print(f"🔗 Stage 2/4: Entity Linking...")
            rel_output = author_folders['rel'] / f"{author_name}_entities.json"
            
            success, output = self.run_in_venv(
                ".rel_env",
                "modules/rel_linker/rel_runner_fixed.py",
                ["--infile", str(author_file), "--outfile", str(rel_output), "--confidence", "0.35"]
            )
            
            result['stages']['rel'] = {'success': success, 'output': output}
            if not success:
                result['error'] = f"Entity linking failed: {output}"
                return result
            
            # Count entities
            with open(rel_output, 'r', encoding='utf-8') as f:
                rel_data = json.load(f)
                result['entities_count'] = len(rel_data.get('entities', []))
            
            # Stage 3: RDF Conversion (using DeepInfra for coreference)
            print(f"🔄 Stage 3/4: RDF Conversion...")
            rdf_output = author_folders['rdf'] / f"{author_name}_rdf.ttl"
            deepinfra_key = os.getenv("DEEPINFRA_API_KEY", "WJkNzU3cHwGGC6d5nGrGGCoFF9qIW8li")
            
            print(f"Using DeepInfra API key: {deepinfra_key[:10]}...")

            rdf_args = [
                "--frames", str(srl_output),
                "--entities", str(rel_output),
                "--outfile", str(rdf_output),
                "--deepinfra-model", "meta-llama/Llama-2-70b-chat-hf"
            ]
            rdf_args.extend(["--deepinfra-api-key", deepinfra_key])
            
            success, output = self.run_in_venv(
                ".orchestrator_env",
                "modules/orchestrator/rdfify_improved.py",
                rdf_args
            )
            
            result['stages']['rdf'] = {'success': success, 'output': output}
            if not success:
                result['error'] = f"RDF conversion failed: {output}"
                return result
            
            # Count triples
            with open(rdf_output, 'r', encoding='utf-8') as f:
                rdf_content = f.read()
                result['triples_count'] = rdf_content.count('.')  # Rough count
            
            # Stage 4: Graph Visualization and Queryable Formats
            print(f"📊 Stage 4/5: Graph Visualization and Queryable Formats...")
            graph_output = self.output_dir / "graphs" / f"{author_name}_graph.png"
            
            success, output = self.run_in_venv(
                ".orchestrator_env",
                "visualize_rdf_simple.py",
                ["--rdf", str(rdf_output), "--output", str(graph_output)]
            )
            
            # Stage 5: LLM-based Evaluation
            print(f"🤖 Stage 5/6: LLM-based Evaluation...")
            eval_success = self._run_evaluation(srl_output, rel_output, rdf_output, author_name, str(author_file), author_folders)
            result['stages']['evaluation'] = {'success': eval_success}
            
            # Load evaluation results if available
            eval_file = author_folders['evaluations'] / f"{author_name}_llm_evaluation.json"
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    result['evaluation_results'] = json.load(f)
            
            # Stage 6: Export RAG-friendly formats (after evaluation for confidence data)
            print(f"📤 Stage 6/6: Exporting RAG-friendly formats...")
            export_files = self.export_triples_for_rag(rdf_output, author_name, result.get('evaluation_results', {}), author_folders)
            result['export_files'] = export_files
            
            result['stages']['visualization'] = {'success': success, 'output': output}
            
            result['success'] = True
            print(f"✅ Successfully processed {author_name}")
            
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            print(f"❌ Error processing {author_name}: {e}")
        
        return result
    
    def _run_evaluation(self, srl_file: Path, rel_file: Path, rdf_file: Path, author_name: str, input_file: str, author_folders: Dict = None) -> bool:
        """Run LLM-based evaluation for a single author."""
        try:
            # Get DeepSeek API key with default
            deepseek_key = os.getenv('DEEPSEEK_API_KEY', 'sk-3dbf2f6094d744eea9f335e5e504b87d')
            
            print("🤖 Running LLM-based evaluation with DeepSeek...")
            print(f"Using DeepSeek API key: {deepseek_key[:10]}...")
            
            # Run triple evaluation using DeepSeek
            if author_folders:
                eval_output = author_folders['evaluations'] / f"{author_name}_llm_evaluation.json"
            else:
                eval_output = self.output_dir / "evaluations" / f"{author_name}_llm_evaluation.json"
            
            success, output = self.run_in_venv(
                ".orchestrator_env",
                "evaluation_pipeline/triple_evaluator_fast.py",
                ["--rdf-file", str(rdf_file), "--paragraph-file", input_file, "--output", str(eval_output), "--max-triples", "50", "--batch-size", "5"]
            )
            
            if success:
                print("✅ LLM evaluation completed successfully!")
                return True
            else:
                print(f"⚠️  LLM evaluation failed: {output}")
                return False
                
        except Exception as e:
            print(f"⚠️  LLM evaluation error: {e}")
            return False
    
    def _save_queryable_formats(self, rdf_file: Path, author_name: str):
        """Save RDF in multiple queryable formats."""
        try:
            from rdflib import Graph
            
            # Load the RDF file
            g = Graph()
            g.parse(str(rdf_file), format='turtle')
            
            # Create queryable formats directory
            queryable_dir = self.output_dir / "queryable"
            queryable_dir.mkdir(exist_ok=True)
            
            # Save in multiple formats
            formats = {
                'rdf': 'xml',
                'jsonld': 'json-ld', 
                'nt': 'nt',
                'ttl': 'turtle'
            }
            
            for ext, fmt in formats.items():
                output_file = queryable_dir / f"{author_name}_graph.{ext}"
                g.serialize(destination=str(output_file), format=fmt)
                print(f"  Saved {ext.upper()}: {output_file.name}")
                
        except Exception as e:
            print(f"  Warning: Could not save queryable formats: {e}")
    
    def process_all_authors(self, max_workers: int = None, limit: int = None) -> Dict:
        """Process all author files through the pipeline with multiprocessing."""
        author_files = list(self.authors_dir.glob("*.txt"))
        
        if not author_files:
            print(f"No author files found in {self.authors_dir}")
            return {}
        
        # Apply limit if specified
        if limit and limit < len(author_files):
            author_files = author_files[:limit]
            print(f"Limited to first {limit} authors")
        
        print(f"Found {len(author_files)} author files to process")
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(author_files), 4)  # Limit to 4 to avoid overwhelming the system
        
        print(f"Using {max_workers} parallel workers")
        
        # Process authors in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_author = {
                executor.submit(self.process_author, author_file): author_file 
                for author_file in author_files
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_author):
                author_file = future_to_author[future]
                completed += 1
                
                try:
                    result = future.result()
                    self.results[author_file.stem] = result
                    print(f"[{completed}/{len(author_files)}] ✅ Completed: {author_file.name}")
                    
                    # Save intermediate results every 5 completions
                    if completed % 5 == 0:
                        self.save_results()
                        
                except Exception as e:
                    print(f"[{completed}/{len(author_files)}] ❌ Failed: {author_file.name} - {e}")
                    self.results[author_file.stem] = {
                        'author': author_file.stem,
                        'file': str(author_file),
                        'success': False,
                        'error': str(e)
                    }
        
        return self.results

    def process_all_authors_sequential(self, limit: int = None) -> Dict:
        """Process all author files through the pipeline sequentially (original method)."""
        author_files = list(self.authors_dir.glob("*.txt"))
        
        if not author_files:
            print(f"No author files found in {self.authors_dir}")
            return {}
        
        # Apply limit if specified
        if limit and limit < len(author_files):
            author_files = author_files[:limit]
            print(f"Limited to first {limit} authors")
        
        print(f"Found {len(author_files)} author files to process")
        
        for i, author_file in enumerate(author_files, 1):
            print(f"\n[{i}/{len(author_files)}] Processing: {author_file.name}")
            
            result = self.process_author(author_file)
            self.results[author_file.stem] = result
            
            # Save intermediate results
            self.save_results()
            
            # Brief pause between authors
            time.sleep(2)
        
        return self.results
    
    def save_results(self):
        """Save processing results to JSON file."""
        results_file = self.output_dir / "processing_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def generate_summary(self):
        """Generate a summary of the processing results."""
        if not self.results:
            print("No results to summarize")
            return
        
        successful = sum(1 for r in self.results.values() if r['success'])
        total = len(self.results)
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total authors processed: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total*100:.1f}%")
        
        # Stage-wise success rates
        stages = ['srl', 'rel', 'rdf', 'visualization']
        for stage in stages:
            stage_success = sum(1 for r in self.results.values() 
                              if r['stages'].get(stage, {}).get('success', False))
            print(f"{stage.upper()} success rate: {stage_success}/{total} ({stage_success/total*100:.1f}%)")
        
        # Statistics
        total_triples = sum(r['triples_count'] for r in self.results.values())
        total_entities = sum(r['entities_count'] for r in self.results.values())
        total_frames = sum(r['frames_count'] for r in self.results.values())
        
        print(f"\nTotal triples generated: {total_triples}")
        print(f"Total entities found: {total_entities}")
        print(f"Total frames processed: {total_frames}")
        
        # Failed authors
        failed_authors = [name for name, result in self.results.items() if not result['success']]
        if failed_authors:
            print(f"\nFailed authors: {', '.join(failed_authors)}")

    def export_triples_for_rag(self, rdf_file: Path, file_name: str, evaluation_results: Dict = None, author_folders: Dict = None) -> Dict[str, str]:
        """Export triples in multiple RAG-friendly formats."""
        try:
            print("📤 Exporting triples for RAG applications...")
            
            # Use author-specific folders if provided, otherwise use main folders
            if author_folders:
                rdf_dir = author_folders['rdf']
                graph_dir = author_folders['graph']
            else:
                rdf_dir = self.output_dir / "rdf"
                graph_dir = self.output_dir / "graph"
                graph_dir.mkdir(exist_ok=True)
            
            # Read the custom RDF format
            with open(rdf_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse triples from custom format with evaluation data
            triples = self._parse_custom_rdf_triples(content, evaluation_results)
            
            if not triples:
                print("⚠️  No triples found for export")
                return {}
            
            export_files = {}
            
            # 1. JSON Export (Best for RAG)
            json_file = rdf_dir / f"{file_name}_triples.json"
            self._export_json_triples(triples, json_file, file_name)
            export_files['json'] = str(json_file)
            
            # 2. CSV Export (Easy Analysis)
            csv_file = rdf_dir / f"{file_name}_triples.csv"
            self._export_csv_triples(triples, csv_file)
            export_files['csv'] = str(csv_file)
            
            # 3. SPARQL Queryable Format
            sparql_file = rdf_dir / f"{file_name}_queryable.ttl"
            self._export_sparql_triples(triples, sparql_file)
            export_files['sparql'] = str(sparql_file)
            
            # 4. SPARQL Query Templates
            query_file = rdf_dir / f"{file_name}_queries.sparql"
            self._export_sparql_queries(triples, query_file, file_name)
            export_files['queries'] = str(query_file)
            
            # 5. Graph CSV Export (Source,Target,Label format)
            graph_csv_file = graph_dir / f"{file_name}_graph_edges.csv"
            self._export_graph_csv(triples, graph_csv_file)
            export_files['graph_csv'] = str(graph_csv_file)
            
            # 6. Graph Visualization (in graph folder)
            dot_file, png_file = self._export_graph_visualization(triples, file_name, graph_dir)
            if dot_file:
                export_files['dot'] = dot_file
            if png_file:
                export_files['png'] = png_file
            
            print(f"✅ Exported {len(triples)} triples in {len(export_files)} formats")
            for format_type, file_path in export_files.items():
                print(f"   📄 {format_type.upper()}: {file_path}")
            
            return export_files
            
        except Exception as e:
            print(f"⚠️  Export error: {e}")
            return {}
    
    def _parse_custom_rdf_triples(self, content: str, evaluation_results: Dict = None) -> List[Dict[str, str]]:
        """Parse triples from custom RDF format with metadata."""
        triples = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('@') and not line.startswith('#'):
                # Remove trailing semicolon or period
                line = re.sub(r'[;.]$', '', line)
                # Split into parts
                parts = line.split()
                if len(parts) >= 3:
                    subject = parts[0]
                    predicate = parts[1]
                    object_val = ' '.join(parts[2:])  # Join remaining parts as object
                    
                    # Clean up the values
                    subject_clean = subject.strip('"\'')
                    predicate_clean = predicate.strip('"\'')
                    object_clean = object_val.strip('"\'')
                    
                    # Get evaluation data if available
                    confidence = 0.9  # Default confidence
                    source_sentence = 'Unknown'
                    
                    if evaluation_results and 'triples' in evaluation_results:
                        # Try to find matching triple in evaluation results
                        for eval_triple in evaluation_results['triples']:
                            if (eval_triple.get('triple', '').strip() == line.strip() or
                                self._triples_match(eval_triple.get('triple', ''), line)):
                                confidence = eval_triple.get('confidence', 0.9)
                                source_sentence = eval_triple.get('source_sentence', 'Unknown')
                                break
                    
                    triples.append({
                        'subject': subject_clean,
                        'predicate': predicate_clean,
                        'object': object_clean,
                        'confidence': confidence,
                        'source_sentence': source_sentence,
                        'extractable': evaluation_results.get('extractable', True) if evaluation_results else True
                    })
        
        return triples
    
    def _triples_match(self, eval_triple: str, rdf_triple: str) -> bool:
        """Check if two triples match (for evaluation data lookup)."""
        if not eval_triple or not rdf_triple:
            return False
        
        # Simple matching - could be enhanced
        eval_clean = eval_triple.strip().lower()
        rdf_clean = rdf_triple.strip().lower()
        
        return eval_clean == rdf_clean or eval_clean in rdf_clean or rdf_clean in eval_clean
    
    def _export_json_triples(self, triples: List[Dict], output_file: Path, file_name: str) -> None:
        """Export triples in RAG-optimized JSON format with query-friendly structure."""
        # Categorize triples by entity types for better RAG retrieval
        categorized_triples = self._categorize_triples_for_rag(triples)
        
        data = {
            'metadata': {
                'total_triples': len(triples),
                'source_file': file_name,
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'format': 'RAG-optimized JSON',
                'description': 'Triples exported for Retrieval-Augmented Generation applications',
                'entity_types': list(categorized_triples.keys()),
                'query_examples': [
                    "Find all people mentioned",
                    "What locations are referenced?",
                    "What events happened?",
                    "What concepts are discussed?",
                    "Find relationships between entities"
                ]
            },
            'triples': {
                'all': triples,
                'by_category': categorized_triples
            },
            'entities': self._extract_entity_index(triples),
            'relationships': self._extract_relationship_index(triples)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _categorize_triples_for_rag(self, triples: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize triples by entity type for better RAG retrieval."""
        categories = {
            'people_related': [],
            'location_related': [],
            'event_related': [],
            'concept_related': [],
            'other': []
        }
        
        for triple in triples:
            subject = triple['subject'].lower()
            predicate = triple['predicate'].lower()
            object_val = triple['object'].lower()
            
            # Categorize based on content
            if any(indicator in subject or indicator in object_val for indicator in ['christie', 'agatha', 'person', 'author']):
                categories['people_related'].append(triple)
            elif any(indicator in subject or indicator in object_val for indicator in ['torquay', 'england', 'place', 'location']):
                categories['location_related'].append(triple)
            elif any(indicator in predicate or indicator in object_val for indicator in ['born', 'death', 'event', 'happened']):
                categories['event_related'].append(triple)
            elif any(indicator in subject or indicator in object_val for indicator in ['mystery', 'novel', 'book', 'writing']):
                categories['concept_related'].append(triple)
            else:
                categories['other'].append(triple)
        
        return categories
    
    def _extract_entity_index(self, triples: List[Dict]) -> Dict[str, List[str]]:
        """Extract entity index for quick lookup."""
        entities = {
            'subjects': [],
            'objects': [],
            'all_unique': []
        }
        
        all_entities = set()
        for triple in triples:
            subj = triple['subject']
            obj = triple['object']
            
            entities['subjects'].append(subj)
            entities['objects'].append(obj)
            all_entities.add(subj)
            all_entities.add(obj)
        
        entities['all_unique'] = list(all_entities)
        return entities
    
    def _extract_relationship_index(self, triples: List[Dict]) -> Dict[str, List[str]]:
        """Extract relationship index for quick lookup."""
        relationships = {
            'predicates': [],
            'unique_predicates': [],
            'predicate_counts': {}
        }
        
        for triple in triples:
            pred = triple['predicate']
            relationships['predicates'].append(pred)
            
            if pred not in relationships['predicate_counts']:
                relationships['predicate_counts'][pred] = 0
            relationships['predicate_counts'][pred] += 1
        
        relationships['unique_predicates'] = list(set(relationships['predicates']))
        return relationships
    
    def _export_graph_csv(self, triples: List[Dict], output_file: Path) -> None:
        """Export graph edges in CSV format: Source,Target,Label,Frame"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Source', 'Target', 'Label', 'Frame'])
            
            # Write edges
            for triple in triples:
                source = self._clean_node_name(triple['subject'])
                target = self._clean_node_name(triple['object'])
                label = self._clean_edge_label(triple['predicate'])
                frame = self._extract_frame_from_predicate(triple['predicate'])
                
                writer.writerow([source, target, label, frame])
    
    def _extract_frame_from_predicate(self, predicate: str) -> str:
        """Extract frame name from predicate (e.g., 'Being_born:has_location' -> 'Being_born')."""
        if ':' in predicate:
            return predicate.split(':')[0]
        return predicate
    
    def _get_frame_color(self, frame_index: int) -> str:
        """Get a distinct color for each frame."""
        colors = [
            "lightblue", "lightcoral", "lightgreen", "lightyellow", "lightpink",
            "lightcyan", "lightsteelblue", "lightgray", "lightgoldenrodyellow", "lightseagreen",
            "lightsalmon", "lightgoldenrod", "lightpink", "lightsteelblue", "lightcoral"
        ]
        return colors[frame_index % len(colors)]
    
    def _get_edge_style_with_frame(self, predicate: str, frame: str, frame_color: str) -> str:
        """Get edge style with frame-based color coding."""
        base_style = self._get_edge_style(predicate)
        
        # Add frame-based color
        if "color=" in base_style:
            # Replace existing color
            base_style = base_style.replace("color=gray", f"color={frame_color}")
        else:
            # Add color
            base_style += f", color={frame_color}"
        
        return base_style
    
    def _export_csv_triples(self, triples: List[Dict], output_file: Path) -> None:
        """Export triples in CSV format for easy analysis."""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['subject', 'predicate', 'object', 'confidence', 'source_sentence', 'extractable'])
            # Write data
            for triple in triples:
                writer.writerow([
                    triple['subject'],
                    triple['predicate'], 
                    triple['object'],
                    triple['confidence'],
                    triple['source_sentence'],
                    triple['extractable']
                ])
    
    def _export_sparql_triples(self, triples: List[Dict], output_file: Path) -> None:
        """Export triples in SPARQL queryable Turtle format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# SPARQL Queryable RDF Triples\n")
            f.write("# Generated for RAG applications\n\n")
            
            # Add namespace declarations
            f.write("@prefix : <http://example.org/> .\n")
            f.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
            f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n")
            
            # Write triples
            for triple in triples:
                subject = triple['subject']
                predicate = triple['predicate']
                object_val = triple['object']
                
                # Convert to proper Turtle format
                if not subject.startswith('<'):
                    subject = f"<{subject}>"
                if not predicate.startswith('<'):
                    predicate = f"<{predicate}>"
                if not object_val.startswith('<') and not object_val.startswith('"'):
                    object_val = f'"{object_val}"'
                
                f.write(f"{subject} {predicate} {object_val} .\n")
    
    def _export_sparql_queries(self, triples: List[Dict], output_file: Path, file_name: str) -> None:
        """Export SPARQL query templates for common RAG queries."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# SPARQL Query Templates for {file_name}\n")
            f.write("# Generated for RAG applications\n\n")
            
            f.write("# PREFIX definitions\n")
            f.write("PREFIX : <http://example.org/>\n")
            f.write("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n")
            f.write("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\n")
            
            f.write("# Query 1: Find all people mentioned\n")
            f.write("SELECT ?person ?predicate ?value WHERE {\n")
            f.write("  ?person ?predicate ?value .\n")
            f.write("  FILTER(CONTAINS(STR(?person), \"christie\") || CONTAINS(STR(?person), \"agatha\"))\n")
            f.write("}\n\n")
            
            f.write("# Query 2: Find all locations\n")
            f.write("SELECT ?location ?predicate ?value WHERE {\n")
            f.write("  ?location ?predicate ?value .\n")
            f.write("  FILTER(CONTAINS(STR(?location), \"torquay\") || CONTAINS(STR(?location), \"england\"))\n")
            f.write("}\n\n")
            
            f.write("# Query 3: Find all events (birth, death, etc.)\n")
            f.write("SELECT ?event ?predicate ?value WHERE {\n")
            f.write("  ?event ?predicate ?value .\n")
            f.write("  FILTER(CONTAINS(STR(?predicate), \"born\") || CONTAINS(STR(?predicate), \"death\"))\n")
            f.write("}\n\n")
            
            f.write("# Query 4: Find all relationships for a specific entity\n")
            f.write("SELECT ?subject ?predicate ?object WHERE {\n")
            f.write("  ?subject ?predicate ?object .\n")
            f.write("  FILTER(?subject = <http://example.org/entity/Agatha_Christie>)\n")
            f.write("}\n\n")
            
            f.write("# Query 5: Find all triples with specific predicate type\n")
            f.write("SELECT ?subject ?predicate ?object WHERE {\n")
            f.write("  ?subject ?predicate ?object .\n")
            f.write("  FILTER(CONTAINS(STR(?predicate), \"has_location\"))\n")
            f.write("}\n\n")
            
            f.write("# Query 6: Count triples by predicate type\n")
            f.write("SELECT ?predicate (COUNT(*) as ?count) WHERE {\n")
            f.write("  ?subject ?predicate ?object .\n")
            f.write("} GROUP BY ?predicate ORDER BY DESC(?count)\n\n")
            
            f.write("# Query 7: Find all unique entities\n")
            f.write("SELECT DISTINCT ?entity WHERE {\n")
            f.write("  { ?entity ?p ?o } UNION { ?s ?p ?entity }\n")
            f.write("}\n\n")
            
            f.write("# Query 8: Find entities connected to a specific concept\n")
            f.write("SELECT ?entity ?predicate ?concept WHERE {\n")
            f.write("  ?entity ?predicate ?concept .\n")
            f.write("  FILTER(CONTAINS(STR(?concept), \"mystery\") || CONTAINS(STR(?concept), \"novel\"))\n")
            f.write("}\n")
    
    def _export_graph_visualization(self, triples: List[Dict], file_name: str, graph_dir: Path) -> Tuple[Optional[str], Optional[str]]:
        """Export graph visualization (DOT + PNG)."""
        try:
            # Create DOT file in graph folder
            dot_file = graph_dir / f"{file_name}_graph.dot"
            png_file = graph_dir / f"{file_name}_graph.png"
            
            # Generate DOT content
            dot_content = self._generate_dot_content_from_triples(triples, file_name)
            
            # Save DOT file
            with open(dot_file, 'w', encoding='utf-8') as f:
                f.write(dot_content)
            
            # Try to generate PNG using Graphviz
            try:
                import subprocess
                result = subprocess.run(
                    ['dot', '-Tpng', str(dot_file), '-o', str(png_file)],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    return str(dot_file), str(png_file)
                else:
                    return str(dot_file), None
                    
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return str(dot_file), None
                
        except Exception as e:
            print(f"⚠️  Graph visualization error: {e}")
            return None, None
    
    def _generate_dot_content_from_triples(self, triples: List[Dict], file_name: str) -> str:
        """Generate professional DOT file content with color-coded entities."""
        dot_lines = [
            f"digraph {file_name.replace(' ', '_')} {{",
            "    rankdir=TB;",
            "    compound=true;",
            "    node [fontname=\"Arial\", fontsize=12, style=filled];",
            "    edge [fontname=\"Arial\", fontsize=10, color=gray];",
            "    ",
            "    // Graph styling",
            "    bgcolor=white;",
            "    ",
            "    // Node type definitions",
            "    subgraph cluster_people {{",
            "        label=\"People\";",
            "        style=filled;",
            "        fillcolor=lightcoral;",
            "        color=red;",
            "        node [fillcolor=lightcoral, color=red, shape=circle];",
            "    }}",
            "    ",
            "    subgraph cluster_locations {{",
            "        label=\"Locations\";",
            "        style=filled;",
            "        fillcolor=lightgreen;",
            "        color=green;",
            "        node [fillcolor=lightgreen, color=green, shape=box];",
            "    }}",
            "    ",
            "    subgraph cluster_concepts {{",
            "        label=\"Concepts\";",
            "        style=filled;",
            "        fillcolor=lightblue;",
            "        color=blue;",
            "        node [fillcolor=lightblue, color=blue, shape=ellipse];",
            "    }}",
            "    ",
            "    subgraph cluster_events {{",
            "        label=\"Events\";",
            "        style=filled;",
            "        fillcolor=lightyellow;",
            "        color=orange;",
            "        node [fillcolor=lightyellow, color=orange, shape=diamond];",
            "    }}",
            "    ",
            "    subgraph cluster_other {{",
            "        label=\"Other\";",
            "        style=filled;",
            "        fillcolor=lightgray;",
            "        color=gray;",
            "        node [fillcolor=lightgray, color=gray, shape=hexagon];",
            "    }}",
            ""
        ]
        
        # Categorize nodes by type
        node_categories = self._categorize_nodes(triples)
        edges = []
        
        # Track frames for color coding
        frame_colors = {}
        frame_counter = 0
        
        for triple in triples:
            subj_clean = self._clean_node_name_for_dot(triple['subject'])
            obj_clean = self._clean_node_name_for_dot(triple['object'])
            pred_clean = self._clean_edge_label(triple['predicate'])
            frame = self._extract_frame_from_predicate(triple['predicate'])
            
            # Assign color to frame if not already assigned
            if frame not in frame_colors:
                frame_colors[frame] = self._get_frame_color(frame_counter)
                frame_counter += 1
            
            # Create edge with styled predicate, frame, and confidence
            edge_style = self._get_edge_style_with_frame(triple['predicate'], frame, frame_colors[frame])
            confidence = triple.get('confidence', 0.9)
            extractable = triple.get('extractable', True)
            
            # Add frame and confidence to edge label
            pred_with_frame = f"{pred_clean} [{frame}]"
            if confidence < 0.5:
                pred_with_frame = f"{pred_with_frame} (low conf)"
            elif confidence < 0.8:
                pred_with_frame = f"{pred_with_frame} (med conf)"
            
            # Add extractable status
            if not extractable:
                pred_with_frame = f"{pred_with_frame} (not extractable)"
            
            edges.append(f'    "{subj_clean}" -> "{obj_clean}" [label="{pred_with_frame}", {edge_style}];')
        
        # Add categorized nodes
        for category, nodes in node_categories.items():
            if nodes:
                dot_lines.append(f"    // {category.title()} nodes")
                for node in nodes:
                    node_style = self._get_node_style(category)
                    dot_lines.append(f'    "{node}" [label="{node}", {node_style}];')
                dot_lines.append("")
        
        # Add edges
        dot_lines.append("    // Relationships")
        dot_lines.extend(edges)
        
        # Add frame legend
        if frame_colors:
            dot_lines.append("")
            dot_lines.append("    // Frame Legend")
            dot_lines.append("    subgraph cluster_legend {")
            dot_lines.append("        label=\"Semantic Frames\";")
            dot_lines.append("        style=filled;")
            dot_lines.append("        fillcolor=white;")
            dot_lines.append("        color=black;")
            dot_lines.append("        rank=sink;")
            for frame, color in frame_colors.items():
                dot_lines.append(f'        "{frame}_legend" [label="{frame}", fillcolor="{color}", style=filled, shape=box, fontsize=8];')
            dot_lines.append("    }")
        
        dot_lines.append("}")
        
        return '\n'.join(dot_lines)
    
    def _categorize_nodes(self, triples: List[Dict]) -> Dict[str, List[str]]:
        """Categorize nodes by type for color coding."""
        categories = {
            'people': [],
            'locations': [],
            'concepts': [],
            'events': [],
            'other': []
        }
        
        all_nodes = set()
        for triple in triples:
            all_nodes.add(self._clean_node_name_for_dot(triple['subject']))
            all_nodes.add(self._clean_node_name_for_dot(triple['object']))
        
        for node in all_nodes:
            category = self._classify_node_type(node)
            categories[category].append(node)
        
        return categories
    
    def _classify_node_type(self, node: str) -> str:
        """Classify node type based on content and patterns."""
        node_lower = node.lower()
        
        # People indicators
        if any(indicator in node_lower for indicator in ['christie', 'agatha', 'person', 'author', 'writer']):
            return 'people'
        
        # Location indicators
        if any(indicator in node_lower for indicator in ['torquay', 'england', 'place', 'location', 'city', 'country']):
            return 'locations'
        
        # Event indicators
        if any(indicator in node_lower for indicator in ['born', 'death', 'died', 'event', 'happened']):
            return 'events'
        
        # Concept indicators
        if any(indicator in node_lower for indicator in ['mystery', 'novel', 'book', 'work', 'writing', 'literature']):
            return 'concepts'
        
        return 'other'
    
    def _get_node_style(self, category: str) -> str:
        """Get DOT styling for different node categories."""
        styles = {
            'people': 'fillcolor=lightcoral, color=red, shape=circle',
            'locations': 'fillcolor=lightgreen, color=green, shape=box',
            'concepts': 'fillcolor=lightblue, color=blue, shape=ellipse',
            'events': 'fillcolor=lightyellow, color=orange, shape=diamond',
            'other': 'fillcolor=lightgray, color=gray, shape=hexagon'
        }
        return styles.get(category, styles['other'])
    
    def _get_edge_style(self, predicate: str) -> str:
        """Get DOT styling for different predicate types."""
        pred_lower = predicate.lower()
        
        if 'has_location' in pred_lower or 'location' in pred_lower:
            return 'color=green, penwidth=2'
        elif 'has_person' in pred_lower or 'person' in pred_lower:
            return 'color=red, penwidth=2'
        elif 'has_time' in pred_lower or 'time' in pred_lower:
            return 'color=purple, penwidth=2'
        elif 'has_topic' in pred_lower or 'topic' in pred_lower:
            return 'color=blue, penwidth=2'
        else:
            return 'color=gray, penwidth=1'
    
    def _clean_node_name(self, name: str) -> str:
        """Clean node name for CSV format."""
        # Remove quotes and clean up
        name = name.strip('"\'')
        # Don't truncate URLs - they should be complete
        if name.startswith('http://') or name.startswith('https://'):
            return name
        # For other names, only clean problematic characters but don't truncate
        name = re.sub(r'[<>{}[\]()]', '', name)
        return name
    
    def _clean_node_name_for_dot(self, name: str) -> str:
        """Clean node name for DOT format (with truncation for visualization)."""
        # Remove quotes and clean up
        name = name.strip('"\'')
        # Replace problematic characters
        name = re.sub(r'[<>{}[\]()]', '', name)
        
        # For URIs, extract the meaningful part instead of truncating
        if name.startswith('http://en.wikipedia.org/wiki/'):
            # Extract the entity name after the last slash
            entity_name = name.split('/')[-1]
            # Replace underscores with spaces for better readability
            entity_name = entity_name.replace('_', ' ')
            return entity_name
        elif name.startswith('http://'):
            # For other URIs, extract the meaningful part
            entity_name = name.split('/')[-1]
            return entity_name
        else:
            # For non-URIs, limit length for DOT visualization
            if len(name) > 30:
                name = name[:27] + "..."
            return name
    
    def _clean_edge_label(self, predicate: str) -> str:
        """Clean edge label for DOT format."""
        # Extract the meaningful part after the colon
        if ':' in predicate:
            predicate = predicate.split(':')[-1]
        # Clean up
        predicate = predicate.replace('_', ' ').replace('#', '')
        # Limit length
        if len(predicate) > 20:
            predicate = predicate[:17] + "..."
        return predicate

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch SRL-to-RDF Pipeline with Multiprocessing")
    parser.add_argument("--authors-dir", default="inputs/authors_reduced_more", help="Directory containing author files")
    parser.add_argument("--output-dir", default="evaluation_outputs", help="Output directory")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: auto)")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing instead of parallel")
    parser.add_argument("--limit", type=int, help="Limit number of authors to process (for testing)")
    
    args = parser.parse_args()
    
    pipeline = BatchPipeline(args.authors_dir, args.output_dir)
    
    if args.sequential:
        print("🔄 Running in sequential mode...")
        results = pipeline.process_all_authors_sequential(args.limit)
    else:
        print("🚀 Running in parallel mode...")
        results = pipeline.process_all_authors(args.workers, args.limit)
    
    pipeline.generate_summary()

if __name__ == "__main__":
    main()
