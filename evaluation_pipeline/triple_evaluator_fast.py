#!/usr/bin/env python3
"""
Fast LLM-based triple extraction evaluation using Deepseek v3 API.
Optimized for large RDF files with batching and limits.
"""

import argparse
import json
import os
import re
import requests
from typing import Dict, List, Tuple
from pathlib import Path

class FastTripleEvaluator:
    def __init__(self, deepseek_api_key: str, model: str = "deepseek-chat"):
        self.api_key = deepseek_api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    def evaluate_triples_batch(self, triples: List[str], paragraph: str, batch_size: int = 5) -> List[Dict]:
        """Evaluate multiple triples in batches for efficiency."""
        results = []
        
        # Process triples in batches
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(triples) + batch_size - 1)//batch_size} ({len(batch)} triples)")
            
            batch_results = self._evaluate_batch(batch, paragraph)
            results.extend(batch_results)
        
        return results
    
    def _evaluate_batch(self, triples: List[str], paragraph: str) -> List[Dict]:
        """Evaluate a batch of triples together."""
        try:
            # Create batch evaluation prompt
            prompt = self._create_batch_evaluation_prompt(triples, paragraph)
            
            # Call Deepseek API
            response = self._call_deepseek(prompt)
            
            # Parse response
            return self._parse_batch_response(response, triples)
            
        except Exception as e:
            print(f"Batch evaluation failed: {e}")
            # Return default results for this batch
            return [{"triple": triple, "extractable": False, "confidence": 0.0, "reason": f"Error: {e}"} for triple in triples]
    
    def _create_batch_evaluation_prompt(self, triples: List[str], paragraph: str) -> str:
        """Create evaluation prompt for a batch of triples."""
        triples_text = "\n".join([f"{i+1}. {triple}" for i, triple in enumerate(triples)])
        
        prompt = f"""You are an expert in information extraction and RDF triple validation. 

Given the following paragraph and a list of RDF triples, evaluate whether each triple is REASONABLY EXTRACTABLE from the paragraph. Be LENIENT and focus on semantic meaning.

Paragraph: "{paragraph}"

Triples to evaluate:
{triples_text}

IMPORTANT EVALUATION GUIDELINES - BE LENIENT:
- Be GENEROUS with "extractable": true if the triple represents information that has any semantic connection to the paragraph
- Don't require exact word matches - look for related concepts, themes, and relationships
- Consider indirect connections, implied meanings, and contextual relationships
- If the paragraph contains any related information that could support the triple's meaning, mark it as extractable
- Only mark as false if the information is completely unrelated or contradictory
- Use high confidence (0.8-1.0) for clear matches, medium (0.5-0.8) for reasonable matches, low (0.2-0.5) for weak but valid matches
- Even weak semantic connections should be marked as extractable with low confidence

For each triple, determine:
1. Can this triple be reasonably extracted from the paragraph? (true/false) - BE LENIENT
2. How confident are you? (0.0-1.0)
3. Brief reason for your decision

Respond with a JSON array where each element has:
- "triple": the original triple
- "extractable": true/false
- "confidence": 0.0-1.0
- "reason": brief explanation

Example format:
[
  {{"triple": "frame:Being_born a frame:Being_born", "extractable": true, "confidence": 0.9, "reason": "Birth information is clearly stated"}},
  {{"triple": "frame:Writing a frame:Writing", "extractable": true, "confidence": 0.7, "reason": "Writing activities are mentioned throughout the text"}}
]

Respond with ONLY the JSON array, no other text."""
        
        return prompt
    
    def _call_deepseek(self, prompt: str, max_retries: int = 3) -> str:
        """Call Deepseek API with retry logic and longer timeout."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                print(f"    API call attempt {attempt + 1}/{max_retries}...")
                response = requests.post(
                    self.base_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=120  # Increased from 30 to 120 seconds
                )
                response.raise_for_status()
                
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
                
            except requests.exceptions.Timeout:
                print(f"    Timeout on attempt {attempt + 1}, retrying...")
                if attempt == max_retries - 1:
                    raise Exception(f"Deepseek API call failed after {max_retries} attempts: Timeout")
                continue
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Deepseek API call failed: {e}")
                print(f"    Error on attempt {attempt + 1}: {e}, retrying...")
                continue
        
        raise Exception(f"Deepseek API call failed after {max_retries} attempts")
    
    def _parse_batch_response(self, response: str, original_triples: List[str]) -> List[Dict]:
        """Parse Deepseek response for batch evaluation."""
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                results = json.loads(json_str)
                
                # Ensure we have results for all triples
                if len(results) != len(original_triples):
                    print(f"Warning: Expected {len(original_triples)} results, got {len(results)}")
                    # Pad with default results if needed
                    while len(results) < len(original_triples):
                        results.append({"triple": original_triples[len(results)], "extractable": False, "confidence": 0.0, "reason": "Not evaluated"})
                
                # Apply confidence-based extractable override
                for result in results:
                    confidence = result.get('confidence', 0.0)
                    original_extractable = result.get('extractable', False)
                    
                    # Override: if confidence >= 0.5, set extractable = true
                    if confidence >= 0.5:
                        result['extractable'] = True
                        # Update reason to reflect the override only if it changed
                        if not original_extractable:
                            original_reason = result.get('reason', '')
                            result['reason'] = f"{original_reason}"
                    else:
                        # If confidence < 0.5, set extractable = false (override LLM decision)
                        result['extractable'] = False
                        # Update reason to reflect the override only if it changed
                        if original_extractable:
                            original_reason = result.get('reason', '')
                            result['reason'] = f"{original_reason}"
                
                return results
            else:
                raise ValueError("No JSON array found in response")
                
        except Exception as e:
            print(f"Error parsing batch response: {e}")
            # Return default results
            return [{"triple": triple, "extractable": False, "confidence": 0.0, "reason": f"Parse error: {e}"} for triple in original_triples]

def extract_triples_from_rdf(rdf_file: str, max_triples: int = 60) -> List[str]:
    """Extract triples from RDF file with limit."""
    triples = []
    
    try:
        with open(rdf_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse custom RDF format: subject predicate object
        lines = content.split('\n')
        for line in lines:
            if len(triples) >= max_triples:
                break
                
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('@') and not line.startswith('#'):
                # Remove trailing semicolon or period
                line = re.sub(r'[;.]$', '', line)
                # Check if it's a valid triple (has at least 2 spaces separating subject, predicate, object)
                parts = line.split()
                if len(parts) >= 3:
                    triples.append(line)
    
    except Exception as e:
        print(f"Error reading RDF file {rdf_file}: {e}")
    
    return triples

def main():
    parser = argparse.ArgumentParser(description="Fast RDF triple extraction evaluation using Deepseek v3")
    parser.add_argument("--rdf-file", required=True, help="Path to RDF file")
    parser.add_argument("--paragraph-file", required=True, help="Path to source paragraph file")
    parser.add_argument("--output", required=True, help="Output evaluation file")
    parser.add_argument("--deepseek-api-key", default="your-api-key", help="Deepseek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--model", default="deepseek-chat", help="Deepseek model")
    parser.add_argument("--max-triples", type=int, default=60, help="Maximum number of triples to evaluate")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Get API key - try argument first, then environment, then default
    api_key = args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or "your-api-key"
    if not api_key:
        print("Error: Deepseek API key required. Set DEEPSEEK_API_KEY env var or use --deepseek-api-key")
        return 1
    else:
        print(f"Using DeepSeek API key: {api_key[:10]}...")
    
    # Load paragraph
    with open(args.paragraph_file, 'r', encoding='utf-8') as f:
        paragraph = f.read()
    
    # Extract triples (limited)
    triples = extract_triples_from_rdf(args.rdf_file, args.max_triples)
    print(f"Found {len(triples)} triples to evaluate (limited to {args.max_triples})")
    
    if not triples:
        print("No triples found to evaluate")
        return 1
    
    # Evaluate
    evaluator = FastTripleEvaluator(api_key, args.model)
    
    print(f"Evaluating {len(triples)} triples in batches of {args.batch_size}...")
    results = evaluator.evaluate_triples_batch(triples, paragraph, args.batch_size)
    
    # Calculate summary
    extractable_count = sum(1 for r in results if r.get('extractable', False))
    avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
    
    # Calculate confidence distribution statistics
    high_confidence_50 = sum(1 for r in results if r.get('confidence', 0) >= 0.5)
    high_confidence_80 = sum(1 for r in results if r.get('confidence', 0) >= 0.8)
    percent_50 = (high_confidence_50 / len(triples)) * 100
    percent_80 = (high_confidence_80 / len(triples)) * 100

    evaluation_result = {
        "evaluations": results,
        "summary": {
            "total_triples": len(triples),
            "extractable_triples": extractable_count,
            "accuracy": extractable_count / len(triples) if triples else 0,
            "avg_confidence": avg_confidence,
            "high_confidence_50": high_confidence_50,
            "high_confidence_80": high_confidence_80,
            "percent_50": percent_50,
            "percent_80": percent_80,
            "max_triples_limit": args.max_triples
        }
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation completed!")
    print(f"Extractable: {extractable_count}/{len(triples)} ({extractable_count/len(triples)*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.2f}")

    
    print(f"High confidence (>= 0.5): {high_confidence_50}/{len(triples)} ({percent_50:.1f}%)")
    print(f"Very high confidence (>= than 0.8): {high_confidence_80}/{len(triples)} ({percent_80:.1f}%)")
    print(f"Results saved to: {args.output}")
    
    
    return 0

if __name__ == "__main__":
    exit(main())
