#!/usr/bin/env python3
"""
Entity linker that splits long text into chunks to avoid API timeouts.
"""

import json
import re
from pathlib import Path
from modules.rel_linker.rel_runner_fixed import extract_entities_rel

def split_text_into_chunks(text: str, max_chars: int = 500) -> list:
    """Split text into chunks of maximum length."""
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_chars, save current chunk
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_entities_chunked(text: str, confidence_threshold: float = 0.5) -> list:
    """Extract entities from text by processing it in chunks."""
    print(f"Original text length: {len(text)} characters")
    
    # Split into chunks
    chunks = split_text_into_chunks(text, max_chars=500)
    print(f"Split into {len(chunks)} chunks")
    
    all_entities = []
    offset = 0
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        
        # Extract entities from this chunk
        chunk_entities = extract_entities_rel(chunk, confidence_threshold)
        
        if chunk_entities:
            # Adjust start/end positions for the full text
            for entity in chunk_entities:
                entity['start'] += offset
                entity['end'] += offset
                all_entities.append(entity)
                print(f"  Found: {entity['mention']} -> {entity['entity']}")
        
        # Update offset for next chunk
        offset += len(chunk) + 2  # +2 for ". " separator
    
    # Remove duplicates based on mention and position
    unique_entities = []
    seen = set()
    
    for entity in all_entities:
        key = (entity['mention'], entity['start'], entity['end'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    print(f"Total unique entities found: {len(unique_entities)}")
    return unique_entities

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract entities from long text using chunking")
    parser.add_argument("--infile", required=True, help="Input text file")
    parser.add_argument("--outfile", required=True, help="Output JSON file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Read input text
    with open(args.infile, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract entities using chunking
    entities = extract_entities_chunked(text, args.confidence)
    
    # Create output
    output = {
        "doc_id": "doc",
        "entities": entities
    }
    
    # Save results
    with open(args.outfile, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Entities saved to {args.outfile}")

if __name__ == "__main__":
    main()
