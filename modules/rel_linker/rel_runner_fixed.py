#!/usr/bin/env python3
"""
Fixed REL linker to JSON (mention -> Wikipedia entity) using REL API
"""

import argparse
import json
import requests
from typing import List, Dict
import time

def split_text_into_chunks(text: str, max_chars: int = 500) -> list:
    """Split text into chunks of maximum length."""
    import re
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

def extract_entities_rel(text: str, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Extract entities using REL API with chunking for long texts.
    REL API: https://rel.cs.ru.nl/api
    """
    # If text is short, process directly
    if len(text) <= 500:
        return _extract_entities_single_chunk(text, confidence_threshold)
    
    # For long text, use chunking
    print(f"Text is long ({len(text)} chars), using chunking approach...")
    return _extract_entities_chunked(text, confidence_threshold)

def _extract_entities_single_chunk(text: str, confidence_threshold: float = 0.5) -> List[Dict]:
    """Extract entities from a single chunk of text."""
    try:
        # REL API expects text and spans
        payload = {
            "text": text,
            "spans": []  # Empty spans for entity linking
        }
        
        response = requests.post(
            "https://rel.cs.ru.nl/api",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"REL API error: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        
        # REL API returns a list of [start, end, mention, entity, confidence]
        entities = []
        if isinstance(data, list):
            for item in data:
                if len(item) >= 5:
                    start, end, mention, entity, confidence = item[:5]
                    
                    # Filter by confidence threshold
                    if confidence >= confidence_threshold:
                        # Create Wikipedia URI
                        wiki_uri = f"http://en.wikipedia.org/wiki/{entity}"
                        
                        entities.append({
                            "mention": mention,
                            "entity": entity,
                            "uri": wiki_uri,
                            "start": start,
                            "end": end,
                            "confidence": confidence
                        })
                    else:
                        # Handle Unicode characters safely
                        try:
                            print(f"Filtered out '{mention}' (confidence: {confidence:.3f} < {confidence_threshold})")
                        except UnicodeEncodeError:
                            mention_safe = mention.encode('ascii', 'replace').decode('ascii')
                            print(f"Filtered out '{mention_safe}' (confidence: {confidence:.3f} < {confidence_threshold})")
        
        return entities
        
    except requests.exceptions.RequestException as e:
        print(f"Network error calling REL API: {e}")
        return None  # Return None to indicate API error
    except Exception as e:
        print(f"Error processing REL API response: {e}")
        return None  # Return None to indicate API error

def _extract_entities_chunked(text: str, confidence_threshold: float = 0.5) -> List[Dict]:
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
        chunk_entities = _extract_entities_single_chunk(chunk, confidence_threshold)
        
        if chunk_entities:
            # Adjust start/end positions for the full text
            for entity in chunk_entities:
                entity['start'] += offset
                entity['end'] += offset
                all_entities.append(entity)
                # Handle Unicode characters safely
                try:
                    print(f"  Found: {entity['mention']} -> {entity['entity']}")
                except UnicodeEncodeError:
                    # Fallback for Unicode characters that can't be displayed
                    mention_safe = entity['mention'].encode('ascii', 'replace').decode('ascii')
                    entity_safe = entity['entity'].encode('ascii', 'replace').decode('ascii')
                    print(f"  Found: {mention_safe} -> {entity_safe}")
        
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
    parser = argparse.ArgumentParser(description="Extract entities using REL API")
    parser.add_argument("--infile", required=True, help="Input text file")
    parser.add_argument("--outfile", required=True, help="Output JSON file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Read input text
    try:
        with open(args.infile, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return 1
    
    if not text.strip():
        print("Warning: Input text is empty")
        return 1
    
    print(f"Processing text with REL API...")
    print(f"Text length: {len(text)} characters")
    print(f"Confidence threshold: {args.confidence}")
    
    # Extract entities
    entities = extract_entities_rel(text, args.confidence)
    
    if entities is None:
        print("API Error: Could not connect to REL API")
        # Create empty output structure
        output = {"doc_id": "doc", "entities": []}
    elif not entities:
        print("API Success: No entities found above confidence threshold")
        # Create empty output structure
        output = {"doc_id": "doc", "entities": []}
    else:
        print(f"API Success: Found {len(entities)} entities")
        
        # Create output structure
        output = {
            "doc_id": "doc",
            "entities": entities
        }
    
    # Save results
    try:
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Entities saved to {args.outfile}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
