#!/usr/bin/env python3
"""
Text chunker to split long author biographies into smaller chunks for SRL processing.
This prevents timeouts and memory issues with very long texts.
"""

import os
import json
from pathlib import Path
from typing import List, Dict

def chunk_text(text: str, max_chars: int = 10000) -> List[str]:
    """
    Split text into chunks of maximum max_chars characters.
    Tries to split at sentence boundaries when possible.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        # Add period back if it's not the last sentence
        if not sentence.endswith('.') and sentence != sentences[-1]:
            sentence += '.'
        
        # Check if adding this sentence would exceed max_chars
        if len(current_chunk) + len(sentence) + 2 <= max_chars:  # +2 for space
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def process_author_chunks(author_file: Path, output_dir: Path, max_chars: int = 10000):
    """
    Process an author file by splitting it into chunks and saving them.
    """
    print(f"📄 Processing {author_file.name}...")
    
    # Read the full text
    with open(author_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    print(f"  Original text: {len(full_text)} characters")
    
    # Split into chunks
    chunks = chunk_text(full_text, max_chars)
    print(f"  Split into {len(chunks)} chunks")
    
    # Save each chunk
    author_name = author_file.stem
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        chunk_file = output_dir / f"{author_name}_chunk_{i+1}.txt"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
        chunk_files.append(str(chunk_file))
        print(f"  Chunk {i+1}: {len(chunk)} characters -> {chunk_file.name}")
    
    return chunk_files

def main():
    """Main function to chunk author files."""
    input_dir = Path("inputs/authors")
    output_dir = Path("inputs/authors_chunked")
    output_dir.mkdir(exist_ok=True)
    
    # Process all author files
    author_files = list(input_dir.glob("*.txt"))
    print(f"Found {len(author_files)} author files to chunk")
    
    all_chunks = {}
    
    for author_file in author_files:
        try:
            chunk_files = process_author_chunks(author_file, output_dir)
            all_chunks[author_file.stem] = chunk_files
        except Exception as e:
            print(f"❌ Error processing {author_file.name}: {e}")
    
    # Save chunk mapping
    mapping_file = output_dir / "chunk_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Chunking complete!")
    print(f"📁 Chunks saved to: {output_dir}")
    print(f"📋 Mapping saved to: {mapping_file}")
    print(f"📊 Total chunks created: {sum(len(chunks) for chunks in all_chunks.values())}")

if __name__ == "__main__":
    main()
