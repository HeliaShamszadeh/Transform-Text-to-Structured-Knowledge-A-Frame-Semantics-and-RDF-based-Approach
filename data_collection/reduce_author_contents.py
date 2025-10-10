#!/usr/bin/env python3
"""
Reduce author biography lengths to prevent SRL processing timeouts.
Truncates very long texts while preserving the most important information.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

def truncate_text(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to max_chars while trying to end at a complete sentence.
    """
    if len(text) <= max_chars:
        return text
    
    # Find the last complete sentence within the limit
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    
    # Find the last sentence ending
    last_sentence_end = max(last_period, last_exclamation, last_question)
    
    if last_sentence_end > max_chars * 0.8:  # If we can end at a sentence within 80% of limit
        return text[:last_sentence_end + 1]
    else:
        # Just truncate and add ellipsis
        return text[:max_chars - 3] + "..."

def process_author_file(input_file: Path, output_file: Path, max_chars: int = 2000) -> Dict:
    """
    Process a single author file, truncating if necessary.
    """
    print(f"📄 Processing {input_file.name}...")
    
    # Read the full text
    with open(input_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    original_length = len(original_text)
    
    # Truncate if necessary
    if original_length > max_chars:
        truncated_text = truncate_text(original_text, max_chars)
        was_truncated = True
        print(f"  ⚠️  Truncated from {original_length:,} to {len(truncated_text):,} characters")
    else:
        truncated_text = original_text
        was_truncated = False
        print(f"  ✅ Kept original length: {original_length:,} characters")
    
    # Write truncated text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(truncated_text)
    
    return {
        "original_file": str(input_file),
        "output_file": str(output_file),
        "original_length": original_length,
        "final_length": len(truncated_text),
        "was_truncated": was_truncated,
        "reduction_percent": round((1 - len(truncated_text) / original_length) * 100, 1) if was_truncated else 0
    }

def main():
    """Main function to reduce author contents."""
    input_dir = Path("inputs/authors")
    output_dir = Path("inputs/authors_reduced_more")
    output_dir.mkdir(exist_ok=True)
    
    # Process all author files
    author_files = list(input_dir.glob("*.txt"))
    print(f"Found {len(author_files)} author files to process")
    print(f"Target: Reduce to maximum 5,000 characters each")
    print()
    
    results = []
    total_original = 0
    total_final = 0
    
    for author_file in author_files:
        try:
            output_file = output_dir / author_file.name
            result = process_author_file(author_file, output_file)
            results.append(result)
            
            total_original += result["original_length"]
            total_final += result["final_length"]
            
        except Exception as e:
            print(f"❌ Error processing {author_file.name}: {e}")
    
    # Save processing summary
    summary_file = output_dir / "reduction_summary.json"
    summary = {
        "total_files": len(results),
        "total_original_chars": total_original,
        "total_final_chars": total_final,
        "total_reduction_percent": round((1 - total_final / total_original) * 100, 1),
        "files_truncated": sum(1 for r in results if r["was_truncated"]),
        "files_kept_original": sum(1 for r in results if not r["was_truncated"]),
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Content reduction complete!")
    print(f"📁 Reduced files saved to: {output_dir}")
    print(f"📊 Summary:")
    print(f"  • Total files processed: {len(results)}")
    print(f"  • Files truncated: {summary['files_truncated']}")
    print(f"  • Files kept original: {summary['files_kept_original']}")
    print(f"  • Total characters: {total_original:,} → {total_final:,}")
    print(f"  • Overall reduction: {summary['total_reduction_percent']}%")
    print(f"📋 Detailed summary: {summary_file}")

if __name__ == "__main__":
    main()
