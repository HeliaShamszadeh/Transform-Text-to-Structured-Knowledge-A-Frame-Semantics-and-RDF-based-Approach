#!/usr/bin/env python3
"""
Wikipedia text scraper that reads author names from a file,
extracts their biographical information, and saves it for pipeline processing.
"""

import requests
import json
import time
import re
import sys
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup

class WikipediaScraper:
    def __init__(self, output_dir: str = "inputs/authors"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SRL-RDF-Evaluation/1.0 (Educational Research Project)'
        })

    def read_author_names_from_file(self, filepath: str = "authors.txt") -> List[str]:
        """Reads author names from a given text file, one name per line."""
        authors_file = Path(filepath)
        if not authors_file.exists():
            print(f"❌ ERROR: The file '{filepath}' was not found.")
            print("Please create it and add one author name per line.")
            return []
        
        with open(authors_file, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, and filter out empty lines
            names = [line.strip() for line in f if line.strip()]
        
        print(f"✅ Found {len(names)} author names in '{filepath}'.")
        return names

    def get_wikipedia_page_title(self, author_name: str) -> Optional[str]:
        """Searches Wikipedia and returns the exact page title for the best match."""
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query', 'format': 'json', 'list': 'search',
            'srsearch': author_name, 'srlimit': 1
        }
        try:
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['query']['search']:
                return data['query']['search'][0]['title']
        except requests.RequestException as e:
            print(f"  - 🌐 Network error while searching for {author_name}: {e}")
        return None

    def get_page_content(self, page_title: str) -> Optional[str]:
        """Retrieves the main body content of a Wikipedia page, excluding titles, bibliography, and further reading."""
        # Get the HTML content instead of plain text
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'parse', 'format': 'json', 'page': page_title,
            'prop': 'text', 'disableeditsection': True
        }
        try:
            response = self.session.get(api_url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if 'parse' in data and 'text' in data['parse']:
                html_content = data['parse']['text']['*']
                return self.extract_main_content(html_content)
        except requests.RequestException as e:
            print(f"  - 🌐 Network error fetching content for {page_title}: {e}")
        return None

    def extract_main_content(self, html_content: str) -> str:
        """Extract only the main body content from Wikipedia HTML, excluding unwanted sections."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted sections
        unwanted_sections = [
            'References', 'Bibliography', 'Further reading', 'External links',
            'See also', 'Notes', 'Citations', 'Sources', 'Works cited',
            'Selected works', 'Awards and honors', 'Honors', 'Legacy',
            'Personal life', 'Death', 'Family', 'Early life', 'Education'
        ]
        
        # Find and remove unwanted sections
        for section in unwanted_sections:
            # Look for headings with these titles
            headings = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(section, re.IGNORECASE))
            for heading in headings:
                # Remove the heading and all content until the next heading
                current = heading
                while current and current.name not in ['h1', 'h2', 'h3', 'h4']:
                    next_sibling = current.next_sibling
                    current.decompose()
                    current = next_sibling
                if current and current.name in ['h1', 'h2', 'h3', 'h4']:
                    current.decompose()
        
        # Remove infoboxes and navigation boxes
        for element in soup.find_all(['table', 'div'], class_=re.compile(r'infobox|navbox|sidebar', re.IGNORECASE)):
            element.decompose()
        
        # Remove reference lists and citations
        for element in soup.find_all(['ol', 'ul'], class_=re.compile(r'references|citations', re.IGNORECASE)):
            element.decompose()
        
        # Remove edit links and other Wikipedia-specific elements
        for element in soup.find_all(['span', 'a'], class_=re.compile(r'edit|mw-editsection', re.IGNORECASE)):
            element.decompose()
        
        # Get the main content area
        main_content = soup.find('div', class_='mw-parser-output')
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Extract text and clean it
            text = main_content.get_text(separator=' ', strip=True)
            return self.clean_text(text)
        
        return ""

    def clean_text(self, text: str) -> str:
        """Removes common Wikipedia artifacts from the text."""
        if not text: 
            return ""
        
        # Remove Wikipedia-specific patterns
        text = re.sub(r'\[\s*\d+\s*\]', '', text)  # Remove citation numbers like [1], [ 182 ], [2 ]
        text = re.sub(r'\[edit\]', '', text)  # Remove edit links
        text = re.sub(r'\[citation needed\]', '', text)  # Remove citation needed tags
        text = re.sub(r'\[when\?\]', '', text)  # Remove when tags
        text = re.sub(r'\[where\?\]', '', text)  # Remove where tags
        text = re.sub(r'\[note\s+\d+\]', '', text)  # Remove note references like [note 1]
        text = re.sub(r'\[[a-z]\]', '', text)  # Remove single letter references like [a], [b]
        text = re.sub(r'\[[A-Z]\]', '', text)  # Remove single letter references like [A], [B]
        text = re.sub(r'\[[a-z]\s*\]', '', text)  # Remove letter references with spaces like [a ]
        text = re.sub(r'\[[A-Z]\s*\]', '', text)  # Remove letter references with spaces like [A ]
        
        # Remove section headers that might have slipped through
        text = re.sub(r'===\s*(.*?)\s*===', r'\1.', text)
        text = re.sub(r'==\s*(.*?)\s*==', r'\n\n\1\n', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Remove leading/trailing whitespace from lines
        
        return text.strip()

    def process_and_save_author(self, author_name: str):
        """Main workflow for a single author: find, fetch, clean, and save."""
        print(f"Processing: {author_name}")
        page_title = self.get_wikipedia_page_title(author_name)
        if not page_title:
            print(f"  - ❌ Could not find a Wikipedia page for '{author_name}'. Skipping.")
            return

        print(f"  - 📄 Found page title: '{page_title}'")
        content = self.get_page_content(page_title)
        if not content:
            print(f"  - ❌ Failed to retrieve content for '{page_title}'. Skipping.")
            return
            
        cleaned_content = self.clean_text(content)
        
        # Create a filesystem-safe name for the file
        safe_filename = re.sub(r'[^\w\s-]', '', author_name).strip()
        safe_filename = re.sub(r'[-\s]+', '_', safe_filename) + ".txt"
        filepath = self.output_dir / safe_filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"  - ✅ Saved {len(cleaned_content):,} characters to {filepath}")
        except IOError as e:
            print(f"  - ❌ Error saving file: {e}")

    def run(self, author_file: str):
        """Runs the complete scraping process."""
        authors = self.read_author_names_from_file(author_file)
        if not authors:
            return
            
        print("\n" + "="*50)
        print(" Scraping Wikipedia for author biographies...")
        print("="*50 + "\n")
        
        for author in authors:
            self.process_and_save_author(author)
            time.sleep(1) # Be polite to the Wikipedia API
        
        print("\n🎉 Scraping complete.")

def main():
    # Check for required libraries
    try:
        import bs4
    except ImportError:
        print("Missing library. Please install with: pip install beautifulsoup4")
        sys.exit(1)

    scraper = WikipediaScraper()
    scraper.run("data_collection/authors_data/author_names.txt")

if __name__ == "__main__":
    main()