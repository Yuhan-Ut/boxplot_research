#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract elementID values from HTML files.

This script processes HTML files and extracts all non-empty elementID values
from elements that have a 'data-rbd-draggable-id' attribute. The attribute
value is expected to be a JSON string containing an 'elementID' field.

For each HTML file, it creates a corresponding .txt file with one elementID
per line.
"""
import json
import os
from pathlib import Path
from html.parser import HTMLParser

class ElementIDExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.element_ids = []
    
    def handle_starttag(self, tag, attrs):
        # Look for data-rbd-draggable-id attribute
        for attr_name, attr_value in attrs:
            if attr_name == 'data-rbd-draggable-id' and attr_value and attr_value.strip():
                try:
                    # Parse the JSON string
                    data = json.loads(attr_value)
                    # Extract elementID if it exists and is non-empty
                    element_id = data.get('elementID', '')
                    if element_id and element_id.strip():
                        self.element_ids.append(element_id.strip())
                except (json.JSONDecodeError, AttributeError):
                    # If it's not JSON or doesn't have elementID, skip
                    pass

def extract_element_ids(html_file):
    """Extract all non-empty elementID values from data-rbd-draggable-id attributes.
    
    Args:
        html_file: Path to the HTML file to process
        
    Returns:
        List of elementID strings
    """
    parser = ElementIDExtractor()
    
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
    except Exception as e:
        print(f"Error reading {html_file}: {e}")
        return []
    
    return parser.element_ids

def main():
    # Find all HTML files in the htmls directory
    html_dir = Path('htmls')
    if not html_dir.exists():
        print(f"Directory {html_dir} does not exist!")
        return
    
    html_files = list(html_dir.glob('*.html'))
    
    if not html_files:
        print(f"No HTML files found in {html_dir}")
        return
    
    print(f"Found {len(html_files)} HTML files")
    
    # Process each HTML file
    for html_file in html_files:
        print(f"Processing {html_file.name}...")
        element_ids = extract_element_ids(html_file)
        
        # Create output text file
        output_file = html_file.with_suffix('.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for element_id in element_ids:
                f.write(f"{element_id}\n")
        
        print(f"  Extracted {len(element_ids)} non-empty element IDs -> {output_file.name}")

if __name__ == '__main__':
    main()

