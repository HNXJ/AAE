import sys
import argparse
import urllib.request
from urllib.error import URLError
import re

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup not found. Please install: pip install beautifulsoup4")
    sys.exit(1)

def extract_content(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
        with urllib.request.urlopen(req) as response:
            html = response.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try to find the main content area (standard for MkDocs/Sphinx/ReadTheDocs)
        main_content = soup.find('main') or soup.find('article') or soup.find(role='main') or soup.find(class_='document') or soup.body
        
        if not main_content:
            return "Could not find main content area."

        # Remove irrelevant elements like sidebars, footers, scripts
        for element in main_content(['nav', 'footer', 'script', 'style', 'header']):
            element.decompose()

        # Format code blocks nicely into markdown
        for pre in main_content.find_all('pre'):
            code_text = pre.get_text()
            new_pre = soup.new_string(f"\n```python\n{code_text}\n```\n")
            pre.replace_with(new_pre)

        # Format headers into markdown
        for h in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(h.name[1])
            h_text = h.get_text().replace('¶', '').strip() # Remove common anchor links
            new_h = soup.new_string(f"\n\n{'#' * level} {h_text}\n\n")
            h.replace_with(new_h)
            
        # Get cleaned text
        text = main_content.get_text(separator='\n')
        
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
        
    except Exception as e:
        return f"Error fetching {url}: {e}"

def main():
    parser = argparse.ArgumentParser(description="doc_surfer: Extract technical documentation into Markdown.")
    parser.add_argument("url", help="URL of the documentation page")
    parser.add_argument("-o", "--output", help="Output Markdown file", default=None)
    args = parser.parse_args()

    print(f"🌊 Surfing: {args.url}")
    content = extract_content(args.url)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Saved to: {args.output}")
    else:
        print("\n--- Extracted Content ---\n")
        print(content)

if __name__ == "__main__":
    main()
