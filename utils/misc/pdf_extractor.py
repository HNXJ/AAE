import fitz  # PyMuPDF
import os
import sys
import re

def extract_pdf_content(pdf_path, output_img_dir, output_txt_dir):
    filename = os.path.basename(pdf_path).replace('.pdf', '')
    doc = fitz.open(pdf_path)
    
    markdown_content = f"# Paper: {filename}\n\n"
    img_count = 0
    
    print(f"📄 Processing: {filename}")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 1. Extract Text
        text = page.get_text("text")
        # Basic heuristic for section detection (all caps or specific headers)
        lines = text.split('\n')
        for line in lines:
            if re.match(r'^(ABSTRACT|INTRODUCTION|METHODS|RESULTS|DISCUSSION|REFERENCES|ACKNOWLEDGEMENTS)', line.strip().upper()):
                markdown_content += f"\n## {line.strip()}\n"
            else:
                markdown_content += line + "\n"
        
        # 2. Extract Images
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            
            img_filename = f"{filename}_page{page_num+1}_img{img_index+1}.{ext}"
            img_path = os.path.join(output_img_dir, img_filename)
            
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            img_count += 1
            
    # Save Markdown
    txt_path = os.path.join(output_txt_dir, f"{filename}.md")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"✅ Finished {filename}: Extracted {img_count} images and saved markdown to {txt_path}")
    doc.close()

if __name__ == "__main__":
    pdf_dir = "/Users/hamednejat/workspace/Research_Assets/media/pdfs/"
    img_dir = "/Users/hamednejat/workspace/Research_Assets/media/pdfs/img/"
    txt_dir = "/Users/hamednejat/workspace/Research_Assets/media/pdfs/txt/"
    
    # Process all PDFs in the directory
    import glob
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    print(f"🔎 Found {len(pdf_files)} PDFs to process.")
    for full_path in pdf_files:
        extract_pdf_content(full_path, img_dir, txt_dir)
