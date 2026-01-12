from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import re 
import os 
import json

FILEPATH_PDF_LECTURES = "pdfs\\OS-LEKTsII_vse.pdf"  # Fixed backslash escaping
FILEPATH_MD_LECTURES = "./pdfs/OS-LEKTsII_vse.md"
filename1 = "Все лекции Операционные системы Сущенко"
filename2 = "Пособие Операционные системы Сущенко"

with open(FILEPATH_MD_LECTURES, "r", encoding="utf-8") as f:
    md_content = f.read()

document = Document(
    page_content=md_content,
    metadata={"source": filename1}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents([document])

def get_page_boundaries(text):
    """
    Find all page markers and their positions in the text
    Returns list of dicts with page numbers and character positions
    """
    boundaries = []
    
    # Find all page markers with their positions
    pattern = r'--- end of page\.page_number=(\d+) ---'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        # If no page markers found, treat entire document as page 1
        boundaries.append({
            'page_num': 1,
            'start_pos': 0,
            'end_pos': len(text)
        })
        return boundaries
    
    # Add start boundary (page 1 starts at position 0)
    boundaries.append({
        'page_num': 1,
        'start_pos': 0,
        'end_pos': matches[0].start()
    })
    
    # Add boundaries for each page marker found
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        marker_end_pos = match.end()
        
        # Next page starts after this marker
        next_page_num = page_num + 1
        next_start_pos = marker_end_pos
        
        # Next page ends at next marker or end of text
        if i + 1 < len(matches):
            next_end_pos = matches[i + 1].start()
        else:
            next_end_pos = len(text)
        
        boundaries.append({
            'page_num': next_page_num,
            'start_pos': next_start_pos,
            'end_pos': next_end_pos
        })
    
    return boundaries

# Get the page boundaries
page_boundaries = get_page_boundaries(md_content)

def assign_detailed_page_metadata(chunks, page_boundaries, full_text):
    """
    Assign detailed page metadata including EXACT text portions per page
    """
    enhanced_chunks = []
    
    for chunk in chunks:
        chunk_start = full_text.find(chunk.page_content)
        if chunk_start == -1:  # Fallback if exact match fails
            chunk_start = 0
        chunk_end = chunk_start + len(chunk.page_content)
        
        # Find which pages this chunk overlaps with
        overlapping_pages = []
        page_text_mapping = {}  # Maps page_num to text portions from that page
        
        for boundary in page_boundaries:
            # Check if chunk overlaps with this page
            if chunk_end > boundary['start_pos'] and chunk_start < boundary['end_pos']:
                overlapping_pages.append(boundary['page_num'])
                
                # Calculate the exact overlap region
                overlap_start = max(chunk_start, boundary['start_pos'])
                overlap_end = min(chunk_end, boundary['end_pos'])
                
                if overlap_end > overlap_start:
                    # Extract the exact text portion from this page
                    text_portion = full_text[overlap_start:overlap_end]
                    
                    # Clean up the text portion (remove leading/trailing whitespace)
                    text_portion = text_portion.strip()
                    
                    if text_portion:  # Only add non-empty portions
                        if boundary['page_num'] not in page_text_mapping:
                            page_text_mapping[boundary['page_num']] = []
                        page_text_mapping[boundary['page_num']].append(text_portion)
        
        # Create proper metadata
        if overlapping_pages:
            page_range = f"{min(overlapping_pages)}-{max(overlapping_pages)}" if len(overlapping_pages) > 1 else str(overlapping_pages[0])
        else:
            page_range = "unknown"
        
        enhanced_metadata = {
            **chunk.metadata,
            'source_pages': sorted(set(overlapping_pages)),
            'page_range': page_range,
            'chunk_start_pos': chunk_start,
            'chunk_end_pos': chunk_end,
            'page_text_mapping': page_text_mapping,  # EXACT mapping of text portions to pages
            'text_portion_lengths': {page_num: sum(len(text) for text in texts) 
                                   for page_num, texts in page_text_mapping.items()}
        }
        
        enhanced_chunks.append(
            Document(
                page_content=chunk.page_content + f"\nИсточник: {chunk.metadata.get('source')}, \n",
                metadata=enhanced_metadata
            )
        )
    
    return enhanced_chunks

def save_chunks_with_mapping(chunks, output_file="chunk_page_mapping.json"):
    """
    Save chunks with their detailed page mapping to a JSON file
    """
    mapping_data = []
    
    for i, chunk in enumerate(chunks):
        chunk_data = {
            'chunk_id': i,
            'content': chunk.page_content,
            'metadata': {
                'source': chunk.metadata.get('source'),
                'source_pages': chunk.metadata.get('source_pages', []),
                'page_range': chunk.metadata.get('page_range', 'unknown'),
                'page_text_mapping': chunk.metadata.get('page_text_mapping', {}),
                'text_portion_lengths': chunk.metadata.get('text_portion_lengths', {})
            }
        }
        mapping_data.append(chunk_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved detailed chunk-to-page mapping to: {output_file}")
    return output_file

def extract_pages_as_png(pdf_path, page_numbers, output_dir="page_images"):
    """
    Extract specified pages from PDF and save as PNG files
    Returns list of saved image paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF
    doc = pymupdf.open(pdf_path)
    saved_paths = []
    
    for page_num in page_numbers:
        # Check if page number is valid
        if 1 <= page_num <= len(doc):
            # Get page (0-indexed, so subtract 1)
            page = doc[page_num - 1]
            
            # Render page to image with good quality (2x zoom)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            
            # Create filename
            filename = f"page_{page_num}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save image
            pix.save(output_path)
            saved_paths.append(output_path)
            print(f"Saved page {page_num} as: {output_path}")
        else:
            print(f"Page {page_num} is out of range (PDF has {len(doc)} pages)")
    
    # Close PDF
    doc.close()
    return saved_paths

# Get final chunks with detailed page metadata
final_chunks = assign_detailed_page_metadata(chunks, page_boundaries, md_content)

# Save the detailed mapping to a file
mapping_file = save_chunks_with_mapping(final_chunks)

# Print results to verify with detailed text mapping
print(f"\n{'='*80}")
print("DETAILED CHUNK-TO-PAGE MAPPING SAVED TO:", mapping_file)
print(f"{'='*80}")

for i, chunk in enumerate(final_chunks[3:10], 1):  # Show first 3 chunks
    print(f"\n{'='*50}")
    print(f"Chunk {i}: Pages {chunk.metadata['page_range']}")
    print(f"Source pages: {chunk.metadata['source_pages']}")
    print(f"Content preview: {chunk.page_content[:150]}...")
    
    print("\n🔍 EXACT TEXT-TO-PAGE MAPPING:")
    page_text_mapping = chunk.metadata.get('page_text_mapping', {})
    for page_num, text_portions in page_text_mapping.items():
        print(f"  📄 Page {page_num}:")
        for j, text in enumerate(text_portions[:2], 1):  # Show first 2 portions per page
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"    • Portion {j}: \"{preview}\"")
            print(f"      Length: {len(text)} characters")
    
    # Extract page images for this chunk
    extract_pages_as_png(FILEPATH_PDF_LECTURES, chunk.metadata['source_pages'], output_dir=f"chunk_{i}_pages")

print(f"\n{'='*80}")
print("✅ All page images have been saved to respective chunk directories")
print("✅ Detailed mapping saved to:", mapping_file)
print("💡 You can now load this mapping file anytime to get EXACT text-to-page information")
print(f"{'='*80}")