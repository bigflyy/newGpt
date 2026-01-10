from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import re 
import os 

FILEPATH_PDF_LECTURES = "pdfs\OS-LEKTsII_vse.pdf"
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
    
    # Add start boundary (page 1 starts at position 0)
    boundaries.append({
        'page_num': 1,
        'start_pos': 0,
        'end_pos': matches[0].start() if matches else len(text)
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

def assign_page_metadata(chunks, page_boundaries, full_text):
    """
    Assign accurate page numbers to each chunk based on character positions
    """
    enhanced_chunks = []
    
    for chunk in chunks:
        # Find exact position of chunk in full text
        chunk_start = full_text.find(chunk.page_content)
        if chunk_start == -1:  # Fallback if exact match fails
            chunk_start = 0
        chunk_end = chunk_start + len(chunk.page_content)
        
        # Find which pages this chunk overlaps with
        overlapping_pages = []
        
        for boundary in page_boundaries:
            # Check if chunk overlaps with this page's content area
            if (chunk_end > boundary['start_pos'] and 
                chunk_start < boundary['end_pos']):
                overlapping_pages.append(boundary['page_num'])
        
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
            'chunk_end_pos': chunk_end
        }
        
        enhanced_chunks.append(
            Document(
                page_content=chunk.page_content + f"\nИсточник: {chunk.metadata.get('source')}, \n",
                metadata=enhanced_metadata
            )
        )
    
    return enhanced_chunks

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

def clean_markdown_text(text):
    """
    Remove markdown formatting like ##, **, etc.
    """
    # Remove headers (##, ###, etc.)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic (**text**, *text*)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    
    # Remove image descriptions blocks
    text = re.sub(r'\*{2,}==>\s*picture\s*\[\d+\s*x\s*\d+\]\s*intentionally omitted\s*<==\*{2,}', '', text)
    text = re.sub(r'\*{2,}-----\s*Start of picture text\s*-----\*{2,}.*?\*{2,}-----\s*End of picture text\s*-----\*{2,}', '', text, flags=re.DOTALL)
    
    # Remove extra newlines and whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    
    return text

# Get final chunks with page metadata
final_chunks = assign_page_metadata(chunks, page_boundaries, md_content)
# Print results to verify
for i, chunk in enumerate(final_chunks[:10], 1):
    print(f"\n{'='*50}")
    print(f"Chunk {i}: Pages {chunk.metadata['page_range']}")
    print(f"Source pages: {chunk.metadata['source_pages']}")
    print(f"Content preview: {chunk.page_content[:150]}...")
    extract_pages_as_png(FILEPATH_PDF_LECTURES, chunk.metadata['source_pages'], output_dir="foundchunksdir")


