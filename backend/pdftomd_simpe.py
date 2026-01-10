import pymupdf.layout
import pymupdf4llm

import os
os.makedirs("output_images", exist_ok=True)



doc = pymupdf.open("./pdfs/OS-LEKTsII_vse.pdf")
md = pymupdf4llm.to_markdown(doc, page_separators=True, page_chunks=False, force_text=True)
print(md)
print(md[0])
# for m in md:
#     print(m.get('toc_items'))


from pathlib import Path
suffix = ".md" # or ".json" or ".txt"
Path(doc.name).with_suffix(suffix).write_bytes(md.encode())

# Method 3: Using regex (more flexible if you need variations)
import re

def find_headers_regex(filename):
    pattern = r'^## .*$'
    headers = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if re.match(pattern, line):
                headers.append(line.strip())
    return headers

# Usage:
headers = find_headers_regex('md/OS.md')
for header in headers:
    print(header)


h1 = ['**ОБЗОР СОДЕРЖАНИЯ ДИСЦИПЛИНЫ «ОПЕРАЦИОННЫЕ СИСТЕМЫ»**',
      '**ЛЕКЦИЯ 1. ОС – Функции и эксплуатационные требования**',
      '**ЛЕКЦИЯ 2. ОС – Процессы**',
      '**ЛЕКЦИЯ** 3. **Синхронизация процессов**',
      ]
h2 = ['', 'Литература', # Л1
      '', '**Функции супервизора:**', '**Преимущества:**', '**Недостатки:**', #Л2
      '']