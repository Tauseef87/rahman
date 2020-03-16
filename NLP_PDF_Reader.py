import sys
sys.path.append("F:\Algorithmica\MyCodes")
import os
import PyPDF2
import io
dir = 'F:\Algorithmica\MyCodes'
f = open(os.path.join(dir,'DAAI Newbie document.pdf'),'rb')
pdf_reader = PyPDF2.PdfFileReader(f)
pdf_reader.numPages
page_one = pdf_reader.getPage(1)
page_one_text = page_one.extractText()
page_one_text
pdf_reader.getDocumentInfo()

#-------- PDF Miner________________

import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
def extract_text_by_page(pdf_path):
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
            yield text
            # close open handles
            converter.close()
            fake_file_handle.close()
def extract_text(pdf_path):
    for page in extract_text_by_page(pdf_path):
        print(page)
        print()
if __name__ == '__main__':
    print(extract_text('F:\Algorithmica\MyCodes\DAAI Newbie document.pdf'))
# ------------------- PDF to Json------------------------------------------------------
import json
import os
from miner_text_generator import extract_text_by_page
def export_as_json(pdf_path, json_path):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    data = {'Filename': filename}
    data['Pages'] = []
    counter = 1
    for page in extract_text_by_page(pdf_path):
        text = page[0:100]
        page = {'Page_{}'.format(counter): text}
        data['Pages'].append(page)
        counter += 1
    with open(json_path, 'w') as fh:
        json.dump(data, fh)
if __name__ == '__main__':
    pdf_path = 'w9.pdf'
    json_path = 'w9.json'
    export_as_json(pdf_path, json_path)