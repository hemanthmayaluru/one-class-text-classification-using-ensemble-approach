# Imported modules
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBox
import re
from io import BytesIO
from urllib.request import urlopen, Request
import urllib
import json
import time
import pandas as pd
import os.path
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup

# Functions for extracting texts from arxiv PDFs
stop = stopwords.words('english')
def get_LTText_Props(tb):
    """ Extract properties from the text box of pdf miner
	
	Args: 
		tb: a text box created by pdf miner
	
	Returns: a dict of properties of that text box: text, and font/position of first character
	
	"""
    props = {
        'text': tb.get_text(),
        'fontname': '',
        'fontsize': 0,
        'x': 0,
        'y': 0,
        'endfontname': '',
        'endfontsize': 0    
        }
    char_found = False
    for line in tb:
        for char in line:
            if char.get_text().isalnum():
                props['fontname'] = char.fontname
                props['fontsize'] = char.size
                props['x'] = char.x0
                props['y'] = char.y0
                char_found = True
                break
        if char_found:
            break
    for line in reversed(list(tb)):
        for char in reversed(list(line)):
            if char.get_text().isalnum():
                props['endfontname'] = char.fontname
                props['endfontsize'] = char.size
                return props
    return props


def scrape_arxiv(path):
    """ 
	Extract the required data from the arxiv pdf and stores as json
	
	Args: 
		path: URL of pdf to be scraped
    
	Returns: Pair of 
		1. True/False whether something was found at that URL
		2. JSON structured output of analyzed document, or None if not of expected structure
    
	"""
    outstring = ''
    try:
        text = urlopen(Request(path)).read()
        if 'PDF unavailable for' in text.decode(errors='replace'):
            return False, None
        else:
            memory_file = BytesIO(text)
            parser = PDFParser(memory_file)
            document = PDFDocument(parser)
            # Check if the document allows text extraction. If not, return None.
            if not document.is_extractable:
                return True, None
            rsrcmgr = PDFResourceManager()
            laparams = LAParams(line_margin=0.4, detect_vertical=True)
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            # Process each page contained in the document.
            for page in PDFPage.get_pages(memory_file):
                interpreter.process_page(page)
                layout = device.get_result()
                for element in layout:
                    if isinstance(element, LTTextBox):
                        box_props = get_LTText_Props(element)
                        if box_props['text'].strip() == 'References':
                            return True, outstring
                        else:
                            #f = open('box_propyxs',"a", encoding = 'utf-8')
                            #f.write(str(box_props))
                            #f.close()
                            if len(pre_process(box_props['text'])) > 140:
                            #outlist += analyze_box(box_props)
                                outstring = outstring + box_props['text']
            return True, outstring
    except urllib.error.HTTPError:
        # If URL does not exist return None
        return False, None

def pre_process(text):
    
    processed_text =  " ".join(x.lower() for x in text.split()) # lowercase
    processed_text = processed_text.replace('[^\w\s]','')                                 # punctuation removal
    processed_text = " ".join(x for x in processed_text.split() if x not in stop) # stopwords
    processed_text = processed_text.replace('- ','')                               # replace -
    processed_text = re.sub("[\(\[].*?[\)\]]", "", processed_text)           # remove bracktes and the text between that  
    processed_text = re.sub(r"\d+", "", processed_text)                     # remove numbers
    #processed_text = ' '.join(word for word in processed_text.split() if len(word)>3)
    #processed_text = processed_text.replace('-',' ') 
    processed_text = re.sub('[!@#$:,”“•]', '', processed_text)
    processed_text = ' '.join(word for word in processed_text.split() if len(word)>3)
    return processed_text

if __name__ == "__main__":

#########################################################################
# 1 The following block extracts text from pdf Documents (Arxiv)
#########################################################################
    #input_file = 'C:/Users/maya_he/Desktop/CVDD-PyTorch-master/src/datasets/arxivData.json'
    stop_words = set(stopwords.words('english'))
    #regex = re.compile(".*?\((.*?)\)")
    doc_df = pd.DataFrame()
    text_preprocessed = {}
    
		#Reading the Json file 
    '''with open(input_file, 'r') as file:
        json_data = file.read()
    input_list = json.loads(json_data,strict=False)
    i = 4983
    start_time = time.time()
    for elem1 in input_list[4983:]:
        docum = json.loads(elem1['link'].replace("'", '"'))
        tags = json.loads(elem1['tag'].replace("'", '"').replace('None', '"None"'))
        tag_list = []
        for link in docum:
            if 'pdf' in link['href']:
                train_text_link = link['href']
                break
        for tag in tags:
            tag_list.append(tag['term'])
        url_found, pdf_extract = scrape_arxiv(train_text_link)
        if pdf_extract:
            text_preprocessed['title'] = elem1['title']
            text_preprocessed['tag'] = tag_list
            text_preprocessed['abstract'] = elem1['summary']
            text_preprocessed['text'] =  pre_process(pdf_extract)
            text_preprocessed['year'] = elem1['year']
            outfile = 'D:/Thesis/arxiv_full_data/'+str(i)+'.txt'
            json_string = json.dumps(text_preprocessed,ensure_ascii=False)
            print('pdf-',i)
            f = open(outfile,"w", encoding="utf-8")
            f.write(json_string.replace('\\n',' ').replace('  ',' '))
            f.close()
            i = i + 1
            json_string = None
    print('Time taken: ', time.time() - start_time)'''
    url_list = []
    i = 661
    page_url = 'http://export.arxiv.org/oai2?verb=ListRecords&from=2018-01-01&until=2018-10-31&metadataPrefix=arXiv&set=eess'
    raw_response = requests.Session().get(page_url)
    page_html = raw_response.text
    # TODO: fill json_response
    parsed_html = BeautifulSoup(page_html, 'html.parser')
    entries = parsed_html.findAll('metadata')
    for entry in entries[661:]:
        _id = entry.find('id').text
        full_url = 'https://arxiv.org/pdf/'+ str(_id) 
        url_list.append(full_url)
        url_found, pdf_extract = scrape_arxiv(full_url)
        if pdf_extract:
            text_preprocessed['title'] = str(entry.find('title').text)
            text_preprocessed['tag'] = 'eess'
            text_preprocessed['abstract'] = str(entry.find('abstract').text)
            text_preprocessed['text'] =  pre_process(pdf_extract)
            text_preprocessed['year'] = str(entry.find('created').text).split('-')[0]
            outfile = 'D:/Thesis/arxiv_full_data_test_eess/'+str(i)+'.txt'
            json_string = json.dumps(text_preprocessed,ensure_ascii=False)
            print('pdf-',i)
            f = open(outfile,"w", encoding="utf-8")
            f.write(json_string.replace('\\n',' ').replace('  ',' '))
            f.close()
            i = i + 1
            json_string = None