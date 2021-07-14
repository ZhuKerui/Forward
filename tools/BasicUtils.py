from typing import Iterable, List
import numpy as np
from heapq import nlargest
import multiprocessing
from threading import Thread
from math import ceil
import subprocess
import wikipedia
import json
import csv
import re

def ugly_normalize(vecs:np.ndarray):
    normalizers = np.sqrt((vecs * vecs).sum(axis=1))
    normalizers[normalizers==0]=1
    return (vecs.T / normalizers).T

def simple_normalize(vec:np.ndarray):
    normalizer = np.sqrt(np.matmul(vec, vec))
    if normalizer == 0:
        normalizer = 1
    return vec / normalizer

def ntopidx(n, score:Iterable):
    s = nlargest(n, zip(np.arange(len(score)), score), key = lambda x: x[1])
    return [item[0] for item in s]

def my_read(file_name:str):
    return open(file_name, 'r').read().split('\n')

def my_write(file_name:str, content:List[str]):
    with open(file_name, 'w') as f_out:
        f_out.write('\n'.join(content))

def my_json_read(file_name:str):
    return json.load(open(file_name, 'r'))

def my_csv_read(file_name:str, delimiter:str=','):
    return csv.reader(open(file_name, 'r'), delimiter=delimiter)
    
class MultiProcessing:
    def __line_process_wrapper(self, temp_obj, input_list:list, output_list):
        for line in input_list:
            temp_obj.line_operation(line)
        output_list.append(temp_obj)

    def run(self, obj_generator, input_list:list, thread_num:int=1, post_operation=None):
        if thread_num <= 0:
            return
        line_count = len(input_list)
        print('Number of lines is %d' % (line_count))
        unit_lines = ceil(line_count / thread_num)

        manager = multiprocessing.Manager()
        result = manager.list()
        processing = []
        for i in range(thread_num):
            processing.append(multiprocessing.Process(target=self.__line_process_wrapper, args=(obj_generator(), input_list[unit_lines*i : unit_lines*(i+1)], result)))

        for i in range(thread_num):
            processing[i].start()

        for i in range(thread_num):
            processing[i].join()
        if post_operation:
            return post_operation(result)
        else:
            ret = []
            for sub in result:
                ret += sub.line_record
            return ret

class MultiThreading:
    def __line_process_wrapper(self, line_operation, input_list:list, output_list:list):
        for line in input_list:
            result = line_operation(line)
            if result is not None:
                output_list.append(result)

    def run(self, line_operation, input_list:list, thread_num:int=1):
        if thread_num <= 0:
            return
        line_count = len(input_list)
        print('Number of lines is %d' % (line_count))
        unit_lines = ceil(line_count / thread_num)

        result = [[] for i in range(thread_num)]
        threads = [Thread(target=self.__line_process_wrapper, args=(line_operation, input_list[unit_lines*i : unit_lines*(i+1)], result[i])) for i in range(thread_num)]

        for i in range(thread_num):
            threads[i].setDaemon(True)
            threads[i].start()

        for i in range(thread_num):
            threads[i].join()
        sub_result = ['\n'.join(sub) for sub in result]
        return '\n'.join(sub_result)

def my_email(title:str, message:str, email:str):
    cmd = 'echo "%s" | mail -s "%s" %s' % (message, title, email)
    subprocess.Popen(cmd, shell=True)

# Collect wiki summary and process the text
def get_wiki_page_from_kw(line:str):
    try:
        page = wikipedia.page(line)
        return page if line.lower() == page.title.lower() else None
    except:
        return None

def get_wiki_summary_from_kw(line:str):
    page = get_wiki_page_from_kw(line)
    return ' '.join(page.summary.split()) if page is not None else None

def get_wiki_context_from_kw(line:str):
    page = get_wiki_page_from_kw(line)
    return ' '.join(page.content.split()) if page is not None else None

def clean_sent(sent:str):
    while re.search(r'{[^{}]*}', sent):
        sent = re.sub(r'{[^{}]*}', '', sent)
    while re.search(r'\([^()]*\)', sent):
        sent = re.sub(r'\([^()]*\)', '', sent)
    return re.sub(r'[^A-Za-z0-9,.\s-]', '', sent.strip())