from collections import defaultdict
import json
import csv
import time
from torch import Tensor as T
from torch.utils.data import Dataset

class EmbDataset(Dataset):    
    def __init__(self, docs):
        self.docs = docs
        
    def __getitem__(self, index):
        x = self.docs[index]        
        return x
    
    def __len__(self):
        return len(self.docs)


class EmbCtxDataset(Dataset):
    def __init__(self, embs, ctxs):
        assert len(embs) == len(ctxs)
        
        self.embs = embs
        self.ctxs = ctxs
        
    def __getitem__(self, index):
        return self.embs[index], self.ctxs[index]
    
    def __len__(self):
        return len(self.embs)
    
    
class DataRepo():
    def __init__(
        self,
        docs_path,
        doc_embs_path,
        training_path,
        test_path,
        dataset_name
    ):
        self.docs_path = docs_path
        self.doc_embs_path = doc_embs_path
        self.training_path = training_path
        self.test_path = test_path
        self.dataset_name = dataset_name
        
        #self.docid2text = defaultdict docid -> doc text
        #self.docid2title = defaultdict docid -> doc title
        #self.doc_embs = # tuple (docid2index (dict docid -> index), doc_embs (tensor of all doc embeddings))
        #self.training = # list of all questions
        #self.test = # ([questions], [answers]) a question may have many answers        
        self.load_datasets()
        
    
    def get_docid2title(self):
        return self.docid2title
    
    def get_docid2text(self):
        return self.docid2text
    
    def get_doc_embs(self):
        return self.doc_embs
    
    def get_training_data(self):
        return self.training
    
    def get_test_data(self):
        return self.test
    
    def get_all_docs(self):
        return self.docs
    
    def get_titles_texts_by_ids(self, doc_ids_per_q):
        titles = []
        texts = []
        for doc_ids in doc_ids_per_q:
            titles.append(self.get_titles_by_ids(doc_ids))
            texts.append(self.get_texts_by_ids(doc_ids))
        return titles, texts
    
    def get_titles_by_ids(self, doc_ids):
        return [self.docid2title[doc_id] for doc_id in doc_ids]
    
    def get_texts_by_ids(self, doc_ids):
        return [self.docid2text[doc_id] for doc_id in doc_ids]
    
    def get_real_docid_by_index(self, indexes_per_q):
        return [list(map(lambda index: self.index2docid[index], indexes)) for indexes in indexes_per_q]
    
    def load_datasets(self): 
        if self.docs_path is not None:
            with open(self.docs_path) as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                next(reader, None)  # skip the headers
                docid2title = defaultdict(str)
                docid2text = defaultdict(str)
                ids = []
                all_docs = []
                for row in reader:
                    # file format: doc_id, title, doc_text
                    doc_id = int(row[0])
                    title = row[1]
                    text = row[2]
                    ids.append(doc_id)
                    docid2title[doc_id] = title
                    docid2text[doc_id] = text
                    all_docs.append(text)
            self.index2docid = ids
            self.docid2title = docid2title
            self.docid2text = docid2text
            self.docs = all_docs
        
        if self.doc_embs_path is not None:
            with open(self.doc_embs_path) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader, None)  # skip the headers      
                ids = []
                embeds = []
                for row in reader:
                    ids.append(int(float(row[0])))
                    embeds.append(list(map(float, row[1:])))                
                doc_embs = T(embeds).to(device)
            self.index2docid = ids
            self.doc_embs = doc_embs
            
        if self.training_path is not None: # For training we only need the question
            q_ctxs = ([],[]) # questions, positive contexts
            passage_id_str = 'passage_id' if self.dataset_name == 'nq' else 'psg_id'
            with open(self.training_path) as jsonfile:
                data = json.load(jsonfile)
                for i in range(len(data)):
                    if len(data[str(i)]['positive_ctxs']) == 0:
                        continue # Sometimes there is not positive context, we skip it
                    
                    q_ctxs[0].append(data[str(i)]['question'])
                    trimmed_pctxs = ([],[]) # id, text
                    for pctx in data[str(i)]['positive_ctxs']:
                        trimmed_pctxs[0].append(int(pctx[passage_id_str]))
                        trimmed_pctxs[1].append(pctx['text'])
                    q_ctxs[1].append(trimmed_pctxs)
            self.training = q_ctxs
                  
        if self.test_path is not None: # For testing we need question and answers
            q_as = ([],[])
            with open(self.test_path) as jsonfile:
                data = json.load(jsonfile)
                for i in range(len(data)):
                    q_as[0].append(data[str(i)]['question'])
                    q_as[1].append(data[str(i)]['answers'])
            self.test = q_as