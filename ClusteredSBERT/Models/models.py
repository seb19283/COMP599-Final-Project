from transformers import AutoModel, AutoTokenizer, DPRReader, DPRReaderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import torch.nn.functional as F
import time
from torch import Tensor as T
from torch import nn
from corpora import Corpora
from repository import DataRepo

class CustomDPRRetriever(nn.Module):
    def __init__(
        self,
        ctx_tokenizer,
        ctx_model,
        q_tokenizer,
        q_model,
        criterion,
        pooling,
        corpora: Corpora
    ):
        super().__init__()
        self.ctx_tokenizer = ctx_tokenizer
        self.ctx_model = ctx_model
        self.q_tokenizer = q_tokenizer if q_tokenizer is not None else ctx_tokenizer
        self.q_model = q_model if q_model is not None else ctx_model
        self.criterion = criterion
        self.pooling = pooling
        self.corpora = corpora
        self.optimizer = self.set_optimizer()
    
    
    def get_corpora(self):
        return self.corpora
    
    def forward(
        self,
        tokenized: T,
        is_question = True,
        fix_model = False
    ):
        model = self.q_model if is_question else self.ctx_model
            
        if fix_model:
            with torch.no_grad():
                return self.get_embedding(tokenized, model)
        else:
            return self.get_embedding(tokenized, model)
    
    def get_doc_embeds(
        self,
        docs
    ):        
        tokenized = self.tokenize_ctx(docs)
        with torch.no_grad(): # Fix model when we are retrieving top k documents because this will only happen during evaluation
            ctx_embeds = self.get_embedding(tokenized, self.ctx_model)
        return ctx_embeds
        
    
    def get_q_embeds(
        self,
        questions
    ):
        tokenized = self.tokenize_question(questions)
        with torch.no_grad(): # Fix model when we are retrieving top k documents because this will only happen during evaluation
            q_embeds = self.get_embedding(tokenized, self.q_model)
        return q_embeds
    
            
    def get_top_k_docs(
        self,
        q_embeds,
        k
    ):
        return self.corpora.get_top_k_for_q_embed(q_embeds, k)
        
        
    def get_optimizer(self):
        return self.optimizer
    
    def set_optimizer(
        self,
        adam_eps = 1e-8,
        weight_decay = 0.0,
        learning_rate = 1e-5 # try 1e-5, 1e-6, 1e-7
    ):        
        return torch.optim.Adam(self.parameters(), eps=adam_eps, weight_decay=weight_decay, lr=learning_rate)
            
        
    def tokenize_question(
        self,
        questions,
        truncation: bool = True,
        padding: bool = True,        
    ):
        return self.q_tokenizer(questions, truncation=truncation, padding=padding, return_tensors='pt').to(device)
    
        
    def tokenize_ctx(
        self,
        docs,
        truncation: bool = True,
        padding: bool = True,        
    ):
        return self.ctx_tokenizer(docs, truncation=truncation, padding=padding, return_tensors='pt').to(device)
        
        
    def get_embedding(
        self,
        tokenized: T,
        model: nn.Module
    ):
        model_output = model(**tokenized, return_dict=True)
        pooled_output = self.pooling(model_output, tokenized['attention_mask'])
        normalized_output = F.normalize(pooled_output, p=2, dim=1)
        return normalized_output    
        
        
class VanillaDPRRetriever(CustomDPRRetriever):
    def __init__(
        self,
        corpora
    ):
        super().__init__(
            DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
            DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device),
            DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
            DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device),
            None, # Will never use for training anyway
            self.pooling,
            corpora
        )
    
    def pooling(self, model_output, attention_mask):
        return model_output.pooler_output

    def forward(
        self,
        tokenized: T,
        is_question: bool
    ):
        return super().forward(tokenized, is_question=is_question, fix_model=True)
    
        
class SBERTRetriever(CustomDPRRetriever):
    def __init__(
        self,
        corpora: Corpora
    ):
        super().__init__(
            AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2'), 
            AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device), 
            None, 
            None,
            self.criterion, 
            self.mean_pooling, 
            corpora
        )
        
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)        
       
    
    def criterion(
        self,
        centroids_embeds: T, 
        docs_embeds: T
    ):
        dot_product_scores = torch.matmul(centroids_embeds, docs_embeds.T)
        softmax_scores = F.log_softmax(dot_product_scores, dim=1)
        labels = torch.arange(len(docs_embeds)).long().to(device)
        # We want the doc embeds that answer the question to be as close to the chosen cluster's centroid as possible
        loss = F.nll_loss(softmax_scores, labels, reduction="mean")        
        return loss     
        
        
    def forward(
        self,
        tokenized: T,
        fix_model: bool = False
    ):
        return super().forward(tokenized, is_question=False)
        

class CustomDPRReader(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        self.model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to(device)
        
    def tokenize(
        self,
        questions,
        texts,
        titles,
        k
    ):     
        return self.tokenizer(
            questions=questions,
            titles=titles,
            texts=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
    def forward(
        self,
        tokenized_inputs, #output of the tokenizer
    ):
        # Max tokens for this model is 512, but tokenizer may output more than that... We trim it out if it happens (should be extremely rare)
        if tokenized_inputs['input_ids'].shape[1] > 512:
            tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'][..., 0:512]
            tokenized_inputs['attention_mask'] = tokenized_inputs['input_ids'][..., 0:512]
        return self.model(**tokenized_inputs)
    
    def decode_best_spans(
        self,
        encoded_inputs,
        outputs,
        num_spans=1,
        num_spans_per_passage=2,
        max_answer_length=25
    ):
        return self.tokenizer.decode_best_spans(
            encoded_inputs,
            outputs,
            num_spans=num_spans,
            num_spans_per_passage=num_spans_per_passage,
            max_answer_length=max_answer_length
        )
        
    

class FullOpenQAModel(nn.Module):
    def __init__(
        self,
        retriever_model: CustomDPRRetriever,
        reader_model: CustomDPRReader,
        repository: DataRepo
    ):
        super().__init__()
        self.retriever = retriever_model
        self.reader = reader_model
        self.repo = repository
        
    def get_retriever(self):
        return self.retriever
    
    def get_reader(self):
        return self.reader
    
    def get_repo(self):
        return self.repo
    
    def forward(
        self,
        questions,
        k
    ):
        q_embeds = self.retriever.get_q_embeds(questions)
        t0 = time.time()
        topk_indexes = self.retriever.get_top_k_docs(q_embeds, k)
        docs_retrieving_time = time.time()-t0
        
        topk_docids = self.repo.get_real_docid_by_index(topk_indexes)
        titles, texts = self.repo.get_titles_texts_by_ids(topk_docids)
        predicted_answers = []
        for i in range(len(questions)):
            question = questions[i]
            text = texts[i]
            title = titles[i]
            
            reader_encoded_inputs = self.reader.tokenize(question, text, title, k)
            outputs = self.reader(reader_encoded_inputs)
            predicted_span = self.reader.decode_best_spans(reader_encoded_inputs,outputs)[0].text
            predicted_answers.append(predicted_span)
        return predicted_answers, docs_retrieving_time  
    