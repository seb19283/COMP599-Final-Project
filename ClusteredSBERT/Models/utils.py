from collections import defaultdict
import re
import string
import math
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from repository import EmbCtxDataset, EmbDataset

class Evaluator():
    def remove_articles(self, text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(self, text):
        return " ".join(text.split())


    def remove_punc(self, text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)


    def lower(self, text):
        return text.lower()

    def normalize_text(self, text):
        for normalization_function in [self.lower, self.remove_articles, self.remove_punc, self.white_space_fix]:
            text = normalization_function(text)
        return text
    
    def exact_match(self, predictions, answers):
        assert len(predictions) == len(answers)
        total = len(predictions)
        correct = 0
        for i in range(total):
            if self.normalize_text(predictions[i]) == self.normalize_text(answers[i][0]):
                correct += 1
        return correct/total   

    
def print_results(self, model_name, nb_questions, em_score, total_doc_retrieval_runtime):
    print(f"Results for {model_name}")
    print(f"Number of questions treated: {nb_questions}")
    print(f"Number of correct answers: {em_score*nb_questions}")
    print(f"EM score for clustered retriever: {em_score*100}%")
    print(f"Total runtime of document retrieval with clustering: {total_doc_retrieval_runtime}s")
    print(f"Average runtime of document retrieval per questions: {total_doc_retrieval_runtime/nb_questions}s")


def evaluate(self, retriever, k, batch_size, data_repo, model_name):
    questions, _ = data_repo.get_test_data()
    
    model_builder = ModelBuilder().set_data_repo(data_repo).set_retriever(retriever)            
    full_model = model_builder.build_full_model()

    full_model.eval()
    questions, answers = full_model.get_repo().get_test_data()
    
    questionsDataset = EmbDataset(questions)
    dataloader = DataLoader(questionsDataset, batch_size=batch_size, shuffle=False)
    
    i = 1
    predictions = []
    total_doc_retrieval_runtime = 0.0
    for qs in tqdm(dataloader):
        if i == 1 or i % 50 == 0:
            print(f"Starting batch #{i}")
        output, doc_retrieval_runtime = full_model(qs, k)
        predictions.extend(output)
        total_doc_retrieval_runtime += doc_retrieval_runtime
        i += 1
    
    em_score = self.exact_match(predictions, answers) 
    self.print_results(model_name, len(questions), em_score, total_doc_retrieval_runtime)
    
    
class ModelBuilder():
    def __init__(self):
        self.retriever = None
        self.reader = CustomDPRReader() # It will always be this reader
        self.data_repo = None    
        
    def set_data_repo(self, data_repo: DataRepo):
        assert self.data_repo is None
        self.data_repo = data_repo
        return self
        
    def set_retriever(self, retriever: CustomDPRRetriever):
        assert self.retriever is None
        self.retriever = retriever
        return self
        
    def build_full_model(self):
        assert self.data_repo is not None
        assert self.retriever is not None
        return FullOpenQAModel(self.retriever, self.reader, self.data_repo)


class Trainer():
    def train_step(
        self,
        model: SBERTRetriever,
        questions,
        positive_ctxs,
        pctx_per_q
    ):
        clustering = model.get_corpora()
        optimizer = model.get_optimizer()

        optimizer.zero_grad()
        
        # q_embeddings shape = (b, 768)
        q_embeddings = model.get_q_embeds(questions) # We don't train on the result of question embedding
        
        # centroids shape = (b, 768)
        centroids = clustering.get_centroids_for_questions(q_embeddings)
        
        flat_pctxs = [positive_ctx for positive_ctxs_for_q in positive_ctxs for positive_ctx in positive_ctxs_for_q]
        docs_tokenized = model.tokenize_ctx(flat_pctxs).to(device)        
        # docs_embeddings shape = (b*pctx_per_q, 768)
        docs_embeddings = model(docs_tokenized, fix_model=False) # We do train on the result of docs embedding
        
        # reshape centroids to match docs_embeddings
        centroids = centroids.unsqueeze(1).repeat(1, pctx_per_q, 1)
        flat_centroids = centroids.view(len(questions)*pctx_per_q, 768)

        loss = model.criterion(flat_centroids, docs_embeddings)
        loss.backward()
        optimizer.step()

        return loss.item()


    def train_retriever(
        self,
        model : SBERTRetriever,
        data_repo : DataRepo,
        batch_size
    ):
        q_train, pctx_train = data_repo.get_training_data()

        model.train()
        total_loss = 0.0

        # Store train data by number of positive contexts, so we can train in batches of same size
        train_data_by_nb_pctxs = defaultdict(list)
        for i in range(len(q_train)):
            q = q_train[i]
            pctx = pctx_train[i]
            pctx_per_q = len(pctx[0])
            if pctx_per_q > 4: # We limit 4 ctxs per question at a time to limit the amount of memory used
                nb_batches = math.floor(pctx_per_q/4)
                for j in range(nb_batches):
                    batch_pctx = (pctx[0][j*4:(j*4)+4], pctx[1][j*4:(j*4)+4])
                    train_data_by_nb_pctxs[4].append((q, batch_pctx))
                if (pctx_per_q % 4) != 0:
                    batch_pctx = (pctx[0][nb_batches*4:], pctx[1][nb_batches*4:])
                    train_data_by_nb_pctxs[len(batch_pctx[1])].append((q, batch_pctx))
            else:
                train_data_by_nb_pctxs[pctx_per_q].append((q, pctx))
        
        batches = 0
        for pctx_per_q, train_data in train_data_by_nb_pctxs.items():
            qs, docids_pctx = list(map(list, zip(*train_data)))
            print(f"Training with questions having {pctx_per_q} positive context passages")
            print(f"Total of {len(qs)} questions with {pctx_per_q} positive context passages")
            _, pctx = list(map(list, zip(*docids_pctx)))
            if len(qs) > batch_size:
                q_pctx_dataset = EmbCtxDataset(qs, pctx)
                loader = DataLoader(q_pctx_dataset, batch_size=batch_size, shuffle=False)
                for qs_batch, pctx_batch in tqdm(loader):
                    total_loss += self.train_step(model, list(qs_batch), list(map(list, pctx_batch)), pctx_per_q)
                    batches += 1
            else:
                total_loss += self.train_step(model, list(qs), list(map(list, pctx)), pctx_per_q)
                batches += 1

        return (total_loss / batches)


def create_doc_embeddings(retriever, data_repo, batch_size):
    print("Starting to update document embeddings")
    t0 = time.time()
    docs = data_repo.get_all_docs()
    docs_dataset = EmbDataset(docs)
    loader = DataLoader(docs_dataset, batch_size=batch_size, shuffle=False)
    print(f"Total nb of docs: {len(docs)}")
    embs_list = []
    i = 1
    for docs in tqdm(loader):
        doc_embeds = retriever.get_doc_embeds(docs)
        embs_list.append(doc_embeds)
        i += 1
    all_embeddings = torch.cat(embs_list, dim=0)
    print(f"Updating document embeddings took {time.time()-t0}s")
    return all_embeddings