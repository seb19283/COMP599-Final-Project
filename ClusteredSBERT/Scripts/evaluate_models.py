import sys
from Models.models import VanillaDPRRetriever, SBERTRetriever
from Models.repository import DataRepo
from Models.utils import Evaluator, create_doc_embeddings

nq_paths = {
    docs_path: 'Data\combined-passages.tsv',
    doc_embs_path_sbert: 'Data\passagesEmbSBERT.csv',
    doc_embs_path_dpr: 'Data\passagesEmbDPR.csv',
    training_path: 'Data\training_data.json',
    test_path: 'Data\retriever-supervision.json',
    dataset_name: 'nq'
}

webq_paths = {
    docs_path: 'Data\webq-combined-passages.tsv',
    doc_embs_path: None,
    doc_embs_path_dpr: None,    
    training_path: 'Data\webq-train-questions.json',
    test_path: 'Data\webq-dev-questions.json',
    dataset_name: 'webq'
}

triviaqa_paths = {
    docs_path: 'Data\trivia-combined-passages.tsv',
    doc_embs_path: 'Data\passagesEmbTriviaQASBERT.csv',
    doc_embs_path_dpr: 'Data\passagesEmbTriviaQADPR.csv',
    training_path: 'Data\trivia-train-questions.json',
    test_path: 'Data\trivia-dev-questions.json',
    dataset_name: 'triviaqa'
}

if __name__ == "__main__":
    nb_clusters = 100
    k = 5
    batch_size = 16
    
    dataset = sys.argv[1]

    data_repo_paths = {}
    if dataset == 'nq':
        data_repo_paths = nq_paths
    elif dataset == 'webq':
        data_repo_paths = webq_paths
    elif dataset == 'triviaqa':
        data_repo_paths = triviaqa_paths

    print("Start uploading SBERT repo")
    data_repo_sbert = DataRepo(
        docs_path=data_repo_paths['docs_path'], 
        doc_embs_path=data_repo_paths['doc_embs_path_sbert'],
        training_path=data_repo_paths['training_path'], 
        test_path=data_repo_paths['test_path'],
        dataset_name=data_repo_paths['dataset_name']
    )
    sbert_doc_embeds = data_repo.get_doc_embs()    
    
    evaluator = Evaluator()

    print("STARTING EVALUATION FOR CLUSTERED SBERT")
    clustered_sbert_retriever = SBERTRetriever(EvidenceClustering(sbert_doc_embeds, nb_clusters))
    if sbert_doc_embeds is None:
        sbert_doc_embeds = create_doc_embeddings(clustered_sbert_retriever, data_repo, batch_size)
        clustered_sbert_retriever.corpora.update_doc_embeddings(sbert_doc_embeds)
        clustered_sbert_retriever.corpora.create_clustering()
    evaluator.evaluate(clustered_sbert_retriever, k, batch_size, data_repo, 'ClusteredSBERT')
    

    print("STARTING EVALUATION FOR BRUTEFORCE SBERT")
    bruteforce_sbert_retriever = SBERTRetriever(Bruteforce(sbert_doc_embeds))
    evaluator.evaluate(bruteforce_sbert_retriever, k, batch_size, data_repo, 'BruteforceSBERT')
    

    print("Start uploading DPR repo")
    data_repo_dpr = DataRepo(
        docs_path=data_repo_paths['docs_path'], 
        doc_embs_path=data_repo_paths['doc_embs_path_dpr'],
        training_path=data_repo_paths['training_path'], 
        test_path=data_repo_paths['test_path'],
        dataset_name=data_repo_paths['dataset_name']
    )
    dpr_doc_embeds = data_repo.get_doc_embs()

    print("STARTING EVALUATION FOR BRUTEFORCE DPR")
    bruteforce_dpr_retriever = VanillaDPRRetriever(Bruteforce(dpr_doc_embeds))
    if dpr_doc_embeds is None:
        dpr_doc_embeds = create_doc_embeddings(bruteforce_dpr_retriever, data_repo, batch_size)
        bruteforce_dpr_retriever.corpora.update_doc_embeddings(dpr_doc_embeds)
    evaluator.evaluate(bruteforce_dpr_retriever, k, batch_size, data_repo, 'BruteforceDPR')
    
    