import sys
from Models.repository import DataRepo
from Models.utils import Evaluator, Trainer, create_doc_embeddings
from Models.models import SBERTRetriever

nq_paths = {
    docs_path: 'Data\combined-passages.tsv',
    doc_embs_path_sbert: 'Data\passagesEmbSBERT.csv',
    training_path: 'Data\training_data.json',
    test_path: 'Data\retriever-supervision.json',
    dataset_name: 'nq'
}

webq_paths = {
    docs_path: 'Data\webq-combined-passages.tsv',
    doc_embs_path: None,
    training_path: 'Data\webq-train-questions.json',
    test_path: 'Data\webq-dev-questions.json',
    dataset_name: 'webq'
}

triviaqa_paths = {
    docs_path: 'Data\trivia-combined-passages.tsv',
    doc_embs_path: 'Data\passagesEmbTriviaQASBERT.csv',
    training_path: 'Data\trivia-train-questions.json',
    test_path: 'Data\trivia-dev-questions.json',
    dataset_name: 'triviaqa'
}

if __name__ == "__main__":
    nb_clusters = 100
    k = 5
    batch_size = 4
    
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
    
    print("Start training retriever")
    trainer = Trainer()
    loss = trainer.train_retriever(retriever, data_repo, batch_size)
    print(f"Loss after training: {loss}")
    
    print("REARRANGING CLUSTERS")
    all_embeddings = create_doc_embeddings(clustered_sbert_retriever, data_repo, batch_size)   
    print("Starting clustering update")
    t1 = time.time()
    retriever.corpora.update_clustering(all_embeddings, batch_size)
    print(f"Updating clusters took {time.time()-t1}s")

    print("STARTING EVALUATION FOR CLUSTERED SBERT - after training - rearranging clusters")
    evaluator.evaluate(clustered_sbert_retriever, k, batch_size, data_repo, 'ClusteredSBERT')
    
    print("RECREATING CLUSTERS")
    clustered_sbert_retriever.corpora.create_clustering()
    
    print("STARTING EVALUATION FOR CLUSTERED SBERT - after training - reclustering")
    evaluator.evaluate(clustered_sbert_retriever, k, batch_size, data_repo, 'ClusteredSBERT')