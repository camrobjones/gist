from celery import shared_task

from gist import synspaces

# explore bind=True?
@shared_task()
def get_synspace(queries, model_name="bert", embeddings_key="bert_l11_wordnet", n=20,
    dimred="pca", metric="cosine"):
	"""Task that calls synspaces.get_synspace."""
	return synspaces.get_synspace(
		queries=queries,
		model_name=model_name,
		embeddings_key=embeddings_key,
		n=n,
		dimred=dimred,
		metric=metric)