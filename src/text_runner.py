try:
    from src.artificial_analysis.catalog import get_models
    from src.artificial_analysis.resolver import resolve
    from src.routing_types import SelectionPolicy
    from src.text_model_selector import select_text_model
except ImportError:
    from artificial_analysis.catalog import get_models
    from artificial_analysis.resolver import resolve
    from routing_types import SelectionPolicy
    from text_model_selector import select_text_model


policy = SelectionPolicy.NOPREF
query = "What are the key differences between supervised and unsupervised learning in machine learning, and can you provide examples of algorithms used in each category?"
selection = select_text_model(query, selection_policy=policy)
candidates = get_models()  # Load the model candidates
result = resolve(selection, candidates=candidates, big3_only=True)
print(result.to_json(indent=2))
# print(result.get_name())
print(result.get_name_and_co())
