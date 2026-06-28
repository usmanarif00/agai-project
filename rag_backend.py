def run_pipeline(user_query):
    state = {
        "original_query": user_query,
        "optimized_query": "",
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "hallucination": ""
    }

    result = app.invoke(state)
    return result["generation"]
