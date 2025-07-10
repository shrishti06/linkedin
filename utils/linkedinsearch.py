from serpapi import GoogleSearch

def search_linkedin(query, serpapi_api_key, num_results=5):
    params = {
        "engine": "google",
        "q": f"{query} site:linkedin.com/in/",
        "num": num_results,
        "api_key": serpapi_api_key,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    profiles = []

    if "organic_results" in results:
        for result in results["organic_results"]:
            profiles.append({
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet", "")
            })
    return profiles