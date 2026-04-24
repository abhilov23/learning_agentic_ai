from langchain_community.utilities import DuckDuckGoSearchAPIWrapper 
from typing import List


# this will fetch the top n results for the query and print the links
def web_search(web_query:str, num_results:int) -> List[str]:
    return [r["link"]
            for r in DuckDuckGoSearchAPIWrapper().results(web_query, num_results)]

