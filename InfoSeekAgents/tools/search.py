from itertools import islice
import json
import os
import random
import traceback
import time
import requests
from bs4 import BeautifulSoup as soup
from duckduckgo_search import DDGS
from serpapi import GoogleSearch

from InfoSeekAgents.tools.base import BaseResult, BaseTool
from InfoSeekAgents.utils.selenium_utils import get_pagesource_with_selenium
from InfoSeekAgents.config import Config
from InfoSeekAgents.tools.search_engines import Google, Bing, Yahoo


class SearchResult(BaseResult):
    @property
    def answer(self):
        if not self.json_data:
            return ""
        else:
            rst = ""
            for item in self.json_data:
                rst += f'title: {item["title"]}\nbody: {item["body"]}\nurl: {item["href"]}\n'
            return rst.strip()

    @property
    def answer_md(self):
        if not self.json_data:
            return ""
        else:
            return "\n" + "\n".join([f'{idx + 1}. <a href="{item["href"]}" target="_blank"><b>{item["title"]}</b></a>' + " | " + item["body"] 
                for idx, item in enumerate(self.json_data)])

    @property
    def answer_full(self):
        if not self.json_data:
            return ""
        else:
            return json.dumps(self.json_data, ensure_ascii=False, indent=2)


class SearchTool(BaseTool):
    """
    Perform an internet search.

    Args:
        text (str): Search query.

    Returns:
        str: Multiple webpage links along with brief descriptions.
    """
    name = "web_search"
    zh_name = "网页搜索"
    description = "Web Search:\"web_search\",args:\"text\":\"<search>\""
    tips = ""
    
    def __init__(self, cfg=None, max_search_nums=5, lang="wt-wt", max_retry_times=5, *args, **kwargs):
        self.cfg = cfg if cfg else Config()
        self.max_search_nums = max_search_nums
        self.max_retry_times = max_retry_times
        self.lang = lang
        self.driver = None
        self.driver_cnt = 0
        self.search_type = cfg.search_type
        print(f'--------------Search Type:{self.search_type}---------------')

    def set_driver(self, driver):
        self.driver_cnt += 1
        if self.driver_cnt >= 20:
            self.driver_cnt = 0
            self.driver = None
        else:
            self.driver = driver
            
    def get_results_by_selenium(self, keyword):
        url = f"https://duckduckgo.com/?q={keyword}&t=h_&ia=web"
        driver, page_source = get_pagesource_with_selenium(url, "chrome", self.driver)
        self.set_driver(driver)
        page_soup = soup(page_source, "html.parser")
        articles = page_soup.find_all("article")

        results = list()
        for idx, article in enumerate(articles):
            if idx >= self.max_search_nums:
                break
            href = article.find_all("a")[1]['href']
            title = article.find(class_="EKtkFWMYpwzMKOYr0GYm kY2IgmnCmOGjharHErah").text  #
            body = article.find(class_="OgdwYG6KE2qthn9XQWFC").text
            results.append({
                "title": title,
                "href": href,
                "body": body
            })
        print('Searching DuckDuckGo by crawler')
        return results

    def get_search_results_by_crawler(self, keyword):
        if self.search_type.lower() == 'ddg':
            return self.get_results_by_selenium(keyword)
        proxy = os.environ.get('http_proxy')
        if self.search_type.lower() == 'google':
            search_engine = Google(proxy=proxy)
        elif self.search_type.lower() == 'bing':
            search_engine = Bing(proxy=proxy)
        elif self.search_type.lower() == 'yahoo':
            search_engine = Yahoo(proxy=proxy)
        else:
            raise NotImplementedError
        search_results = search_engine.search(keyword, pages=1)
        time.sleep(random.uniform(4, 6))  # 适当延时，避免请求过快

        results = list()
        for idx in range(len(search_results)):
            if idx >= self.max_search_nums:
                break
            results.append({
                "title": search_results[idx]['title'],
                "href": search_results[idx]['link'],
                "body": search_results[idx]['text']
            })
        return results

    def get_results_by_ddg(self, keyword):
        search_results = list()
        my_proxy = os.getenv("http_proxy")
        with DDGS(proxy=my_proxy, timeout=20) as ddgs:
            ddgs_gen = ddgs.text(keyword, timelimit='y')
            for r in islice(ddgs_gen, self.max_search_nums):
                search_results.append(r)
        return search_results

    def get_results_by_bing_serp(self, keyword):
        serp_key = os.environ.get("SERP_API_KEY", '')
        if not serp_key:
            return None

        params = {
            "engine": "bing",
            "q": keyword,
            "cc": "US",
            "api_key": serp_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        search_results = list()
        for res in results["organic_results"]:
            search_results.append({
                "title": res['title'],
                "href": res['link'],
                "body": res.get("snippet", '')
            })
        return search_results

    def get_results_by_yahoo_serp(self, keyword):
        serp_key = os.environ.get("SERP_API_KEY", '')
        if not serp_key:
            return None

        params = {
            "engine": "yahoo",
            "p": keyword,
            "api_key": serp_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        search_results = list()
        for res in results["organic_results"]:
            search_results.append({
                "title": res['title'],
                "href": res['link'],
                "body": res.get("snippet", '')
            })
        return search_results

    def get_results_by_google_serper(self, keyword):
        serper_key = os.environ.get('SERPER_API_KEY', '')
        if not serper_key:
            return None
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": keyword
        })
        headers = {
            'X-API-KEY': serper_key,
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        search_results = list()
        for res in json.loads(response.text)["organic"]:
            search_results.append({
                "title": res['title'],
                "href": res['link'],
                "body": res["snippet"]
            })
        return search_results

    def get_result_by_google_api(self, keyword):
        google_api_key = os.environ.get('GOOGLE_API_KEY', '')
        google_cse_id = os.environ.get('GOOGLE_CSE_ID', '')
        if not google_api_key or not google_cse_id:
            return None

        data = {
            "key": os.environ.get('GOOGLE_API_KEY'),
            "cx": os.environ.get('GOOGLE_CSE_ID'),
            "q": keyword
        }
        url = "https://www.googleapis.com/customsearch/v1"
        results = requests.get(url, params=data).json()
        results = results.get("items", [])
        search_results = list()
        for result in results:
            search_results.append({
                "title": result['title'],
                "href": result['link'],
                "body": result["snippet"]
            })
        return search_results

    def get_results_by_api(self, keyword):
        if self.search_type.lower() == 'ddg':
            results = self.get_results_by_ddg(keyword)
        elif self.search_type.lower() == 'google':
            results = self.get_results_by_google_serper(keyword)
        elif self.search_type.lower() == 'bing':
            results = self.get_results_by_bing_serp(keyword)
        elif self.search_type.lower() == 'yahoo':
            results = self.get_results_by_yahoo_serp(keyword)
        else:
            results = None
        if results:
            return results, False
        else:
            return results, True

    def _retry_search_result(self, keyword, counter=0):
        counter += 1
        if counter > self.max_retry_times:
            print("Search failed after %d retrying" % counter, ', return search failed')
            return [{
                "title": "Search Failed",
                "href": "",
                "body": ""
            }]
        try:
            search_results = None
            error_message = ''
            try:
                search_results, use_selenium = self.get_results_by_api(keyword)
            except:
                error_message = traceback.format_exc()
                print(error_message)
                use_selenium = True
            if not search_results and (counter >= 2 or 'Ratelimit' in error_message):
                use_selenium = True
            if use_selenium:
                search_results = self.get_search_results_by_crawler(keyword)
            if search_results and (
                    "Google Patents" in search_results[0]["body"] or "patent" in search_results[0]["href"]):
                search_results = list()
            if not search_results:
                time.sleep(random.uniform(1, 5))  # 适当延时，避免请求过快
                return self._retry_search_result(keyword, counter)
            print('Num of search results', len(search_results))
            return search_results
        except:
            print(traceback.format_exc())
            print("Retry search...")
            time.sleep(random.uniform(1, 5))  # 适当延时，避免请求过快
            return self._retry_search_result(keyword, counter)

    def __call__(self, text):
        return SearchResult(self._retry_search_result(text))
