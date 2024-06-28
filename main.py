import asyncio
import aiohttp
from urllib.parse import urlparse
from webpage_fetcher import WebpageFetcher
from content_extractor import ContentExtractor
from data_cleaner import DataCleaner
from result_outputter import ResultOutputter

class OpenLexicalAnalysis:
    def __init__(self):
        self.fetcher = WebpageFetcher()
        self.extractor = ContentExtractor()
        self.cleaner = DataCleaner()
        self.outputter = ResultOutputter()

    async def analyze_url(self, url):
        try:
            domain = urlparse(url).netloc
            html_content = await self.fetcher.fetch(url)
            raw_content = self.extractor.extract(html_content, domain)
            cleaned_content = self.cleaner.clean(raw_content)
            self.outputter.output(cleaned_content)
            return cleaned_content
        except Exception as e:
            print(f"Error analyzing {url}: {str(e)}")
            return None

    async def analyze_urls(self, urls):
        async with aiohttp.ClientSession() as session:
            self.fetcher.session = session
            tasks = [self.analyze_url(url) for url in urls]
            results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

if __name__ == "__main__":
    urls = [
        "https://x.com/gloridust1024/status/1805669956647866807"
    ]
    
    analyzer = OpenLexicalAnalysis()
    results = asyncio.run(analyzer.analyze_urls(urls))
    print(f"Successfully analyzed {len(results)} out of {len(urls)} URLs")