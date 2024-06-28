import aiohttp
from newspaper import Article

class WebpageFetcher:
    def __init__(self):
        self.session = None

    async def fetch(self, url):
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(url) as response:
                html_content = await response.text()
            
            article = Article(url)
            article.set_html(html_content)
            article.parse()
            
            return {
                'html': html_content,
                'article': article
            }
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None