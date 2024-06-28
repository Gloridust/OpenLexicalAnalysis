import aiohttp

class WebpageFetcher:
    def __init__(self):
        self.session = None

    async def fetch(self, url):
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with self.session.get(url, headers=headers) as response:
                html_content = await response.text()
            
            return {'url': url, 'html_content': html_content}
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None