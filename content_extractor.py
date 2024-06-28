import json
import re
from bs4 import BeautifulSoup

class ContentExtractor:
    def extract(self, content, url):
        if content is None:
            return None

        if 'twitter.com' in url or 'x.com' in url:
            return self.extract_twitter(content['html_content'], url)
        else:
            return self.extract_general(content['html_content'], url)

    def extract_twitter(self, html_content, url):
        tweet_data = self.extract_tweet_data(html_content)
        if not tweet_data:
            return {"error": "Could not extract tweet data", "url": url}

        return {
            'url': url,
            'author': tweet_data.get('user', {}).get('name'),
            'username': tweet_data.get('user', {}).get('screen_name'),
            'text': tweet_data.get('text', ''),
            'date': tweet_data.get('created_at'),
            'retweet_count': tweet_data.get('retweet_count'),
            'favorite_count': tweet_data.get('favorite_count'),
            'images': [media['media_url_https'] for media in tweet_data.get('entities', {}).get('media', []) if media['type'] == 'photo']
        }

    def extract_tweet_data(self, html_content):
        match = re.search(r'<script type="application/ld\+json".*?>(.*?)</script>', html_content, re.DOTALL)
        if match:
            try:
                json_data = json.loads(match.group(1))
                return json_data
            except json.JSONDecodeError:
                return None
        return None

    def extract_general(self, html_content, url):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        title = soup.find('title')
        title = title.text if title else "No title found"

        author = soup.find('meta', {'name': 'author'})
        author = author['content'] if author else "No author found"

        date = soup.find('meta', {'property': 'article:published_time'})
        date = date['content'] if date else "No date found"

        content = soup.find('article') or soup.find('div', class_='content')
        content = content.get_text(strip=True) if content else "No content found"

        images = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs]

        return {
            'url': url,
            'title': title,
            'author': author,
            'date': date,
            'content': content[:500] + "..." if len(content) > 500 else content,  # Truncate long content
            'images': images[:5]  # Limit to first 5 images
        }