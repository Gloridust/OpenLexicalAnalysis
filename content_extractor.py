from bs4 import BeautifulSoup
import re

class ContentExtractor:
    def extract(self, content, domain):
        if content is None:
            return None

        article = content['article']
        html = content['html']
        soup = BeautifulSoup(html, 'html5lib')

        extracted_content = {
            'url': article.url,
            'title': article.title,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'text': article.text,
            'top_image': article.top_image,
            'images': article.images
        }

        # Domain-specific extraction
        if 'twitter.com' in domain:
            extracted_content.update(self.extract_twitter(soup))
        elif 'news.ycombinator.com' in domain:
            extracted_content.update(self.extract_hacker_news(soup))

        return extracted_content

    def extract_twitter(self, soup):
        tweet = soup.find('div', {'data-testid': 'tweetText'})
        tweet_text = tweet.get_text() if tweet else ""
        user = soup.find('div', {'data-testid': 'User-Name'})
        username = user.get_text() if user else ""
        return {'tweet_text': tweet_text, 'username': username}

    def extract_hacker_news(self, soup):
        title = soup.find('td', class_='title')
        title_text = title.a.get_text() if title and title.a else ""
        score = soup.find('span', class_='score')
        score_text = score.get_text() if score else ""
        return {'hn_title': title_text, 'hn_score': score_text}