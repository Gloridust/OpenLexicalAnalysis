
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class OpenLexicalAnalysis:
    def __init__(self):
        self.title = ""
        self.author = ""
        self.date = ""
        self.paragraphs = []
        self.images = []

    def analyze(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        self.extract_title(soup)
        self.extract_author(soup)
        self.extract_date(soup)
        self.extract_paragraphs(soup)
        self.extract_images(soup, url)
        
        self.print_results()
        return self.get_results()

    def extract_title(self, soup):
        title_tag = soup.find('h1') or soup.find('title')
        self.title = title_tag.text.strip() if title_tag else "Title not found"

    def extract_author(self, soup):
        author_tag = soup.find('meta', {'name': 'author'}) or soup.find(class_='author')
        self.author = author_tag['content'] if author_tag.has_attr('content') else author_tag.text.strip() if author_tag else "Author not found"

    def extract_date(self, soup):
        date_tag = soup.find('meta', {'property': 'article:published_time'}) or soup.find(class_='date')
        self.date = date_tag['content'] if date_tag and date_tag.has_attr('content') else date_tag.text.strip() if date_tag else "Date not found"

    def extract_paragraphs(self, soup):
        content = soup.find('article') or soup.find('div', class_='content')
        if content:
            self.paragraphs = [p.text.strip() for p in content.find_all('p') if p.text.strip()]

    def extract_images(self, soup, base_url):
        images = soup.find_all('img')
        for img in images:
            if img.get('src'):
                full_url = urljoin(base_url, img['src'])
                self.images.append(full_url)

    def print_results(self):
        print(f"Title: {self.title}")
        print(f"Author: {self.author}")
        print(f"Date: {self.date}")
        print("Paragraphs:")
        for p in self.paragraphs:
            print(f"- {p[:100]}...")  # Print first 100 characters of each paragraph
        print("Images:")
        for img in self.images:
            print(f"- {img}")

    def get_results(self):
        return {
            'title': self.title,
            'author': self.author,
            'date': self.date,
            'paragraphs': self.paragraphs,
            'images': self.images
        }

# Usage example
if __name__ == "__main__":
    analyzer = OpenLexicalAnalysis()
    url = "https://gloridust.xyz/%E6%8A%80%E6%9C%AF/2024/02/10/Job-submission-status-Check-tool.html"
    results = analyzer.analyze(url)