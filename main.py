import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import re
from datetime import datetime
import hashlib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class OpenLexicalAnalysis:
    def __init__(self):
        self.config = self.load_config()
        self.ml_model = self.train_ml_model()

    def load_config(self):
        with open('config.json', 'r') as f:
            return json.load(f)

    def train_ml_model(self):
        X = ["这是一个标题", "这是作者名", "2021-01-01", "这是正文内容..."]
        y = ["title", "author", "date", "content"]
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)
        model = MultinomialNB()
        model.fit(X_vec, y)
        return (vectorizer, model)

    def analyze(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        result = {
            'title': self.extract_title(soup),
            'author': self.extract_author(soup),
            'date': self.extract_date(soup),
            'content': self.extract_content(soup),
            'images': self.extract_images(soup, url)
        }
        
        self.save_as_markdown(result, url)
        return result

    def extract_title(self, soup):
        candidates = []
        for selector in self.config['title_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))
        for h1 in soup.find_all('h1'):
            candidates.append((h1.text.strip(), 4))
        for element in soup.find_all(['h1', 'h2', 'title']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'title':
                candidates.append((text, 3))
        return max(candidates, key=lambda x: x[1])[0] if candidates else "Untitled"

    def extract_author(self, soup):
        candidates = []
        for selector in self.config['author_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))
        for element in soup.find_all(class_=re.compile('author', re.I)):
            candidates.append((element.text.strip(), 4))
        for element in soup.find_all(id=re.compile('author', re.I)):
            candidates.append((element.text.strip(), 4))
        for element in soup.find_all(['span', 'div', 'p']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'author':
                candidates.append((text, 3))
        return max(candidates, key=lambda x: x[1])[0] if candidates else "Unknown Author"

    def extract_date(self, soup):
        candidates = []
        for selector in self.config['date_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))
        date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        for element in soup.find_all(text=re.compile(date_pattern)):
            candidates.append((element.strip(), 4))
        for element in soup.find_all(['span', 'div', 'time']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'date':
                candidates.append((text, 3))
        return max(candidates, key=lambda x: x[1])[0] if candidates else datetime.now().strftime("%Y-%m-%d")

    def extract_content(self, soup):
        candidates = []
        for selector in self.config['content_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element, 5))
        paragraphs = soup.find_all('p')
        if paragraphs:
            content = soup.new_tag('div')
            for p in paragraphs:
                content.append(p)
            candidates.append((content, len(content.text)))
        for element in soup.find_all(['div', 'article']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'content':
                candidates.append((element, 3))
        return max(candidates, key=lambda x: x[1])[0] if candidates else soup.new_tag('div')

    def extract_images(self, soup, base_url):
        images = []
        for img in soup.find_all('img'):
            if img.get('src'):
                full_url = urljoin(base_url, img['src'])
                images.append((img, full_url))
        return images

    def save_as_markdown(self, result, url):
        title = re.sub(r'[<>:"/\\|?*]', '', result['title'])  # 移除不合法的文件名字符
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_path = f"./article/{title}"
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(f"{folder_path}/src", exist_ok=True)

        content = result['content']
        markdown_content = f"# {result['title']}\n\n"
        markdown_content += f"作者: {result['author']}\n\n"
        markdown_content += f"日期: {result['date']}\n\n"
        markdown_content += f"原文链接: {url}\n\n"
        markdown_content += "---\n\n"

        for element in content.children:
            if element.name == 'p':
                markdown_content += f"{element.text.strip()}\n\n"
            elif element.name == 'img':
                img_url = urljoin(url, element['src'])
                img_filename = self.download_image(img_url, f"{folder_path}/src")
                if img_filename:
                    markdown_content += f"![{element.get('alt', 'image')}](src/{img_filename})\n\n"

        for img, img_url in result['images']:
            if img not in content.find_all('img'):
                img_filename = self.download_image(img_url, f"{folder_path}/src")
                if img_filename:
                    markdown_content += f"![{img.get('alt', 'image')}](src/{img_filename})\n\n"

        with open(f"{folder_path}/{title}.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)

    def download_image(self, url, folder):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    extension = content_type.split('/')[-1]
                    filename = f"{hashlib.md5(url.encode()).hexdigest()}.{extension}"
                    with open(f"{folder}/{filename}", 'wb') as f:
                        f.write(response.content)
                    return filename
        except Exception as e:
            print(f"Error downloading image {url}: {str(e)}")
        return None

# 配置文件示例 (config.json)
'''
{
    "title_selectors": [".article-title", "#post-title", "h1.title"],
    "author_selectors": [".author-name", "#writer", "span.byline"],
    "date_selectors": [".publish-date", "#post-date", "time.entry-date"],
    "content_selectors": [".article-body", "#post-content", "div.entry-content"]
}
'''

# 使用示例
if __name__ == "__main__":
    analyzer = OpenLexicalAnalysis()
    url = "https://gloridust.xyz/%E6%8A%80%E6%9C%AF/2024/01/12/RuijieWIFI-AutoLogin.html"
    results = analyzer.analyze(url)