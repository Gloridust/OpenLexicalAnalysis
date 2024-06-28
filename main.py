import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class OpenLexicalAnalysis:
    def __init__(self):
        self.config = self.load_config()
        self.ml_model = self.train_ml_model()
        self.structure_model = self.train_structure_model()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename='open_lexical_analysis.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error("Config file not found. Using default configuration.")
            return {
                "title_selectors": [".article-title", "#post-title", "h1.title"],
                "author_selectors": [".author-name", "#writer", "span.byline"],
                "date_selectors": [".publish-date", "#post-date", "time.entry-date"],
                "content_selectors": [".article-body", "#post-content", "div.entry-content"]
            }

    def train_ml_model(self):
        X = ["这是一个标题", "这是作者名", "2021-01-01", "这是正文内容..."]
        y = ["title", "author", "date", "content"]
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)
        model = MultinomialNB()
        model.fit(X_vec, y)
        return (vectorizer, model)

    def train_structure_model(self):
        # 这里使用K-means聚类作为示例
        # 在实际应用中，你可能需要使用更复杂的模型和更多的训练数据
        X = np.random.rand(100, 10)  # 假设我们有100个样本，每个样本有10个特征
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X)
        return kmeans

    def analyze(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            result = {
                'title': self.extract_title(soup),
                'author': self.extract_author(soup),
                'date': self.extract_date(soup),
                'content': self.extract_content(soup),
                'images': self.extract_images(soup, url)
            }
            
            result = self.clean_and_format(result)
            self.learn_structure(soup, result)
            
            self.print_results(result)
            return result
        except requests.RequestException as e:
            logging.error(f"Error fetching URL {url}: {str(e)}")
            return None

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

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Title not found"

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

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Author not found"

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

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Date not found"

    def extract_content(self, soup):
        candidates = []
        for selector in self.config['content_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))

        paragraphs = soup.find_all('p')
        if paragraphs:
            content = ' '.join([p.text.strip() for p in paragraphs])
            candidates.append((content, len(content)))

        for element in soup.find_all(['div', 'article']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'content':
                candidates.append((text, 3))

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Content not found"

    def extract_images(self, soup, base_url):
        images = []
        for img in soup.find_all('img'):
            if img.get('src'):
                full_url = urljoin(base_url, img['src'])
                images.append(full_url)
        return images

    def clean_and_format(self, result):
        for key in ['title', 'author', 'date', 'content']:
            if key in result:
                result[key] = self.clean_text(result[key])
        
        # 将内容分成段落
        if 'content' in result:
            result['content'] = self.format_content(result['content'])
        
        return result

    def clean_text(self, text):
        # 分词
        words = word_tokenize(text.lower())
        # 去除停用词和非字母字符
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        # 词形还原
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def format_content(self, content):
        # 将内容分成段落
        paragraphs = content.split('\n')
        # 去除空段落并清理每个段落
        paragraphs = [self.clean_text(p) for p in paragraphs if p.strip()]
        return paragraphs

    def learn_structure(self, soup, result):
        # 这里我们使用一个非常简单的方法来"学习"网页结构
        # 在实际应用中，你可能需要使用更复杂的方法
        structure = {
            'title_tag': soup.find(text=result['title']).parent.name,
            'author_tag': soup.find(text=result['author']).parent.name,
            'date_tag': soup.find(text=result['date']).parent.name,
            'content_tags': Counter([p.parent.name for p in soup.find_all(text=result['content'])])
        }
        
        # 更新配置
        self.update_config(structure)

    def update_config(self, structure):
        # 这里我们简单地将学到的结构添加到配置中
        # 在实际应用中，你可能需要更复杂的逻辑来决定是否和如何更新配置
        self.config['title_selectors'].append(structure['title_tag'])
        self.config['author_selectors'].append(structure['author_tag'])
        self.config['date_selectors'].append(structure['date_tag'])
        self.config['content_selectors'].extend([tag for tag, count in structure['content_tags'].items() if count > 1])
        
        # 保存更新后的配置
        with open('config.json', 'w') as f:
            json.dump(self.config, f)

    def print_results(self, result):
        logging.info("Extraction results:")
        print(f"Title: {result['title']}")
        print(f"Author: {result['author']}")
        print(f"Date: {result['date']}")
        print("Content:")
        for i, paragraph in enumerate(result['content'], 1):
            print(f"Paragraph {i}: {paragraph[:100]}...")  # 打印每个段落的前100个字符
        print("Images:")
        for img in result['images']:
            print(f"- {img}")


if __name__ == "__main__":
    analyzer = OpenLexicalAnalysis()
    url = "https://example.com/article"
    results = analyzer.analyze(url)