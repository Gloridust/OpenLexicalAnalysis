import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
import os

class OpenLexicalAnalysis:
    def __init__(self):
        self.config = self.load_config()
        self.ml_model = self.train_ml_model()

    def load_config(self):
        default_config = {
            "title_selectors": [".article-title", "#post-title", "h1.title"],
            "author_selectors": [".author-name", "#writer", "span.byline"],
            "date_selectors": [".publish-date", "#post-date", "time.entry-date"],
            "content_selectors": [".article-body", "#post-content", "div.entry-content"]
        }
        
        config_file = 'config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {config_file}. Using default configuration.")
                return default_config
        else:
            print(f"{config_file} not found. Using default configuration.")
            # 创建默认配置文件
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Default {config_file} has been created.")
            return default_config

    def train_ml_model(self):
        # 训练一个简单的机器学习模型来识别内容类型
        # 这里使用 TF-IDF 和朴素贝叶斯分类器作为示例
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
        
        self.print_results(result)
        return result

    def extract_title(self, soup):
        candidates = []
        # 使用启发式规则
        for selector in self.config['title_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))  # 权重为5

        # 查找所有 h1 标签
        for h1 in soup.find_all('h1'):
            candidates.append((h1.text.strip(), 4))  # 权重为4

        # 使用机器学习模型
        for element in soup.find_all(['h1', 'h2', 'title']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'title':
                candidates.append((text, 3))  # 权重为3

        # 选择得分最高的候选项
        return max(candidates, key=lambda x: x[1])[0] if candidates else "Title not found"

    def extract_author(self, soup):
        candidates = []
        # 使用启发式规则
        for selector in self.config['author_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))

        # 查找带有 "author" 类或者 id 的元素
        for element in soup.find_all(class_=re.compile('author', re.I)):
            candidates.append((element.text.strip(), 4))
        for element in soup.find_all(id=re.compile('author', re.I)):
            candidates.append((element.text.strip(), 4))

        # 使用机器学习模型
        for element in soup.find_all(['span', 'div', 'p']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'author':
                candidates.append((text, 3))

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Author not found"

    def extract_date(self, soup):
        candidates = []
        # 使用启发式规则
        for selector in self.config['date_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))

        # 查找带有日期格式的文本
        date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        for element in soup.find_all(text=re.compile(date_pattern)):
            candidates.append((element.strip(), 4))

        # 使用机器学习模型
        for element in soup.find_all(['span', 'div', 'time']):
            text = element.text.strip()
            vec = self.ml_model[0].transform([text])
            if self.ml_model[1].predict(vec)[0] == 'date':
                candidates.append((text, 3))

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Date not found"

    def extract_content(self, soup):
        candidates = []
        # 使用启发式规则
        for selector in self.config['content_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), 5))

        # 查找最长的段落集合
        paragraphs = soup.find_all('p')
        if paragraphs:
            content = ' '.join([p.text.strip() for p in paragraphs])
            candidates.append((content, len(content)))

        # 使用机器学习模型
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

    def print_results(self, result):
        print(f"Title: {result['title']}")
        print(f"Author: {result['author']}")
        print(f"Date: {result['date']}")
        print(f"Content: {result['content'][:200]}...")  # 打印前200个字符
        print("Images:")
        for img in result['images']:
            print(f"- {img}")


if __name__ == "__main__":
    analyzer = OpenLexicalAnalysis()
    url = "https://gloridust.xyz/%E6%8A%80%E6%9C%AF/2024/02/10/Job-submission-status-Check-tool.html"
    results = analyzer.analyze(url)