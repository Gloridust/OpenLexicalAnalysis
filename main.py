import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
import transformers
import nltk
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor
import logging
import pickle
from sklearn.cluster import KMeans

class OpenLexicalAnalysis:
    def __init__(self):
        self.config = self.load_config()
        self.dl_model = self.load_dl_model()
        self.structure_learner = self.load_structure_learner()
        self.setup_logging()
        nltk.download('punkt')

    def load_config(self):
        with open('config.json', 'r') as f:
            return json.load(f)

    def load_dl_model(self):
        # 加载预训练的BERT模型用于文本分类
        model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        return model, tokenizer

    def load_structure_learner(self):
        # 加载或初始化结构学习器
        try:
            with open('structure_learner.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return KMeans(n_clusters=5)  # 假设我们有5种主要的内容类型

    def setup_logging(self):
        logging.basicConfig(filename='open_lexical_analysis.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def analyze(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            result = {
                'title': self.extract_title(soup),
                'author': self.extract_author(soup),
                'date': self.extract_date(soup),
                'content': self.extract_content(soup),
                'images': self.extract_images(soup, url)
            }
            
            result['content'] = self.clean_and_format_content(result['content'])
            
            self.learn_structure(soup, result)
            
            self.print_results(result)
            return result
        except Exception as e:
            logging.error(f"Error analyzing {url}: {str(e)}")
            raise

    def extract_title(self, soup):
        candidates = []
        for selector in self.config['title_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), self.classify_text(element.text.strip())))

        for h1 in soup.find_all('h1'):
            candidates.append((h1.text.strip(), self.classify_text(h1.text.strip())))

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Title not found"

    def extract_author(self, soup):
        candidates = []
        for selector in self.config['author_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), self.classify_text(element.text.strip())))

        for element in soup.find_all(class_=re.compile('author', re.I)):
            candidates.append((element.text.strip(), self.classify_text(element.text.strip())))

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Author not found"

    def extract_date(self, soup):
        candidates = []
        for selector in self.config['date_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), self.classify_text(element.text.strip())))

        date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        for element in soup.find_all(text=re.compile(date_pattern)):
            candidates.append((element.strip(), self.classify_text(element.strip())))

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Date not found"

    def extract_content(self, soup):
        candidates = []
        for selector in self.config['content_selectors']:
            element = soup.select_one(selector)
            if element:
                candidates.append((element.text.strip(), self.classify_text(element.text.strip())))

        paragraphs = soup.find_all('p')
        if paragraphs:
            content = ' '.join([p.text.strip() for p in paragraphs])
            candidates.append((content, self.classify_text(content)))

        return max(candidates, key=lambda x: x[1])[0] if candidates else "Content not found"

    def extract_images(self, soup, base_url):
        images = []
        for img in soup.find_all('img'):
            if img.get('src'):
                full_url = urljoin(base_url, img['src'])
                images.append(full_url)
        return images

    def classify_text(self, text):
        # 使用BERT模型进行文本分类
        inputs = self.dl_model[1](text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.dl_model[0](**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities[0].max().item()

    def clean_and_format_content(self, content):
        # 使用NLTK进行句子分割
        sentences = sent_tokenize(content)
        # 简单的清理：移除多余的空白字符
        cleaned_sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in sentences]
        # 将句子重新组合成段落
        paragraphs = []
        current_paragraph = []
        for sentence in cleaned_sentences:
            current_paragraph.append(sentence)
            if len(' '.join(current_paragraph)) > 500:  # 假设每个段落约500字符
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        return '\n\n'.join(paragraphs)

    def learn_structure(self, soup, result):
        # 提取页面结构特征
        features = self.extract_structure_features(soup)
        # 使用KMeans进行聚类
        cluster = self.structure_learner.predict([features])[0]
        # 更新配置
        self.update_config(cluster, result)
        # 保存更新后的结构学习器
        with open('structure_learner.pkl', 'wb') as f:
            pickle.dump(self.structure_learner, f)

    def extract_structure_features(self, soup):
        # 这里只是一个简单的示例，你可以根据需要提取更多特征
        return [
            len(soup.find_all('div')),
            len(soup.find_all('p')),
            len(soup.find_all('h1')),
            len(soup.find_all('h2')),
            len(soup.find_all('img'))
        ]

    def update_config(self, cluster, result):
        # 根据聚类结果和提取结果更新配置
        # 这只是一个简单的示例，你可能需要更复杂的逻辑
        if result['title'] not in self.config['title_selectors']:
            self.config['title_selectors'].append(f"h1:contains('{result['title']}')")
        # 对其他字段进行类似的更新...
        with open('config.json', 'w') as f:
            json.dump(self.config, f)

    def print_results(self, result):
        print(f"Title: {result['title']}")
        print(f"Author: {result['author']}")
        print(f"Date: {result['date']}")
        print(f"Content: {result['content'][:200]}...")  # 打印前200个字符
        print("Images:")
        for img in result['images']:
            print(f"- {img}")

def parallel_analyze(urls):
    analyzer = OpenLexicalAnalysis()
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(analyzer.analyze, urls))


if __name__ == "__main__":
    urls = ["https://gloridust.xyz/%E6%8A%80%E6%9C%AF/2024/02/10/Job-submission-status-Check-tool.html", 
            "https://x.com/gloridust1024/status/1805669956647866807"
            ]
    
    results = parallel_analyze(urls)
    for result in results:
        print(result)