import re

class DataCleaner:
    def clean(self, content):
        if content is None:
            return None

        cleaned_content = content.copy()

        # Clean text
        if 'text' in cleaned_content:
            cleaned_content['text'] = self.clean_text(content['text'])

        # Clean author name
        if 'author' in cleaned_content:
            cleaned_content['author'] = self.clean_text(content['author'])

        return cleaned_content

    def clean_text(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        return text