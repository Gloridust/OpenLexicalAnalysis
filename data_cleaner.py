import re

class DataCleaner:
    def clean(self, content):
        if content is None:
            return None

        cleaned_content = content.copy()

        # Clean text
        cleaned_content['text'] = self.clean_text(content['text'])

        # Clean title
        cleaned_content['title'] = self.clean_text(content['title'])

        # Clean authors
        cleaned_content['authors'] = [self.clean_text(author) for author in content['authors']]

        return cleaned_content

    def clean_text(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text