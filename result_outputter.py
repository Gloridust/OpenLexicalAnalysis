import json

class ResultOutputter:
    def output(self, content):
        if content is None:
            print("No content to output")
            return

        print("=" * 50)
        print(f"URL: {content['url']}")
        print(f"Title: {content['title']}")
        print(f"Authors: {', '.join(content['authors'])}")
        print(f"Publish Date: {content['publish_date']}")
        print(f"Text: {content['text'][:500]}...")  # Print first 500 characters
        print(f"Top Image: {content['top_image']}")
        print(f"Images: {', '.join(list(content['images'])[:5])}")  # Print first 5 images

        # Output domain-specific content
        if 'tweet_text' in content:
            print(f"Tweet: {content['tweet_text']}")
            print(f"Username: {content['username']}")
        elif 'hn_title' in content:
            print(f"HN Title: {content['hn_title']}")
            print(f"HN Score: {content['hn_score']}")

        print("=" * 50)

        # Save to JSON file
        with open(f"output_{content['url'].replace('/', '_')}.json", 'w') as f:
            json.dump(content, f, indent=4, default=str)