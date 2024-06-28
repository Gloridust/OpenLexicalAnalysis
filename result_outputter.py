import json

class ResultOutputter:
    def output(self, content):
        if content is None:
            print("No content to output")
            return

        print("=" * 50)
        print(f"URL: {content['url']}")
        print(f"Author: {content.get('author', 'N/A')}")
        print(f"Username: {content.get('username', 'N/A')}")
        print(f"Date: {content.get('date', 'N/A')}")
        print(f"Text: {content.get('text', 'N/A')}")
        print(f"Retweet Count: {content.get('retweet_count', 'N/A')}")
        print(f"Favorite Count: {content.get('favorite_count', 'N/A')}")
        print(f"Images: {', '.join(content.get('images', []))}")
        print("=" * 50)

        # Save to JSON file
        with open(f"output_{content['url'].split('/')[-1]}.json", 'w') as f:
            json.dump(content, f, indent=4, default=str)