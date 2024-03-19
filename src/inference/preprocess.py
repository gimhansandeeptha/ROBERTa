import re
from bs4 import BeautifulSoup


def normalize_texts(text):
    NON_ALPHANUM = re.compile(r'[^a-z0-9,.\s]')  # Exclude all non-alphanumeric characters except comma and dot
    NON_ASCII = re.compile(r'[\x20-\x7E]+')      # Exclude all characters that are not lowercase letters, digits, or whitespace

    ascii_chars = NON_ASCII.findall(text)
    lower_text = ascii_chars[0].lower()
    alphanumeric_text = NON_ALPHANUM.sub(r'', lower_text)
    return alphanumeric_text

def html_to_text(html_content):
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text) 
    return text

# # Example HTML content
# html_content = """
# <html>
# <head><title>Test HTML</title></head>
# <body>
# <h1>This is a heading</h1>
# <p>This is a paragraph with <a href="https://example.com">a link</a>.</p>
# </body>
# </html>
# """

# # Convert HTML to plain text
# plain_text = html_to_text(html_content)
# print(f"plain_text: {plain_text}")

# normalized_text = normalize_texts(plain_text)
# print(f"normalized Text: {normalized_text}")

