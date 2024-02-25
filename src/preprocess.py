# import re
# class Preprocess():
#     def 
#     def normalize_texts(df, text_column):
#         # Exclude all non-alphanumeric characters except comma and dot
#         NON_ALPHANUM = re.compile(r'[^a-z0-9,.\s]')

#         # Exclude all characters that are not lowercase letters, digits, or whitespace
#         NON_ASCII = re.compile(r'[\x20-\x7E]+')

#         normalized_texts = []
#         for text in df[text_column]:
#             ascii_chars = NON_ASCII.findall(text)
#             lower_text = ascii_chars[0].lower()
#             alphanumeric_text = NON_ALPHANUM.sub(r'', lower_text)
#             normalized_texts.append(alphanumeric_text)

#         return normalized_texts



