from ..model.app import App
from ..model.roberta import RobertaClass
# Replace the models file path in the models directory. 
robertaApp = App(metadata_path = "C:\\Users\\gimhanSandeeptha\\Gimhan Sandeeptha\\Sentiment Project\\ROBERTa\\metadata\\roberta.json")
robertaApp.start_model()

def get_one_sentiment(text):
    prediction = robertaApp.predict([text])[0]
    sentiment = "unknown"
    if prediction == 0:
        sentiment = 'Negative'
    elif prediction == 1:
        sentiment = 'Neutral'
    elif prediction == 2:
        sentiment = 'Positive'
    return sentiment

def get_sentiments(response_list):
    for case_number in range(len(response_list)):
        case = response_list[case_number]
        sentiment_lst =[]
        for comment_number in range(len(case[2])):
            comment = case[2][comment_number]
            sentiment = get_one_sentiment(comment)
            sentiment_lst.append(sentiment)
        case.append(sentiment_lst)    # implement to check whether the two lists are in the same length
    return response_list


# response_list =[["Case_id01","Acc01",["Hi how are you?", "it is not a fair deal. i am dissapointed"]],["Case_id02","Acc2",["It is great keep it up", "Hi this is a normal day","can i have your pen for a moment"]]]
# print(get_sentiments(response_list))
    