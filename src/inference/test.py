from src.model.roberta import RobertaClass
from src.servicenow.data_object import SentimentData
from src.inference.main import ModelPrediction

GPT_OUTPUT =  [{'case_id': 'CS0432550', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 03:39:58', 'entries': [{'value': 'solution accepted', 'created_on': '2024-03-19 03:43:50'}, {'value': 'proposed solution rejected i prefer different solution...', 'created_on': '2024-03-19 03:42:37'}, {'value': 'customer commented here..', 'created_on': '2024-03-19 03:41:08'}, {'value': 'description test description goes here', 'created_on': '2024-03-19 03:39:58'}]}, 
               {'case_id': 'CS0432551', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 03:44:08', 'entries': [{'value': 'closed by customer.', 'created_on': '2024-03-19 03:44:26'}, {'value': 'description  this is the previous description edit or delete if you want to alter  test description goes here', 'created_on': '2024-03-19 03:44:08'}]}, 
               {'case_id': 'CS0432552', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 03:46:44', 'entries': [{'value': 'solution accepted', 'created_on': '2024-03-19 03:50:28'}, {'value': 'proposed solution rejected i prefer different solution...', 'created_on': '2024-03-19 03:49:16'}, {'value': 'customer commented here..', 'created_on': '2024-03-19 03:47:51'}, {'value': 'description test description goes here', 'created_on': '2024-03-19 03:46:44'}]}, 
               {'case_id': 'CS0432553', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 03:50:49', 'entries': [{'value': 'closed by customer.', 'created_on': '2024-03-19 03:51:05'}, {'value': 'description  this is the previous description edit or delete if you want to alter  test description goes here', 'created_on': '2024-03-19 03:50:49'}]}, 
               {'case_id': 'CS0432555', 'account': 'DemoCloud', 'sys_created_on': '2024-03-19 04:02:44', 'entries': [{'value': 'solution accepted', 'created_on': '2024-03-19 04:06:51'}, {'value': 'proposed solution rejected i prefer different solution...', 'created_on': '2024-03-19 04:05:59'}, {'value': 'agent commented in here with different proposal,', 'created_on': '2024-03-19 04:05:15'}, {'value': 'case moved to waiting on wso2', 'created_on': '2024-03-19 04:04:43'}, {'value': 'customer commented here..', 'created_on': '2024-03-19 04:04:32'}, {'value': 'description test goes here', 'created_on': '2024-03-19 04:02:44'}]}, 
               {'case_id': 'CS0432557', 'account': 'DemoCloud', 'sys_created_on': '2024-03-19 04:19:05', 'entries': [{'value': 'solution accepted', 'created_on': '2024-03-19 04:23:17'}, {'value': 'proposed solution rejected i prefer different solution...', 'created_on': '2024-03-19 04:22:25'}, {'value': 'agent commented in here with different proposal,', 'created_on': '2024-03-19 04:21:38'}, {'value': 'case moved to waiting on wso2', 'created_on': '2024-03-19 04:21:06'}, {'value': 'customer commented here..', 'created_on': '2024-03-19 04:20:54'}, {'value': 'description test goes here', 'created_on': '2024-03-19 04:19:05'}]}, 
               {'case_id': 'CS0432559', 'account': 'DemoCloud', 'sys_created_on': '2024-03-19 04:28:10', 'entries': [{'value': 'solution accepted', 'created_on': '2024-03-19 04:32:24'}, {'value': 'proposed solution rejected i prefer different solution...', 'created_on': '2024-03-19 04:31:30'}, {'value': 'agent commented in here with different proposal,', 'created_on': '2024-03-19 04:30:43'}, {'value': 'case moved to waiting on wso2', 'created_on': '2024-03-19 04:30:09'}, {'value': 'customer commented here..', 'created_on': '2024-03-19 04:29:57'}, {'value': 'description test goes here', 'created_on': '2024-03-19 04:28:10'}]}, 
               {'case_id': 'CS0432563', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 04:46:50', 'entries': [{'value': 'incident impact description test impact description test description', 'created_on': '2024-03-19 04:46:50'}]}, 
               {'case_id': 'CS0432566', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 05:07:32', 'entries': [{'value': 'closed by customer.', 'created_on': '2024-03-19 05:08:07'}, {'value': 'priority was updated from medium p3 to high p2', 'created_on': '2024-03-19 05:07:54'}, {'value': 'incident impact description test impact description test description', 'created_on': '2024-03-19 05:07:32'}]}, 
               {'case_id': 'CS0432567', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 05:08:34', 'entries': [{'value': 'incident impact description test impact description test description', 'created_on': '2024-03-19 05:08:34'}]}, 
               {'case_id': 'CS0432577', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 06:28:52', 'entries': [{'value': 'closed by customer.', 'created_on': '2024-03-19 06:29:25'}, {'value': 'priority was updated from medium p3 to high p2', 'created_on': '2024-03-19 06:29:13'}, {'value': 'incident impact description test impact description test description', 'created_on': '2024-03-19 06:28:52'}]}, 
               {'case_id': 'CS0432578', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 06:29:49', 'entries': [{'value': 'priority was updated from medium p3 to high p2', 'created_on': '2024-03-19 06:30:11'}, {'value': 'incident impact description test impact description test description', 'created_on': '2024-03-19 06:29:49'}]}, 
               {'case_id': 'CS0432582', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 10:30:45', 'entries': [{'value': 'closed by customer.', 'created_on': '2024-03-19 10:31:19'}, {'value': 'priority was updated from medium p3 to high p2', 'created_on': '2024-03-19 10:31:05'}, {'value': 'incident impact description test impact description test description', 'created_on': '2024-03-19 10:30:45'}]}, 
               {'case_id': 'CS0432583', 'account': 'ParaTestAccount6', 'sys_created_on': '2024-03-19 10:31:43', 'entries': [{'value': 'priority was updated from medium p3 to high p2', 'created_on': '2024-03-19 10:32:02'}, {'value': 'incident impact description test impact description test description', 'created_on': '2024-03-19 10:31:43'}]}
            ]

def get_mock_data() -> SentimentData:
    mock_sentiment_data = SentimentData()
    mock_sentiment_data.load_data(GPT_OUTPUT)
    return mock_sentiment_data

model_predict = ModelPrediction()
data = get_mock_data()
model_predict.get_sentiments(data)
print(data.cases)

# import torch
# model = RobertaClass()
# model = torch.load('models\\pytorch_roberta_sentiment_3_classes_0.1.3.bin', map_location=torch.device('cpu'))

# # Step 2: Save the loaded model as a .pth file
# checkpoint = {
#             'model_state_dict': model.state_dict()
#         }
# torch.save(checkpoint, 'models\\your_model.pth')
