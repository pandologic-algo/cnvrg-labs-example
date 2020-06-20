import os
from dotenv import load_dotenv


# load env
load_dotenv('research/utils/.env')

# env file
slack_token = os.getenv('SLACK_TOKEN')
