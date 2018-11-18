# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterMessagesDatabase.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterMessagesDatabase.db models/finalized_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `cd app; python run.py`

3. Go to http://0.0.0.0:3001/
