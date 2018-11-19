# Disaster Response Pipeline Project

### Project summary:

This project is the part of [Udacity](http://udacity.com) Data Science Nanodegree. It is created by Udacity together with their partner [Figure Eight](https://www.figure-eight.com/), see [this video](https://www.youtube.com/watch?v=4kTn7E3uTGA) for more detailed description.

### Files:

1. `data` directory:

    - `messages.csv` - CSV file that contains sample messages labeled by their id, together with their English translation, and message genres;
    - `categories.csv` - CSV file that contains 36 categories for each message id;
    - `process_data.py` - Python script that launches data processing of `messages.csv` and `categories.csv` creating a cleaned database `DisasterMessagesDatabase.db`.

2. `models` directory:
    - `train_classifier.py` - Python script that launches data modeling of `DisasterMessagesDatabase.db` and saves the model at `finalized_model.pkl`.

3. `app` directory:
    - `run.py` - Python script that locally run the Flask app with two html pages, `templates/master.html` and `templates/go.html`.


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterMessagesDatabase.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterMessagesDatabase.db models/finalized_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `cd app; python run.py`

3. Go to http://0.0.0.0:3001/

