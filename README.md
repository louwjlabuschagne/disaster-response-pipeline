# Disaster Response Pipeline Project

### Instructions:

0.  Set up your virtual environment by running the following in Unix

Identify where your `python3` installation resides by running

    which python3

This should return something like `/usr/local/bin/python3`. Now we create a new python 3 virtual environment by running


    virtualenv -p /usr/local/bin/python3 venv


Now you can activate the virtual environment by running

    source venv/bin/activate

Lastly you can install the required packages by running:

    pip install -r requirements.txt

1.  Run the following commands in the project's root directory to set up your database and model.

    -   To run ETL pipeline that cleans data and stores in database

    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    -   To run ML pipeline that trains classifier and saves

    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2.  Run the following command in the app's directory to run your web app.

          python run.py

3.  Go to <http://0.0.0.0:3001/>
