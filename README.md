# FootPredict
Free football predictor. Data: football-data.co.uk (results, odds, fixtures), TheSportsDB (club info). No keys.

## Run
    pip install -r requirements.txt
    python app.py
Computer: http://localhost:5000
Phone (same Wi-Fi): http://YOUR-COMPUTER-IP:5000  (Windows: `ipconfig`, Mac/Linux: `ifconfig`)

## Test the model honestly
    python backtest.py E0 150

## GitHub
GitHub Pages only hosts static files, so push this repo and deploy it free on Render.com or PythonAnywhere (start command: `gunicorn app:app`, add gunicorn to requirements).
