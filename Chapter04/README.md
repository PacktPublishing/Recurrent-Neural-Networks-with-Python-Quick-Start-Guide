# Chapter 4 - Create a Spanish to English translator

## Instructions
1. Make sure you have python3 installed on your machine.
2. Download "en-es" sentences from the 2nd table (Statistics and TMX/Moses Downloads) found here http://opus.nlpl.eu/OpenSubtitles.php
3. Unzip the file and rename its content to `data.en` and `data.es`. Create a new folder `data/` inside the project structure and add `data.en` and `data.es` inside. 
4. Run `pip3 install -r requirements.txt`.
5. Run `python3 data_utils.py`. This will create a `data.pkl` file.
6. Run `python3 neural_machine_translation.py`. This will create a `ckpt/` folder. Wait for the whol operation - if will perform training so might take several hours.
7. Run `python3 predict.py` and you should see the English translations of several Spanish sentences defined inside the file.

