# Lexical similarity code

This code was developed in order to create items for an aphasia study.  Aphasia is a condition often preceded by a stroke or some other type of head trauma that impairs either a person's ability to retrieve words, or a person's ability to process meaning, or some combination. While historically aphasia has been split into Wernicke and Broca's aphasia the truth of the condition is that it is much more muddled, with most aphasia patients not fully representing the archetype of either of these pathologies.

One research challenge that has hovered over this area of research is attempting to separate semantic knowledge from lexical knowledge, semantic knowledge being closer to ground truth and the reality of things, and lexical knowledge being more oriented towards sometimes figurative, more language based word relationships.  This is clinically important as in aphasia rehabilitation settings it could be possible to train patients to use semantic knowledge that they still have access to to inform lexical knowledge etc, because of how we know these types of knowledge to be distributed throughout the brain.

### Instructions:

<<<<<<< HEAD
1. The folder test code is just a draft essentially.  There are some old scripts related to pmi, and just mutual informativity that were relevant at the start of this project, however, the word vectors we've borrowed from spacy eclipse the need for hand calculated mutual informativity.  I used borrowed word vectors because there was no "corpus" to begin with.  
=======
1. The folder test code is just a draft essentially
>>>>>>> a6cd60949894a039b10406f006b1b41c9b50bc07

2. Run the following command in the app's directory to run your web app.  Since this is a Dash app I believe that the normal port used is 8080 for local machines.
    `python run.py 8080`

3. Go to http://0.0.0.0:8080/ or localhost:8080, depending on your computer



# Folders and files included:

* **app**:
	* **run.py**:  This will launche the web app
* **test code**: 
	* **reduced_model**: This folder is supposed to be an import/export of spacy en_core_web_md with pruned vectors.  In all honesty, I'm not sure how well it worked.  On my local machine I am able to use it to run the app easily, but the build currently fails with an msgpack extra values error when writing from bytes in heroku.  I have yet to test this in a virtual environment, which may be part of the problem.  However, if you have the skills to run this locally, you can also just load the normal spacy models (md or lg because of the vector functions used) instead of this import.
	* **FigureofSpeechPrediction.ipynb**, **latent_meaning.py**, **mutual_informativity.py**: These contain the scripts from various stages of development.  It is simple enough to just be included in the run file, although it could be read in from these.
	* **test_doc***, quotes from Moby Dick, just used to test code. 

# Current Requirements:
chardet==3.0.4
click==6.7
dash==1.16.3
dash-bootstrap-components==0.10.7
dash-core-components==1.12.1
dash-html-components==1.1.1
scikit-learn==0.22.2
pandas==1.1.2
idna==2.6
importlib-metadata==1.4.0
itsdangerous==0.24
msgpack==0.5.4
msgpack-numpy==0.4.1
msgpack-python==0.5.1
numpy==1.18.1
nltk==3.2.5
plac==0.9.6
plotly==4.11.0
gunicorn==19.10.0
pytz==2017.3
python-dateutil==2.8.1
requests==2.23.0
six==1.14.0
scipy==1.3.2
spacy==2.2.3
srsly==1.0.1
thinc==7.3.1
tqdm==4.42.1
urllib3==1.22
wasabi==0.6.0
zipp==2.0.1

# Results:
  
Computational metric of evaluation in the works.  However Native english speakers will recognize that if provided with sufficient text, probable lexical relations are extracted, although there are some errors.

# Notes:



# Contact: 

Madeline Kehl (mad.kehl@gmail.com)

# Acknowledgements:

* LRDC, University of Pittsburgh
* Dr. Tessa Warren, Haley Dresang

# MIT License

Copyright (c) 2020 Madeline Kehl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

