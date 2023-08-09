# Automatically analyze a set of BYU comments

This is a set of scripts and templates to automatically analyze BYU student comments.  It uses a suite of ML algorithms to analyze each comment, and automatically generates a set of static HTML files with various interactive visualizations of the results.

To run the system, the user must provide a BYU "CSV" file containing the comments.  This should be structured as a tab-delimited file with four columns, containing a semester indicator, a section indicator, a student ID (currently ignored), and the text of the comment:
```
F 2016	C S 501R (001)	737967	Best ever!
```
However, note that this is NOT a standard CSV, as BYU by default generates a CSV with additional lines separating out different questions; the fourth column will sometimes contain a string like the following:
```
Semester	Course	Random ID	Question: Explained concepts effectively
```
The reports generated will use these questions to additionally segment the comments.

BYU provides a mechanism to download all of the comments in the proper format; see instructions below.

## Quickstart

To run the system, first create a virtual environment and install dependencies:

```
python -m venv ./autocomment
. ./autocomment/bin/activate
pip install -r requirements.txt
```
Fire up a python terminal and make sure that the proper NLTK packages are installed:
```
import nltk
nltk.download('wordnet')
```

Many of the visualizations rely on sentiment analysis (specifically as provided by Google Cloud NLP).  Make sure that you have appropriate credentials stored in a json file. The output files also include the name of the professor; this must be supplied as an additional argument

Run the script (assuming our CSV is called `comments.csv` and our Google credentials are called `google.json`:

```
python ./main.py --credentials ./google.json --comments comments.csv --name="John Doe"
```

The output will be stored in the `output/` subdirectory.

## Downloading comments from BYU's student rating system

The goal is to generate a CSV file containing the comment you want analyzed; for your notebook, this would probably be the comments for all classes that you've taught since your last promotion.

1. Go to ratings.byu.edu
2. Select "View Student Ratings Reports"
3. Click the button "View Reports" next to "Comprehensive report for all courses, beginning with the selected semester/term (2015 to present)"
4. Pick the semester and year you want the reports to start from
5. Click generate report
6. Scroll down about a page.  There are a couple of different sections; you need to find the comments section.  At the top of the comments section is a button labeled "Download comments .csv file". Click that and save the file.
