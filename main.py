'''
TODO:
  click to see more
  color words in the clouds by sentiment?
  for most pos/neg, if a student has commented multiple times, pick the most extreme comment (right now, it only picks the first one)
'''

import argparse
import os
from pathlib import Path

import pickle
import time
import re
import operator

import numpy as np
from tqdm import tqdm

from google.cloud import language_v1

from jinja2 import Environment, FileSystemLoader, select_autoescape

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from wordcloud import WordCloud

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import tensorflow_hub as hub

#
# ============================================================
#

# These are column numbers

SEMESTER=0
SECTION=1
RANDOMID=2
COMMENT=3
QUESTION=4
SCORE=5
MAG=6
JSCORE=7
JMAG=8
TOPICX=9
TOPICY=10

#
# ============================================================
#

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--credentials', type=str, default="./google.json" )
    parser.add_argument( '--comments', type=str, default="./comments.csv" )

    parser.add_argument( '--name', type=str, default="Jane Doe" )

    parser.add_argument( '--html_output', type=str, default="./output" )
    parser.add_argument( '--data_cache', type=str, default="./data_cache" )

    parser.add_argument( '--tfhub_cache', type=str, default="./tfhub_cache" )
    parser.add_argument( '--templates', type=str, default="./templates" )

    # parser.add_argument('--dim', type=int, default=10)
    # parser.add_argument('--num_epochs', type=int, default=1)
    # parser.add_argument('--num_training_steps', type=int, default=20000)
    # parser.add_argument('--num_warmup_steps', type=int, default=0)
    # parser.add_argument('--context', type=str, default="boi_small_alltox")
    # parser.add_argument('--lr', type=float, default=1e-1)
    # parser.add_argument('--lrs_type', type=str, default="linear") # or "cosine"
    # parser.add_argument('--cache_dir', type=str, default="/home/wingated/.cache/huggingface/transformers/")
    # parser.add_argument('--train_datafile', type=str, default="./small_training_data.jsonl")
    # parser.add_argument('--batch_size', type=int, default=1)  # XXX not fully implemented

    return parser.parse_args()

#
# ============================================================
#

args = parse_args()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials
os.environ["TFHUB_CACHE_DIR"] = args.tfhub_cache

DATA_CACHE_DIR = args.data_cache
OUTPUT_DIR = args.html_output
TEMPLATE_DIR = args.templates

WORDCLOUD_DIR = OUTPUT_DIR + "/clouds/"

COMMENTS_FN = args.comments
ANALYZED_FN = DATA_CACHE_DIR + "/analyzed_comments.pkl"
ENTITIES_FN = DATA_CACHE_DIR + "/entities_result.pkl"

Path( OUTPUT_DIR ).mkdir( parents=True, exist_ok=True )
Path( WORDCLOUD_DIR ).mkdir( parents=True, exist_ok=True )
Path( DATA_CACHE_DIR ).mkdir( parents=True, exist_ok=True )
Path( args.tfhub_cache ).mkdir( parents=True, exist_ok=True )

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml'])
)

stop_word_list = "i,me,my,myself,we,us,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,whose,this,that,these,those,am,is,are,was,were,be,been,being,have,has,had,having,do,does,did,doing,will,would,should,can,could,ought,i'm,you're,he's,she's,it's,we're,they're,i've,you've,we've,they've,i'd,you'd,he'd,she'd,we'd,they'd,i'll,you'll,he'll,she'll,we'll,they'll,isn't,aren't,wasn't,weren't,hasn't,haven't,hadn't,doesn't,don't,didn't,won't,wouldn't,shan't,shouldn't,can't,cannot,couldn't,mustn't,let's,that's,who's,what's,here's,there's,when's,where's,why's,how's,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,about,against,between,into,through,during,before,after,above,below,to,from,up,upon,down,in,out,on,off,over,under,again,further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,same,so,than,too,very,say,says,said,shall".split(',')

#
# ============================================================
#

def load_full_comments():
    orig_lines = open(COMMENTS_FN,"r").readlines()

    # We expect something like this: ['F 2016', 'C S 501R (001)', '737967', 'Best ever!' ]

    cur_question = "unknown"
    full_comments = []

    for line_num, l in enumerate( orig_lines ):
        l = l.rstrip()
        parts = l.split("\t")

        if len(parts) != 4:
            print( f"Error on line {line_num+1}, skipping...  [{l}]" )

        if parts[3].startswith("Question: "):
            cur_question = parts[3].replace("Question: ", "")
            continue

        parts[COMMENT] = re.sub('[^A-Za-z0-9 ,.!:;#/-]+', '', parts[COMMENT]).strip() # XXX leave parens?

        parts.append( cur_question )

        full_comments.append( parts )

    return full_comments

def get_comments_and_sentiments():
    try:
        f = open( ANALYZED_FN, "rb" )
        full_comments = pickle.load( f )
        f.close()
        return full_comments
    except:
        pass

    full_comments = load_full_comments()
    analyze_comments( full_comments )
    f = open( ANALYZED_FN, "wb" )
    pickle.dump( full_comments, f )
    f.close()
    return full_comments

def get_entity_sentiment( full_comments ):
    try:
        f = open( ENTITIES_FN, "rb" )
        full_es_results = pickle.load( f )
        f.close()
        return full_es_results
    except:
        pass

    full_es_results = calc_entity_sentiment( full_comments )
    f = open( ENTITIES_FN, "wb" )
    pickle.dump( full_es_results, f )
    f.close()
    return full_es_results

def analyze_comments( full_comments ):
    # run a sentiment analysis on each comment.  Modifies full_comments in place!!!

    error_cnt = 0

    client = language_v1.LanguageServiceClient()

    for comment_ind in tqdm( range(len(full_comments)) ):

        time.sleep(0.1)  # need this to avoid rate-based throttling

        comment_text = full_comments[comment_ind][COMMENT]

        document = { "type_": language_v1.Document.Type.PLAIN_TEXT, "content": comment_text }

        analyzed = False
        for ind in range(10):
            try:
                sentiment = client.analyze_sentiment( request={"document":document} ).document_sentiment
                full_comments[comment_ind] += [ sentiment.score, sentiment.magnitude ] # SCORE AND MAG
                analyzed = True
                break
            except Exception as e:
                print( f"Problem with comment {comment_text}" )
                print( e )
                time.sleep(5.0)  # need this to avoid rate-based throttling
                continue

        if not analyzed:
            full_comments[comment_ind] += [ 0.0, 0.0 ]  # SCORE AND MAG
            error_cnt += 1

    print( f"There were {error_cnt} errors" )

    # response = client.analyze_entities(document, encoding_type=encoding_type)


def calc_entity_sentiment( full_comments ):
    # run an entity sentiment analysis on each comment.

    error_cnt = 0
    client = language_v1.LanguageServiceClient()
    full_results = []

    for comment_ind in tqdm( range(len(full_comments)) ):

        time.sleep(0.1)  # need this to avoid rate-based throttling
        comment_text = full_comments[comment_ind][COMMENT]

        document = { "type_": language_v1.Document.Type.PLAIN_TEXT, "content": comment_text }

        for ind in range(10):  # try up to 10 times
            try:
                response = client.analyze_entity_sentiment( request={"document":document} )
                full_results.append( response )
                break
            except Exception as e:
                print( f"Problem with comment {comment_text}" )
                print( e )
                time.sleep(5.0)  # need this to avoid rate-based throttling
                continue

    print( f"There were {error_cnt} errors" )

    return full_results


def analyze_entity_sentiment( full_results, thresh=10 ):
    results = {}

    for response in full_results:
        for entity in response.entities:
            word = entity.name.lower()
            word = re.sub('[^A-Za-z0-9 ]+', '', word )

            parts = word.split()
            if len(parts) == 1:
                word = lemmatizer.lemmatize(word) # XXX not sure how this works with bigrams?

            if not word in results:
                results[word] = []
            results[word].append( (entity.sentiment.score, entity.sentiment.magnitude ) )

    final_results = {}
    for word in results:
        if len(results[word]) >= thresh:
            final_results[word] = results[word]

    return final_results

#
# ============================================================
#

def composite_score( score, mag ):
    cmag = np.clip( mag, 0, 5 )
    return 7.5*score+cmag

def save_page( data, fn ):
    f = open( OUTPUT_DIR+"/"+fn, "w" )
    f.write( data )
    f.close()

def generate_comment_data( full_comments ):
    template = jinja_env.get_template( 'comment_data.js')
    rendered_result = template.render( full_comments=full_comments, name=args.name );
    save_page( rendered_result, "comment_data.js" )

def generate_dashboard( full_comments ):

    total_comments = len(full_comments)
    total_years = 42
    total_sections = 43

    template = jinja_env.get_template( 'dashboard.html')
    rendered_result = template.render( total_comments=total_comments, total_years=total_years, total_sections=total_sections, name=args.name )
    save_page( rendered_result, "dashboard.html" )

def generate_mostposneg( full_comments ):

    # get rid of multiple comments by the same person
    sm_full_comments = []
    used = {}
    for c in full_comments:
        commenter_id = c[RANDOMID]
        if commenter_id in used:
            continue
        used[commenter_id] = True
        sm_full_comments.append( c )

    pos_sorted_results = sorted( sm_full_comments, key=lambda item: composite_score(item[SCORE],item[MAG]) )
    pos_comments = pos_sorted_results[-10:]

    neg_sorted_results = sorted( sm_full_comments, key=lambda item: composite_score(-item[SCORE],item[MAG]) )
    neg_comments = neg_sorted_results[-10:]

    template = jinja_env.get_template( 'mostposneg.html')
    rendered_result = template.render( pos_comments=pos_comments, neg_comments=neg_comments, name=args.name )
    save_page( rendered_result, "mostposneg.html" )

def generate_senttime( full_comments ):
    template = jinja_env.get_template('senttime.html')
    rendered_result = template.render( full_comments=full_comments, name=args.name )
    save_page( rendered_result, "senttime.html" )

def compute_topic_embedding( full_comments ):
    act = [ c[COMMENT] for c in full_comments ]
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")  # this is way faster
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-qa/3")

    embeddings = embed( act )
    npl = [ e.numpy() for e in embeddings ]

    data_matrix = np.vstack( npl )  # now num_comments x 512

    tmp = PCA(n_components=5).fit_transform(data_matrix)

    results = TSNE(n_components=2,perplexity=30,early_exaggeration=20,metric='cosine',learning_rate=10,n_iter=5000).fit_transform(tmp)

#    results = TSNE(n_components=2,perplexity=50,early_exaggeration=20,metric='cosine',learning_rate=10,n_iter=5000).fit_transform(data_matrix)
#    plt.clf()
#    plt.scatter( results[:,0], results[:,1], c=[c[4] for c in full_comments], alpha=0.5, cmap='RdBu' )
#    plt.colorbar()

    return results

#    plt.clf()
#    plt.scatter( results[:,0], results[:,1], c=[c[4] for c in full_comments], alpha=0.5, cmap='RdBu' )
#    plt.colorbar()


def generate_topiccluster( full_comments ):
    template = jinja_env.get_template('topiccluster.html')
    rendered_result = template.render( full_comments=full_comments, name=args.name )
    save_page( rendered_result, "topiccluster.html" )

def generate_entity_sentiment( results ):
    data = []
    for w in results:
        avg_score = np.average( [ d[0] for d in results[w] ] )
        avg_mag = np.clip( np.average( [ d[1] for d in results[w] ] ), 0.0, 2.0 )
        cnt = len( results[w] )
        ryval = np.random.rand()
        data.append( [ w, avg_score, ryval, avg_mag ] )  # cnt?

    template = jinja_env.get_template('entity_sentiment.html')
    rendered_result = template.render( data=data, name=args.name )
    save_page( rendered_result, "entity_sentiment.html" )

def generate_wordclouds( full_comments ):

    # still want year, semester, question
    # word and count
    # top 100 words

    # lemmatizer sometimes produces "wa"???
    global stop_word_list
    stop_word_list += [ "wa", "dr", "class" ]

    tmp = args.name.split()
    tmp = [ x.lower().strip() for x in tmp ]
    stop_word_list += tmp

    class_fd = {}
    question_fd = {}
    all_fd = {}

    for comment in full_comments:
        ctext = comment[COMMENT].lower()
        clean_comment = re.sub( '[^a-z0-9 ]+', ' ', ctext )

        # remove stop words, etc.
        dirty_words = clean_comment.split()
        words = []
        for w in dirty_words:
            w = lemmatizer.lemmatize(w) # lemmatize the word
            if w in stop_word_list: # ignore stop words
                continue;
            words.append(w)

        # generate all bigrams
        bigrams = [ words[ind]+" "+words[ind+1] for ind in range(len(words)-1) ]

        words += bigrams

        ze_class = comment[SEMESTER] + " - " + comment[SECTION]
        question = comment[QUESTION]

        used_words = {}  # only count each word once (???)
        for w in words:
            if w in used_words:
                continue
            used_words[w] = True

            if not ze_class in class_fd:
                class_fd[ze_class] = {}
            if not question in question_fd:
                question_fd[question] = {}

            if not w in class_fd[ze_class]:
                class_fd[ze_class][w] = 0
            class_fd[ze_class][w] += 1

            if not w in question_fd[question]:
                question_fd[question][w] = 0
            question_fd[question][w] += 1

            if not w in all_fd:
                all_fd[w] = 0
            all_fd[w] += 1

    class_list = class_fd.keys()
    question_list = question_fd.keys()

    wc = WordCloud( width=800, height=600, background_color='white' )
    wc_img = wc.generate_from_frequencies( all_fd )
    wc_img.to_file( WORDCLOUD_DIR + "all_comments.png" )

    for ze_class in class_list:
        wc_img = wc.generate_from_frequencies( class_fd[ze_class] )
        wc_img.to_file( WORDCLOUD_DIR + ze_class + ".png" )

    for question in question_list:
        wc_img = wc.generate_from_frequencies( question_fd[question] )
        wc_img.to_file( WORDCLOUD_DIR + question + ".png" )

    template = jinja_env.get_template('wordclouds.html')
    rendered_result = template.render( class_list=class_list, question_list=question_list, name=args.name )
    save_page( rendered_result, "wordclouds.html" )

def generate_senthist( full_comments ):
    scores = np.atleast_1d( [ c[SCORE] for c in full_comments ] )
    template = jinja_env.get_template( 'senthist.html')
    rendered_result = template.render( full_comments=full_comments, scores=scores, name=args.name )
    save_page( rendered_result, "senthist.html" )

def generate_sentmagscatter( full_comments ):
    template = jinja_env.get_template( 'sentmagscatter.html')
    rendered_result = template.render( full_comments=full_comments, name=args.name )
    save_page( rendered_result, "sentmagscatter.html" )


def generate_all_comments( full_comments ):
    template = jinja_env.get_template('allcomments.html')
    rendered_result = template.render( full_comments=full_comments, name=args.name )
    save_page( rendered_result, "allcomments.html" )

def generate_about():
    template = jinja_env.get_template('about.html')
    rendered_result = template.render( name=args.name )
    save_page( rendered_result, "about.html" )

def generate_index():
    template = jinja_env.get_template('index.html')
    rendered_result = template.render( name=args.name )
    save_page( rendered_result, "index.html" )

def generate_generate():
    template = jinja_env.get_template('generate.html')
    rendered_result = template.render( name=args.name )
    save_page( rendered_result, "generate.html" )


def generate_byquest( full_comments ):
    question_fd = {}

    for comment in full_comments:
        question = comment[QUESTION]
        if not question in question_fd:
            question_fd[question] = []
        question_fd[question].append( comment[SCORE] )  # pull out sentiment

    question_list = question_fd.keys()

    avgs = []
    for q in question_list:
        val = np.average( question_fd[q] )
        avgs.append( [ q, val ] )

    template = jinja_env.get_template('byquest.html')
    rendered_result = template.render( avgs=avgs, name=args.name )
    save_page( rendered_result, "byquest.html" )


def generate_byclass( full_comments ):
    class_fd = {}

    for comment in full_comments:
        ze_class = comment[SEMESTER] + " - " + comment[SECTION]
        if not ze_class in class_fd:
            class_fd[ze_class] = []
        class_fd[ze_class].append( comment[SCORE] )  # pull out sentiment

    class_list = class_fd.keys()

    avgs = []
    for c in class_list:
        val = np.average( class_fd[c] )
        avgs.append( [ c, val ] )

    template = jinja_env.get_template('byclass.html')
    rendered_result = template.render( avgs=avgs, name=args.name )
    save_page( rendered_result, "byclass.html" )

def copy_additional_assets():
    import glob
    import shutil

    for file in glob.glob( "./additional_assets/*" ):
        shutil.copy( file, OUTPUT_DIR )

#
# ============================================================
#

def jitter_and_clip( full_comments ):
    for ind in range(len(full_comments)):
        jscore = np.clip( full_comments[ind][SCORE]+0.025*np.random.randn(), -1, 1 )
        jmag = np.clip( full_comments[ind][MAG] + 0.25*np.random.randn(), 0, 6 )
        full_comments[ind] += [ jscore, jmag ]  # JSCORE AND JMAG
#
# ============================================================
#

print("Loading...")
full_comments = get_comments_and_sentiments()  # loads from disk, or calculates

# full_comments now looks likes this: [['W 2020', 'C S 330 (001)', '259978165419', 'A lot of', 'Spiritually strengthening', 0.1, 0.5], [...], ...]

#
# from here we can derive a number of visualizations
#
jitter_and_clip( full_comments )

print("Embedding...")
results = compute_topic_embedding( full_comments )
for ind in range(len(full_comments)):
    full_comments[ind] += [ results[ind,0], results[ind,1] ]  # TOPICX AND TOPICY

print("Generating wordclouds...")
generate_wordclouds( full_comments )

print("Doing entity level sentiment analysis...")
full_es_results = get_entity_sentiment( full_comments )
analyzed_es_results = analyze_entity_sentiment( full_es_results )
generate_entity_sentiment( analyzed_es_results )

print("Generating docs...")
generate_comment_data( full_comments )
generate_dashboard( full_comments )
generate_mostposneg( full_comments )
generate_senttime( full_comments )
generate_topiccluster( full_comments )
generate_byquest( full_comments )
generate_byclass( full_comments )
generate_senthist( full_comments )
generate_sentmagscatter( full_comments )
generate_all_comments( full_comments )

generate_index()
generate_generate()
generate_about()

copy_additional_assets()
