# Text Helper Functions
# ---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import string
import os
import tarfile
import collections
import numpy as np
import requests
import pandas as pd
import re


# Normalize text
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]

    return (texts)


# Normalize text
def normalise_kor_text(texts, stops):
    # Lower case
    # texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]

    return (texts)


def filter_rows(question, answer):
    if re.findall(pattern='상담(\s)*접수(\s)*후(\s)*재(\s)*문의', string=answer):
        return 'DROP'
    if re.findall(pattern='신속한(\s)*답변', string=answer):
        return 'DROP'
    if re.findall(pattern='(이미)(\s)*(답변)', string=answer):
        return 'DROP'
    if re.findall(pattern='(업체|담당(\s)*부서)(.)*(확인|연락)(\s)*(하고|하여|중|후|요청)', string=answer):
        return False
    if re.findall(pattern='(전화로)(\s)*(말씀)', string=answer):
        return False
    if re.findall(pattern='(부재)(\s)*(중)(\s)*(이셔서)', string=answer):
        return False
    if re.findall(pattern='(재(\s)*전화|재(\s)*연락)', string=answer):
        return False
    if re.findall(pattern='(오늘)(.)*(확인)(.)*(어려워)', string=answer):
        return False
    if question == None:
        return False
    return True


def clean_answer(answer):
    answer = re.sub(pattern= r'((\@[0-9]+\@)|(오늘도)|(\\x1e)|(&#41;)|(죄송(\s)*합니다)|(안녕(\s)*하세요)'
                             r'|(감사(\s)*합니다)|(고객님)((께서)|(께)|(이)|(의))*(\.|,|\s|!|~|\?)*'
                             r'|(가장(\s)*좋은(\s)*선택)|(수고(\s)하세요)|(문의)(\s)*(주신|하신))+', repl="", string=answer)
    answer = re.sub(pattern='(GS SHOP).{3,15}(입니다)(\.|,|!|~|\?)*', repl="", string=answer)
    answer = re.sub(pattern='((좋은|행복한|즐거운|편안한|기쁨이 가득한)(\s)*(저녁|하루|주말|밤|오후|휴일|시간)(\s)*'
                            '(되세요|보내세요|되십시오|되시기 바랍니다))(\.|,|\s|!|~|\?)*', repl="", string=answer)
    answer = re.sub(pattern='^(이용해)(\s)*(주셔서)', repl="", string=answer)
    answer = re.sub(pattern='((이용)(.))*(참고)(.)*(부탁)(\s)*(드립니다)+(\.|,|\s|!|~|\?)*', repl="", string=answer)
    answer = re.sub(pattern='((이용)(.))*(불편)(.)*((드려)|(드리게)(\s)*(되어)+)(\s)*(대단히)*', repl="", string=answer)
    answer = re.sub(pattern='(기다리게|기다려)(\s)*(해서|해드려|주셔서)', repl="", string=answer)
    answer = re.sub(pattern='((시간)*양해)(\s)*(부탁)(\s)*(드리며|드립니다)(\.|,|\s|!|~|\?)*', repl="", string=answer)
    answer = re.sub(pattern='(기대).{2,6}(주문).{3,6}(실망).{3,6}(정말|너무|대단히|\s)', repl="", string=answer)
    answer = re.sub(pattern='(저희(.))*(GS|gs)(\s)*(SHOP|shop).{3,5}(주문|구매).{3,10}(실망).{3,10}(정말|너무|대단히|\s)'
                    , repl="", string=answer)
    answer = re.sub(pattern='(감기)+(\s)*(조심하시고)', repl="", string=answer)
    answer = re.sub(pattern='도움(\s)*드리지(\s)*못(\s)*해', repl="", string=answer)
    answer = re.sub(pattern='(저희(.))*(GS|gs)(\s)*(SHOP|shop).{1,3}(이용).{3,6}(진심.{3,6})*(\.|,|\s|!|~|\?)'
                    , repl="", string=answer)
    return answer


def clean_question(question):
    question = re.sub(pattern= r'(ㆍ|ㅠ|ㅜ|ㅋ|ㅎ|ㅡ)*', repl="", string=question)
    return question


def filter_pos(answer):
    pos_list = ['EP+EF', 'ETM', 'EC', 'JKO', 'SSC', 'JKG', 'JX', 'JKB', 'SN', 'SC', 'SY', 'JSK', 'SF', 'SSO', 'JKS'
        , 'EF', 'EP', 'VCP+EF', 'JKS']
    return [tup[0] for tup in answer if tup[1] not in pos_list]

def load_dataset():
    df = pd.read_pickle('/home/will/workspace/sr_data/preprocessed_sr_data_180816.pkl')  # TODO file_path
    texts = df.question_pos_text.apply(lambda x: " ".join(x)).drop_duplicates().tolist()
    target = [i for i in range(len(texts))]
    # print(texts[0])
    return texts, target


def load_dataset_QA():
    df = pd.read_pickle('/home/will/workspace/sr_data/preprocessed_sr_data_180816.pkl')  # TODO file_path
    question_texts = df.question_pos_text.apply(lambda x: " ".join(x)).tolist()
    answer_texts = df.answer_pos_text.apply(lambda x: " ".join(x)).tolist()
    target = [i for i in range(len(question_texts))]
    return question_texts, answer_texts, target

def load_dataset_origin():
    def extract_state_from_question(pattern, question):
        assert pattern in ['주문상태', '문의유형']
        try:
            found = re.search(pattern=pattern + ' : <(.*?)>', string=question).group(1)
            if found == "":
                return None
            else:
                return found
        except AttributeError:
            return None

    def cleansing_question(question):
        try:
            found = re.search(pattern='@내용 : (.+?)$', string=question).group(1)
            if found == "":
                return None
            else:
                return found
        except AttributeError:
            return None

    df = pd.read_csv('/home/admin-/PycharmProjects/sr_data/180622_minimal.dat', sep='\t')
    df.columns = ['req_date', 'cate1', 'cate2', 'cate3', 'prd_cd', 'prd_nm', 'answer_date', 'answer_time', 'question',
                  'answer']
    df.drop(['req_date', 'answer_date', 'answer_time'], inplace=True, axis=1)
    for col in ['cate1', 'cate2', 'cate3', 'answer_date', 'answer_time', 'question', 'answer']:
        df[col] = df[col].apply(lambda x: x[1:-1])

    df['order_state'] = df.question.apply(lambda x: extract_state_from_question('주문상태', x))
    df['cate0'] = df.question.apply(lambda x: extract_state_from_question('문의유형', x))
    df['question'] = df.question.apply(lambda x: cleansing_question(x))

    df.drop(['req_date', 'answer_date', 'answer_time'], inplace=True, axis=1)

    return df

# Build dictionary of words
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]

    print("# words : %d ", len(words))

    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]

    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)

    return (word_dict)


# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split():
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return (data)


# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        # len(sentences) 이하의 숫자 하나 임의로 선택

        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]

        # window size 기준으로 sequence 생성
        # doc2vec에서는 window_sequence, label_indicies 불필요

        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in
                            enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

        # Pull out center word of interest for each window and create a tuple for each window
        # 모델의 목적(성격)에 맞게 x, y값 (word pair) 생성
        # doc2vec 의 경우 rand_sentence_ix 를 더해서 doc_id 도 부여
        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
            batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method == 'cbow':
            batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]
            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * window_size]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method == 'doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i + window_size], rand_sentence[i + window_size]) for i in
                                range(0, len(rand_sentence) - window_size)]
            # [([59, 142, 2838], 0), ([142, 2838, 0], 155), ([2838, 0, 155], 222), ([0, 155, 222], 46), ([155, 222, 46], 171), ([222, 46, 171], 7335), ([46, 171, 7335], 5)]

            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]

            # batch = [word_0, word_1, word_2, doc_idx], [word_1, word_2, word_3, doc_idx]
            # labels = [word_3, word_4, word_5 ..]

        else:
            raise ValueError('Method {} not implemented yet.'.format(method))

        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return (batch_data, label_data)


# Load the movie review data
# Check if data was downloaded, otherwise download it and save for future use
def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

    # Check if files are already downloaded
    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open('temp_movie_review_temp.tar.gz', 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        # Extract tar.gz file into temp folder
        tar = tarfile.open('temp_movie_review_temp.tar.gz', "r:gz")
        tar.extractall(path='temp')
        tar.close()

    pos_data = []
    with open(pos_file, 'r', encoding='latin-1') as f:
        for line in f:
            pos_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    pos_data = [x.rstrip() for x in pos_data]

    neg_data = []
    with open(neg_file, 'r', encoding='latin-1') as f:
        for line in f:
            neg_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    neg_data = [x.rstrip() for x in neg_data]

    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] * len(neg_data)

    return (texts, target)