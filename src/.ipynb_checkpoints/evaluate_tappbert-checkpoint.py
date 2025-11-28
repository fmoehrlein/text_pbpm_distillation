import os
import argparse
from pm4py.objects.log.obj import EventLog

def filter_every_nth_trace(log: EventLog, n: int = 10) -> EventLog:
    return [trace for idx, trace in enumerate(log) if idx % n == 0]

def filter_traces_by_max_length(log, max_length):
    return [trace for trace in log if len(trace) <= max_length]

def create_log_statistics(args, log):
    # Analyse log
    print("Create log statistics...")
    language = "english" if args.language is None else args.language[0]
    text_attribute = args.text[0]
    traces = len(log)
    events = sum(len(case) for case in log)
    durations = [(case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp()) / 86400 for case in log]
    docs = [event[text_attribute] for case in log for event in case if text_attribute in event]
    words = [word for doc in docs for word in word_tokenize(doc, language=language)]
    docs_filtered = BoWTextEncoder().preprocess_docs(docs, as_list=False)
    words_filtered = [word for doc in docs_filtered for word in word_tokenize(doc, language=language)]

    log_info = pd.DataFrame(
        [[path,
        traces,
        len(variants_filter.get_variants(log)),
        events,
        events / traces,
        np.median(durations),
        np.mean(durations),
        len(list(dict.fromkeys([event["concept:name"] for case in log for event in case])) if log else []),
        len(words),
        len(words_filtered),
        len(set(words)),
        len(set(words_filtered))]],
        columns=["log", "cases", "trace variants", "events", "events per trace", "median case duration",
                "mean case duration", "activities", "words pre filtering", "words post filtering",
                "vocabulary pre filtering", "vocabulary post filtering"]
    )

    log_info.to_csv("./results/log_info.csv", index=False, sep=";")
    print("Done.")


parser = argparse.ArgumentParser(description='Evaluate Text-Aware Process Prediction')
# ../datasets/werk.xes -a age gender -t question
parser.add_argument('log',
                    help='an event log in XES format (.xes)')
parser.add_argument('-a', '--attributes', nargs='+', required=False,
                    help='list of considered numerical or categorical attributes besides activity and timestamp')
parser.add_argument('-t', '--text', nargs=1, required=False,
                    help='attribute name with textual data', default=['text'])
parser.add_argument('-l', '--language', nargs=1, required=False,
                    help='language of the text in the log')

args = parser.parse_args()

print("Prepare...")
from tapp.tapp_model import TappModel, _get_event_labels
from tapp.log_encoder import LogEncoder
from tapp.text_encoder import BoWTextEncoder
from tapp.text_encoder import BoNGTextEncoder
from tapp.text_encoder import PVTextEncoder
from tapp.text_encoder import LDATextEncoder
from tapp.text_encoder import BERTbaseTextEncoder
from tapp.text_encoder import BERTbaseFineTunedNextActivityTextEncoder
from tapp.text_encoder import BERTbaseFineTunedNextTimeTextEncoder
from tapp.text_encoder import BERTbaseFineTunedNextActivityAndTimeTextEncoder
from tapp.text_encoder import BERTfromScratchTextEncoder
from tapp.text_encoder import BERTAndTokenizerFromScratchTextEncoder
from tapp.text_encoder import SentimentTextEncoder
from nltk.tokenize import word_tokenize
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.conversion.log import converter as log_converter
import datetime
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import log as log_utils

# Workstation
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs detected and configured: {len(gpus)}")
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")
else:
    print("No GPU detected. Ensure that TensorFlow is installed with GPU support.")

# Download text preprocessing resources from nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("corpora/stopwords")
except (LookupError, OSError):
    nltk.download("stopwords")

if not os.path.exists('./results/'):
    os.makedirs('./results/')

# Load event data
print("Load event log...")
path = args.log
variant = xes_importer.Variants.ITERPARSE
parameters = {variant.value.Parameters.TIMESTAMP_SORT: True, variant.value.Parameters.REVERSE_SORT: False}
log = xes_importer.apply(path, variant=variant, parameters=parameters)
print("Global log attributes:")
for key, value in log.attributes.items():
    print(f"  {key}: {value}")
log = filter_every_nth_trace(log, 10)
log = filter_traces_by_max_length(log, 70)
print("Global log attributes:")
for key, value in log.attributes.items():
    print(f"  {key}: {value}")
print("Done.")

# Split data in train and test log
text_attribute = args.text[0]
split = len(log) // 5 * 4
train_log = log[:split]
test_log = log[split:]

# Configure and build model variants
language = "english"

# configure text base for training:
# 'event' -> treat text attributes as event attributes
# 'prefix' -> use concatenation of text attributes from events
text_base_for_training = 'event'

text_models = [
    # --- Baselines from Pegoraro et al. ---
    None,
    BoWTextEncoder(encoding_length=50, language=language),
    BoWTextEncoder(encoding_length=100, language=language),
    BoWTextEncoder(encoding_length=500, language=language),
    BoNGTextEncoder(n=2, encoding_length=50, language=language),
    BoNGTextEncoder(n=2, encoding_length=100, language=language),
    BoNGTextEncoder(n=2, encoding_length=500, language=language),
    PVTextEncoder(encoding_length=10, language=language),
    PVTextEncoder(encoding_length=20, language=language),
    PVTextEncoder(encoding_length=100, language=language),
    LDATextEncoder(encoding_length=10, language=language),
    LDATextEncoder(encoding_length=20, language=language),
    LDATextEncoder(encoding_length=100, language=language),

    # --- TAPPBERT ---
    # Pre-trained BERT
    BERTbaseTextEncoder(encoding_length=768, language=language),
    # Pre-trained + fine-tuned BERT
    # (1) fine-tuned toward next activity prediction
    BERTbaseFineTunedNextActivityTextEncoder(encoding_length=768, language=language, epochs=16, lr=5e-5),
    # (2) fine-tuned toward next timestamp prediction
    BERTbaseFineTunedNextTimeTextEncoder(encoding_length=768, language=language, epochs=16, lr=5e-5),
    # (3) concat. embeddings of BERT fine-tuned toward next activity + next timestamp prediction
    BERTbaseFineTunedNextActivityAndTimeTextEncoder(encoding_length=768, language=language, epochs=16, lr=5e-5),
    # BERT trained from scratch
    # (1) tokenizer is pre-trained
    BERTfromScratchTextEncoder(encoding_length=36, language=language),
    BERTfromScratchTextEncoder(encoding_length=768, language=language),
    # (2) tokenizer is trained from scratch
    BERTAndTokenizerFromScratchTextEncoder(encoding_length=36, language=language, vocab_size=1000),
    BERTAndTokenizerFromScratchTextEncoder(encoding_length=768, language=language, vocab_size=1000),
]

text_models = [None]
#text_models = [None, BERTbaseTextEncoder(encoding_length=768, language=language)]
#text_models = [BERTbaseTextEncoder(encoding_length=768, language=language)]
#text_models = [SentimentTextEncoder()]
'''
text_models = [
    BoWTextEncoder(encoding_length=50, language=language),
    BoNGTextEncoder(n=2, encoding_length=50, language=language),
    PVTextEncoder(encoding_length=10, language=language),
    LDATextEncoder(encoding_length=10, language=language),
    SentimentTextEncoder(),
    BERTbaseTextEncoder(encoding_length=768, language=language),
]
'''

if BERTfromScratchTextEncoder in text_models or BERTAndTokenizerFromScratchTextEncoder in text_models:
    text_data_path = "../../datasets/questions.txt"
    if not os.path.exists(text_data_path):
        # extract text and store in separate file to be used during pretraining BERT from scratch
        df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
        txt = df["question"].dropna().values.tolist()
        with open(text_data_path, "w", encoding="utf-8") as output:
            for doc in txt:
                sentences = nltk.tokenize.sent_tokenize(doc)
                for sentence in sentences:
                    output.write(sentence)
                    output.write('\n')
                output.write('\n')


shared_layers = [1]
special_layers = [1]
neurons = [100]
data_attributes_list = [[]] if args.attributes is None else [args.attributes]
iterations = 1

print("Evaluate prediction models...")
print("This might take a while...")
for text_model in text_models:
    for shared_layer in shared_layers:
        for special_layer in special_layers:
            for neuron in neurons:
                for data_attributes in data_attributes_list:
                    if shared_layer + special_layer == 0:
                        pass
                    else:
                        log_encoder = LogEncoder(text_encoder=text_model, advanced_time_attributes=True,
                                                 text_base_for_training=text_base_for_training)
                        model = TappModel(log_encoder=log_encoder, num_shared_layer=shared_layer,
                                          num_specialized_layer=special_layer, neurons_per_layer=neuron, dropout=0.2,
                                          learning_rate=0.001)
                        model.activities = _get_event_labels(log, "concept:name")
                        log_encoder.fit(log, activities=model.activities, data_attributes=data_attributes,
                                        text_attribute=text_attribute)
                        for iteration in range(iterations):
                            model.fit(train_log, data_attributes=data_attributes, text_attribute=text_attribute, epochs=25)
                            model.evaluate(test_log, "results.csv", num_prefixes=8)
print("Done. Evaluation completed.")
