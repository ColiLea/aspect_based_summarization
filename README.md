## Script to reproduce the synthetic data set used in 

Inducing Document Structure for Aspect-based Summarization, Lea Frermann and Alexandre Klementiev. ACL 2019, Florence, Italy.

 The script is based in Abigail See's procedure for reproducing the CNN/DM data set
 https://github.com/abisee/cnn-dailymail

### Prerequisites
 1) tokenized CNN/DM stories. For download and preprocessing follow the instructions on:
    https://github.com/abisee/cnn-dailymail 
 2) URL_Lists folder from https://github.com/abisee/cnn-dailymail 

### Usage
 python make_aspect_datafiles.py  path/to/url_lists \
                                  path/to/cnn_stories_tokenized  \
                                  path/to/dm_stories_tokenized \
                                  path/to/store/interleaved_stories_tokenized \
                                  path/to/store/interleaved_stories_binary \

 The script will store the plain text stories (train / dev / test) in interleaved_stories_tokenized 
 and the binaries in interleaved_stories_binary
