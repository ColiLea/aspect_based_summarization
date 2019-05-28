import sys
import os
import hashlib
import struct
import collections
import random
from tensorflow.core.example import example_pb2
import numpy as np

# Code for reproducing the MA-News data set used in the paper
# Inducing Structure for Aspect-based Summarization
# 
# The script is based in Abigail See's procedure for reproducing the CNN/DM data set
# https://github.com/abisee/cnn-dailymail
#
# PREREQUISITES
# =============
# 1) tokenized CNN/DM stories. For download and preprocessing follow the instructions on:
#    https://github.com/abisee/cnn-dailymail 
# 2) URL_Lists folder from https://github.com/abisee/cnn-dailymail 
#
# USAGE
# =====
# python make_aspect_datafiles.py  path/to/url_lists \
#                                  path/to/cnn_stories_tokenized  \
#                                  path/to/dm_stories_tokenized \
#                                  path/to/store/interleaved_stories_tokenized \
#                                  path/to/store/interleaved_stories_binary \
#
# The script will store the plain text stories (train / dev / test) in interleaved_stories_tokenized 
# and the binaries in interleaved_stories_binary

dm_single_close_quote = u'\u2019'
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

DOC_2_LENGTHS = {}

# make sure we always get the same output
random.seed(0)
np.random.seed(0)

# Define target aspects and some helper IDs
corpus_aspect_id = {'cnn':2, 'dailymail':1}
TARGET_ASPECTS = {'tvshowbiz':0, 'travel':1, 'health':2, 'sciencetech':3, 'sport':4, 'news':5}
TARGET_ASPECTS_IDS = ['tvshowbiz', 'travel', 'health', 'sciencetech', 'sport', 'news']

MAX_OUTPUT_DOC_LEN = 1500
MAX_DOC_LENGTH = 1000
MAX_DOCS_PER_ASPECT = 4000
MIN_DOCS_PER_ASPECT = 500

def chunk_file(bin_dir, set_name, chunks_dir):
  in_file =os.path.join(bin_dir, set_name+'.bin')
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(bin_dir):
  # Make a dir to hold the chunks
  chunks_dir = os.path.join(bin_dir, "chunked")
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print "Splitting %s data into chunks..." % set_name
    chunk_file(bin_dir, set_name, chunks_dir)
  print "Saved chunked data in %s" % chunks_dir



def read_text_file(text_file):
  """Reads a text file line by line into list"""
  lines = []
  with open(text_file, "r") as f:
    for line in f:
        lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_binary_inputs(story_file):
  """reads plain txt version of interleaved story for turning into binary"""
  blocks = open(story_file, 'r').read().split("\n\n")
  aspect = blocks[0]
  urls   = blocks[1]
  hashes = blocks[2]
  article_b= blocks[3]
  summary_b= blocks[4]
  
  # get article line by line
  article_lines = ["%s %s %s" % (SENTENCE_START, line.split("\t")[1].strip(), SENTENCE_END) for line in article_b.split("\n") if line != ""]
  article       = ' '.join(article_lines)

  # get sentence-level aspect labels
  article_sentence_aspects = [line.split("\t")[0].strip() for line in article_b.split("\n") if line != ""]
  article_sentence_aspects = ' '.join(article_sentence_aspects)

  # get summary
  summary       = summary_b.split("\t")[1]
      
  return article, summary, aspect, article_sentence_aspects, urls, hashes


def get_art_abs(story_file):
  """reads plain txt version of tokenized CNN / DM document"""
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract, article_lines

def write_to_bin(url_list, url_hashes, in_dir, out_dir, out_file, makevocab=False, data_split=""):
  """Reads the tokenized .story files corresponding to the urls listed in the url_list and writes them to a out_file."""
  print "Making bin file for URLs listed in %s..." % data_split
  story_fnames = os.listdir(in_dir)
  num_stories = len(url_list)
  
  if len(story_fnames) > len(url_list):
      story_fnames = story_fnames[:len(url_list)]
  print(len(url_list), len(story_fnames))
  #assert len(url_list) == len(story_fnames)
  
  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories))

      # Look in the fake story dir to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(in_dir, s)):
        story_file = os.path.join(in_dir, s)
      else:
        raise Exception("Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir))

      # Get the strings to write to .bin file
      article, abstract, aspect, article_sentence_aspects, urls, hashes = get_binary_inputs(story_file)
      
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['aspect'].bytes_list.value.extend([aspect])
      tf_example.features.feature['urls'].bytes_list.value.extend([urls])
      tf_example.features.feature['hashes'].bytes_list.value.extend([hashes])
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example.features.feature['article_sentence_aspects'].bytes_list.value.extend([article_sentence_aspects])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        art_tokens = [t for t in art_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print "Finished writing file %s\n" % out_file

  # write vocab to file
  if makevocab:
    print "Writing vocab file..."
    with open(os.path.join(out_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print "Finished writing vocab file"



def aspect_to_hashes(url_list):
    aspect2hashes = {}
    url_hashes = get_url_hashes(url_list)
    for idx, url in enumerate(url_list):
        
        if idx % 10000 == 0:
            print "Checked %d / %d " % (idx, len(url_list))
        
        url = url.replace("https", "http")
        url = url.replace("http://www.edition.cnn", "http://www.cnn")
        url = url.replace("//", "/")
        
        assert len(url.split("http:/")) == 3
    
        main_cont = url.split("http:/")[2].strip()
        corpus    = main_cont.split(".")[1]        # get corpus name (cnn vs dailymail )

        if corpus in ['cnn', 'dailymail']:
            # get aspect from URL
            aspect = main_cont.split("/")[corpus_aspect_id[corpus]]
            
            # match slightly wider set of CNN / DM topics
            if aspect.lower()=='showbiz':
                aspect = 'tvshowbiz'
            elif aspect.lower()=='tech':
                aspect = 'sciencetech'
            elif aspect=='football':
                aspect = 'sport'

            if aspect.isdigit():
                aspect = main_cont.split("/")[4]
                
            if  aspect in TARGET_ASPECTS:
                try:
                    doc = open(os.path.join(dm_tokenized_stories_dir, url_hashes[idx]+".story"), 'r').read().split("@highlight")[0]
                except:
                    doc = open(os.path.join(cnn_tokenized_stories_dir, url_hashes[idx]+".story"), 'r').read().split("@highlight")[0]

                # make sure that the associated document satisfies length constraint
                doc_length = len(doc.split(" "))
                if doc_length <= MAX_DOC_LENGTH and doc_length >= 100:
                    if TARGET_ASPECTS[aspect] not in aspect2hashes:
                        aspect2hashes[TARGET_ASPECTS[aspect]] = []
                    aspect2hashes[TARGET_ASPECTS[aspect]].append(url_hashes[idx])
                    DOC_2_LENGTHS[url_hashes[idx]] = doc_length

    # limit the max number of docs per aspect to balance the data set
    for k,v in aspect2hashes.items():
        if len(v) > MAX_DOCS_PER_ASPECT:
            random.shuffle(aspect2hashes[k])
            aspect2hashes[k] = aspect2hashes[k][:MAX_DOCS_PER_ASPECT]
        print(k, len(aspect2hashes[k]))
    
    return aspect2hashes, url_hashes


def create_interleaved_docs(urls, hashes, aspect2hash, output_directory):
    """interleave paragraphs from original CNN / DM articles into multi-aspect documents"""
        
    # create output dir if it doesn't exist
    if not os.path.exists(output_directory): os.makedirs(output_directory)
    
    # create as many interleaved documents as there are in the CNN / DM dataset
    docs_created = 0
    while docs_created < len(hashes):

        # sample number of aspects for this document in [1..4]
        this_n_aspects = random.randint(1, 4)
        
        # first sample the aspects and a corresponding document for each (subject to length constraint)
        # initialize data structs for this interleaved doc
        this_doc_idxs = [0]*this_n_aspects
        this_doc_hashes = [0]*this_n_aspects
        this_docs = [0]*this_n_aspects
        this_summaries = [0]*this_n_aspects
        this_sentence_aspects = []
        this_output_doc = []

        # sample target aspect WITHOUT REPLACEMENT; and random corresponding document
        this_aspects = np.random.choice(len(TARGET_ASPECTS), this_n_aspects, replace=False)

        while True:
            this_tmp_idxs = [random.randint(0, len(aspect2hash[this_aspect])-1) for this_aspect in this_aspects]
            lengths = [DOC_2_LENGTHS[aspect2hash[this_aspect][this_tmp_idxs[idx]]] for idx, this_aspect in enumerate(this_aspects)]

            if sum(lengths) < MAX_OUTPUT_DOC_LEN:
                break


        this_doc_hashes = [aspect2hash[aspect][this_tmp_idxs[idx]]+".story" for idx, aspect in enumerate(this_aspects)]
        
        for i, this_target_doc_hash in enumerate(this_doc_hashes):
            if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, this_target_doc_hash)):
              story_file = os.path.join(cnn_tokenized_stories_dir, this_target_doc_hash)
            elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, this_target_doc_hash)):
              story_file = os.path.join(dm_tokenized_stories_dir, this_target_doc_hash)
            
            # create and record full document and summary
            _, this_target_sum, this_target_doc = get_art_abs(story_file)
                
            this_docs[i] = this_target_doc
            this_summaries[i] = this_target_sum

            # record aspect, doc index in URL and hash list, and hash itself
            this_doc_idxs[i] = hashes.index(this_target_doc_hash.split(".")[0])
            this_doc_hashes[i] = this_target_doc_hash
                
                           
        
        # now, randomly assemble fake doc as chunks of input_docs and keep track of sequence of sentence IDs
        while sum([len(doc) for doc in this_docs]) > 0:
            
            # choose one of the sampled documents
            chunk_doc_idx = random.randint(0, len(this_aspects)-1)

            if len(this_docs[chunk_doc_idx]) > 0:
                # retrieve corresponding aspect
                chunk_topic   = this_aspects[chunk_doc_idx]
                # sample length of paragraph to append to target doc in [1, 5] (except when <5 sentences left in doc)
                chunk_length  = random.randint(1, min(5, len(this_docs[chunk_doc_idx])))
                # append output doc and sent-level aspects accordingly
                this_output_doc.extend(this_docs[chunk_doc_idx][:chunk_length])
                this_sentence_aspects.extend([chunk_topic]*chunk_length)
                assert len(this_output_doc) == len(this_sentence_aspects)
                # chop off used prefix of this_doc
                this_docs[chunk_doc_idx] = this_docs[chunk_doc_idx][chunk_length:]
                
        # write to file, once with each target summary
        for idx, hhash in enumerate(this_doc_hashes):
            with open(os.path.join(output_directory, hhash+"_"+str(docs_created)+".story"), 'w+') as outfile:
                outfile.write(str(this_aspects[idx])+'\t'+TARGET_ASPECTS_IDS[this_aspects[idx]]+'\n\n')
                outfile.write("\n".join(str(this_aspects[ii])+"\t"+urls[hIdx] if ii != idx else "*"+str(this_aspects[ii])+"\t"+urls[hIdx] for ii, hIdx in enumerate(this_doc_idxs))+'\n\n')
                outfile.write("\n".join(str(this_aspects[ii])+"\t"+hashes[hIdx] if ii != idx else "*"+str(this_aspects[ii])+"\t"+hashes[hIdx] for ii, hIdx in enumerate(this_doc_idxs))+'\n\n')
                outfile.write("\n".join(str(this_sentence_aspects[sID])+'\t'+sent for sID, sent in enumerate(this_output_doc))+'\n\n')
                outfile.write(str(this_aspects[idx])+"\t"+this_summaries[idx])

            docs_created += 1
            if docs_created % 5000  == 0:
                print(docs_created, "out of", len(hashes), "docs created")
        

def move_rare_aspect_docs(source, source_url, source_hash, target, target_url, target_hash):
    for aaspect in TARGET_ASPECTS_IDS:
        aspect = TARGET_ASPECTS[aaspect]
        #first for the validation set
        if len(target[aspect]) < MIN_DOCS_PER_ASPECT:
          while len(target[aspect]) < MIN_DOCS_PER_ASPECT:
            n_a = random.randint(0, len(source[aspect])-1)
            target[aspect].append(source[aspect][n_a])
            hash_idx = source_hash.index(target[aspect][-1])
            target_hash.append(target[aspect][-1])
            target_url.append(source_url[hash_idx])
            del source[aspect][n_a]
            del source_hash[hash_idx]
            del source_url[hash_idx]


if __name__ == '__main__':
  if len(sys.argv) != 6:
    print "USAGE: python make_datafiles.py <url_list> <tokenized_cnn_stories_dir> <tokenized_dailymail_stories_dir> <out_txt> <out_binary>"
    sys.exit()

  urls_dir = sys.argv[1]
  cnn_tokenized_stories_dir = sys.argv[2]
  dm_tokenized_stories_dir = sys.argv[3]
  assembled_txt_dir = sys.argv[4]
  assembled_bin_dir = sys.argv[5]
  
  # read URL files
  all_train_urls = read_text_file(os.path.join(urls_dir, "all_train.txt"))
  all_val_urls = read_text_file(os.path.join(urls_dir, "all_val.txt"))
  all_test_urls = read_text_file(os.path.join(urls_dir, "all_test.txt"))
  
  # Create some new directories
  if not os.path.exists(cnn_tokenized_stories_dir): print("Tokenizec CNN corpus doesn't exist (at this location!)")
  if not os.path.exists(dm_tokenized_stories_dir): print("Tokenizec Daily Mail corpus doesn't exist (at this location!)")
  if not os.path.exists(assembled_bin_dir): os.makedirs(assembled_bin_dir)
  
  # Create Aspect -> story_hash dictionary
  test_aspects, test_url_hashes   = aspect_to_hashes(all_test_urls)
  val_aspects, val_url_hashes     = aspect_to_hashes(all_val_urls)
  train_aspects, train_url_hashes = aspect_to_hashes(all_train_urls)
  
    
  # since some aspects have very few individual docs in test / val we move some over from train (and delete them from train)
  # check this for each aspect individually
  move_rare_aspect_docs(train_aspects, all_train_urls, train_url_hashes, val_aspects, all_val_urls, val_url_hashes)
  move_rare_aspect_docs(train_aspects, all_train_urls, train_url_hashes, test_aspects, all_test_urls, test_url_hashes)
  
  for aspect in TARGET_ASPECTS_IDS:
    print("train", aspect, len(train_aspects[TARGET_ASPECTS[aspect]]))
    print("val  ", aspect, len(val_aspects[TARGET_ASPECTS[aspect]]))
    print("test ", aspect, len(test_aspects[TARGET_ASPECTS[aspect]]))
    print("sum  ", aspect, len(train_aspects[TARGET_ASPECTS[aspect]])+len(val_aspects[TARGET_ASPECTS[aspect]])+len(test_aspects[TARGET_ASPECTS[aspect]]))


  # create interleaved documents
  create_interleaved_docs(all_test_urls, test_url_hashes, test_aspects, os.path.join(assembled_txt_dir, "test"))
  create_interleaved_docs(all_val_urls, val_url_hashes, val_aspects, os.path.join(assembled_txt_dir, "val"))
  create_interleaved_docs(all_train_urls, train_url_hashes, train_aspects, os.path.join(assembled_txt_dir, "train"))

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, test_url_hashes, os.path.join(assembled_txt_dir, "test"), assembled_bin_dir, os.path.join(assembled_bin_dir, "test.bin"), data_split="test")
  write_to_bin(all_val_urls, val_url_hashes, os.path.join(assembled_txt_dir, "val"), assembled_bin_dir, os.path.join(assembled_bin_dir, "val.bin"), data_split="val")
  write_to_bin(all_train_urls, train_url_hashes, os.path.join(assembled_txt_dir, "train"), assembled_bin_dir, os.path.join(assembled_bin_dir, "train.bin"), makevocab=True, data_split="train")

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunked
  chunk_all(assembled_bin_dir)
