#!/usr/bin/python
import sys, getopt

import numpy as np

import utility as util

from keras import preprocessing, Sequential, layers
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding, Dropout, Input, Concatenate
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pdb

#reverse_word = 1

def build_data(words_dict, words, char_dict, cat_dict, word_embeddings, input_length, char_win) :
  assert(word_embeddings or (char_dict and char_win and input_length))
  trn_x = []
  int_words_trn = []
  int_cats_trn = []
  word_embeddings_trn = []
  for v in words :
    if char_dict :
      int_words_trn.append([char_dict[c] for c in v])
    if word_embeddings: 
      word_embeddings_trn.append(word_embeddings[v])
    int_cats_trn.append(cat_dict[words_dict[v]])
  #trn_x = [to_categorical(int_w, len(char_set)) for int_w in int_words_trn]
  if char_dict :
    trn_x_post  = pad_sequences(int_words_trn, padding='post', truncating='post', value=0, maxlen=input_length)
    trn_x_pre  = pad_sequences(int_words_trn, padding='pre', truncating='pre', value=0, maxlen=input_length)
    trn_x  = np.concatenate((trn_x_post[:,0:char_win], trn_x_pre[:,-char_win:]),axis=1) 
  if word_embeddings: 
    word_embeddings_trn = np.array(word_embeddings_trn)
  trn_y  = to_categorical((int_cats_trn), len(cat_dict.keys()))
  return trn_x, word_embeddings_trn, trn_y 

def getoptions(argv) :
  (language, char_win, epochs, print_errors, word_emb_file, extra_features_file, char_emb, char_emb_sz, word_emb_sz) = \
      (None, 0, 0, False, None, None, False, 50, None)
  try:
    opts, args = getopt.getopt(argv,"pcl:w:e:b:f:C:d:",["print-error","char-emb","language=","char-win=","epochs=","emb-file=","feat-file=","char-emb-size=","word-emb-size="])
  except getopt.GetoptError:
    print("Unknown argument")
    sys.exit(2)
 
  for opt, arg in opts:
    if opt in ("-p", "--print-error") :
      print_errors = True
    elif opt in ("-c", "--char-emb"):
      char_emb = True
    elif opt in ("-C", "--char-emb-size"):
      char_emb_sz = int(arg)
    elif opt in ("-d", "--word-emb-size"):
      word_emb_sz = int(arg)
    elif opt in ("-l", "--language"):
      language = arg 
    elif opt in ("-w", "--char-win"):
      char_win = int(arg)
    elif opt in ("-e", "--epochs"):
      epochs = int(arg)
    elif opt in ("-b", "--emb-file"):
      word_emb_file = arg
    elif opt in ("-f", "--feat-file"):
      extra_features_file = arg
      
  return (language, char_win, epochs, print_errors, word_emb_file, extra_features_file, char_emb, char_emb_sz, word_emb_sz) 


#def main(language = "Swahili", char_win=3, epochs=10, print_errors=False, word_emb_file=None, extra_features_file=None, char_emb=True) :
def main(argv) :
    (language, char_win, epochs, print_errors, word_emb_file, extra_features_file, char_emb, char_emb_sz, word_emb_sz) = getoptions(argv)
    if language == "Swedish": 
      base_dict = 'dictionaries/Swedish'
    elif language == "Swahili":
      base_dict = 'dictionaries/Swahili'
    elif language == "French":
      base_dict = 'dictionaries/French'
      #base_dict = 'dictionaries/French_inflected'
    elif language == "German":
      base_dict = 'dictionaries/German'
      #base_dict = 'dictionaries/German_inflected'
    elif language == "Icelandic" :
      base_dict = 'dictionaries/Icelandic'
    elif language == "Russian":
      base_dict = 'dictionaries/Russian'
      #base_dict = 'dictionaries/Russian_inflected'
    else:
      print("Could not find language dictionary for %s" % language) 
      return 

    # extra_features_file is not used
    assert(char_emb or not extra_features_file or word_emb_file)

    vcb_file = base_dict
    trn_file = base_dict + '.train'
    dev_file = base_dict + '.dev'
    tst_file = base_dict + '.test'

    [words, cats] = util.read_vocab_file(vcb_file)
    #[trn_words, trn_cats] = util.read_vocab_file(trn_file)
    #[dev_words, dev_cats] = util.read_vocab_file(dev_file)
    #[tst_words, tst_cats] = util.read_vocab_file(tst_file)
    # random split
    trn_words, tst_words, trn_cats, tst_cats = train_test_split(words, cats, test_size=0.1)
    trn_words, dev_words, trn_cats, dev_cats = train_test_split(trn_words, trn_cats, test_size=0.1)

    words_dict = dict(zip(words, cats)) 
    cats_set  = set(cats) ;
    cat_dict  = dict(zip(cats_set, range(0,len(cats_set))))
    num_classes = len(cat_dict.keys())
    print(cat_dict)

    if extra_features_file :
      [w, extra_features] = util.read_vocab_file(extra_features_file)
      extra_features_dict = dict(zip(w, extra_features)) 
      extra_feature_dict = dict(zip(set(extra_features), range(0,len(set(extra_features)))))
      for word in extra_features_dict.keys(): extra_features_dict[word]=extra_feature_dict[extra_features_dict[word]]
      try :
        trn_ext_feat = [feat for word, feat in extra_features if word in trn_words]
        dev_ext_feat = [feat for word, feat in extra_features if word in dev_words]
        tst_ext_feat = [feat for word, feat in extra_features if word in tst_words]
      except :
        print("cannot process the extra features")

    word_embeddings = {}
    word_embeddings_dim = 0
    if word_emb_file :
      print("loading word embeddings from ",word_emb_file)
      with open(word_emb_file) as wfp :
        line = wfp.readline() # always ignore the first line
        line = wfp.readline()
        while line :
          toks = line.strip().split(" ")
          word = toks[0]
          word_embeddings[word] = [float(f) for f in toks[1:]]
          assert(not word_embeddings_dim or word_embeddings_dim == len(word_embeddings[word]))
          word_embeddings_dim = word_embeddings_dim or len(word_embeddings[word])
          try:
            line = wfp.readline()
          except UnicodeDecodeError:
            # this a hacky solution to resolve the word2vec errors
            line = wfp.readline()
            while line:
              if len(line.strip().split(" ")) == word_embeddings_dim + 1:
                break
              line = wfp.readline()
            #line = "__dummy__ " + ' '.join(["0"]*50)
            pass
      if word_emb_sz is not None: 
        assert(word_emb_sz <= word_embeddings_dim)
        for word in word_embeddings:
          word_embeddings[word] = word_embeddings[word][0:word_emb_sz]
        word_embeddings_dim = word_emb_sz

      words_dict = {w: words_dict[w] for w in set(words_dict.keys()) & set(word_embeddings.keys())}
      trn_words = [w for w in set(trn_words) & set(words_dict.keys())]
      dev_words = [w for w in set(dev_words) & set(words_dict.keys())]
      tst_words = [w for w in set(tst_words) & set(words_dict.keys())]

    char_dict = {}
    input_length = None
    if char_emb :
      #if reverse_word == 1 :
      #  words = [w[::-1] for w in words]
      #  trn_words = [w[::-1] for w in trn_words]
      #  dev_words = [w[::-1] for w in dev_words]
      #  tst_words = [w[::-1] for w in tst_words]

      word_len = list(map(len,words_dict.keys()))
      max_word_len = max(word_len)

      if char_win > max_word_len: char_win = max_word_len
      
      char_set  = list(sorted(set([c for w in words_dict.keys() for c in w ])))
      char_dict = dict(zip(['unk'] + char_set, range(0, len(char_set)+1)))

      num_chars   = len(char_dict.keys())
      input_length = max_word_len
      #print([num_chars, num_classes])

      print("input length = %d\n" % input_length)
    
    [trn_x, trn_embeddings, trn_y] = build_data(words_dict, trn_words, char_dict, cat_dict, word_embeddings, input_length, char_win) 
    [dev_x, dev_embeddings, dev_y] = build_data(words_dict, dev_words, char_dict, cat_dict, word_embeddings, input_length, char_win) 
    [tst_x, tst_embeddings, tst_y] = build_data(words_dict, tst_words, char_dict, cat_dict, word_embeddings, input_length, char_win) 

    if char_emb:
      char_input  = Input(shape = (2*char_win,), name='char_input')
      char_x      = Embedding(num_chars, output_dim=char_emb_sz, input_length = 2*char_win, name='embeding')(char_input)
      char_x      = Flatten()(char_x) 
      char_x      = Dropout(rate=0.5)(char_x)
    
    if word_emb_file: 
      word_emb_input   = Input(shape = (word_embeddings_dim,), name='word_emb_input')
      if char_emb :
        concatenated = Concatenate()([char_x, word_emb_input])#, axis=2)
        dens      = Dense(90,  activation='relu')(concatenated)
      else :
        dens = word_emb_input 
    elif char_emb :
      dens      = Dense(90,  activation='relu')(char_x)
    else :
      raise("Either word or character embeddings should be used") 
    
    dens      = Dropout(rate=0.5)(dens)
    dens      = Dense(20, activation='relu')(dens)
    dens      = Dropout(rate=0.5)(dens)
    output    = Dense(num_classes, activation='softmax', name='output')(dens)

    if word_emb_file and char_emb :
      model = Model(inputs=[char_input, word_emb_input], outputs=output)
    elif word_emb_file and not(char_emb):
      model = Model(inputs=word_emb_input, outputs=output)
    elif not(word_emb_file) and char_emb:
      model = Model(inputs=[char_input], outputs=output)
    else :
      raise("Either word or character embeddings should be used") 


    #model = Sequential() 
    #model.add(Embedding(num_chars, output_dim=50, input_length = 2*char_win, name='embeding'))
    ##model.add(Conv1D(filters=2, kernel_size=5, padding='same'))
    ##model.add(MaxPooling1D(pool_size=5 ))
    #model.add(Flatten())
    ##model.add(Dropout(rate=0.5))
    #model.add(Dense(90, activation='relu'))
    #model.add(Dropout(rate=0.5))
    #model.add(Dense(20, activation='relu'))
    #model.add(Dropout(rate=0.5))
    #model.add(Dense(num_classes, activation='softmax', name='output'))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])

    model.summary()
    verbose = 1
    if word_emb_file and char_emb :
      model_train = model.fit([trn_x, trn_embeddings], trn_y, epochs=epochs, verbose=verbose, validation_data=([dev_x, dev_embeddings],dev_y))
    elif word_emb_file and not(char_emb) :
      model_train = model.fit(trn_embeddings, trn_y, epochs=epochs, verbose=verbose, validation_data=(dev_embeddings, dev_y))
    elif not(word_emb_file) and char_emb :
      model_train = model.fit(trn_x, trn_y, epochs=epochs, verbose=verbose, validation_data=(dev_x, dev_y))

    if word_emb_file and char_emb :
      test_loss, test_acc = model.evaluate([tst_x, tst_embeddings], tst_y)
    elif word_emb_file and not(char_emb) :
      test_loss, test_acc = model.evaluate(tst_embeddings, tst_y)
    elif not(word_emb_file) and char_emb :
      test_loss, test_acc = model.evaluate(tst_x, tst_y)
    print("Accuracy: %f" % test_acc)

    if word_emb_file and char_emb :
      softmax_output = model.predict([tst_x, tst_embeddings], batch_size=None, verbose=0)
    elif word_emb_file and not(char_emb) :
      softmax_output = model.predict(tst_embeddings, batch_size=None, verbose=0)
    elif not(word_emb_file) and char_emb :
      softmax_output = model.predict(tst_x, batch_size=None, verbose=0)

    prd_y = np.argmax(softmax_output, axis=1)
    tst_y_int = np.argmax(tst_y, 1) ;
    print(classification_report(tst_y_int, prd_y, target_names=cat_dict.keys()))

    if print_errors:
        print("word\tgender\tpredicted")
        for i in range(0,len(tst_words)) :
            if (tst_y_int[i] != prd_y[i]):
                yg = util.get_key(cat_dict, tst_y_int[i])
                yp = util.get_key(cat_dict, prd_y[i])
                print("%s\t%s\t%s" % (tst_words[i], yg, yp))

    return model


if __name__ == '__main__' :
  main(sys.argv[1:])
