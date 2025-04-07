# Importing needed libraries for preprocessing and visualization
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Getting the test set
def train_test(datafile, source_language, target_language):
  df = pd.read_csv(datafile)
  print(df.shape)
  df.drop_duplicates(inplace = True)
  print(df.shape)

  # # drop conflicting translations
  # df.drop_duplicates(subset='source_sentence', inplace=True)
  # df.drop_duplicates(subset='target_sentence', inplace=True)
  print(df.shape)

  df.reset_index(drop=True, inplace=True)

  df.dropna(inplace=True)
  print(df.shape)

  train, test = train_test_split(df, test_size=0.2)
  train.to_csv(target_language+'train.csv',index=False)
  test.to_csv('test.csv',index=False)

  # Test
  with open("test."+source_language, "w") as src_file, open("test."+target_language, "w") as trg_file:
    for index, row in test.iterrows():
      src_file.write(row["source_sentence"]+"\n")
      trg_file.write(row["target_sentence"]+"\n")

# Sampling/AL strategies
# 1. Random sampling
def ran_sample(df, number):
  # Shuffle the data
  df = df.sample(frac=1, random_state=42).reset_index(drop=True)
  return df.head(number), df.tail(df.shape[0]-number)

# 2. RTTL
def rttl(files, train_file, number):

  # Getting distribution of training
  train = pd.read_csv(train_file)
  train['tgt_len'] = train.apply(lambda row: len(row.target_sentence.split()), axis = 1)
  bins = [i for i in range(0,141,10)]
  train['dist'] = pd.cut(train['tgt_len'], bins=bins,labels=[i for i in range(1,len(bins))])
  train['dist'].astype('category') 

  props = round((train['dist'].value_counts()/train.shape[0])*number).astype('int').to_dict()
  print(props)

  data = {}
  l  = []
  for i in files: 
    with open(i, "r") as src_file:
        x = src_file.read().splitlines()
        l.append(len(x))
        data[i] = x
  assert l.count(l[0]) == len(l), 'Check your files'

  df = pd.DataFrame(data)
  df.columns = ['source_sentence','target_sentence','translation','scores']
  df['tgt_len'] = df.apply(lambda row: len(row.target_sentence.split()), axis = 1)
  df['dist'] = pd.cut(df['tgt_len'], bins=bins,labels=[i for i in range(1,len(bins))])
  df["scores"] = pd.to_numeric(df["scores"])
  whole = df['dist'].value_counts().to_dict()
  remp = {key: whole[key] - props.get(key, 0) for key in whole}
  selected = df.sort_values(by = ['scores']).groupby('dist', group_keys=False).apply(lambda x: x.head(props[x.name]))
  rem = df.sort_values(by = ['scores']).groupby('dist', group_keys=False).apply(lambda x: x.tail(remp[x.name]))
  print(df.head(1))

  return selected[['source_sentence','target_sentence']], rem[['source_sentence','target_sentence']]

# Setting language and data file:
def data_load(datafile, target_language, query, k, iteration):

  path = os.getcwd() 
  data_path = 'data/'+target_language+'/'+k
  fpath = os.path.join(path, data_path,query)
  print(path)
  print(data_path)
  print(fpath)
  
  if not os.path.exists(fpath):
      os.mkdir(fpath)

  if iteration== str(1):
    df = pd.read_csv(datafile)
    # train, rem = ran_sample(df, 31000)
    print(df.shape)
    # train.to_csv('train.csv',index=False)
    # rem.to_csv('rem.csv',index=False)

  elif iteration== str(2):
    train = pd.read_csv(data_path+'/'+target_language+'train.csv')
    df = pd.read_csv(data_path+'/'+'rem.csv')
    print(train.shape,df.shape)
    t, rem = ran_sample(df, 20000)
    train_full = pd.concat([train,t])

    os.chdir(fpath)

    t.to_csv(iteration+'.csv', index=False)
    train_full.to_csv(query+'_train.csv', index=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)

  else:
    os.chdir(fpath)

    df = pd.read_csv(query+'_rem.csv')
    train, rem = ran_sample(df, 5000)
    print(train.shape)
    train.to_csv(iteration+'.csv', index=False)
    train.to_csv(query+'_train.csv',index=False, mode='a', header=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)

# Setting language and data file:
def rttl_data_load(datafile, target_language, query, k, iteration):

  path = os.getcwd() 
  data_path = 'data/'+target_language+'/'+k
  train_file = '/content/gdrive/Shareddrives/Low_Budget_MT/data/'+target_language+'/'+k+'/'+target_language+'train.csv'
  fpath = os.path.join(path, data_path,query)
  print(path)
  print(data_path)
  print(fpath)
  
  # if not os.path.exists(fpath):
  #     os.mkdir(fpath)

  if iteration== str(2):
    it = str(int(iteration)-1)
    train = pd.read_csv('/content/gdrive/Shareddrives/Low_Budget_MT/data/'+target_language+'/'+k+'/'+target_language+'train.csv')
    print(train.shape)
    t, rem = rttl([fpath+'/1/rem.en', fpath+'/1/rem.'+target_language, fpath+'/1/translation.test', fpath+'/1/translation.test.scores'], train_file, 20000)
    print(t.shape, rem.shape)
    train_full = pd.concat([train,t])

    os.chdir(fpath)

    t.to_csv(iteration+'.csv', index=False)
    train_full.to_csv(query+'_train.csv', index=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)

  else:
    os.chdir(fpath)

    it = str(int(iteration)-1)

    train, rem = rttl([fpath+'/'+it+'/rem.en', fpath+'/'+it+'/rem.'+target_language, fpath+'/'+it+'/translation.test', fpath+'/'+it+'/translation.test.scores'], train_file, 5000)
    print(train.shape)
    train.to_csv(iteration+'.csv', index=False)
    train.to_csv(query+'_train.csv',index=False, mode='a', header=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)

def split_srctgt(train, source_language, target_language, query, k, iteration):

  path = os.getcwd() 
  data_path = 'data/'+target_language+'/'+k+'/'+query
  
  os.chdir(os.path.join(path, data_path))
  if not os.path.exists(iteration):
    os.mkdir(iteration)
  os.chdir(iteration)
  # Splitting train and validation set
  train = pd.read_csv(path+train)
  print(train.shape)
  #num_valid = 1000

  # dev = train.tail(num_valid) 
  # stripped = train.drop(train.tail(num_valid).index)

  # Creating files: Train
  with open("train."+source_language, "w") as src_file, open("train."+target_language, "w") as trg_file:
    for index, row in train.iterrows():
      src_file.write(str(row["source_sentence"])+"\n")
      trg_file.write(str(row["target_sentence"])+"\n")

  # Dev   
  # with open("dev."+source_language, "w") as src_file, open("dev."+target_language, "w") as trg_file:
  #   for index, row in dev.iterrows():
  #     src_file.write(row["source_sentence"]+"\n")
  #     trg_file.write(row["target_sentence"]+"\n")

  os.chdir(path)

def generating_BPE(source_language, target_language, query, k, iteration):
  path = os.getcwd()
  data_path = 'data/'+target_language+'/'+k+'/'+query+'/'+iteration
  dpath = os.getcwd() + '/data/'+target_language+'/'+k
  
  os.chdir(os.path.join(path, data_path))

  os.environ["src"] = source_language 
  os.environ["tgt"] = target_language
  os.environ['k'] = k
  os.environ["iteration"] = iteration
  os.environ['spath'] = dpath
  os.environ['query'] = query
  # Apply BPE splits to the development and test data.
  os.system(' subword-nmt learn-joint-bpe-and-vocab --input train.$src train.$tgt -s 4000 -o bpe.codes.4000 --write-vocabulary vocab.$src vocab.$tgt')

  # Apply BPE splits to the development and test data.
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$src < train.$src > train.bpe.$src')
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$tgt < train.$tgt > train.bpe.$tgt')

  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$src < $spath/dev.$src > dev.bpe.$src')
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$tgt < $spath/dev.$tgt > dev.bpe.$tgt')
  
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$src < $spath/test.$src > test.bpe.$src')
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$tgt < $spath/test.$tgt > test.bpe.$tgt')

  os.chdir(path)
  # Create that vocab using build_vocab
  os.system(' chmod 777 joeynmt/scripts/build_vocab.py')
  os.system('joeynmt/scripts/build_vocab.py data/$tgt/$k/$query/$iteration/train.bpe.$src data/$tgt/$k/$query/$iteration/train.bpe.$tgt --output_path data/$tgt/$k/$query/$iteration/vocab.txt')

def run_model(lan, target_language, source_language, model_name, query, k, iteration):
    path = os.getcwd() 
    data_path = 'data/'+lan+'/'+k+'/'+query+'/'+iteration

    full_path = os.path.join(path, data_path)
    name = '%s%s' % (target_language, source_language)
    
    # Create the config
    config = """
    name: "{model_name}"
    data:
      src: "{target_language}"
      trg: "{source_language}"
      train: "{path}/train.bpe"
      dev:   "{path}/dev.bpe"
      test:  "{path}/test.bpe"
      
      level: "bpe"
      lowercase: False
      max_sent_length: 100
      src_vocab: "{path}/vocab.txt"
      trg_vocab: "{path}/vocab.txt"
    testing:
      score_mode: test
      beam_size: 5
      alpha: 1.0
    training:
      #load_model: path # if uncommented, load a pre-trained model from this checkpoint
      random_seed: 42
      optimizer: "adam"
      normalization: "tokens"
      adam_betas: [0.9, 0.98] 
      scheduling: "plateau"
      patience: 5
      learning_rate_factor: 0.5
      learning_rate_warmup: 1000
      decrease_factor: 0.7
      loss: "crossentropy"
      learning_rate: 0.001
      learning_rate_min: 0.00000001
      weight_decay: 0.0
      label_smoothing: 0.1
      batch_size: 4000
      batch_type: "token"
      eval_batch_size: 3600
      eval_batch_type: "token"
      batch_multiplier: 1
      early_stopping_metric: "ppl"
      epochs: 60  
      validation_freq: 600         # TODO: Set to at least once per epoch.
      logging_freq: 200
      eval_metric: "bleu"
      model_dir: "models/{k}/{model_name}"
      overwrite: True              # TODO: Set to True if you want to overwrite possibly existing models. 
      shuffle: True
      use_cuda: True
      max_output_length: 100
      print_valid_sents: [0, 1, 2, 3]
      keep_best_ckpts: 3
    model:
      initializer: "xavier"
      bias_initializer: "zeros"
      init_gain: 1.0
      embed_initializer: "xavier"
      embed_init_gain: 1.0
      tied_embeddings: True
      tied_softmax: True
      encoder:
          type: "transformer"
          num_layers: 6
          num_heads: 4             
          embeddings:
              embedding_dim: 256
              scale: True
              dropout: 0.2
          # typically ff_size = 4 x hidden_size
          hidden_size: 256  
          ff_size: 1024 
          dropout: 0.3
      decoder:
          type: "transformer"
          num_layers: 6
          num_heads: 4
          embeddings:
              embedding_dim: 256
              scale: True
              dropout: 0.2
          # typically ff_size = 4 x hidden_size
          hidden_size: 256 
          ff_size: 1024
          dropout: 0.3
    """.format(model_name=model_name, k=k, path=full_path, source_language=source_language, target_language=target_language)
    with open("joeynmt/configs/{k}/{model_name}.yaml".format(model_name=model_name, k=k),'w') as f:
      f.write(config)

    return config

def rttl_scoring(rem_file, src, tgt, query, k, iteration, M, Mrev):

  path = '/content/gdrive/Shareddrives/Low_Budget_MT/data/'+tgt+'/'+k+'/'+query+'/'+iteration
  df = pd.read_csv(rem_file)
  print(df.shape)

  os.chdir(path)
  print(os.getcwd())

  # Splitting x and y:
  os.environ['src'] = src
  os.environ['tgt']  = tgt
  with open("rem."+src, "w") as src_file, open("rem."+tgt, "w") as trg_file:
    for index, row in df.iterrows():
      src_file.write(str(row["source_sentence"])+"\n")
      trg_file.write(str(row["target_sentence"])+"\n")

  # Getting bpe for rem
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$tgt < rem.$tgt > rem.bpe.$tgt')
  os.chdir('/content/gdrive/Shareddrives/Low_Budget_MT')
  print(os.getcwd())

  #translate rem
  os.environ["model_path"] = '/content/gdrive/Shareddrives/Low_Budget_MT/joeynmt/models/'+k+'/'+M+'/config.yaml'
  os.environ["x_path"] = path+'/rem.bpe.'+tgt
  os.environ["yhat_path"] = path+'/yhat.'+src

  os.system('cd joeynmt; python -m joeynmt translate $model_path < $x_path > $yhat_path')

  print('Starting here')
  with open(path+'/rem.'+src, "r") as x_file, open(path+'/rem.bpe.'+tgt, "r") as y_file, open(path+'/yhat.'+src, "r") as z_file:
    x = x_file.read().splitlines()
    y = y_file.read().splitlines()
    z = z_file.read().splitlines()
    print(len(x),len(y),len(z))

  with open(path+'/rem.'+tgt, "r") as a_file:
    a = a_file.read().splitlines()

    print(len(a))

  assert len(x)==len(y)==len(z), "Error in translating"

  f = [i for i,j in enumerate(z) if len(j)==0]

  if len(f) !=0:
    x = [i for j, i in enumerate(x) if j not in f]
    y = [i for j, i in enumerate(y) if j not in f]
    z = [i for j, i in enumerate(z) if j not in f]
    a = [i for j, i in enumerate(a) if j not in f]
    
    print(len(x),len(y),len(z))
  
  with open(path+'/rem.'+src, "w") as x_file, open(path+'/rem.bpe.'+tgt, "w") as y_file, open(path+'/yhat.'+src, "w") as z_file:
    for i in zip(x,y,z):
        x_file.write(i[0]+"\n")
        y_file.write(i[1]+"\n")
        z_file.write(i[2]+"\n")

  with open(path+'/rem.'+tgt, "w") as a_file:
    for i in a:
        a_file.write(i+"\n")

  os.chdir(path)
  print(os.getcwd())
  # Getting bpe for rem
  os.system(' subword-nmt apply-bpe -c bpe.codes.4000 --vocabulary vocab.$src < yhat.$src > rem.bpe.$src')
  os.chdir('/content/gdrive/Shareddrives/Low_Budget_MT')
  print(os.getcwd())

  if os.path.exists(path+'/rem.bpe.'+src):
    config = run_model(tgt,src,tgt,Mrev,query,k,iteration)
    reload_config = config.replace(
        f'test:  "{path}/test.bpe"', f'test:  "{path}/rem.bpe"').replace(
            f'score_mode: test', f'score_mode: scoring').replace(
            f'beam_size: 5', f'beam_size: 1')
            
    with open("joeynmt/models/"+k+"/"+Mrev+"/config.yaml",'w') as f:
        f.write(reload_config)

    os.environ["model_path"] = '/content/gdrive/Shareddrives/Low_Budget_MT/joeynmt/models/'+k+'/'+Mrev+'/config.yaml'
    os.environ["output_path"] = path+'/translation'
    print('doing scoring')
    os.system('cd joeynmt; python -m joeynmt test $model_path --output_path $output_path')
  else:
    print('BPE file not created.')

  with open(path+'/translation.test', "r") as x_file, open(path+'/translation.test.scores', "r") as y_file:
    x = x_file.read().splitlines()
    y = y_file.read().splitlines()
    print(len(x),len(y))

# 2. Naive RTTL
def nrttl(files, number):

  data = {}
  l  = []
  for i in files: 
    with open(i, "r") as src_file:
        x = src_file.read().splitlines()
        l.append(len(x))
        data[i] = x
  assert l.count(l[0]) == len(l), 'Check your files'

  df = pd.DataFrame(data)
  df.columns = ['source_sentence','target_sentence','translation','scores']
  df["scores"] = pd.to_numeric(df["scores"])
  df = df.sort_values(by = ['scores'])
  selected = df.head(number)
  rem = df.tail(df.shape[0]-number)
  print(df.head(1))

  return selected[['source_sentence','target_sentence']], rem[['source_sentence','target_sentence']]

# Setting language and data file:
def nrttl_data_load(datafile, target_language, query, k, iteration):

  path = os.getcwd() 
  data_path = 'data/'+target_language+'/'+k
  fpath = os.path.join(path, data_path,query)
  print(path)
  print(data_path)
  print(fpath)
  
  # if not os.path.exists(fpath):
  #     os.mkdir(fpath)

  if iteration== str(2):
    it = str(int(iteration)-1)
    train = pd.read_csv('/content/gdrive/Shareddrives/Low_Budget_MT/data/'+target_language+'/'+k+'/'+target_language+'train.csv')
    print(train.shape)
    t, rem = nrttl([fpath+'/1/rem.en', fpath+'/1/rem.'+target_language, fpath+'/1/translation.test', fpath+'/1/translation.test.scores'], 20000)
    print(t.shape, rem.shape)
    train_full = pd.concat([train,t])

    os.chdir(fpath)

    t.to_csv(iteration+'.csv', index=False)
    train_full.to_csv(query+'_train.csv', index=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)

  else:
    os.chdir(fpath)

    it = str(int(iteration)-1)

    train, rem = nrttl([fpath+'/'+it+'/rem.en', fpath+'/'+it+'/rem.'+target_language, fpath+'/'+it+'/translation.test', fpath+'/'+it+'/translation.test.scores'], 5000)
    print(train.shape)
    train.to_csv(iteration+'.csv', index=False)
    train.to_csv(query+'_train.csv',index=False, mode='a', header=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)

def comet(files, train_file, number):

  data = {}
  l  = []
  for i in files: 
    with open(i, "r") as src_file:
        x = src_file.read().splitlines()
        l.append(len(x))
        data[i] = x
  assert l.count(l[0]) == len(l), 'Check your files'

  df = pd.DataFrame(data)
  df.columns = ['source_sentence','target_sentence','scores']
  df["scores"] = pd.to_numeric(df["scores"])
  df = df.sort_values(by = ['scores'])
  print(df.head(1))

  return df.head(number), df.tail(df.shape[0]-number)

def comet_data_load(datafile, target_language, query, k, iteration):

  path = os.getcwd() 
  data_path = 'data/'+target_language
  train_file = '/content/gdrive/Shareddrives/Low_Budget_MT/data/'+target_language+'/'+k+'/'+target_language+'train.csv'
  fpath = os.path.join(path, data_path,k,query)
  print(path)
  print(data_path)
  print(fpath)

  if iteration== str(2):
    it = str(int(iteration)-1)
    train = pd.read_csv('/content/gdrive/Shareddrives/Low_Budget_MT/data/'+target_language+'/'+k+'/'+target_language+'train.csv')
    print(train.shape)
    t, rem = comet([fpath+'/1/rem.en', fpath+'/1/rem.'+target_language, fpath+'/1/scores'], train_file, 20000)
    print(t.shape, rem.shape)
    train_full = pd.concat([train,t])

    os.chdir(fpath)

    t.to_csv(iteration+'.csv', index=False)
    train_full.to_csv(query+'_train.csv', index=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)

  else:
    os.chdir(fpath)

    it = str(int(iteration)-1)

    train, rem = comet([fpath+'/'+it+'/rem.en', fpath+'/'+it+'/rem.'+target_language, fpath+'/'+it+'/scores'], train_file, 20000)
    print(train.shape)
    train.to_csv(iteration+'.csv', index=False)
    train.to_csv(query+'_train.csv',index=False, mode='a', header=False)
    rem.to_csv(query+'_rem.csv',index=False)

    os.chdir(path)