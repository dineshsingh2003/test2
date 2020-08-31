from mongo_db_model import mongo_db_model
from db_model import db_model
from flask import make_response
import flask
from flask import Flask, render_template
from flask import request
from flask import Response
from flask import jsonify,json
from utility import utility  
from file_writer import file_writer 
import time
#import datetime
import numpy as np
from gensim import models,corpora,similarities
from datetime import datetime
import configparser
from textblob import TextBlob

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from urllib.parse import unquote
from flask_cors import CORS
import indiatoday.mongo_db_model as indiatoday_mongo_db_model
import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model

import lallantop.redis_handler as lallantop_redis_handler

import re

#import yaml
import pickle
import gzip
import nltk

from tags.tags_utility import tags_utility
from tags.tags_file_writer import tags_file_writer 
from tags.tags_mongo_db_file_model import tags_mongo_db_file_model
from tags.tags_db_model import tags_db_model

from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import dill
import html
import random

nltk.download('wordnet')

mdb = mongo_db_model()
mdb_it=indiatoday_mongo_db_model.mongo_db_model()
mdfb_at=aajtak_mongo_db_file_model.mongo_db_file_model()

#When add Redis Handler
ltop_rh=lallantop_redis_handler.redis_handler()
r_handler=lallantop_redis_handler.redis_handler()

mdfb_tags = tags_mongo_db_file_model()
u_tags=tags_utility()

#WordNetLemmatizer
fw=file_writer()
config = configparser.ConfigParser()
config.read_file(open('config.properties'))

filepath=config.get('LOGPATH', 'data_transfer_logpath')
model_path=config.get('FILEPATH', 'model_path')
print('Start loading ....')
print('1....')
t1=datetime.now()
#lda_model=mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='lda_model')
#lda = pickle.loads(gzip.decompress(lda_model))
lda=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model')
lda_it=mdb_it.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
dictionary_it = mdb_it.load_latest_version_file_data_in_gridfs(filename='dic_it')
lda_at=mdfb_at.load_latest_version_file_data_in_gridfs(filename='lda_model_at')
dictionary_at = mdfb_at.load_latest_version_file_data_in_gridfs(filename='dic_at')


redisFlag=True


print('Business Today lda Model=>',lda)
print('India Today lda Model=>',lda_it)
print('Aaj Tak lda Model=>',lda_at)

t2=datetime.now()
d=t2-t1
#print("Total Time = %s "%(d))
print('Start for take request.......')



app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

@app.route("/getarticles", methods=['GET', 'POST'])
def getarticles():
    story_count=5
    news_id = int(request.args.get('newsid'))
    utm_source = request.args.get('utm_source')
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception =>',exp)
        story_count=5
 
    if story_count>10:
        story_count=10

    print("source_newsid = ",news_id)
    print("story_count = ",type(story_count))
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1176113
    t1 = datetime.now()
    news_flag = False
    dictionary=None
    terms=None
    #story_count=8

    portal_corpus=[]
    try:
        terms=mdb.get_corpus_data_from_mongodb(collection_name='bt_term', fieldname='news_id', fieldvalue=news_id)
        print('terms =',terms['term'])
        #dictionary = mdb.load_file_data_from_mongodb(collection_name='bt_file_system', filename='dic')
        dictionary = mdb.load_latest_version_file_data_in_gridfs(filename='dic')
        print('dictionary =',dictionary)
        print('dictionary Length =',len(dictionary))
        portal_corpus = [dictionary.doc2bow(terms['term'])]    
        #print('portal_corpus ==>',portal_corpus)
    except Exception as exp:
        print('Exception to get news Article =>',exp)
        news_flag = False
    lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus')))     
    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid')))
    
    latest_data = []
    
    similar_news=[]
    newslist=[]
    newslist_local=[]
    if terms is not None:    
        similar_news = lda_index[lda[portal_corpus]]
        print('similar_news==>',similar_news)
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        log+='|L1=%d'%(news_id)
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            if mapping_id_newsid_dictionary[x[0]]!=news_id:
                log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count] 
        print('newslist_local ==>',newslist_local)
        for x in newslist_local:
            newslist.append(x)
        #log+='|L1 newsid=%d - ref news=%s|'%(session[0]['newsid'],newslist_local)
        newslist_local=[]
        news_flag=True
    log+='|similar_newsids=%s'%(newslist)
    if news_flag==False:
        print("News id not exits =",news_id)
        db = db_model()
        latest_data = db.get_portal_Data(model="BT",LIMIT=story_count)
        for x in latest_data:
            newslist.append(x['newsid'])
        log+='|newsid not exist latest newsid=%s'%(newslist)    
    elif len(newslist)<story_count:
            db = db_model()
            latest_data = db.get_portal_Data(model="BT",LIMIT=(story_count - len(newslist)))
            for x in latest_data:
                newslist.append(x['newsid'])
            log+='|newsid lesser so final latest newsid=%s'%(newslist)        
    print(news_id," ==> ", newslist)
    log+='|response_newsid=%s'%(newslist)  
    t2 = datetime.now()
    d=t2-t1
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    response="SUCCESS"
    message="OK"
    return jsonify(status=response,newsid=newslist,message=message,source_newsid=news_id)

@app.route("/getsimilarnews", methods=['GET', 'POST'])
def getsimilar_news():
    db=db_model()
    u=utility()
    text = request.args.get('text')
    #text = text.decode("utf-8", errors="ignore")
    #json_data=request.json
    #text=json_data['text']
    #print("json_data = ",json_data)
    #text = "phone radio pencle computer system"
    #text = 'The SBI stock rose in Wednesday\'s trade after the state-owned lender posted its biggest ever quarterly loss at Rs 7,718.17 crore and said it has a recognized major portion of its bad loans during the quarter ending March 2018. At 9:24 am, the stock was trading 3.09% or 7.85 points higher at 262 level on the BSE. The stock rose 5 percent intra day to hit a high of 266.85 its highest level since March 1. Brokerages have become positive on the stock after commentary on bad loans. Motilal Oswal has given a \'buy\' recommendation on the stock with a price target of Rs 365."With significant cleansing of the book largely behind, we expect provisioning expenses to decline sharply, while gradual pick -up in loan growth/margins is expected to further boost earnings. We raise our FY19/20E earnings by 36%/12% and revise our price target to  Rs 365 (1.5x FY20E ABV for bank). Maintain Buy, " brokerage Motilal Oswal said in its report.Brokerage Sharekhan too has given has given a buy recommendation on the stock with a price target of Rs 325.The stock has been gaining for the last three days and has risen 10.08% during the period. Prabhudas Lilladher is too positive on the stock and has given a price target of Rs 349. HDFC Securities too has given a buy call on the stock. Darpin Shah in a note said "GNPA jumped 12% QoQ to Rs 2.23 trillion with fresh slippage at Rs 336.7 bn i.e. 7.16% annually. (lower vs our estimates of Rs 450 bn). NNPA jumped 8% QoQ to Rs 1.1 trillion (5.73%). All SDR and S4A exposures have been recognized as NPAs in 4Q. Provisions for NPL jumped to Rs 241 bn (plus 36% QoQ) i.e. 5.12% annualized  vs. 3.92% QoQ leading to profit before tax loss of Rs 12200 crore. The bank has not utilised the RBI dispensation of mortising the MTM loss over the next three quarters. However with tax write back of Rs 4490 crore, SBIN reported a net loss (2nd consecutive qtr) of Rs 7728 crore.  We have BUY rating on the stock. We will revise our estimates/recommendations post the concall."Jaikishan Parmar, senior equity research analyst at Angel Broking said,"Higher losses in the current fiscal can also be attributed to the consolidation of its five subsidiaries into the parent which was effective from April 2017. For the fourth quarter, SBI has reported the second-highest net loss among banks with only PNB worse off, reporting a net loss of Rs 13,417 crore in the fourth quarter.While the gross NPAs went up marginally from 10.35% to 10.90%, the net NPAs were almost flat at 5.73% in the fourth quarter. Fresh slippages (representing new loans turning bad) was at Rs 33,670 crore in the fourth quarter while fresh provisioning was sharply higher at Rs 28,096 crore. This is despite the fact that RBI had permitted banks to provide only 40% for companies under NCLT, as against the original stipulation of 50% provision. The good news from SBI is that it has provided Rs 6,000 crore in the last 2 quarters for bond losses as a result of rising bond yields. The losses are fully provided for despite RBI giving banks 4 quarters to write off the bond losses.Markets appeared to be impressed by the SBI results for two reasons. Firstly, there is the first indication that the NPA cycle may be turning around and combined with growth in advances, this could result in improved profitability in the coming quarters. Secondly, the NCLT resolution will result in a write-back of close to Rs 1 trillion for Indian banks and SBI is likely to be the biggest beneficiary."The stock closed 3.69% or 9.05 points higher at Rs 254.15 Tuesday after the Q4 earnings of the lender were announced. The share opened at 245.10 and surged as much as 6.03% intra day on Tuesday, its highest in over a month after FY19 slippages were seen at 2% or Rs 45,000 crore for a balance sheet size of Rs 20 lakh crore compared to Rs 150,000 crore slippages in FY 18, according to an analysis done by a business news channel.On Tuesday, the bank reported a net loss of Rs 7,718 crore in Q4 of last fiscal. In comparison, in its December quarter, when it had reported its first quarterly loss in 17 years, the net loss stood at Rs 2,416 crore. And the bank had reported a net profit of over Rs 2,814 crore in Q4FY17. In fact, SBI\'s latest loss is the highest quarterly loss figure reported by any bank after Punjab National Bank\'s Rs 13,417-crore loss. The loss came largely from the huge jump in provisions for non-performing assets (NPAs) under the Reserve Bank of India\'s revised framework for resolving stressed assets. Total provisions went up 66.55 per cent in the quarter under review to Rs 23,601 crore against Rs 14,171 crore in the previous quarter.'
    #text=[]
    print("Headers =>",request.headers)
    print("request.mimetype =>",request.mimetype)
    print("Method = ",request.method)
    print("Query String Type => ",type(request.query_string))
    print("Query String => ",request.query_string.decode("utf-8", errors="ignore"))
    t1 = datetime.now()
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+= "|getsimilarnews"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()
    
    #dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='dic')))
    dictionary = mdb.load_latest_version_file_data_in_gridfs(filename='dic')
    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid')))
    lda=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model')
    lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus'))) 

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text length=%d'%(len(text))
        #response="FAIL"
        #message="Text should be more than 30 characters"
        #db = db_model()
        #data = db.get_portal_Data(model="BT",LIMIT=5)
        #data_final['text']=unquote(data[0]['text'])
    else:
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary.doc2bow(text) for text in [lemma_tokens]]
       
        similar_news = lda_index[lda[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        for x in similar_news[:5]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        for x in newslist_local:
            newslist.append(x)
        db = db_model()    
        data=db.pickData_from_newsid(newslist=newslist) 
            
        for decode_data in data:
            decode_data['uri']=(unquote(decode_data['uri']))
            decode_data['title']=(unquote(decode_data['title']))
            data_final.append(decode_data)
            
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    log+= '|text=%s'%(text)
    filename="flask_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data)
    return jsonify(status=response,message=message,result=data_final)

@app.route("/remodel/getterm", methods=['GET', 'POST'])
def getterm():
    news_id = int(request.args.get('newsid'))
    model = request.args.get('model')
    print("source_newsid = ",news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|getterm"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|model=%s"%(model)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    t1 = datetime.now()
    term_flag = False
    newsid_term=None
    newsid_term_data=None

    try:
        newsid_term_data=mdb.get_corpus_data_from_mongodb(collection_name='bt_term', fieldname='news_id', fieldvalue=news_id)
        print('term =',newsid_term_data['term'])
        newsid_term=newsid_term_data['term']
    except Exception as exp:
        print('Exception to getterm =>',exp)
        term_flag = False
    
    if term_flag==False:
        print("Term do not exits for =",news_id)
        log+='|term not exist for newsid=%s'%(news_id)    
    else:
        print("Term do not exits for =",news_id)
        log+='|term exist for newsid=%s'%(news_id)    
	
    t2 = datetime.now()
    d=t2-t1
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    response="SUCCESS"
    message="OK"
    return jsonify(status=response,term=newsid_term,message=message,source_newsid=news_id)
    #return render_template('hello.html',text=f_text,content_hindi=content_hindi,content_marathi=content_marathi)   

@app.route("/remodel/process", methods=['GET', 'POST'])
def process():
    story_count=5
    news_id = int(request.args.get('newsid'))
    utm_source = request.args.get('utm_source')
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception =>',exp)
        story_count=5
 
    if story_count>10:
        story_count=10

    print("source_newsid = ",news_id)
    print("story_count = ",type(story_count))
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|process-algo"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1176113
    t1 = datetime.now()
    news_flag = False
    dictionary=None
    terms=None
    #story_count=8

    portal_corpus=[]
    try:
        #newsid_corpus_mapping_dictionary=mdb.get_corpus_data_from_mongodb(collection_name='bt_corpus_record', fieldname='news_id', fieldvalue=news_id)
        terms=mdb.get_corpus_data_from_mongodb(collection_name='bt_term', fieldname='news_id', fieldvalue=news_id)
        print('terms =',terms['term'])
        #dictionary = mdb.load_file_data_from_mongodb(collection_name='bt_file_system', filename='dic')
        dictionary = mdb.load_latest_version_file_data_in_gridfs(filename='dic')
        #print('dictionary =',dictionary)
        portal_corpus = [dictionary.doc2bow(terms['term'])]    
        #print('portal_corpus ==>',portal_corpus)
    except Exception as exp:
        print('Exception to get news Article =>',exp)
        news_flag = False
    lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus')))     
    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid')))
    #print('mapping_id_newsid_dictionary ==>',mapping_id_newsid_dictionary)

    #lda_model=mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='lda_model')
    #lda = pickle.loads(gzip.decompress(lda_model))
    
    lda=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model')
    
    latest_data = []
    
    similar_news=[]
    newslist=[]
    newslist_local=[]
    #print('terms start.......', )
    mapping_for_algo='News - '
    if terms is not None:    
        try:
            similar_news = lda_index[lda[portal_corpus]]
            print('similar_news==>',similar_news)
            similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        except Exception as exp:
            print("Exception in similar news =",exp)
        log+='|L1=%d'%(news_id)
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            if mapping_id_newsid_dictionary[x[0]]!=news_id:
                log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
                mapping_for_algo += ', (%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count] 
        print('newslist_local ==>',newslist_local)
        for x in newslist_local:
            newslist.append(x)
        #log+='|L1 newsid=%d - ref news=%s|'%(session[0]['newsid'],newslist_local)
        newslist_local=[]
        news_flag=True
    log+='|similar_newsids=%s'%(newslist)
    if news_flag==False:
        print("News id not exits =",news_id)
        db = db_model()
        latest_data = db.get_portal_Data(model="BT",LIMIT=story_count)
        for x in latest_data:
            newslist.append(x['newsid'])
        log+='|newsid not exist latest newsid=%s'%(newslist)    
    elif len(newslist)<story_count:
            db = db_model()
            latest_data = db.get_portal_Data(model="BT",LIMIT=(story_count - len(newslist)))
            for x in latest_data:
                newslist.append(x['newsid'])
            log+='|newsid lesser so final latest newsid=%s'%(newslist)        
    print(news_id," ==> ", newslist)
    log+='|response_newsid=%s'%(newslist)  
    t2 = datetime.now()
    d=t2-t1
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    
    match_list=[]
    nomatch_list=[]

    for t in terms['term']:
        temp_var = dictionary.doc2idx([t])[0]
        if temp_var>-1:
            match_list.append(t)
            #print('Match')
        else:
            nomatch_list.append(t)
            #print('No Match')
    match_per= round((len(match_list)*100)/(len(terms['term'])),2)
    nomatch_per= round((len(nomatch_list)*100)/(len(terms['term'])),2)  
    
    match_list1=None
    match_list2=None
    match_list3=None
    match_list4=None
    match_list5=None
  
    from gensim.corpora import Dictionary
    dictionary_temp = Dictionary([match_list]) 
    #dictionary_temp.token2id
    news_counter=1
    for news_temp in newslist:
        print('R news id=>',news_temp)
        #news_temp=1171419
        try:
            temp_terms=mdb.get_corpus_data_from_mongodb(collection_name='bt_term', fieldname='news_id', fieldvalue=news_temp)
            #print('terms =',temp_terms['term'])
            match_list_temp=[]
            #len(terms['term'])
            
            for t_dic in temp_terms['term']:
                #print('news_counter =>',news_counter)
                temp_var = dictionary_temp.doc2idx([t_dic])[0]
                #temp_var = dictionary_temp.doc2idx(temp_terms['term'])
                #temp_var = dictionary_temp.doc2idx(["sport","Dinesh","use"])[0]
                if temp_var>-1:
                    match_list_temp.append(t_dic)
                    if news_counter==1:
                        match_list1=match_list_temp
                    if news_counter==2:
                        match_list2=match_list_temp
                    if news_counter==3:
                        match_list3=match_list_temp
                    if news_counter==4:
                        match_list4=match_list_temp
                    if news_counter==5:
                        match_list5=match_list_temp
        except Exception as exp:
            print('Exception to get news Article =>',exp)
            news_flag = False
        news_counter +=1                        

        #terms_percetange={} 
        terms_percetange_list=[]
        terms_list=[]  
        for t in match_list:
            res=lda.get_term_topics(t)
            if res!=[]:
                #res = t + "-" + res
                terms_list.append(t)
                terms_percetange_list.append(res)
                #print(t, " ==> ", lda.get_term_topics(t))
        dict_term_topic_distribution=dict(zip(terms_list,terms_percetange_list)) 
       
        all_topics  = lda.get_document_topics(portal_corpus, per_word_topics=True)
        
        document_topics=[]
        for doc_topics, word_topics, phi_values in all_topics:
            document_topics.append(doc_topics)
        #print('Document topics:', document_topics)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
        

    #return jsonify(status=response,newsid=newslist,message=message,source_newsid=news_id)
    return render_template('lda_algo.html',total_term=(0 if terms['term'] is None else len(terms['term'])),term=terms['term'],match_per=match_per,match_list=match_list,match_list_length=(0 if match_list is None else len(match_list)),nomatch_per=nomatch_per,nomatch_list=nomatch_list,nomatch_list_length=(0 if nomatch_list is None else len(nomatch_list)),match_list1=match_list1,match_list1_length=(0 if match_list1 is None else len(match_list1)),match_list2=match_list2,match_list2_length=(0 if match_list2 is None else len(match_list2)),match_list3=match_list3,match_list3_length=(0 if match_list3 is None else len(match_list3)),match_list4=match_list4,match_list4_length=(0 if match_list4 is None else len(match_list4)),match_list5=match_list5,match_list5_length=(0 if match_list5 is None else len(match_list5)),dict_term_topic_distribution=dict_term_topic_distribution,document_topics=sorted(document_topics[0],key=lambda tup: -tup[1]), mapping_for_algo=mapping_for_algo)
    #return render_template('lda_algo.html',total_term=len(terms['term']),term=terms['term'],match_per=match_per,match_list=match_list,match_list_length=len(match_list),nomatch_per=nomatch_per,nomatch_list=nomatch_list,nomatch_list_length=len(nomatch_list),match_list1=match_list1,match_list2=match_list2,match_list3=match_list3,match_list4=match_list4,match_list5=match_list5)

@app.route("/remodel/topics", methods=['GET', 'POST'])
def topics():
    num = int(request.args.get('num'))
    model = request.args.get('model')
    
    if num<=0:
        num = 5
    elif num>1000:
        num=1000    

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|gettopics"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|model=%s"%(model)
    log+= "|queryString=%s"%(request.query_string)
    log+='|num=%s|sessionid=%s'%(num,'None')
    
    lda_topic_list = lda.show_topics(num_topics=-1, num_words=num, log=False, formatted=True)        

    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    
    return render_template('lda_topic.html',lda_topic_list=lda_topic_list)

@app.route("/hello")
def hello():
    print("1.........")
    response="Dinesh test response-2"
    news_id = int(request.args.get('newsid'))
    utm_source = request.args.get('utm_source')
    r=request.query_string
    print("source_newsid = ",news_id)
    print("utm_source =>",type(utm_source), ' == ' ,utm_source)
    print("Query String =>",r)
    return jsonify(status=response)

@app.route("/recengine/it/getarticles1", methods=['GET', 'POST'])
def getarticles_it():
    resp = Response('{"status": "SUCCESS","message": "OK","source_newsid": 0,"data": [{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"},{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"},{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"},{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"},{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"},{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"},{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"},{"newsid": 1333896,"title": "Doorstep delivery of services: Delhi minister reviews preps ahead of launch","uri": "https://www.indiatoday.in/pti-feed/story/doorstep-delivery-of-services-delhi-minister-reviews-preps-ahead-of-launch-1333896-2018-09-06","mobile_image": "https://smedia2.intoday.in/aajtak/images/stories/082018/dilip_kumar_1024_1536146607_88x50.jpeg"}]}')
    #resp.headers['Access-Control-Allow-Origin'] = 'http://10.5.0.189:9090'
    resp.headers['Access-Control-Allow-Origin'] = '*'

    print('Headres =>',resp.headers)
    return resp

@app.route("/recengine/indiacontent/similarimage1", methods=['GET', 'POST'])
def getarticles_indiacontent():
    resp = Response('{ "status": "SUCCESS","message": "OK","source_image_id": 123, "recommnded_image_id": [ 5428313, 5428314, 1525860619385, 1535353323005, 1535353323797], "data": [ {"image_id": 1535353323797, "prodNameLowerCase": "narendra mohan", "pr_id": "634952", "image_caption": "Narendra Mohan  Owner of Dainik gran  Portrait ",  "imageurl": "https://akm-img-a-in.tosshub.com/sites/indiacontent/0/images/product/public/27082018/00/01/53/53/53/32/37/97/1535353323797/150-narendra-mohan-owner-of-dainik-jagran-image-F161SS17_001407.jpg" },  { "image_id": 1535353323005,  "prodNameLowerCase": "narendra mohan",  "pr_id": "634951",      "image_caption": "Narendra Mohan  Owner of Dainik Jagran  Portrait ",      "imageurl": "https://akm-img-a-in.tosshub.com/sites/indiacontent/0/images/product/public/27082018/00/01/53/53/53/32/30/05/1535353323005/150-narendra-mohan-owner-of-dainik-jagran-image-F161SS16_001407.jpg"   },    {     "image_id": 1525860619385,      "prodNameLowerCase": "jk racing asia series",       "pr_id": "600080",       "image_caption": "GREATER NOIDA INDIA - DECEMBER 02 Actor Gul Panag during JK Racing Asia Series at the Buddh International Circuit in Greater Noida on Sunday. Photo by K Asif India Today Group Gul Panag ",       "imageurl": "https://akm-img-a-in.tosshub.com/sites/indiacontent/0/images/product/public/09052018/00/01/52/58/60/61/93/85/1525860619385/150-88005500_20121203_108.jpg"    },     {      "image_id": 5428313,       "prodNameLowerCase": "bindu mairaa 2",       "pr_id": "531126",       "image_caption": "Crystal Healer and tarot card reader Bibdu Maira at the Gurugram Studio - photo by Vikram SHarma",       "imageurl": "https://akm-img-a-in.tosshub.com/sites/indiacontent/0/images/product/public/00/00/05/42/83/13/5428313/150-Bindu Mairaa_2.jpg"    },     {      "image_id": 5428314,       "prodNameLowerCase": "bindu mairaa 1",       "pr_id": "531125",       "image_caption": "Crystal Healer and tarot card reader Bibdu Maira at the Gurugram Studio - photo by Vikram SHarma",       "imageurl": "https://akm-img-a-in.tosshub.com/sites/indiacontent/0/images/product/public/00/00/05/42/83/14/5428314/150-Bindu Mairaa_1.jpg"    }  ]} ')

    resp.headers['Access-Control-Allow-Origin'] = '*'

    print('Headres =>',resp.headers)
    return resp

@app.route("/recengine/indiacontent/similarimage", methods=['GET', 'POST'])
def indiacontent_getsimilarimage():
    import indiacontent.utility as indiacontent_utility 
    import indiacontent.mongo_db_model as indiacontent_mongo_db_model
    import indiacontent.db_model as db_model

    u=indiacontent_utility.utility()
    mdb=indiacontent_mongo_db_model.mongo_db_model()
    db = db_model.db_model()
    image_id=0
    image_count=0
    data_final=[]
    #story_count=5
    utm_source = request.args.get('utm_source')
    text = request.args.get('text')
    #text='Relative values: With foster family; and Niharika Up close and personal, it was a mixed year for the prime minister. There was both a wedding and a funeral in his family. While Vajpayee attended a niece&#039;s wedding reception in February, he lost his sister Urmila Mishra to cancer in May. But both the grief and the celebrations were in private. The prime ministerial public visage was a pair of Ray-Ban glasses. His own birthday found the 79-year-old in eloquent mood. Addressing a crowd of 400 party supporters, Vajpayee wryly said, &quot; Assi saal ke vyakti ko happy returns kehna bade saahas ka kaam hai (It takes some courage to wish an 80-year-old man happy returns).&quot; And typical of the paradox of his life, the prime minister had three birthday celebrations on December 25. One was with PMO officials while driving from Jaipur to Delhi. At midnight, he and his entourage stopped at a restaurant and cut the famous Alwar milk cake. The next was a public function at his official residence, followed by lunch with his foster family. The cake was pineapple, the food, his favourite Chinese, and the conversation, apolitical. At home, his day begins at 7 a.m. with tea and biscuits. Then he gets on the fitness cycle and for half an hour surfs channels while cycling. Breakfast- usually sprouts, upma or idli, toast and'
    #text='KARNATAKA  INDIA-AUGUST 24  A view of food at Aroy Restaurant in Bangalore.  Photo by Sanjay Ramchandran India Today Group'
    #image_id='1535363944419'
    try:
        image_id = int(request.args.get('image_id'))
    except Exception as exp:
        print('Exception in get image_id=>',exp)
        image_id=0
 
    try:
        image_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Iimage Count =>',exp)
        image_count=5

    if image_count>15:
        image_count=15

    print("source_image_id = ",image_id)
    print("story_image_count = ",image_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|indiacontent_get_similar_image"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(image_id,'None')
    #news_id=790807
    t1 = datetime.now()
    dictionary=None
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    dictionary = mdb.load_latest_version_file_data_in_gridfs(filename='dic_indiacontent')
    mapping_id_image_id_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='indiacontent_file_system',filename='id_image_id')))
    lda=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_indiacontent')
    lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='indiacontent_file_system',filename='portal_corpus'))) 

    imagelist=[]
    imagelist_local=[]
    match_list=[]
    nomatch_list=[]
    response="SUCCESS"
    message="OK"
    if text==None or len(text)<=3:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text length=%d'%(len(text))
    else:
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        image_corpus = [dictionary.doc2bow(text) for text in [lemma_tokens]]
        
        try:
            similar_image = lda_index[lda[image_corpus]]
            similar_image = sorted(enumerate(similar_image[0]), key=lambda item: -item[1])
            #for x in similar_image[:image_count]:
            for x in similar_image[:100]:
                if x[1]>0:
                    imagelist_local.append(mapping_id_image_id_dictionary[x[0]])
                    log+=',(%d-%s)'%(mapping_id_image_id_dictionary[x[0]],x[1])
                else:
                    #imagelist_local.append(mapping_id_image_id_dictionary[x[0]])
                    log+=',(%d-%s)'%(mapping_id_image_id_dictionary[x[0]],x[1])
            #print('3.....',similar_image)  
            #image_id=590252 
            #Fetchning imageid and catid
            print('imagelist_local=>',imagelist_local)
            try:
                imageid_catid=db.get_imageid_catid_based_on_prid(image_id)                  
                log+='Cat id =%s'%(imageid_catid[1])
            except Exception as exp:  
                imageid_catid=None
                log+='Cat id = None'
            #Remove is same content is recommending                
            try:
                if(imagelist_local.index(image_id)>=0):
                    imagelist_local.remove(image_id)
            except Exception as exp:
                image_id=0
                imageid_catid=None
                print('Image_id=>',image_id,' not Found')
                
            prid_catid_dic=None
            if imageid_catid!=None:
                try:
                    prid_catid_dic = db.get_prid_categoryid(imagelist_local)
                except Exception as exp:
                    print('image Exception in get pprid_catid_dicrid_catid_dic:',exp)

                for data in imagelist_local:
                    if prid_catid_dic[data]==imageid_catid[1]:
                        match_list.append(data)
                    else:
                        nomatch_list.append(data)
            else:
                nomatch_list=imagelist_local

            imagelist_local=[]                
            print('Match list count=',len(match_list))
            print('No Match list count=',len(nomatch_list))
            if len(match_list)>=image_count:
                for prid in match_list[:image_count]:
                    imagelist_local.append(prid)
            else:
                imagelist_local=match_list
                for prid in nomatch_list[:(image_count - len(imagelist_local))]:
                    imagelist_local.append(prid)
            
            imagelist=imagelist_local
            #imagelist=[591255, 591251, 591888, 591661, 589836]
            image_id=0
            data_final=[]
            #db = db_model.db_model()
            try:
                data=db.pickData_from_imageid_indiacontent(pr_id_list=imagelist) 
            except Exception as exp:
                print('Exception in get image data', exp)

            #data=db.pickData_from_imageid_indiacontent(pr_id_list=imagelist) 
            for decode_data in data:
                decode_data['prodNameLowerCase']=(unquote(decode_data['prodNameLowerCase']))
                #decode_data['pr_id']=(unquote(decode_data['pr_id']))
                decode_data['image_caption']=(unquote(decode_data['image_caption']))
                decode_data['imageurl']=(unquote(decode_data['imageurl']))
                data_final.append(decode_data)
        except Exception as exp:
            print('Exception in get image_id=>',exp)
            response="FAIL"
            message="NOK"
            imagelist=[]

    #newslist = [275318,275317,275316,275315,275313]
    #print("imagelist =>",imagelist)
    log+='|response=%s'%(imagelist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    log+= '|text=%s'%(text)
    filename="flask_web_application_indiacontent_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_image_id=image_id,recommnded_pr_id=imagelist,data=data_final)

#With only Simlarity no Trend and Popular Item in this
@app.route("/recengine/it/getarticles_org", methods=['GET', 'POST'])
def indiatoday_getarticles_org():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model
    import indiatoday.db_model as indiatoday_db_model
    
    u=indiatoday_utility.utility()
    mdb=indiatoday_mongo_db_model.mongo_db_model()
    db = indiatoday_db_model.db_model() 
    story_count=5
    #news_id=1304234
    text=None
    newsdata=[]
    utm_source = request.args.get('utm_source')
    utm_medium = request.args.get('utm_medium')
    #text = request.args.get('text')
    
    #text='Relative values: With foster family; and Niharika Up close and personal, it was a mixed year for the prime minister. There was both a wedding and a funeral in his family. While Vajpayee attended a niece&#039;s wedding reception in February, he lost his sister Urmila Mishra to cancer in May. But both the grief and the celebrations were in private. The prime ministerial public visage was a pair of Ray-Ban glasses. His own birthday found the 79-year-old in eloquent mood. Addressing a crowd of 400 party supporters, Vajpayee wryly said, &quot; Assi saal ke vyakti ko happy returns kehna bade saahas ka kaam hai (It takes some courage to wish an 80-year-old man happy returns).&quot; And typical of the paradox of his life, the prime minister had three birthday celebrations on December 25. One was with PMO officials while driving from Jaipur to Delhi. At midnight, he and his entourage stopped at a restaurant and cut the famous Alwar milk cake. The next was a public function at his official residence, followed by lunch with his foster family. The cake was pineapple, the food, his favourite Chinese, and the conversation, apolitical. At home, his day begins at 7 a.m. with tea and biscuits. Then he gets on the fitness cycle and for half an hour surfs channels while cycling. Breakfast- usually sprouts, upma or idli, toast and'
    try:
        news_id = int(request.args.get('newsid'))
        newsdata=db.get_news_text_for_it(model='IT',news_id=news_id)
        text=unquote(newsdata[0]['full_description'])
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
 
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid')))
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus'))) 

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        
        #response="FAIL"
        #message="Text should be more than 30 characters"
        #db = db_model()
        #data = db.get_portal_Data(model="BT",LIMIT=5)
        #data_final['text']=unquote(data[0]['text'])
    else:
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
       
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        #for x in newslist_local:
         #   newslist.append(x)
        db = indiatoday_db_model.db_model()    
        data=db.pickData_from_newsid_it(newslist=newslist_local) 
        
    counter=1        
    for decode_data in data:
        decode_data['uri']=(unquote(decode_data['uri']))+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        decode_data['title']=(unquote(decode_data['title']))
        decode_data['mobile_image']=(unquote(decode_data['mobile_image']))
        newslist.append(decode_data['newsid'])
        data_final.append(decode_data)
        counter+=1
            
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=news_id,data=data_final)

@app.route("/recengine/at/getarticles", methods=['GET', 'POST'])
def aajtak_getarticles():
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    
    u=aajtak_utility.utility()
    mdfb=aajtak_mongo_db_file_model.mongo_db_file_model()

    story_count=10
    #news_id='1046690'
    newsdata=None
    utm_source = None
    utm_medium = None
    news_id=None
    latest_Flag=False
    t_psl=False
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdfb.get_aajtak_news_text_from_mongodb(collection_name='at_recom',fieldname='id',fieldvalue=news_id)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id='0'
        newsdata=None
        
    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'

    try:
        story_count=10
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|at-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1044367
    t1 = datetime.now()

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    mapping_id_newsid_dictionary = None
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = None
    
    #print(1044367')
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        data=mdfb.get_latest_news_records(collection_name='at_recom', field_name='modified', LIMIT=story_count)
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='1-mapping-dic'
            key_2='1-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    

        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 4]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        similar_news = lda_index[lda_at[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        data=mdfb.get_aajtak_news_data_for_json(collection_name='at_recom',fieldname='id',fieldvaluelist=newslist_local)

    total_dict_data={}
    counter=1
    for row_data in data:
        dict_data={}
        dict_data['newsid']=row_data['id']
        dict_data['title']=row_data['title']
        #dict_data['uri']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        dict_data['uri']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        dict_data['mobile_image']=row_data['media']['kicker_image2']
        total_dict_data[row_data['id']]=dict_data
        newslist_local.append(row_data['id'])
        counter+=1
    
    data_final=[]
    for newsid in newslist_local:
        if len(data_final)==story_count:
            break
        try:
            data_final.append(total_dict_data[newsid])
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_at_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=news_id,data=data_final)


@app.route("/recengine/at/getarticles_amp", methods=['GET', 'POST'])
def aajtak_getarticles_amp():
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    
    u=aajtak_utility.utility()
    mdfb=aajtak_mongo_db_file_model.mongo_db_file_model()

    story_count=10
    #news_id='1046690'
    newsdata=None
    utm_source = None
    utm_medium = None
    news_id=None
    latest_Flag=False
    t_psl=False
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdfb.get_aajtak_news_text_from_mongodb(collection_name='at_recom',fieldname='id',fieldvalue=news_id)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id='0'
        newsdata=None
        
    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'

    try:
        story_count=10
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|at-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    mapping_id_newsid_dictionary = None
    lda_index = None    
    #print(1044367')
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        data=mdfb.get_latest_news_records(collection_name='at_recom', field_name='modified', LIMIT=story_count)
    else:
        if redisFlag:
            #print('5......')  
            key_1='1-mapping-dic'
            key_2='1-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 4]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        similar_news = lda_index[lda_at[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        data=mdfb.get_aajtak_news_data_for_json(collection_name='at_recom',fieldname='id',fieldvaluelist=newslist_local)

    total_dict_data={}
    counter=1
    for row_data in data:
        dict_data={}
        dict_data['newsid']=row_data['id']
        dict_data['title']=row_data['title']
        #dict_data['uri']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        #dict_data['uri']=row_data['amp_url']+'?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d'%(counter)
        dict_data['uri']=row_data['amp_url']+'?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=amp&t_content=footerstrip-%d&t_psl=%s'%(counter,counter,t_psl)
        dict_data['mobile_image']=row_data['media']['kicker_image2']
        total_dict_data[row_data['id']]=dict_data
        newslist_local.append(row_data['id'])
        counter+=1
    
    data_final=[]
    for newsid in newslist_local:
        if len(data_final)==story_count:
            break
        try:
            data_final.append(total_dict_data[newsid])
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_at_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=news_id,items=data_final)

    
@app.route("/recengine/at/unpublish",methods=['GET', 'POST'])
def unpublish_aajtak():
    newsid=None
    model=None
    ctype=None
    key=None
    unpublishtime=None
    update_status=True
    message='SUCCESS'
    try:
        print('1.....')
        model = request.args.get('model',None)
        print('2.....',model)
        newsid = request.args.get('newsid',None)
        print('3.....',newsid)
        ctype = request.args.get('ctype',None)
        print('4.....',ctype)
        unpublishtime = request.args.get('unpublishtime',None)
        print('5.....',key)
        key = request.args.get('key',None)
        print("=====>",model,' -- ',newsid,'  --  ',ctype,' -- ',unpublishtime,' -- ',key)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        update_status=False
        message='FAIL'
        
    if model==None or newsid==None or ctype==None or unpublishtime==None:
        update_status=False
        message='FAIL'
        
    if update_status==True and key=='aajtak$33wfdvdv123':
        update_status=True
    else:
        update_status=False
        message='FAIL'
 
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+= "|unpublish_aajtak"
    print("Flask Time",datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    log+= "|model=%s|newsid=%s|ctype=%s|unpublishtime=%s|key=%s|status=%s"%(model,newsid,ctype,unpublishtime,key,update_status)
    print("log=>",log)
    filename="flask_unpublish_model" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return jsonify(model=model,ctype=ctype,update_status=update_status,newsid=newsid,message=message)


@app.route("/recengine/it/unpublish",methods=['GET', 'POST'])
def unpublish_indiatoday():
    newsid=None
    model=None
    ctype=None
    key=None
    unpublishtime=None
    update_status=True
    message='SUCCESS'
    try:
        print('1.....')
        model = request.args.get('model',None)
        print('2.....',model)
        newsid = request.args.get('newsid',None)
        print('3.....',newsid)
        ctype = request.args.get('ctype',None)
        print('4.....',ctype)
        unpublishtime = request.args.get('unpublishtime',None)
        print('5.....',key)
        key = request.args.get('key',None)
        print("=====>",model,' -- ',newsid,'  --  ',ctype,' -- ',unpublishtime,' -- ',key)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        update_status=False
        message='FAIL'
        
    if model==None or newsid==None or ctype==None or unpublishtime==None:
        update_status=False
        message='FAIL'
        
    if update_status==True and key=='indiatoday$33wfdvdv123':
        update_status=True
    else:
        update_status=False
        message='FAIL'
 
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+= "|unpublish_indiatoday"
    print("Flask Time",datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    log+= "|model=%s|newsid=%s|ctype=%s|unpublishtime=%s|key=%s|status=%s"%(model,newsid,ctype,unpublishtime,key,update_status)
    print("log=>",log)
    filename="flask_unpublish_model" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return jsonify(model=model,ctype=ctype,update_status=update_status,newsid=newsid,message=message)

@app.route("/recengine/bt/unpublish",methods=['GET', 'POST'])
def unpublish_businesstoday():
    newsid=None
    model=None
    ctype=None
    key=None
    unpublishtime=None
    update_status=True
    message='SUCCESS'
    try:
        print('1.....')
        model = request.args.get('model',None)
        print('2.....',model)
        newsid = request.args.get('newsid',None)
        print('3.....',newsid)
        ctype = request.args.get('ctype',None)
        print('4.....',ctype)
        unpublishtime = request.args.get('unpublishtime',None)
        print('5.....',key)
        key = request.args.get('key',None)
        print("=====>",model,' -- ',newsid,'  --  ',ctype,' -- ',unpublishtime,' -- ',key)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        update_status=False
        message='FAIL'
        
    if model==None or newsid==None or ctype==None or unpublishtime==None:
        update_status=False
        message='FAIL'
        
    if update_status==True and key=='businesstoday$bfeuw93d':
        update_status=True
    else:
        update_status=False
        message='FAIL'
 
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+= "|unpublish_businesstoday"
    print("Flask Time",datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    log+= "|model=%s|newsid=%s|ctype=%s|unpublishtime=%s|key=%s|status=%s"%(model,newsid,ctype,unpublishtime,key,update_status)
    print("log=>",log)
    filename="flask_unpublish_model" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return jsonify(model=model,ctype=ctype,update_status=update_status,newsid=newsid,message=message)
    
#==========

@app.route("/translator/lang")
def textlanguage_converter():
    print('Starting ......' + request.method)    
    print("1.......") 
    text = request.args.get('text')
    lang = request.args.get('lang')
    
    status= 'SUCCESS'
    message = 'OK'
    #text='Textblob is amazingly simple to use. What great fun!'
    #lang='hi'
    #text='At the airport of Guwahati, the CISF security personnel allegedly fabricated a pregnant woman and examined whether she is pregnant or not. Dolly Goswami\'s husband Shivam Sarmah has tweeted a complaint with CISF. He said, CISF should learn how pregnancy works with a woman. The CISF staff named Sujata forced my wife to wear clothes to confirm the pregnancy. Is crime in pregnancy in this country?'
    #print('text ==>',text)

    get_language={}
    get_language['hi']='Hindi'
    get_language['en']='English'
    get_language['mr']='Marathi'
    get_language['ml']='Malayalam'
    get_language['kn']='Kannada'
    get_language['gu']='Gujarati'
    get_language['pa']='Punjabi'
    get_language['bn']='Bengali'
    get_language['ta']='Tamil'
    get_language['te']='Telugu'
    get_language['ur']='Urdu'

    blob=TextBlob(text)
    print("blob =>",blob)
    target_text=None
    target_language=None
    source_language=None
    
    try:
        source_language = get_language[blob.detect_language()]
        target_language = get_language[lang]
        target_text = blob.translate(to=lang)
        
        print('target_text=',target_text) 
        
    except Exception as exp:
        print('Exception =>',exp)
        status='FAIL'
        message='Language not detected'
        source_language=None
        target_language=None
    #return jsonify(status=status,message=message,source_language=source_language,target_language=target_language,text=target_text)
    return jsonify(status=status,message=message,source_language=source_language,target_language=target_language,text=str(target_text))
#With Trend:1, Popular:3 , rest Similarity
@app.route("/recengine/it/getarticles_0409", methods=['GET', 'POST'])
def indiatoday_getarticles_0409():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model
    import indiatoday.db_model as indiatoday_db_model
    
    u=indiatoday_utility.utility()
    mdb=indiatoday_mongo_db_model.mongo_db_model()
    db = indiatoday_db_model.db_model() 
    story_count=8
    t=2
    p=2
    #news_id=1449317
    text=None
    newsdata=[]
    utm_source = request.args.get('utm_source')
    utm_medium = request.args.get('utm_medium')
    #text = request.args.get('text')
    
    #text='Relative values: With foster family; and Niharika Up close and personal, it was a mixed year for the prime minister. There was both a wedding and a funeral in his family. While Vajpayee attended a niece&#039;s wedding reception in February, he lost his sister Urmila Mishra to cancer in May. But both the grief and the celebrations were in private. The prime ministerial public visage was a pair of Ray-Ban glasses. His own birthday found the 79-year-old in eloquent mood. Addressing a crowd of 400 party supporters, Vajpayee wryly said, &quot; Assi saal ke vyakti ko happy returns kehna bade saahas ka kaam hai (It takes some courage to wish an 80-year-old man happy returns).&quot; And typical of the paradox of his life, the prime minister had three birthday celebrations on December 25. One was with PMO officials while driving from Jaipur to Delhi. At midnight, he and his entourage stopped at a restaurant and cut the famous Alwar milk cake. The next was a public function at his official residence, followed by lunch with his foster family. The cake was pineapple, the food, his favourite Chinese, and the conversation, apolitical. At home, his day begins at 7 a.m. with tea and biscuits. Then he gets on the fitness cycle and for half an hour surfs channels while cycling. Breakfast- usually sprouts, upma or idli, toast and'
    try:
        news_id = int(request.args.get('newsid'))
        newsdata=db.get_news_text_for_it(model='IT',news_id=news_id)
        text=unquote(newsdata[0]['full_description'])
        #t = int(request.args.get('t'))
        #p = int(request.args.get('p'))
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
        #t=2
        #p=2
 
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid')))
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus'))) 

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        
        #response="FAIL"
        #message="Text should be more than 30 characters"
        #db = db_model()
        #data = db.get_portal_Data(model="BT",LIMIT=5)
        #data_final['text']=unquote(data[0]['text'])
    else:
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
       
        trendnews_mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_trend_news')))
        trendnews_lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='trend_news_corpus'))) 
        
        popularnews_mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_popular_news')))
        popularnews_lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='popular_news_corpus'))) 
        
        trend_similar_news = trendnews_lda_index[lda_it[news_corpus]]
        trend_similar_news = sorted(enumerate(trend_similar_news[0]), key=lambda item: -item[1])

        popular_similar_news = popularnews_lda_index[lda_it[news_corpus]]
        popular_similar_news = sorted(enumerate(popular_similar_news[0]), key=lambda item: -item[1])

        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        
        newslist_local_trend=[]
        for x in trend_similar_news[:t]:
            newslist_local_trend.append(trendnews_mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(trendnews_mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local_trend=list(filter(lambda x:x!=news_id,newslist_local_trend))[:t]    
        
        newslist_local_popular=[]
        for x in popular_similar_news[:p]:
            newslist_local_popular.append(popularnews_mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(popularnews_mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local_popular=list(filter(lambda x:x!=news_id,newslist_local_popular))[:(t+p)]    
        
        s_counter=1
        for x in similar_news[:(story_count + 1)]:
            if s_counter==1:
                newslist_local.append(newslist_local_trend[0])
            if s_counter==3:
                newslist_local.append(newslist_local_popular[0])
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count + t + p]    
        #for x in newslist_local:
         #   newslist.append(x)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         

        db = indiatoday_db_model.db_model()    
        data=db.pickData_from_newsid_it(newslist=newslist_local_final) 
    #utm_medium='WEB'    
    #data_final=[]    
    counter=1        
    for decode_data in data[:story_count]:
        decode_data['uri']=(unquote(decode_data['uri']))+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        decode_data['title']=(unquote(decode_data['title']))
        decode_data['mobile_image']=(unquote(decode_data['mobile_image']))
        newslist.append(decode_data['newsid'])
        data_final.append(decode_data)
        counter+=1
            
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=news_id,data=data_final)


@app.route("/headlinestoday/get_en_keyword", methods=['GET', 'POST'])
def headlinestoday_get_en_keyword():
    print('Start......')
    count=10
    stop=[]
    stop_list=[]
    count_c=count
    try:
        print('Start......2')
        count = int(request.args.get('count'))
    except Exception as exp:
        print('Exception =>',exp)
        count=10
        stop=[]
        
    try:
        stop = request.args.get('stop')
        stop_list=stop.split(',')
        count_c=count + len(stop_list)
        print('Start......3 count', count)
    except Exception as exp:
        print('Exception =>',exp)
        stop=[]
        stop_list=[]
        count_c=count

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|headlinestoday-get_en_keyword"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)
    log+= "|stop=%s"%(request.stop)

    if count>20:
        count=20

    #stop_words=mdfb.load_latest_version_file_data_in_gridfs(filename='stop_words')
    #tag_corpus=mdfb.load_latest_version_file_data_in_gridfs(filename='tag_corpus')
    #feature_names=mdfb.load_latest_version_file_data_in_gridfs(filename='feature_names')
    #tfidf_transformer=mdfb.load_latest_version_file_data_in_gridfs(filename='tfidf_transformer')
    
    X=mdfb_tags.load_latest_version_file_data_in_gridfs(filename='X')
    cv=mdfb_tags.load_latest_version_file_data_in_gridfs(filename='cv')
    print('X=>',X)
    print('cv=>',cv)

    #Most frequently occuring words
    #cv=CountVectorizer(max_df=0.2,min_df=0.001, stop_words=stop_words, max_features=15000, ngram_range=(1,1))
    #X=cv.fit_transform(corpus)
    sum_words = X.sum(axis=0) 
    print('1....')
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    print('2....')
    words_freq =sorted(words_freq, key = lambda x: x[1],reverse=True)
    print('3....')
    
        
    keywords=words_freq[:100]
    #print("\nKeywords:\n========\n", keywords)

    print('4....')    
    klist=[]
    for k in keywords:
        #print(k[0])
        klist.append(k[0])
        #print('Type=>',type(klist))
    print('5....')    
    for s in stop_list:
        #print('6....')
        try:
            klist.remove(s)
        except:
            pass        
    print('7....')

    s_l=['the','will','also','can','just','one','get','take','say','help','due','use','give','think','happen','know','now','new','make','take','come','know','show','see','first','put','much','away','may','local','thing','as','it','his','its','an','he']
    for s in s_l:
        #print('6....')
        try:
            klist.remove(s)
        except:
            pass        
    
    flist=klist[:count]  
    print('flist=>',flist)    
    #log+= "|keyword=%s"%(str(flist))
        #print(k,keywords[k])
    #resp.headers['Access-Control-Allow-Origin'] = '*'
    response="SUCCESS"
    #print('Headres =>',resp.headers)
    return jsonify(status=response,count=count,keyword=flist)
    
@app.route("/headlinestoday/get_en_topic", methods=['GET', 'POST'])
def headlinestoday_get_en_topic():
    print('Start......')
    count=10
    stop=[]
    stop_list=[]
    count_c=count
    try:
        print('Start......2')
        count = int(request.args.get('count'))
    except Exception as exp:
        print('Exception =>',exp)
        count=10
        stop=[]
        
    try:
        stop = request.args.get('stop')
        stop_list=stop.split(',')
        count_c=count + len(stop_list)
        print('Start......3 count', count)
    except Exception as exp:
        print('Exception =>',exp)
        stop=[]
        stop_list=[]
        count_c=count

    if count>20:
        count=20

    #stop_words=mdfb.load_latest_version_file_data_in_gridfs(filename='stop_words')
    #tag_corpus=mdfb.load_latest_version_file_data_in_gridfs(filename='tag_corpus')
    #feature_names=mdfb.load_latest_version_file_data_in_gridfs(filename='feature_names')
    #tfidf_transformer=mdfb.load_latest_version_file_data_in_gridfs(filename='tfidf_transformer')
    
    X=mdfb_tags.load_latest_version_file_data_in_gridfs(filename='X_2')
    cv=mdfb_tags.load_latest_version_file_data_in_gridfs(filename='cv_2')
    print('X=>',X)
    print('cv=>',cv)

    #Most frequently occuring words
    #cv=CountVectorizer(max_df=0.2,min_df=0.001, stop_words=stop_words, max_features=15000, ngram_range=(1,1))
    #X=cv.fit_transform(corpus)
    sum_words = X.sum(axis=0) 
    print('1....')
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    print('2....')
    words_freq =sorted(words_freq, key = lambda x: x[1],reverse=True)
    print('3....')
    
        
    keywords=words_freq[:100]
    s_l=['the','will','also','can','just','one','get','take','say','help','due','use','give','think','happen','know','now','new','make','take','come','know','show','see','first','put','much','away','may','local','thing','as','it','his','its','an','he']
    #print("\nKeywords:\n========\n", keywords)

    print('4....')    
    klist=[]
    for k in keywords:
        #print(k[0])
        klist.append(k[0])
        #print('Type=>',type(klist))
    print('5....')    
    for s in stop_list:
        #print('6....')
        try:
            klist.remove(s)
        except:
            pass        
    print('7....')

    for s in s_l:
        #print('6....')
        try:
            klist.remove(s)
        except:
            pass        

    
    flist=klist[:count]  
    print('flist=>',flist)
               
        #print(k,keywords[k])
    #resp.headers['Access-Control-Allow-Origin'] = '*'
    response="SUCCESS"
    #print('Headres =>',resp.headers)
    return jsonify(status=response,count=count,topic=flist)

@app.route("/headlinestoday/get_hi_keyword", methods=['GET', 'POST'])
def headlinestoday_get_hi_keyword():
    import requests
    import json   
    import uuid
    url='https://us-central1-speech-conversion-213608.cloudfunctions.net/function-1'
    headers = {'Content-Type': 'application/json','Accept':'*','Authorization':'aajtak:aajtak$n738ffwfdd'}
    
    count=10
    stop=[]
    stop_list=[]
    count_c=count
    try:
        print('Start......2')
        count = int(request.args.get('count'))
    except Exception as exp:
        print('Exception =>',exp)
        count=10
        stop=[]
        
    try:
        stop = request.args.get('stop')
        stop_list=stop.split(',')
        count_c=count + len(stop_list)
        print('Start......3 count', count)
    except Exception as exp:
        print('Exception =>',exp)
        stop=[]
        stop_list=[]
        count_c=count



    if count>20:
        count=20

    #stop_words=mdfb.load_latest_version_file_data_in_gridfs(filename='stop_words')
    #tag_corpus=mdfb.load_latest_version_file_data_in_gridfs(filename='tag_corpus')
    #feature_names=mdfb.load_latest_version_file_data_in_gridfs(filename='feature_names')
    #tfidf_transformer=mdfb.load_latest_version_file_data_in_gridfs(filename='tfidf_transformer')
    
    X_hindi=mdfb_tags.load_latest_version_file_data_in_gridfs_dillpackage(filename='X_hindi')
    cv_hindi=mdfb_tags.load_latest_version_file_data_in_gridfs_dillpackage(filename='cv_hindi')
    print('X=>',X_hindi)
    print('cv=>',cv_hindi)

    #Most frequently occuring words
    #cv=CountVectorizer(max_df=0.2,min_df=0.001, stop_words=stop_words, max_features=15000, ngram_range=(1,1))
    #X=cv.fit_transform(corpus)
    sum_words = X_hindi.sum(axis=0) 
    print('1....')
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv_hindi.vocabulary_.items()]
    print('2....')
    words_freq =sorted(words_freq, key = lambda x: x[1],reverse=True)
    print('3....')
    
        
    keywords=words_freq[:100]
    #print("\nKeywords:\n========\n", keywords)

    print('4....')    
    klist=[]
    for k in keywords:
        #print(k[0])
        klist.append(k[0])
        #print('Type=>',type(klist))
    print('5....')    
    for s in stop_list:
        print('6....')
        try:
            klist.remove(s)
        except:
            pass        
    print('7....')
    '''
    s_l=['the','will','also','can','just','one','get','take','say','help','due','use','give','think','happen','know','now','new','make','take','come','know','show','see','first','put','much','away','may','local','thing']
    for s in s_l:
        print('6....')
        try:
            klist.remove(s)
        except:
            pass        
    '''
    flist=klist[:count]  
    
    data_text=[]
    counter=1
    for s in flist:
       print(s)
       if counter>1:
           data_text.append('-')
       data_text.append(s)    
       counter +=1
  
    data ={"lang":"en","text":''.join(data_text),"requestId":"123456789"}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    d=json.loads(r.text)
    c_text=d['convertedText']
    eng_text = c_text.split('-')
    #fflist.append(s + '-' + c_text)
        #fflist.append("-")
        #fflist.append(c_text)
        
    #print(fflist)
            
    #print(k,keywords[k])
    #resp.headers['Access-Control-Allow-Origin'] = '*'
    response="SUCCESS"
    #print('Headres =>',resp.headers)
    return jsonify(status=response,count=count,keyword=flist,eng_keyword=eng_text)

@app.route("/headlinestoday/get_hi_topic", methods=['GET', 'POST'])
def headlinestoday_get_hi_topic():
    import requests
    import json   
    import uuid
    url='https://us-central1-speech-conversion-213608.cloudfunctions.net/function-1'
    headers = {'Content-Type': 'application/json','Accept':'*','Authorization':'aajtak:aajtak$n738ffwfdd'}
    
    print('Start......')
    count=10
    stop=[]
    stop_list=[]
    count_c=count
    try:
        print('Start......2')
        count = int(request.args.get('count'))
    except Exception as exp:
        print('Exception =>',exp)
        count=10
        stop=[]
        
    try:
        stop = request.args.get('stop')
        stop_list=stop.split(',')
        count_c=count + len(stop_list)
        print('Start......3 count', count)
    except Exception as exp:
        print('Exception =>',exp)
        stop=[]
        stop_list=[]
        count_c=count

    if count>20:
        count=20

    #stop_words=mdfb.load_latest_version_file_data_in_gridfs(filename='stop_words')
    #tag_corpus=mdfb.load_latest_version_file_data_in_gridfs(filename='tag_corpus')
    #feature_names=mdfb.load_latest_version_file_data_in_gridfs(filename='feature_names')
    #tfidf_transformer=mdfb.load_latest_version_file_data_in_gridfs(filename='tfidf_transformer')
    
    X_hindi_2=mdfb_tags.load_latest_version_file_data_in_gridfs_dillpackage(filename='X_hindi_2')
    cv_hindi_2=mdfb_tags.load_latest_version_file_data_in_gridfs_dillpackage(filename='cv_hindi_2')
    print('X=>',X_hindi_2)
    print('cv=>',cv_hindi_2)

    #Most frequently occuring words
    #cv=CountVectorizer(max_df=0.2,min_df=0.001, stop_words=stop_words, max_features=15000, ngram_range=(1,1))
    #X=cv.fit_transform(corpus)
    sum_words = X_hindi_2.sum(axis=0) 
    print('1....')
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv_hindi_2.vocabulary_.items()]
    print('2....')
    words_freq =sorted(words_freq, key = lambda x: x[1],reverse=True)
    print('3....')
    
        
    keywords=words_freq[:100]
    #s_l=['the','will','also','can','just','one','get','take','say','help','due','use','give','think','happen','know','now','new','make','take','come','know','show','see','first','put','much','away','may','local','thing']
    #print("\nKeywords:\n========\n", keywords)

    print('4....')    
    klist=[]
    for k in keywords:
        #print(k[0])
        klist.append(k[0])
        #print('Type=>',type(klist))
        
    print('5....')    
    for s in stop_list:
        #print('6....')
        try:
            klist.remove(s)
        except:
            pass        
    print('7....')
 
    
    flist=klist[:count]  

    data_text=[]
    counter=1
    for s in flist:
       print(s)
       if counter>1:
           data_text.append('-')
       data_text.append(s)    
       counter +=1
  
    data ={"lang":"en","text":''.join(data_text),"requestId":"987654321"}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    d=json.loads(r.text)
    c_text=d['convertedText']
    eng_text = c_text.split('-')
   
               
    #print(k,keywords[k])
    #resp.headers['Access-Control-Allow-Origin'] = '*'
    response="SUCCESS"
    #print('Headres =>',resp.headers)
    #return jsonify(status=response,count=count,topic=fflist)
    return jsonify(status=response,count=count,topic=flist,eng_topic=eng_text)

@app.route("/headlinestoday/groupdata", methods=['GET', 'POST'])
def headlinestoday_groupdata():
    groupdata=[]
    response="SUCCESS"
    try:
        groupdata = pickle.loads(gzip.decompress(mdfb_tags.get_data_record_from_mongodb(collection_name='cluster_hl',filename='data')))     
    except Exception as exp:
        response="FAILED"
       
    #print('Headres =>',resp.headers)
    return jsonify(status=response,groupdata=groupdata)

@app.route("/headlinestoday/wordcloud_org", methods=['GET', 'POST'])
def headlinestoday_wordcloud_org():
    wordcloud=[]
    response="SUCCESS"
    try:
        wordcloud = pickle.loads(gzip.decompress(mdfb_tags.get_data_record_from_mongodb(collection_name='cluster_hl',filename='word_cloud_data')))     
    except Exception as exp:
        response="FAILED"
       
    return jsonify(status=response,wordcloud=wordcloud)

@app.route("/recengine/at/video/getarticles", methods=['GET', 'POST'])
def aajtak_video_getarticles():
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    
    u=aajtak_utility.utility()
    mdfb=aajtak_mongo_db_file_model.mongo_db_file_model()

    video_count=5
    #video_id='1095702'
    video_id='0'
    videodata=None
    utm_source = None
    utm_medium = None
    t_psl = False
    
    try:
        video_id = request.args.get('videoid')
        videodata=mdfb.get_aajtak_video_text_from_mongodb(collection_name='at_recom_video',fieldname='id',fieldvalue=video_id)
    except Exception as exp:
        print('Exception in get video id=>',exp)
        video_id='0'
        data=None

    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown' 
    
    
    try:
        video_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        video_count=5

    if video_count>15:
        video_count=15
        
    print("source_videoid = ",video_id)
    print("video_count = ",video_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|at-video-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_video_id=%s|sessionid=%s'%(video_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid_video')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus_video'))) 
    mapping_id_newsid_dictionary = None
    lda_index = None
    
    #print(1044367')
    videolist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if videodata==None or len(videodata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(videodata)
        data=mdfb.get_latest_news_records(collection_name='at_recom_video', field_name='modified', LIMIT=video_count)
    else:
        if redisFlag:
            #print('5......')  
            key_1='1-mapping-dic-video'
            key_2='1-lda-video'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid_video')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus_video'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(videodata)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 3]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        video_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        similar_news = lda_index[lda_at[video_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(video_count + 1)]:
            videolist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        videolist_local=list(filter(lambda x:x!=video_id,videolist_local))[:video_count]    
        
        data=mdfb.get_aajtak_news_data_for_json(collection_name='at_recom_video',fieldname='id',fieldvaluelist=videolist_local)

    total_dict_data={}
    counter=1
    videolist_local_new=[]
    
    #from random import randint
    #hls_link = ['https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_09_baj_gye_1024/02_jul_19_09_baj_gye_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_01_jul_19_smv_rohitanil_1024/02_jul_19_01_jul_19_smv_rohitanil_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19__smv_sunilnew_1024/02_jul_19__smv_sunilnew_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_10_minute_50_khabre_morning_1024/02_jul_19_10_minute_50_khabre_morning_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_smv_gvl_new_1024/02_jul_19_smv_gvl_new_1024.m3u8']
    #mp4_link = ['https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_09_baj_gye_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_01_jul_19_smv_rohitanil_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19__smv_sunilnew_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_10_minute_50_khabre_morning_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_smv_gvl_new_1024_512.mp4']
        
    for row_data in data:
        dict_data={}
        #sources={}
        #sources['type']='application/vnd.apple.mpegurl'
        #sources['file']=hls_link[randint(0, 4)]
        
        dict_data['videoid']=row_data['id']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        #dict_data['link']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['link']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d&t_source=recengine&t_medium=%s&t_content=video_player-slot-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        dict_data['image']=row_data['media']['image']
        #dict_data['sources']=[sources]
        dict_data['duration']=row_data['fileduration']
        
        #dict_data['hls_link']=hls_link[randint(0, 4)]
        #dict_data['mp4_link']=mp4_link[randint(0, 4)]
        
        total_dict_data[row_data['id']]=dict_data
        videolist_local_new.append(row_data['id'])
        counter+=1
    
    data_final=[]
    for videoid in videolist_local_new:
        if len(data_final)==video_count:
            break
        try:
            data_final.append(total_dict_data[videoid])
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(videolist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_at_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_videoid=video_id,playlist=data_final)


@app.route("/recengine/at/story_to_video/getarticles", methods=['GET', 'POST'])
def aajtak_story_to_video_getarticles():
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    
    u=aajtak_utility.utility()
    mdfb=aajtak_mongo_db_file_model.mongo_db_file_model()


    video_count=5
    #news_id='1095736'
    news_id='0'
    newsdata=None
    utm_source = None
    utm_medium = None
    t_psl = False
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdfb.get_aajtak_news_text_from_mongodb(collection_name='at_recom',fieldname='id',fieldvalue=news_id)
        print('newsdata=',newsdata)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id='0'
        newsdata=None

    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown' 
 
    try:
        video_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        video_count=5

    if video_count>15:
        video_count=15
        
    print("source_newsid = ",news_id)
    print("video_count = ",video_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|at-story_to_video-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid_video')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus_video'))) 
    mapping_id_newsid_dictionary = None
    lda_index = None
    
    #print(1044367')
    videolist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        data=mdfb.get_latest_news_records(collection_name='at_recom_video', field_name='modified', LIMIT=video_count)
    else:
        if redisFlag:
            #print('5......')  
            key_1='1-mapping-dic-story-to-video'
            key_2='1-lda-story-to-video'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid_video')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus_video'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 3]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        video_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        similar_news = lda_index[lda_at[video_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(video_count + 1)]:
            videolist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        videolist_local=list(filter(lambda x:x!=news_id,videolist_local))[:video_count]    
        
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        data=mdfb.get_aajtak_news_data_for_json(collection_name='at_recom_video',fieldname='id',fieldvaluelist=videolist_local)

    #from random import randint
    #hls_link = ['https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_09_baj_gye_1024/02_jul_19_09_baj_gye_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_01_jul_19_smv_rohitanil_1024/02_jul_19_01_jul_19_smv_rohitanil_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19__smv_sunilnew_1024/02_jul_19__smv_sunilnew_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_10_minute_50_khabre_morning_1024/02_jul_19_10_minute_50_khabre_morning_1024.m3u8','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_smv_gvl_new_1024/02_jul_19_smv_gvl_new_1024.m3u8']
    #mp4_link = ['https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_09_baj_gye_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_01_jul_19_smv_rohitanil_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19__smv_sunilnew_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_10_minute_50_khabre_morning_1024_512.mp4','https://aajtak-pdelivery.akamaized.net/aajtak/video/2019_07/02_jul_19_smv_gvl_new_1024_512.mp4']

    total_dict_data={}
    counter=1
    videolist_local_new=[]
    for row_data in data:
        dict_data={}
        #sources={}
        #sources['type']='application/vnd.apple.mpegurl'
        #sources['file']=hls_link[randint(0, 4)]
        
        dict_data['videoid']=row_data['id']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['link']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d&t_source=recengine&t_medium=%s&t_content=video_player-slot-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        dict_data['image']=row_data['media']['image']
        #dict_data['sources']=[sources]
        dict_data['duration']=row_data['fileduration']
        
        #dict_data['hls_link']=hls_link[randint(0, 4)]
        #dict_data['mp4_link']=mp4_link[randint(0, 4)]
        
        total_dict_data[row_data['id']]=dict_data
        videolist_local_new.append(row_data['id'])
        counter+=1

    
    data_final=[]
    for videoid in videolist_local_new:
        if len(data_final)==video_count:
            break
        try:
            data_final.append(total_dict_data[videoid])
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(videolist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_at_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=news_id,playlist=data_final)


@app.route("/recengine/ht/getarticles", methods=['GET', 'POST'])
def headlinetoday_getarticles():
    import tags.tags_utility as tags_utility 
    import tags.tags_mongo_db_file_model_recommnedation as tags_mongo_db_file_model_recommnedation
    import tags.tags_db_model as tags_db_model
    
    u=tags_utility.tags_utility()
    mdb=tags_mongo_db_file_model_recommnedation.tags_mongo_db_file_model_recommnedation()
    db = tags_db_model.tags_db_model() 
    story_count=5
    id=0
    text=None
    newsdata=[]
    #utm_source = request.args.get('utm_source')
    #utm_medium = request.args.get('utm_medium')
    #text = request.args.get('text')
    
    #text='Relative values: With foster family; and Niharika Up close and personal, it was a mixed year for the prime minister. There was both a wedding and a funeral in his family. While Vajpayee attended a niece&#039;s wedding reception in February, he lost his sister Urmila Mishra to cancer in May. But both the grief and the celebrations were in private. The prime ministerial public visage was a pair of Ray-Ban glasses. His own birthday found the 79-year-old in eloquent mood. Addressing a crowd of 400 party supporters, Vajpayee wryly said, &quot; Assi saal ke vyakti ko happy returns kehna bade saahas ka kaam hai (It takes some courage to wish an 80-year-old man happy returns).&quot; And typical of the paradox of his life, the prime minister had three birthday celebrations on December 25. One was with PMO officials while driving from Jaipur to Delhi. At midnight, he and his entourage stopped at a restaurant and cut the famous Alwar milk cake. The next was a public function at his official residence, followed by lunch with his foster family. The cake was pineapple, the food, his favourite Chinese, and the conversation, apolitical. At home, his day begins at 7 a.m. with tea and biscuits. Then he gets on the fitness cycle and for half an hour surfs channels while cycling. Breakfast- usually sprouts, upma or idli, toast and'
    try:
        #text = request.args.get('text')
        #news_id = int(request.args.get('newsid'))
        #id=10888803
        id = int(request.args.get('id'))
        newsdata=db.getTextData_tag(id=id)
        text=newsdata[0]['text']
    except Exception as exp:
        print('Exception in get news id=>',exp)
        #id=0
        #text=None

    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    #print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ht-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    #log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    
    newslist=[]
    newslist_local=[]
    data_final = []
    id_status=False
    source_id=id
    language=None
    newslist_dic={}
    try:
        source_id=[]
        source_id.append(newsdata[0]['id'])
        source_id.append(newsdata[0]['site_name'])
        source_id.append(newsdata[0]['title'])
        source_id.append(newsdata[0]['Cat_Name'])
        source_id.append(newsdata[0]['URL'])
        source_id.append(newsdata[0]['Description'])
        #source_id.append(newsdata[0]['language'])
        language=newsdata[0]['language']
        print('language =>',language)
        
        #text=list(d)[2] + ' ' + list(d)[5]
        id_status=True
    except Exception as exp:
        print('Exception in get news id=>',exp)
        text=None
        source_id=id
        id_status=False
        
    if id_status:
            if language=='hn':
                mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_hindi',filename='id_newsid')))
                lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_hindi',filename='portal_corpus'))) 
                latest_data = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_hindi',filename='latest_data'))) 
            elif language=='en':
                mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='id_newsid')))
                lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='portal_corpus'))) 
                latest_data = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='latest_data'))) 
            else:
                mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='id_newsid')))
                lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='portal_corpus'))) 
                latest_data = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='latest_data'))) 
                
    response="SUCCESS"
    message="OK"
    data=""
    print('text=>',text)
    if text==None or len(text)<=10:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        
        #response="FAIL"
        #message="Text should be more than 30 characters"
        #db = db_model()
        #data = db.get_portal_Data(model="BT",LIMIT=5)
        #data_final['text']=unquote(data[0]['text'])
    else:
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
       
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            newslist_dic[mapping_id_newsid_dictionary[x[0]]]=x[1]
            log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=id,newslist_local))[:story_count]    
        #for x in newslist_local:
         #   newslist.append(x)
        #db = indiatoday_db_model.db_model()    
        #data=db.pickData_from_newsid_it(newslist=newslist_local) 
    data_final=[]    
    counter=1        
    for table_id in newslist_local:
        try:
            latest_data_1=latest_data[table_id]
            latest_data_2=latest_data_1 + (str(newslist_dic[table_id]),)
            data_final.append(latest_data_2)
        except Exception as exp:
            print('Exception =>',exp)
        counter+=1
        
        #len(data_final)
            
    #newslist = [275318,275317,275316,275315,275313]
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_ht_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    #return jsonify(status=response,message=message,source_newsid=news_id,data=data_final)
    #return jsonify(status=response,message=message,source_text=text,data=data_final)
    return jsonify(source=source_id,id_status=id_status,data=data_final)

@app.route("/recengine/lt/unpublish",methods=['GET', 'POST'])
def unpublish_lallantop():
    import tts.mongo_db_file_model_tts as mongo_db_file_model_tts
    
    mdfb=mongo_db_file_model_tts.mongo_db_file_model_tts()
    newsid=None
    model=None
    ctype=None
    key=None
    unpublishtime=None
    update_status=True
    message='SUCCESS'
    try:
        print('1.....')
        model = request.args.get('model',None)
        print('2.....',model)
        newsid = request.args.get('newsid',None)
        print('3.....',newsid)
        ctype = request.args.get('ctype',None)
        print('4.....',ctype)
        unpublishtime = request.args.get('unpublishtime',None)
        print('5.....',key)
        key = request.args.get('key',None)
        print("=====>",model,' -- ',newsid,'  --  ',ctype,' -- ',unpublishtime,' -- ',key)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        update_status=False
        message='FAIL'
        
    if model==None or newsid==None or ctype==None:
        update_status=False
        message='FAIL'
    if update_status==True and key=='lallantop$fbwu223' and model.lower()=='lt':
        #update_status=True
        siteid="4"
        update_status=mdfb.update_tts_news(collection_name='data_record',siteid=siteid,newsid=newsid,status=0)    
        if update_status==True:
            message='SUCCESS'
        else:    
            message='FAIL'
    else:
        update_status=False
        message='FAIL'
 
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+= "|unpublish_lallantop"
    print("Flask Time",datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    log+= "|model=%s|newsid=%s|ctype=%s|unpublishtime=%s|key=%s|status=%s"%(model,newsid,ctype,unpublishtime,key,update_status)
    print("log=>",log)
    filename="flask_unpublish_model_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return jsonify(model=model,ctype=ctype,update_status=update_status,newsid=newsid,message=message)

@app.route("/remodel/ht/topics", methods=['GET', 'POST'])
def topics_ht():
    num = int(request.args.get('num'))
    model = request.args.get('model')
    
    if num<=0:
        num = 5
    elif num>1000:
        num=1000    

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|gettopics"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|model=%s"%(model)
    log+= "|queryString=%s"%(request.query_string)
    log+='|num=%s|sessionid=%s'%(num,'None')
    
    lda_topic_list = lda_it.show_topics(num_topics=-1, num_words=num, log=False, formatted=True)        

    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_ht_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    
    return render_template('ht/lda_ht_topic.html',lda_topic_list=lda_topic_list)


@app.route("/remodel/ht/getterm", methods=['GET', 'POST'])
def getterm_ht():
    import tags.tags_utility as tags_utility 
    #import tags.tags_mongo_db_file_model_recommnedation as tags_mongo_db_file_model_recommnedation
    import tags.tags_db_model as tags_db_model

    u=tags_utility.tags_utility()
    #mdb=tags_mongo_db_file_model_recommnedation.tags_mongo_db_file_model_recommnedation()
    db = tags_db_model.tags_db_model() 

    try:
        language=request.args.get('lang')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        language='en'

    try:
        #id=78782520
        id = int(request.args.get('id'))
        newsdata=db.getTextData_tag(id=id,lang=language)
        text=newsdata[0]['text']
    except Exception as exp:
        print('Exception in get news id=>',exp)


    #news_id = int(request.args.get('newsid'))
    model = request.args.get('model')
    print("source_newsid = ",id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|getterm"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|model=%s"%(model)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(id,'None')
    t1 = datetime.now()
    term_flag = False
    newsid_term=None

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    response="SUCCESS"
    message="OK"
    portal_corpus=[]
    document_topics=[]
    dict_term_topic_distribution={}

    try:
        length=len(text)
        print('length=>',length)
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #newsid_term_data=mdb.get_corpus_data_from_mongodb(collection_name='bt_term', fieldname='news_id', fieldvalue=news_id)
        #print('term =',newsid_term_data['term'])
        newsid_term=lemma_tokens
        portal_corpus = [dictionary_it.doc2bow(newsid_term)] 
        term_flag=True
        
        #terms_percetange={} 
        terms_percetange_list=[]
        terms_list=[]  
        newsid_term_temp=['default','august']
        print('1.....')
        for t in newsid_term:
            print('2.....')
            res=None
            try:
                res=lda_it.get_term_topics(t)
            except Exception as exp:
                print('Exception inner :', exp)
                res=None
            print('3.....')
            print(res)
            print('4.....')
            if res!=None and res!=[]:
                #res = t + "-" + res
                terms_list.append(t)
                terms_percetange_list.append(res)
                #print(t, " ==> ", lda.get_term_topics(t))
            print('5.....')    
        dict_term_topic_distribution=dict(zip(terms_list,terms_percetange_list)) 
        print('6.....')
        '''   
        for key, value in dict_term_topic_distribution.items():
            print(key, ' -- ', value)        
        '''
    except Exception as exp:
        print('Exception to getterm =>',exp)
        term_flag = False

    
    if term_flag==False:
        print("Term do not exits for =",id)
        log+='|term not exist for newsid=%s'%(id)    
        response="FAIL"
        message="NOK"
    #else:
    all_topics  = lda_it.get_document_topics(portal_corpus, per_word_topics=True)
    
    for doc_topics, word_topics, phi_values in all_topics:
        print(doc_topics)
        print("\n")
        document_topics.append(doc_topics)
    print("Term exits for =",id)
    log+='|term exist for newsid=%s'%(id)    
	
    t2 = datetime.now()
    d=t2-t1
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_ht_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,message=message,source_newsid=id,terms=newsid_term,document_topics_distribution=str(document_topics),term_topic_distribution=str(dict_term_topic_distribution))
    return render_template('ht/term_ht.html',terms=newsid_term,document_topics_distribution=document_topics,dict_term_topic_distribution=dict_term_topic_distribution,length=length)   

@app.route("/remodel/ht/process", methods=['GET', 'POST'])
def process_ht():
    import tags.tags_utility as tags_utility 
    import tags.tags_mongo_db_file_model_recommnedation as tags_mongo_db_file_model_recommnedation
    import tags.tags_db_model as tags_db_model
    from gensim import similarities

    u=tags_utility.tags_utility()
    mdb=tags_mongo_db_file_model_recommnedation.tags_mongo_db_file_model_recommnedation()
    db = tags_db_model.tags_db_model() 
    
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()
    process_Flag=False

    story_count=5
    text=None
    newsdata=[]
    language='en'
    news_topics_distribution={}
    lemma_tokens=[]

    dictionary=None
    terms=None
    similar_news=[]
    newslist=[]
    newslist_local=[]


    try:
        language=request.args.get('lang')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        language='en'

    try:
        #id=77611841
        id = int(request.args.get('id'))
        newsdata=db.getTextData_tag(id=id,lang=language)
        text=newsdata[0]['text']
    except Exception as exp:
        print('Exception in get news id=>',exp)


    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ht-process-algo"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(id,'None')


    print("source_newsid = ",id)
    #print("story_count = ",type(story_count))

    
    if text==None or len(text)<=10:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
    else:
        if language=='en':
            log+= '|SUCCESS'
            log+= '|Result'
            text = text.lower()
            text = u.clean_doc(text)
            tokens = tokenizer.tokenize(text)
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
            #print(stopped_tokens)
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            #print(stemmed_tokens)
            lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
            terms=lemma_tokens
            #print("lemma_tokens => ",lemma_tokens)
            lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='portal_corpus')))    
            portal_corpus = [dictionary_it.doc2bow(lemma_tokens)]    
            
            #print('101......portal_corpus=',type(portal_corpus),portal_corpus)
            #print('lda_it=',type(lda_it),lda_it)
            #print('lda_index=',type(lda_index),lda_index)
            #lda_p_corpas=lda_it[portal_corpus]
            #print('lda_p_corpas=>',lda_p_corpas)
            s_news = lda_index[lda_it[portal_corpus]]
            print('102......')
            similar_news = sorted(enumerate(s_news[0]), key=lambda item: -item[1])
            #print('103......',similar_news[:6])
            print('103......')
            process_Flag=True

        elif language=='hn':
            hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
            log+= '|SUCCESS'
            log+= '|Result'
            clean_text = u.clean_doc_hindi(text)
            cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
            tokens = cleaned_tokens_n.split(' ')
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
            stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        else:
            print("No else Data........")
            log+= '|FAIL'
            log+= '|No Process,text=%s'%(text)



    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='id_newsid')))
    
    mapping_for_algo='News - '
    print('100......')
    #if terms is not None: 
    if process_Flag:
        log+='|L1=%d'%(id)
        print('104......')
        #for x in similar_news[:(story_count + 1)]:
        for x in similar_news[:6]:
            print('105......')
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            if mapping_id_newsid_dictionary[x[0]]!=id:
                log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
                mapping_for_algo += ', (%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            print('106......')    
        newslist_local=list(filter(lambda x:x!=id,newslist_local))[:story_count] 
        print('newslist_local ==>',newslist_local)
        for x in newslist_local:
            newslist.append(x)
    
    match_list=[]
    nomatch_list=[]
    
    dictionary=dictionary_it
    
    #len(match_list)

    for t in terms:
        temp_var = dictionary.doc2idx([t])[0]
        if temp_var>-1:
            match_list.append(t)
            #print('Match =>',t)
        else:
            nomatch_list.append(t)
            #print('No Match =>',t)
    
    match_per='100'
    nomatch_per='0'
    
    if len(terms)>0: 
        match_per= round((len(match_list)*100)/(len(terms)),2)
        nomatch_per= round((len(nomatch_list)*100)/(len(terms)),2)  
    
    match_list1=None
    match_list2=None
    match_list3=None
    match_list4=None
    match_list5=None
  
    news_counter=1
    for news_temp in newslist:
        
        print('R news id=>',news_temp)
        #news_temp=77611967
        try:
            newsdata=db.getTextData_tag(id=news_temp,lang=language)
            text=newsdata[0]['text']
            
            match_list_temp=[]
            #len(terms['term'])

            text = text.lower()
            text = u.clean_doc(text)
            tokens = tokenizer.tokenize(text)
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
            #print(stopped_tokens)
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            #print(stemmed_tokens)
            lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
            
            temp_terms=lemma_tokens
            #dictionary_temp = Dictionary([temp_terms]) 
            #print(dictionary_temp)
            #len(terms)
            #match_list_temp=[]
            for t_dic in match_list:
                #print('news_counter =>',news_counter)
                #print(t_dic)
                #temp_var = dictionary_temp.doc2idx([t_dic])[0]
                #print(t_dic,' == ',temp_var)
                
                try:
                    temp_var=temp_terms.index(t_dic)
                    #print(t_dic)
                except Exception as exp:
                    temp_var=-1
                    #print("Exception in similar news =",exp,' :x=',x)
                
                if temp_var>-1:
                    match_list_temp.append(t_dic)
            if news_counter==1:
                match_list1=match_list_temp
            if news_counter==2:
                match_list2=match_list_temp
            if news_counter==3:
                match_list3=match_list_temp
            if news_counter==4:
                match_list4=match_list_temp
            if news_counter==5:
                match_list5=match_list_temp
           
            lda_it_topic=lda_it
            #local_portal_corpus=[]    
            #local_portal_corpus = [dictionary_it.doc2bow(match_list_temp)] 
            #topics_distribution  = lda_it.get_document_topics(local_portal_corpus, per_word_topics=True)
            topics_distribution  = lda_it_topic.get_document_topics([dictionary_it.doc2bow(match_list_temp)])
            news_topics_distribution[news_temp]=topics_distribution[0]

        except Exception as exp:
            print('Exception to get news Article =>',exp)
            #news_flag = False
        print('news_topics_distribution =>',news_topics_distribution)    
        news_counter +=1                        

    #all_topics  = lda_it.get_document_topics(portal_corpus, per_word_topics=True)
    all_topics  = lda_it.get_document_topics(portal_corpus)
    
    document_topics=all_topics[0]
    #for doc_topics, word_topics, phi_values in all_topics:
        #print(doc_topics)
        #print("\n")
        #document_topics.append(doc_topics)
    
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_ht_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return render_template('ht/ht_process_algo.html',total_term=(0 if terms is None else len(terms)),term=terms,match_per=match_per,match_list=match_list,match_list_length=(0 if match_list is None else len(match_list)),nomatch_per=nomatch_per,nomatch_list=nomatch_list,nomatch_list_length=(0 if nomatch_list is None else len(nomatch_list)),match_list1=match_list1,match_list1_length=(0 if match_list1 is None else len(match_list1)),match_list2=match_list2,match_list2_length=(0 if match_list2 is None else len(match_list2)),match_list3=match_list3,match_list3_length=(0 if match_list3 is None else len(match_list3)),match_list4=match_list4,match_list4_length=(0 if match_list4 is None else len(match_list4)),match_list5=match_list5,match_list5_length=(0 if match_list5 is None else len(match_list5)), mapping_for_algo=mapping_for_algo,document_topics=document_topics,news_topics_distribution=news_topics_distribution)

@app.route("/remodel/ht/process_test", methods=['GET', 'POST'])
def process_ht_test():
    import tags.tags_utility as tags_utility 
    import tags.tags_mongo_db_file_model_recommnedation as tags_mongo_db_file_model_recommnedation
    import tags.tags_db_model as tags_db_model
    from gensim import similarities

    u=tags_utility.tags_utility()
    mdb=tags_mongo_db_file_model_recommnedation.tags_mongo_db_file_model_recommnedation()
    db = tags_db_model.tags_db_model() 
    
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    text=None
    newsdata=[]
    language='en'
    lemma_tokens=[]
    portal_corpus=[]
    similar_news=[]
    process_Flag=False
    language='en'
    news_topics_distribution={}
    document_topics=[]
    terms=[]


    try:
        id = int(request.args.get('id'))
        newsdata=db.getTextData_tag(id=id,lang=language)
        text=newsdata[0]['text']
    except Exception as exp:
        print('Exception in get news id=>',exp)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ht-process-algo"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(id,'None')

    if text==None or len(text)<=10:
        print("No Source Data........")
    else:
        if language=='en':
            try:
                text = text.lower()
                text = u.clean_doc(text)
                tokens = tokenizer.tokenize(text)
                cleaned_tokens = [word for word in tokens if len(word) > 2]
                stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
                stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
                lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
                lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='portal_corpus')))    
                terms=lemma_tokens
                lda_it_temp=lda_it
                portal_corpus = [dictionary_it.doc2bow(terms)]    
                print('99.......')    
                #all_topics  = lda_it.get_document_topics(portal_corpus, per_word_topics=True)
                all_topics  = lda_it_temp.get_document_topics([dictionary_it.doc2bow(terms)])
                #print('all_topics =>',all_topics[0][0])
                
                '''
                #for doc_topics, word_topics, phi_values in all_topics:
                for doc_topics in all_topics[0]:
                    print('doc_topics =>',doc_topics)
                    #print("\n")
                    document_topics.append(doc_topics)
                '''    
                document_topics=all_topics[0]
                print('document_topics=>',document_topics)    
                print('101......portal_corpus=',type(portal_corpus),portal_corpus)
                print('lda_it=',type(lda_it_temp),lda_it_temp)
                print('lda_index=',type(lda_index),lda_index)
                #lda_p_corpas=lda_it_temp[portal_corpus]
                #print('lda_p_corpas=>',lda_p_corpas)
                s_news = lda_index[lda_it[portal_corpus]]
                print('102......')
                similar_news = sorted(enumerate(s_news[0]), key=lambda item: -item[1])
                print('103......')
                process_Flag=True
            except Exception as exp:
                print("Exception in similar news test=",exp)

    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system',filename='id_newsid')))
    
    mapping_for_algo='News - '
    print('100......')
    #if terms is not None: 
    newslist_local=[]
    newslist=[]
    if process_Flag:
        print('103.5......')
        #similar_news = sorted(enumerate(s_news[0]), key=lambda item: -item[1])
        log+='|L1=%d'%(id)
        print('104......')
        #for x in similar_news[:(story_count + 1)]:
        for x in similar_news[:6]:
            print('105......')
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            if mapping_id_newsid_dictionary[x[0]]!=id:
                log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
                mapping_for_algo += ', (%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            print('106......')    
        newslist_local=list(filter(lambda x:x!=id,newslist_local))[:5] 
        print('newslist_local ==>',newslist_local)
        for x in newslist_local:
            newslist.append(x)

    match_list=[]
    nomatch_list=[]
    
    
    dictionary=dictionary_it
    
    #len(match_list)

    for t in terms:
        temp_var = dictionary.doc2idx([t])[0]
        if temp_var>-1:
            match_list.append(t)
            #print('Match =>',t)
        else:
            nomatch_list.append(t)
            #print('No Match =>',t)
    
    match_per='100'
    nomatch_per='0'
    
    if len(terms)>0: 
        match_per= round((len(match_list)*100)/(len(terms)),2)
        nomatch_per= round((len(nomatch_list)*100)/(len(terms)),2)  
    
    match_list1=None
    match_list2=None
    match_list3=None
    match_list4=None
    match_list5=None
    lda_it_topic=lda_it

    news_counter=1
    for news_temp in newslist:
        print('R news id=>',news_temp)
        #news_temp=77611967
        try:
            newsdata=db.getTextData_tag(id=news_temp,lang=language)
            text=newsdata[0]['text']
            
            match_list_temp=[]
            #len(terms['term'])

            text = text.lower()
            text = u.clean_doc(text)
            tokens = tokenizer.tokenize(text)
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
            #print(stopped_tokens)
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            #print(stemmed_tokens)
            lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
            
            temp_terms=lemma_tokens
            #dictionary_temp = Dictionary([temp_terms]) 
            #print(dictionary_temp)
            #len(terms)
            #match_list_temp=[]
            for t_dic in match_list:
                #print('news_counter =>',news_counter)
                #print(t_dic)
                #temp_var = dictionary_temp.doc2idx([t_dic])[0]
                #print(t_dic,' == ',temp_var)
                
                try:
                    temp_var=temp_terms.index(t_dic)
                    #print(t_dic)
                except Exception as exp:
                    temp_var=-1
                    #print("Exception in similar news =",exp,' :x=',x)
                
                if temp_var>-1:
                    match_list_temp.append(t_dic)
                    #print('match_list_temp=>',match_list_temp)
            if news_counter==1:
                match_list1=match_list_temp
            if news_counter==2:
                match_list2=match_list_temp
            if news_counter==3:
                match_list3=match_list_temp
            if news_counter==4:
                match_list4=match_list_temp
            if news_counter==5:
                match_list5=match_list_temp
           
            #local_portal_corpus=[]    
            #local_portal_corpus = [dictionary_it.doc2bow(match_list_temp)] 
            #print('local_portal_corpus=>',local_portal_corpus)
            #topics_distribution  = lda_it_topic.get_document_topics(portal_corpus, per_word_topics=True)
            topics_distribution  = lda_it_topic.get_document_topics([dictionary_it.doc2bow(match_list_temp)])
            print('topics_distribution=>',topics_distribution)
            print('news_counter =>',news_counter)
            news_topics_distribution[news_temp]=topics_distribution[0]
            print('news_topics_distribution=>',news_topics_distribution)

        except Exception as exp:
            print('Exception to get news Article =>',exp)
            #news_flag = False
        #print('news_topics_distribution =>',news_topics_distribution)    
        news_counter +=1                        

    #return jsonify(status=str(similar_news[:6]),newslist=str(newslist),match_per=match_per,document_topics=str(document_topics))
    return render_template('ht/ht_process_algo.html',total_term=(0 if terms is None else len(terms)),term=terms,match_per=match_per,match_list=match_list,match_list_length=(0 if match_list is None else len(match_list)),nomatch_per=nomatch_per,nomatch_list=nomatch_list,nomatch_list_length=(0 if nomatch_list is None else len(nomatch_list)),match_list1=match_list1,match_list1_length=(0 if match_list1 is None else len(match_list1)),match_list2=match_list2,match_list2_length=(0 if match_list2 is None else len(match_list2)),match_list3=match_list3,match_list3_length=(0 if match_list3 is None else len(match_list3)),match_list4=match_list4,match_list4_length=(0 if match_list4 is None else len(match_list4)),match_list5=match_list5,match_list5_length=(0 if match_list5 is None else len(match_list5)), mapping_for_algo=mapping_for_algo,document_topics=document_topics,news_topics_distribution=news_topics_distribution)


@app.route("/recengine/it/video/getarticles", methods=['GET', 'POST'])
def indiatoday_video_getarticles():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_file_model as indiatoday_mongo_db_file_model
    
    u=indiatoday_utility.utility()
    mdfb=indiatoday_mongo_db_file_model.mongo_db_file_model()

    video_count=5
    #video_id='1590753'
    video_id='0'
    videodata=None
    utm_source = None
    utm_medium = None
    t_psl = False
    
    try:
        video_id = request.args.get('videoid')
        videodata=mdfb.get_indiatoday_video_text_from_mongodb(collection_name='it_recom_video',fieldname='n_id',fieldvalue=video_id)
        utm_medium = request.args.get('utm_medium')
        #utm_medium='web'
    except Exception as exp:
        print('Exception in get video id=>',exp)
        video_id='0'

    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown' 
        
    try:
        video_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        video_count=5

    if video_count>15:
        video_count=15
        
    print("source_videoid = ",video_id)
    print("video_count = ",video_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-video-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_video_id=%s|sessionid=%s'%(video_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()
    
    #hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_video')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_video'))) 

    mapping_id_newsid_dictionary = None
    lda_index = None
    
    #print(1044367')
    videolist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if videodata==None or len(videodata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(videodata)
        data=mdfb.get_latest_news_records(collection_name='tbl_it_videodata', field_name='publishdate', LIMIT=video_count)
    else:
        if redisFlag:
            #print('5......')  
            key_1='2-mapping-dic-video'
            key_2='2-lda-video'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_video')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_video'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        videodata = videodata.lower()
        videodata = u.clean_doc(videodata)
        tokens = tokenizer.tokenize(videodata)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        video_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        similar_news = lda_index[lda_it[video_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(video_count + 1)]:
            videolist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        videolist_local=list(filter(lambda x:x!=video_id,videolist_local))[:video_count]    
        
        print('videolist_local =>', videolist_local)
        
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        data=mdfb.get_aajtak_news_data_for_json(collection_name='tbl_it_videodata',fieldname='videoid',fieldvaluelist=videolist_local)

    total_dict_data={}
    counter=1
    videolist_local_new=[]
    
    for row_data in data:
        dict_data={}
        dict_data['videoid']=row_data['videoid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d&t_source=recengine&t_medium=%s&t_content=video_player-slot-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        dict_data['image']=row_data['mobile_image']
        total_dict_data[row_data['videoid']]=dict_data
        videolist_local_new.append(row_data['videoid'])
        counter+=1
    
    data_final=[]
    for videoid in videolist_local_new:
        if len(data_final)==video_count:
            break
        try:
            data_final.append(total_dict_data[videoid])
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(videolist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_videoid=video_id,playlist=data_final)

@app.route("/recengine/it/story_to_video/getarticles", methods=['GET', 'POST'])
def indiatoday_story_to_video_getarticles():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_file_model as indiatoday_mongo_db_file_model
    #import indiatoday.db_model as indiatoday_db_model
    
    u=indiatoday_utility.utility()
    mdfb=indiatoday_mongo_db_file_model.mongo_db_file_model()
    #db = indiatoday_db_model.db_model() 

    video_count=5
    #news_id=1591282
    news_id='0'
    newsdata=None
    utm_source = None
    utm_medium = None
    t_psl = False
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdfb.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=news_id)
        text=newsdata
    except Exception as exp:
        print('Exception in get video id=>',exp)
        news_id='0'
        text=None
 
    try:
        utm_medium = request.args.get('utm_medium')
        #utm_medium='web'
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'
        
    try:
        utm_source = request.args.get('utm_source')
        #utm_source='recengine'
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'


    try:
        video_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        video_count=5

    if video_count>15:
        video_count=15
        
    print("source_videoid = ",news_id)
    print("video_count = ",video_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-story_to_video-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()
    
    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_video')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_video'))) 
    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_video')))
    lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_video'))) 
    
    #print(1044367')
    videolist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        data=mdfb.get_latest_news_records(collection_name='tbl_it_videodata', field_name='publishdate', LIMIT=video_count)
    else:
        if redisFlag:
            #print('5......')  
            key_1='2-mapping-dic-story-2-video'
            key_2='2-lda-story-2-video'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_video')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_video')))
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    

        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()   
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        #similar_news[:5]
        for x in similar_news[:(video_count + 1)]:
            videolist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        videolist_local=list(filter(lambda x:x!=news_id,videolist_local))[:video_count]    
        
        print('videolist_local =>', videolist_local)
        
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        data=mdfb.get_aajtak_news_data_for_json(collection_name='tbl_it_videodata',fieldname='videoid',fieldvaluelist=videolist_local)

    total_dict_data={}
    counter=1
    videolist_local_new=[]
    
    for row_data in data:
        dict_data={}
        dict_data['videoid']=row_data['videoid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d&t_source=recengine&t_medium=%s&t_content=video_player-slot-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        
        dict_data['image']=row_data['mobile_image']
        total_dict_data[row_data['videoid']]=dict_data
        videolist_local_new.append(row_data['videoid'])
        counter+=1
    
    data_final=[]
    for videoid in videolist_local_new:
        if len(data_final)==video_count:
            break
        try:
            data_final.append(total_dict_data[videoid])
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(videolist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=news_id,playlist=data_final)

@app.route("/headlinestoday/wordcloud", methods=['GET', 'POST'])
def headlinestoday_wordcloud():
    import tags.tags_mongo_db_file_model as tags_mongo_db_file_model
    mdfb_tags=tags_mongo_db_file_model.tags_mongo_db_file_model()
    print('1.....')
    cat=None
    lang=None
    site='ht'
    word_cloud_filename='word_cloud_data'

    try:
        cat = request.args.get('cat')
    except Exception as exp:
        cat=None
    print('2.....')
    try:
        lang = request.args.get('lang')
        if lang==None:
            lang='en'
    except Exception as exp:
        lang=None
    print('3.....')

    try:
        site = request.args.get('site')
        if site==None:
            site='ht'
    except Exception as exp:
        site='ht'
    print('4.....')
        
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|headlinestoday_wordcloud_categorywise"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    t1 = datetime.now()
    print('5.....')

    #cat=None
    #lang='hn'
    #site='ht'

    print(site,' = ',lang, ' = ' ,cat)
    if site=='ht':
        if lang=='en':
            word_cloud_filename+= "_" + lang
            print('6.....')

            if cat!=None and cat!='':
                print('final wc en')
                word_cloud_filename+= "_" + cat
            else:
                print('wc en all')
        elif lang=='hn':
            word_cloud_filename+= "_" + lang
            print('7.....', word_cloud_filename)

            if cat!=None and cat!='':
                print('final wc hn')
                word_cloud_filename+= "_" + cat
            else:
                print('wc hn all')
        else:
            #lang='en'
            word_cloud_filename+= "_" + lang
            print('wc en all')
    else:
        print('No site....')
        #lang='en'
        print('wc en all')
    print('8.....')

    log+='|cat=%s,lang=%s,site=%s,word_cloud_filename=%s'%(cat,lang,site,word_cloud_filename)

    print('word_cloud_filename =>', word_cloud_filename)
    
    #word_cloud_filename='word_cloud_data_en'
    wordcloud=[]
    response="SUCCESS"
    try:
        print('9.....')
        wordcloud = pickle.loads(gzip.decompress(mdfb_tags.get_data_record_from_mongodb(collection_name='cluster_hl',filename=word_cloud_filename)))     
        print(wordcloud)
        
        y=[]
        if lang=='hn':
            for i in wordcloud:
                z=[]
                for j in i:
                    z.append(str(j))
                y.append(z)    
        
            wordcloud=y
        
    except Exception as exp:
        print('Exception =>',exp)
        response="FAILED"

    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="word_cloud_ht_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    print('10.....')
       
    return jsonify(status=response,wordcloud=wordcloud)

@app.route("/headlinestoday/cloud/stopword", methods=['GET', 'POST'])
def headlinestoday_wordcloud_stopword():
    import tags.tags_mongo_db_file_model as tags_mongo_db_file_model
    
    mdfb_tags=tags_mongo_db_file_model.tags_mongo_db_file_model()
    
    action=0
    word=None
    status=False
    description=None
    try:
        action = int(request.args.get('action'))
    except Exception as exp:
        action=0

    try:
        word = request.args.get('word')
    except Exception as exp:
        word=None

    try:
        site = request.args.get('site')
    except Exception as exp:
        site=None

        
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|headlinestoday_wordcloud_stopword"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    #log+= "|utm_source=%s"%(utm_source)
    #news_id=1044367
    t1 = datetime.now()
        
        
    if action==1 and word!=None and site=='ht':
        if mdfb_tags.is_record_exist(collection_name='cloud_stopword',fieldname='data',fieldvalue=word):
            description='Already exist'
            print('No Insert...')
        else:
            #Insert Data into DB
            mdfb_tags.save_file(collection_name='cloud_stopword',filename='stopword',data=word)
            status=True
            description='Success'
            print('Insert...')
    elif action==2 and word!=None and site=='ht':    
        if mdfb_tags.is_record_exist(collection_name='cloud_stopword',fieldname='data',fieldvalue=word):
            #Delete word from DB
            mdfb_tags.remove_collection_from_at_recom_updated(collection_name='cloud_stopword',fieldname='data',fieldvalue=word)
            print('Delete...')
            status=True
            description='Success'
        else:    
            print('No Delete...')
            description='Word Does Not exist'
    elif action==3 and site=='ht':    
        #show stopword from DB
        word=mdfb_tags.get_alldata_record_from_mongodb(collection_name='cloud_stopword',filename='stopword')
        print('Show...')
        status=True        
        description='Success'
    else:
        print('No Action...')
        status=False        
        description='No Action'
        word=None
        
    log+='|action=%d,word=%s,site=%s,status=%s,description=%s'%(action,word,site,status,description)

    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="cloud_stopword_ht_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return jsonify(status=status,stopword=word,description=description)

@app.route("/recengine/it/getarticles", methods=['GET', 'POST'])
def indiatoday_getarticles():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model

    print('1.....')
    
    u=indiatoday_utility.utility()
    mdb=indiatoday_mongo_db_model.mongo_db_model()
    
    import boto3
    from boto3.dynamodb.conditions import Key

    news_count=10
    #newsid='1715391'
    newsid=0
    newsdata=None
    utm_source = None
    utm_medium = None
    source_newsid = 0
    uid=None
    min_story_count=5
    t_psl=False    
    text=None

    latest_Flag=False
    ad_slot_flag=True
    
    try:
        newsid = request.args.get('newsid',0)
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0

    try:
        uid = request.args.get('uid',None)
        print('uid =>', uid)
    except Exception as exp:
        print('Exception in get uid=>',exp)
        uid = None
        
    try:
        utm_source = request.args.get('utm_source',None)
        print('uid =>', utm_source)
    except Exception as exp:
        print('Exception in get utm_source=>',exp)
        utm_source = 'Unknown'        
    try:
        utm_medium = request.args.get('utm_medium','Unknown')
        print('utm_medium =>', utm_medium)
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    try:
        news_count=int(request.args.get('no',10))
        print('news_count =>', news_count)
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        news_count=10        

    if news_count>20:
        news_count=20
        
    print("source_newsid = ",newsid)
    print("news_count = ",news_count)


    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    log+='|request_uid=%s'%(uid)


    p1_ttl = 15 * 60
    p2_ttl = 60 * 60
    p3_ttl = 15 * 60
    #Temp
    p4_ttl = 2 * 24 * 60 *60
    newslist_ttl = 10 * 60
    
    
    
    #p1_ttl = 3 * 60
    #p2_ttl = 2 * 60
    #p3_ttl = 2 * 60
    #p4_ttl = 3 * 60
    #newslist_ttl = 10 * 60


    t1 = datetime.now()
    dynamodb=None
    interaction_item = None
    
    #This is for Lallantop =4
    site_id="2"
    
    newsrc_Flag=False
    uidrc_Flag=False
    history_db_Flag=False

    news_rc_list=None
    u_history_list=[]   
    u_history_for_save=[]
    history_rc=[]

    data_final = []
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    
    print('1.....')
    
    mapping_id_newsid_dictionary = None
    lda_index = None 


    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = None
    lda_index = None


    if redisFlag:
        #print('5......')  
        key_1=site_id + '-mapping-dic'
        key_2=site_id + '-lda'
        try:
            print('2......')  
            if r_handler.exists_key(key=key_1)==1 and r_handler.exists_key(key=key_2)==1:
                print('2.1......') 
                mapping_id_newsid_dictionary = pickle.loads(r_handler.get_data_from_cache(key=key_1))
                lda_index = pickle.loads(r_handler.get_data_from_cache(key=key_2))
        except Exception as exp:
            print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
            mapping_id_newsid_dictionary=None
            lda_index=None

        if mapping_id_newsid_dictionary==None or lda_index==None:        
            print('3......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_new')))
            lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_new'))) 
            
            if redisFlag:
                set_flag = r_handler.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = r_handler.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    

    def recommended_newsarray_english(text=None,newsid='0'):
        rc_news_array=[] 
        try:
            text = text.lower()
            text = u.clean_doc(text)
            tokens = tokenizer.tokenize(text)
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
            news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
            similar_news = lda_index[lda_it[news_corpus]]
            similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
            for x in similar_news[:(news_count + 1)]:
                rc_news_array.append(mapping_id_newsid_dictionary[x[0]])
                #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            rc_news_array = list(filter(lambda x:x!=newsid,rc_news_array))  
            uniq = []
            [uniq.append(x) for x in rc_news_array if x not in uniq]
            rc_news_array = uniq 
        except:
            rc_news_array=[] 
        return rc_news_array   

    #newsid='1713175'    
    if newsid!=None and int(newsid)>0:
        print('4.....')
        key = site_id + '-p1-' + newsid
        try:
            if r_handler.exists_key(key=key)==1:
                print('5.....')
                try:
                    news_rc_list = pickle.loads(r_handler.get_data_from_cache(key=key))
                    print('5.1.....news_rc_list=',news_rc_list)
                    news_rc_list = list(filter(lambda x:x!=str(newsid),news_rc_list))
                    newsrc_Flag=True
                    print('5.1 news_rc_list => ', news_rc_list)
                except:
                    print('Exception in get recommnedation news for passing newsid')
                    newsrc_Flag=False
            if newsrc_Flag==False:
                print('6.....')
                try:
                    newsdata=mdb.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=newsid)
                except:
                    newsdata=None
                    print('6.1.....')
 
                if newsdata!=None:
                    news_rc_list = recommended_newsarray_english(text=newsdata,newsid=str(newsid))
                    print('6.1.0.....news_rc_list=',news_rc_list)
                    news_rc_list = list(filter(lambda x:x!=str(newsid),news_rc_list))
                    newsrc_Flag=True
                    print('6.2.....news_rc_list=>', news_rc_list)
                    if len(news_rc_list)>0:
                        r_handler.set_data_in_cache(key=key, data=pickle.dumps(news_rc_list), ttl=p1_ttl) 
                else:
                    print('No News Found.....')
        except:
            print('7 ...exp')
            newsrc_Flag=False

    min_story_Flag=False
    if uid!=None and uid!='':
        print('8 ....')
        #Temporary
        #uid='04b024e2-8285-4a83-b8cc-7706587fc1cc'

        key = site_id + '-p4-' + uid
        try:
            if r_handler.exists_key(key=key)==1:
                t_psl=True
                u_history_list = pickle.loads(r_handler.get_data_from_cache(key=key))
                print('8.1 ....u_history_list=> ', u_history_list)
                u_history_for_save = u_history_list
                u_history_list = list(reversed(u_history_list))
                uidrc_Flag=True
                if u_history_list!=None and len(u_history_list)>=min_story_count:
                    print('8.2 ....')
                    min_story_Flag=True    
        except:
            u_history_list=[]
            u_history_for_save = []
            print('8.3 ....')
        
        if min_story_Flag==False:
            print('8.4 ....')
            u_interaction_list=[]
            key = site_id + '-p2-' + uid
            try:
                if r_handler.exists_key(key=key)==1:
                    print('9 ....')
                    try:
                        u_interaction_list = pickle.loads(r_handler.get_data_from_cache(key=key))
                        history_db_Flag=True
                        uidrc_Flag=True
                    except:
                        print('Exception')
                        #history_db_Flag=True
                        #uidrc_Flag=True
                else:
                    try:
                        #import boto3
                        dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
                        #dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1", aws_access_key_id="AKIAJBYGBQ4PH5FQQSBQ", aws_secret_access_key= "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")
                        response = dynamodb.Table("itgd_cs_interaction_data_prod").query(IndexName="final_id-index",KeyConditionExpression=Key('final_id').eq(uid))
                        print('11 ....')
                        interaction_item = response['Items'] 
                        print('11.1 ....interaction_item=>', interaction_item)
                        if len(interaction_item)>0:
                            history_db_Flag=True
                            uidrc_Flag=True
                            interaction_item = sorted(interaction_item, key = lambda i: i['ist_tstamp'],reverse=True)[:20]     
                            print('12 ....interaction_item=>', interaction_item)
                            #interaction_list=list(map(str, list(map(lambda x: x['newsid'] , list(filter(lambda x: x['site_id']==int(site_id), interaction_item))))))
                            interaction_item=list(map(str, list(map(lambda x: x['newsid'] , list(filter(lambda x: x['site_id']==int(site_id), interaction_item))))))
                            print('13 ....', interaction_item)
                            
                            if len(interaction_item)>0:
                                u_interaction_list = []
                                [x for x in interaction_item if x not in u_interaction_list and u_interaction_list.append(x)]
                                print('14 ...unique .len(u_interaction_list)', len(u_interaction_list))
                                key = site_id + '-p2-' + uid
                                set_flag = r_handler.set_data_in_cache(key=key, data=pickle.dumps(u_interaction_list), ttl=p2_ttl) 
                                #set_flag = ltop_rh.set_data_in_cache(key=key, data=str(u_interaction_list), ttl=p2_ttl)    
                                print('user interacted set  =>', set_flag)
                    except Exception as e:
                        history_db_Flag=False
                        print('Except=>',e)
            except:
                u_interaction_list=[]
                print('15 ....')   

        if history_db_Flag:   
            t_psl=True
            u_history_list.extend(u_interaction_list)   
            print('15.1 ....u_history_list =>', u_history_list) 
            
        if uidrc_Flag:                 
            try:
                print('16 ....')
                if len(u_history_list)>0:
                    u_history_list_unique = []
                    [x for x in u_history_list if x not in u_history_list_unique and u_history_list_unique.append(x)]
                
                    u_history_list=u_history_list_unique
                    

                u_history_list = list(filter(lambda x: x not in set([newsid]), list(map(str, u_history_list))))
                
                history_stack_length=10
                u_history_list=u_history_list[:history_stack_length]
                
                print('16.05.....u_history_list(with db records) =>',u_history_list)
                
                #For process
                #u_history_list=u_history_list[-(history_stack_length -1):]

                print('16.1 ....')
                counter=0
                #for nid in reversed(u_history_list):
                for nid in u_history_list:
                    #print('nid =>', nid)
                    news_rc_temp_list=[]
                    key = site_id + '-p1-' + str(nid)
                    counter += 1
                    try:
                        print('17 ....', counter,' => ', nid)
                        rc_content_flag=False
                        if r_handler.exists_key(key=key)==1:
                            print('18 ....in Redis exists', nid)
                            try:
                                news_rc_temp_list = pickle.loads(r_handler.get_data_from_cache(key=key))
                                print('18.1 ....news_rc_temp_list=>',str(nid),' =>', news_rc_temp_list)
                                
                                news_rc_temp_list = list(filter(lambda x:x!=str(nid),news_rc_temp_list))
                                print('18.1 ....news_rc_temp_list=>', news_rc_temp_list)
                                rc_content_flag=True
                            except:
                                print('18.2......Exception')
                                rc_content_flag=False
                        #else:
                        if rc_content_flag==False:
                            print('19.....')
                            #Process from DB only first 5 news Rest will check only from Cache
                            if counter<=5:
                                try:
                                    #newsdata=mdb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(nid))
                                    newsdata=mdb.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=str(nid))
                                except:
                                    newsdata=None
                                    print('19.1.....exp')
                                
                                #print('19.2.....newsdata=>', newsdata)
                                if newsdata!=None:
                                    print('19.3.....')
                                    news_rc_temp_list = recommended_newsarray_english(text=newsdata,newsid=str(nid))
                                    #news_rc_temp_list = list(filter(lambda x:x!=str(newsid),news_rc_temp_list))
                                    print('19.4.....news_rc_temp_list')
                                    if len(news_rc_temp_list)>0:
                                        r_handler.set_data_in_cache(key=key, data=pickle.dumps(news_rc_temp_list), ttl=p1_ttl) 

                        if len(news_rc_temp_list)>0:
                            #newsrc_Flag=True
                            history_rc.extend(news_rc_temp_list)
                            print('20.....len(history_rc)', len(history_rc))
                            print('20.1 news_rc_temp_list => ', news_rc_temp_list)
                    except:
                        print('21 ...exp')
                        
            except Exception as e:
                print('22 ...',e)
                print('Exception to set user list for key', key)
            
            try:
                subtract_newslist=[]
                if newsrc_Flag:
                    history_rc.extend(news_rc_list[2:])
                    print('22.1.....history_rc=>', history_rc)
                    subtract_newslist = news_rc_list[:2]
                    print('22.2.....subtract_newslist=>', subtract_newslist)
                    
                subtract_newslist.append(newsid)
                print('22.3.....subtract_newslist=>', subtract_newslist)    
                #history_rc = list(set(history_rc) - set([newsid]) -  set(news_rc_list[:2]))
                history_rc = list(filter(lambda x: x not in set(subtract_newslist), history_rc))
                print('22.4.....history_rc=>=>', history_rc)
                h_uniq = []
                [h_uniq.append(x) for x in history_rc if x not in h_uniq]
                history_rc = h_uniq      
            
                if history_rc!=None and len(history_rc)<8:
                    uidrc_Flag=False
                    print('22.5.....')
            except Exception as e:    
                print('22.6.....')
                print('Exception to set user list for key', key)
                
    try:
        u_history_for_save = list(filter(lambda x: x not in set([newsid]), u_history_for_save))
        
        #u_history_for_save = list(set(history_rc) - set(newsid))
        u_history_for_save = u_history_for_save[-(news_count -1):]
        u_history_for_save.append(newsid)
        print('23.....u_history_for_save=>', u_history_for_save)
     
        if uid!=None and len(u_history_for_save)>0:
            print('24.....u_history_for_save=>',u_history_for_save)
            key = site_id + '-p4-' + uid
            try:
                set_flag = r_handler.set_data_in_cache(key=key, data=pickle.dumps(u_history_for_save), ttl=p4_ttl) 
                print('user history set  =>', set_flag)
            except:
                print('Error to set History')
    except Exception as e: 
        print('24.1.....')    


    final_rc_array = []
    try:
        if newsrc_Flag==True and uidrc_Flag==True:
            print('25.....')
            final_rc_array.extend(news_rc_list[:2])
            random.shuffle(history_rc)
            print('25.1 Final history_rc after shuffle =>', history_rc)
            final_rc_array.extend(history_rc[:(news_count - 2)])
        elif newsrc_Flag==True and uidrc_Flag==False:  
            print('26.....')
            final_rc_array=news_rc_list[:news_count]
        elif newsrc_Flag==False and uidrc_Flag==True:    
            print('27.....')
            random.shuffle(history_rc)
            final_rc_array = history_rc[:news_count]
        else:
            print('28.....')
            latest_Flag=True
    except Exception as e:
        print('28.1.....') 
        latest_Flag=True

    if latest_Flag==False:        
        print('29.....')
        #final_rc_array = list(map(int, final_rc_array))
        print('final_rc_array=>',final_rc_array)
        data=mdb.get_indiatoday_news_data_for_json(collection_name='tbl_it_newsdata',fieldname='newsid',fieldvaluelist=final_rc_array)
    else:
        print('30.....')
        data=mdb.get_latest_news_records(collection_name='tbl_it_newsdata', field_name='publishdate', LIMIT=news_count)

    newslist=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"

    #create a Dictionary for all news content to maintain sequence
    news_data={}
    latest_list=[]
    for decode_data in data:
            temp_data={}
            temp_data['newsid']=int(decode_data['newsid'])
            temp_data['title']=html.unescape(decode_data['title'])
            temp_data['uri']=decode_data['uri']
            temp_data['mobile_image']=decode_data['mobile_image']
            news_data[decode_data['newsid']]=temp_data
            latest_list.append(decode_data['newsid'])
    
    #ad_slot_flag=False        
    ad_data={}    
    if ad_slot_flag:
        ad_data_rec=mdb.get_ad_record(collection_name='it_recom_ad_raw',field_name='publish_status',LIMIT=10)
        for ad in ad_data_rec:
            #print(ad)
            d={}
            try:
                d['newsid']=int(ad['content_id'])
                d['title']=ad['title']
                d['uri']=ad['url']
                d['mobile_image']=ad['img_url']
                d['utm_campaign']=ad['utm_campaign']
                ad_data[ad['content_position']]=d
            except:
                print('Exception in ad....')
                
    if latest_Flag:
        final_rc_array=latest_list            
     
    counter=1     
    for rc_newsid in final_rc_array[:news_count]:
        if bool(ad_data) and counter in ad_data:
            temp_data={}
            temp_data['newsid']=ad_data[counter]['newsid']
            temp_data['title']=html.unescape(ad_data[counter]['title'])
            temp_data['uri']=ad_data[counter]['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&utm_campaign=%s&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,ad_data[counter]['utm_campaign'],utm_medium,counter,t_psl)
            temp_data['mobile_image']=ad_data[counter]['mobile_image']
            newslist.append(ad_data[counter]['newsid'])
            #print('newslist =>', newslist)
            data_final.append(temp_data)
            #print('data_final =>', data_final)
        else:    
            #print(counter, ' = ','3....')
            temp_data=news_data[rc_newsid]
            temp_data['uri']=temp_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
            newslist.append(rc_newsid)
            data_final.append(temp_data)
        counter+=1        
        
    print("newslist =>",newslist)
    
    
    print("data_final =>",len(data_final))
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=int(newsid),data=data_final)


@app.route("/recengine/it/getarticles_28082020", methods=['GET', 'POST'])
def indiatoday_getarticles_28082020():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model

    print('1.....')
    u=indiatoday_utility.utility()
    mdb=indiatoday_mongo_db_model.mongo_db_model()
    story_count=10
    #news_id='1700334'
    text=None
    newsdata=[]
    utm_source = None
    utm_medium = None
    print('2.....')

    news_id=None
    latest_Flag=False
    t_psl = False
    ad_slot_flag=True
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdb.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=news_id)
        text=newsdata
        print('text =>',text)
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
    print('4.....')    

    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'

    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=10
        
    print('5.....')
    if story_count>20:
        story_count=20
        


    print("source_newsid = ",news_id)
    print("story_count = ",story_count)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = None
    lda_index = None


    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdb.get_latest_news_records(collection_name='tbl_it_newsdata', field_name='publishdate', LIMIT=story_count)
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='2-mapping-dic'
            key_2='2-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_new')))
            lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_new')))
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
    
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        print('9.....',similar_news[:3])

        s_counter=1
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
            
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        data=mdb.get_indiatoday_news_data_for_json(collection_name='tbl_it_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        print('14.....data',data)

    #Set all published Ad in dictionary ad_data
    ad_data={}    
    if ad_slot_flag:
        ad_data_rec=mdb.get_ad_record(collection_name='it_recom_ad_raw',field_name='publish_status',LIMIT=10)
        for ad in ad_data_rec:
            #print(ad)
            d={}
            try:
                d['newsid']=int(ad['content_id'])
                d['title']=ad['title']
                d['uri']=ad['url']
                d['mobile_image']=ad['img_url']
                d['utm_campaign']=ad['utm_campaign']
                ad_data[ad['content_position']]=d
            except:
                print('Exception in ad....')
     
    counter=1     
    for decode_data in data[:story_count]:
        #print(decode_data)
        if bool(ad_data) and counter in ad_data:
            temp_data={}
            temp_data['newsid']=ad_data[counter]['newsid']
            temp_data['title']=html.unescape(ad_data[counter]['title'])
            temp_data['uri']=ad_data[counter]['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&utm_campaign=%s&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,ad_data[counter]['utm_campaign'],utm_medium,counter,t_psl)
            temp_data['mobile_image']=ad_data[counter]['mobile_image']
            newslist.append(ad_data[counter]['newsid'])
            data_final.append(temp_data)
        else:    
            temp_data={}
            temp_data['newsid']=int(decode_data['newsid'])
            temp_data['title']=html.unescape(decode_data['title'])
            temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
            temp_data['mobile_image']=decode_data['mobile_image']
            newslist.append(decode_data['newsid'])
            data_final.append(temp_data)
        counter+=1        
    print("newslist =>",newslist)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=int(news_id),data=data_final)


@app.route("/recengine/it/getarticles_uid", methods=['GET', 'POST'])
def indiatoday_getarticles_uid():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model

    print('1.....')
    u=indiatoday_utility.utility()
    mdb=indiatoday_mongo_db_model.mongo_db_model()
    #mdb=indiatoday_mongo_db_model()
    #db = indiatoday_db_model.db_model() 
    story_count=8
    t=2
    p=2
    #news_id='1594854'
    text=None
    newsdata=[]
    utm_source = None
    utm_medium = None
    print('2.....')

    news_id=None
    latest_Flag=False
    t_psl = False
    
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdb.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=news_id)
        text=newsdata
        print('text =>',text)
        #t = int(request.args.get('t'))
        #p = int(request.args.get('p'))
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
        #t=2
        #p=2
    print('4.....')    

    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'

    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5
        
    print('5.....')
    if story_count>15:
        story_count=15
        


    print("source_newsid = ",news_id)
    print("story_count = ",story_count)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_new')))
    #lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_new'))) 
    mapping_id_newsid_dictionary = None
    lda_index = None


    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdb.get_latest_news_records(collection_name='tbl_it_newsdata', field_name='publishdate', LIMIT=story_count)
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='2-mapping-dic'
            key_2='2-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_new')))
            lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_new')))
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
    
    #print('6.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        
        #trendnews_mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_trend_news')))
        #trendnews_lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='trend_news_corpus'))) 

        #popularnews_mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_popular_news')))
        #popularnews_lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='popular_news_corpus'))) 
        
        #trend_similar_news = trendnews_lda_index[lda_it[news_corpus]]
        #trend_similar_news = sorted(enumerate(trend_similar_news[0]), key=lambda item: -item[1])
        #print('7.....',trend_similar_news[:3])

        #popular_similar_news = popularnews_lda_index[lda_it[news_corpus]]
        #popular_similar_news = sorted(enumerate(popular_similar_news[0]), key=lambda item: -item[1])
        #print('8.....',popular_similar_news[:3])
        
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        print('9.....',similar_news[:3])
        #similar_news[:5]
        #story_count=8
        '''    
        newslist_local_trend=[]
        for x in trend_similar_news[:t]:
            newslist_local_trend.append(trendnews_mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(trendnews_mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local_trend=list(filter(lambda x:x!=news_id,newslist_local_trend))[:t]    
        
        print('10.....',newslist_local_trend)
        newslist_local_popular=[]
        for x in popular_similar_news[:p]:
            newslist_local_popular.append(popularnews_mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(popularnews_mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local_popular=list(filter(lambda x:x!=news_id,newslist_local_popular))[:(t+p)]    
        print('11.....',newslist_local_popular)
        
        s_counter=1
        #newslist_local=[]
        for x in similar_news[:(story_count + t + p)]:
            if s_counter==1:
                newslist_local.append(newslist_local_trend[0])
            elif s_counter==3:
                newslist_local.append(newslist_local_popular[0])
            else:
                newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
        '''
        s_counter=1
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
            
        #newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count + t + p]    
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        #newslist_local_final=['1595355','1595354','1595339','1595350','1595044']
        #db = indiatoday_db_model.db_model()    
        #data=db.pickData_from_newsid_it(newslist=newslist_local_final) 
        data=mdb.get_indiatoday_news_data_for_json(collection_name='tbl_it_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        print('14.....data',data)
    #utm_medium='WEB'    
    
    #data[0]
    #data_final=[]
    counter=1        
    for decode_data in data[:story_count]:
        if counter==2:
            temp_data={}
            temp_data['newsid']=1708943
            temp_data['title']='Modi the King of Indian Politics'
            temp_data['uri']='https://www.indiatoday.in/magazine/india/story/20200817-mood-of-the-nation-pm-modi-still-king-of-indian-politics-with-78-per-cent-approval-rating-1708943-2020-08-07?utm_source=recengine&utm_medium=%s&utm_campaign=internal&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
            temp_data['mobile_image']='https://akm-img-a-in.tosshub.com/indiatoday/images/story/202008/RTX775HX.png?Y6wbUVhOgMS947dvzpOIRFG0kkl60VhZ&size=170:96'
            data_final.append(temp_data)
        elif counter==10:     
            temp_data={}
            temp_data['newsid']=1708966
            temp_data['title']='Corona-The biggest concern'
            temp_data['uri']='https://www.indiatoday.in/magazine/mood-of-the-nation/story/20200817-coronavirus-pandemic-biggest-concern-for-india-finds-mood-of-the-nation-poll-1708966-2020-08-07?utm_source=recengine&utm_medium=%s&utm_campaign=internal&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
            temp_data['mobile_image']='https://akm-img-a-in.tosshub.com/indiatoday/images/story/202008/AP20208360284573.jpeg?FzmuuAai.jXIqw4N_hYkPdZJnOleI_gP&size=170:96'
            data_final.append(temp_data)
        else:    
            temp_data={}
            temp_data['newsid']=int(decode_data['newsid'])
            temp_data['title']=html.unescape(decode_data['title'])
            temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
            temp_data['mobile_image']=decode_data['mobile_image']
            newslist.append(decode_data['newsid'])
            data_final.append(temp_data)
        counter+=1
    #type(data_final)  
    #type(newslist)      
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=int(news_id),data=data_final)


@app.route("/recengine/it/getmagazine", methods=['GET', 'POST'])
def indiatoday_getmagazine():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model
    #import indiatoday.mongo_db_file_model as indiatoday_mongo_db_file_model
    print('1.....')
    u=indiatoday_utility.utility()
    mdb=indiatoday_mongo_db_model.mongo_db_model()
    #mdb=indiatoday_mongo_db_model()
    t_psl = False
    story_count=8
    #news_id='1594856'
    text=None
    newsdata=[]
    utm_source = request.args.get('utm_source')
    utm_medium = request.args.get('utm_medium')
    print('2.....')
    try:
        news_id = request.args.get('newsid')
        newsdata=mdb.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=news_id)
        text=newsdata
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
    print('4.....')    
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5
    print('5.....')
    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-get-magazine"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_magazine')))
    lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='magazine_corpus'))) 

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        data=mdb.get_latest_news_records(collection_name='tbl_it_magazine_data', field_name='publishdate', LIMIT=story_count)

    else:
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]

        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        
        #similar_news[:5]
        #story_count=8
        s_counter=1
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         

        #db = indiatoday_db_model.db_model()    
        #data=db.pickData_from_newsid_it(newslist=newslist_local_final) 
        data=mdb.get_indiatoday_news_data_for_json(collection_name='tbl_it_magazine_data',fieldname='newsid',fieldvaluelist=newslist_local_final)
    #utm_medium='WEB'    
    
    #data[0]
    #data_final=[]
    counter=1        
    for decode_data in data[:story_count]:
        temp_data={}
        temp_data['newsid']=int(decode_data['newsid'])
        temp_data['title']=decode_data['title']
        #temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=magazinestrip-%d'%(utm_medium,counter)
        temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=magazinestrip-%d&t_source=recengine&t_medium=%s&t_content=magazinestrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        temp_data['mobile_image']=decode_data['mobile_image']
        newslist.append(decode_data['newsid'])
        data_final.append(temp_data)
        counter+=1
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=int(news_id),data=data_final)
    #return jsonify(status=response,message=message,source_newsid=news_id)


#With BT:2, IT:2,AT:2
@app.route("/recengine/it/mixstory/getarticles", methods=['GET', 'POST'])
def indiatoday_mixstory_getarticles():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model
    
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    
    from mongo_db_model import mongo_db_model
    from db_model import db_model
    from db_model_live import db_model_live
    
    #u=aajtak_utility.utility()
    mdfb_at=aajtak_mongo_db_file_model.mongo_db_file_model()
    mdb_bt=mongo_db_model()
    db = db_model()
    db_live = db_model_live()
    
    #import indiatoday.mongo_db_file_model as indiatoday_mongo_db_file_model
    #import indiatoday.db_model as indiatoday_db_model
    print('1.....')
    u=indiatoday_utility.utility()
    mdb_it=indiatoday_mongo_db_model.mongo_db_model()
    
    t_psl = False
    #mdb=indiatoday_mongo_db_model()
    #db = indiatoday_db_model.db_model() 
    story_count=9
    #news_id='1594858'
    text=None
    newsdata=[]
    
    utm_source=None
    utm_medium=None
    try:
        utm_source = request.args.get('utm_source')
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        utm_source=None
        utm_medium=None
        
    print('2.....')
    #text='Relative values: With foster family; and Niharika Up close and personal, it was a mixed year for the prime minister. There was both a wedding and a funeral in his family. While Vajpayee attended a niece&#039;s wedding reception in February, he lost his sister Urmila Mishra to cancer in May. But both the grief and the celebrations were in private. The prime ministerial public visage was a pair of Ray-Ban glasses. His own birthday found the 79-year-old in eloquent mood. Addressing a crowd of 400 party supporters, Vajpayee wryly said, &quot; Assi saal ke vyakti ko happy returns kehna bade saahas ka kaam hai (It takes some courage to wish an 80-year-old man happy returns).&quot; And typical of the paradox of his life, the prime minister had three birthday celebrations on December 25. One was with PMO officials while driving from Jaipur to Delhi. At midnight, he and his entourage stopped at a restaurant and cut the famous Alwar milk cake. The next was a public function at his official residence, followed by lunch with his foster family. The cake was pineapple, the food, his favourite Chinese, and the conversation, apolitical. At home, his day begins at 7 a.m. with tea and biscuits. Then he gets on the fitness cycle and for half an hour surfs channels while cycling. Breakfast- usually sprouts, upma or idli, toast and'
    try:
        news_id = request.args.get('newsid')
        newsdata=mdb_it.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=news_id)
        text=newsdata
        print('text =>',text)
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
    print('4.....')    
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=8
    print('5.....')
    if story_count>=9:
        story_count=9
    else:    
        story_count=6
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb_it.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_new')))
    lda_index = pickle.loads(gzip.decompress(mdb_it.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_new'))) 

    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb_it.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid')))
    #lda_index = pickle.loads(gzip.decompress(mdb_it.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus'))) 


    lda_index_bt = pickle.loads(gzip.decompress(mdb_bt.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus')))
    mapping_id_newsid_dictionary_bt = pickle.loads(gzip.decompress(mdb_bt.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid')))

    newslist=[]
    newslist_local=[]
    data_final = []
    bt_data=[] 
    it_data=[]
    at_data=[]
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdb_it.get_latest_news_records(collection_name='tbl_it_newsdata', field_name='publishdate', LIMIT=story_count)
    else:
        print('6.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]

        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        print('9.....',similar_news[:4])

        s_counter=1
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
            
        #newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count + t + p]    
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)

        #===================BT
        portal_corpus=[]
        newslist_local_bt=[]
        dict_bt = mdb_bt.load_latest_version_file_data_in_gridfs(filename='dic')
        portal_corpus = [dict_bt.doc2bow(lemma_tokens)]   
        
        similar_news_bt = lda_index_bt[lda[portal_corpus]]        
        similar_news_bt = sorted(enumerate(similar_news_bt[0]), key=lambda item: -item[1])
        for x in similar_news_bt[:3]:
            newslist_local_bt.append(mapping_id_newsid_dictionary_bt[x[0]])
        print('newslist_local_bt =>',newslist_local_bt)    
         
        bt_data=[]    
        for x in newslist_local_bt:
            print('type=>',type(x), ' =',x)
            temp_data={}
            res=db_live.get_data(newsid=x)  
            
            temp_data['newsid']=res[0]['newsid']
            temp_data['title']=res[0]['title']
            temp_data['uri']='https://www.businesstoday.in/' + res[0]['uri']
            temp_data['mobile_image']='https://smedia2.intoday.in/btmt/images/stories/' + res[0]['mobile_image']
            temp_data['site']='BT'
            temp_data['site_name']='businesstoday.in'
            temp_data['logo']='http://media2.intoday.in/microsites/site-logo/bt_logo.png'
            bt_data.append(temp_data)
            
            #print('res=>',res)
            #data_final.append(temp_data)  
        print('bt_data=>',bt_data)    
                            
        #===================
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        #newslist_local_final=[1578284, 1580069, 1575899, 1578626, 1580675, 1575523]
        #db = indiatoday_db_model.db_model()    
        #data=db.pickData_from_newsid_it(newslist=newslist_local_final) 
        data_it=mdb_it.get_indiatoday_news_data_for_json(collection_name='tbl_it_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        data_at=mdfb_at.get_latest_news_records(collection_name='at_recom', field_name='modified', LIMIT=3)
        print('14.....data',data_it)
        #print('14.....data',data_it[0])
    

        counter=1        
        for decode_data in data_it[:4]:
            print('decode_data=>',decode_data)

            temp_data={}
            temp_data['newsid']=int(decode_data['newsid'])
            temp_data['title']=decode_data['title']
            temp_data['uri']=decode_data['uri']
            temp_data['mobile_image']=decode_data['mobile_image']
            temp_data['site']='IT'
            temp_data['site_name']='indiatoday.in'
            temp_data['logo']='http://media2.intoday.in/microsites/site-logo/it_logo.png'
            newslist.append(decode_data['newsid'])
            #data_final.append(temp_data)
            it_data.append(temp_data)
            counter+=1
        
        print('it_data=>',it_data)    


        #data[0]
        #data_final=[]
        counter=1
        for row_data in data_at:
            dict_data={}
            dict_data['newsid']=row_data['id']
            dict_data['title']=row_data['title']
            dict_data['uri']=row_data['url']
            dict_data['mobile_image']=row_data['media']['kicker_image2']
            dict_data['site']='AT'
            dict_data['site_name']='aajtak.intoday.in'
            dict_data['logo']='http://media2.intoday.in/microsites/site-logo/at_logo.png'
            #total_dict_data[row_data['id']]=dict_data
            data_final.append(dict_data)
            at_data.append(dict_data)
            counter+=1
            
    data_final=[]
    if story_count==6:    
        for t in bt_data[:2]:
            data_final.append(t)
        for t in it_data[:2]:
            data_final.append(t)
        for t in at_data[:2]:
            data_final.append(t)
    else:        
        for t in bt_data[:2]:
            data_final.append(t)
        for t in it_data[:2]:
            data_final.append(t)
        for t in at_data[:2]:
            data_final.append(t)
        data_final.append(bt_data[2])
        data_final.append(it_data[2])
        data_final.append(at_data[2])
            
    #type(data_final)  
    #type(newslist)      
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    #return jsonify(status=response,message=message,source_newsid=int(news_id),data=it_data,bt_data=bt_data,at_data=at_data)
    return jsonify(status=response,message=message,source_newsid=int(news_id),data=data_final)

@app.route("/recengine/ht_hn/getarticles", methods=['GET', 'POST'])
def headlinetoday_hn_getarticles():
    import tags.tags_utility as tags_utility 
    import tags.tags_mongo_db_file_model_recommnedation as tags_mongo_db_file_model_recommnedation
    import tags.tags_db_model as tags_db_model
    #print('1.....')
    u=tags_utility.tags_utility()
    mdb=tags_mongo_db_file_model_recommnedation.tags_mongo_db_file_model_recommnedation()
    db = tags_db_model.tags_db_model() 
    #print('2.....')
    story_count=5
    id=0
    text=None
    newsdata=[]
    #utm_source = request.args.get('utm_source')
    #utm_medium = request.args.get('utm_medium')
    #text = request.args.get('text')
    #print('3.....')
    #text='Relative values: With foster family; and Niharika Up close and personal, it was a mixed year for the prime minister. There was both a wedding and a funeral in his family. While Vajpayee attended a niece&#039;s wedding reception in February, he lost his sister Urmila Mishra to cancer in May. But both the grief and the celebrations were in private. The prime ministerial public visage was a pair of Ray-Ban glasses. His own birthday found the 79-year-old in eloquent mood. Addressing a crowd of 400 party supporters, Vajpayee wryly said, &quot; Assi saal ke vyakti ko happy returns kehna bade saahas ka kaam hai (It takes some courage to wish an 80-year-old man happy returns).&quot; And typical of the paradox of his life, the prime minister had three birthday celebrations on December 25. One was with PMO officials while driving from Jaipur to Delhi. At midnight, he and his entourage stopped at a restaurant and cut the famous Alwar milk cake. The next was a public function at his official residence, followed by lunch with his foster family. The cake was pineapple, the food, his favourite Chinese, and the conversation, apolitical. At home, his day begins at 7 a.m. with tea and biscuits. Then he gets on the fitness cycle and for half an hour surfs channels while cycling. Breakfast- usually sprouts, upma or idli, toast and'
    try:
        #text = request.args.get('text')
        #news_id = int(request.args.get('newsid'))
        #id=102744488
        id = int(request.args.get('id'))
        newsdata=db.getTextData_tag(id=id,lang='hn')
        text=newsdata[0]['text']
        print('4.....',newsdata)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        #id=0
        #text=None

    try:
        #print('5.....')
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    #print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)
    #hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    hi_stop=u.getHindiStopWords()

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ht_hn-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    #log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(id,'None')
    #news_id=790807
    t1 = datetime.now()
    
    newslist=[]
    newslist_local=[]
    data_final = []
    id_status=False
    source_id=id
    language=None
    newslist_dic={}
    try:
        source_id=[]
        source_id.append(newsdata[0]['id'])
        source_id.append(newsdata[0]['site_name'])
        source_id.append(newsdata[0]['title'])
        source_id.append(newsdata[0]['Cat_Name'])
        source_id.append(newsdata[0]['URL'])
        source_id.append(newsdata[0]['Description'])
        #source_id.append(newsdata[0]['language'])
        language=newsdata[0]['language']
        print('language =>',language)

        #text=list(d)[2] + ' ' + list(d)[5]
        id_status=True
    except Exception as exp:
        print('Exception in get news id=>',exp)
        text=None
        source_id=id
        id_status=False
        
    if id_status:
            if language=='hn':
                mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_hindi',filename='id_newsid')))
                lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_hindi',filename='portal_corpus'))) 
                latest_data = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_hindi',filename='latest_data'))) 
                
    response="SUCCESS"
    message="OK"
    data=""
    print('text=>',text)
    newslist_dic={}
    if text==None or len(text)<=10 or language!='hn':
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
    else:
        log+= '|SUCCESS'
        log+= '|Result'
        #t='class="acl-ttl">पीएम मोदी ने भारतीय विदेश नीति के स्थापित सिद्धांतों को तोड़ा- कांग्रेस प्रधानमंत्री नरेंद्र मोदी ने अमेरिका के ह्यूस्टन में ‘हाउडी मोदी’ कार्यक्रम में डोनाल्ड ट्रम्प को 2020 में फिर से जिताने के लिए रविवार को लोगों का उत्साहवर्धन करते हुए कहा था, \'अबकी बार, ट्रम्प सरकार\'. &nbsp;|&nbsp;Updated: 23 Sep 2019 06:52 PM Updated: 23 Sep 2019 06:52 PM नई दिल्लीः कांग्रेस ने सोमवार को आरोप लगाया कि प्रधानमंत्री नरेंद्र मोदी ने ‘अमेरिकी राष्ट्रपति डोनाल्ड ट्रंप के पक्ष में प्रचार कर’ दूसरे देशों के चुनावों में दखल नहीं देने के भारतीय विदेश नीति के स्थापित सिद्धांत का उल्लंघन किया है. पार्टी के वरिष्ठ प्रवक्ता आनंद शर्मा ने यह दावा भी किया कि यह भारत के दीर्घकालिक कूटनीतिक हितों के लिए बड़ा झटका है. उन्होंने ट्वीट कर कहा, "आपको (मोदी) याद दिलाना चाहता हूं कि आप अमेरिकी चुनाव में स्टार प्रचारक की तरह नहीं, बल्कि भारत के प्रधानमंत्री के तौर पर अमेरिका गए हैं." Reminding you that you are in the USA as our Prime Minister and not a star campaigner in US elections. पीएम मोदी ने भारतीय विदेश नीति के स्थापित सिद्धांतों को तोड़ा- कांग्रेस प्रधानमंत्री नरेंद्र मोदी ने अमेरिका के ह्यूस्टन में ‘हाउडी मोदी’ कार्यक्रम में डोनाल्ड ट्रम्प को 2020 में फिर से जिताने के लिए रविवार को लोगों का उत्साहवर्धन करते हुए कहा था, \'अबकी बार, ट्रम्प सरकार\'.'
        
        #print('6 : ',text)
        clean_text = u.clean_doc_hindi(text)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        #print('7 : ',cleaned_tokens_n)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        #print('8 : ',stemmed_tokens)
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        #print('9 : ',news_corpus)
        similar_news = lda_index[lda_at[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            newslist_dic[mapping_id_newsid_dictionary[x[0]]]=x[1]
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=id,newslist_local))[:story_count]    
        #print('10....',newslist_local)
    data_final=[]    
    counter=1      
    
    #table_id=102644169
    for table_id in newslist_local:
        try:
            latest_data_1=latest_data[table_id]
            #print('11.....',counter, ' - ', latest_data_1)
            latest_data_2=latest_data_1 + (str(newslist_dic[table_id]),)
            #type(latest_data_1)
            #latest_data_2=latest_data_1 + (str(newslist_dic[table_id]),)
            #data_final.append(list(latest_data_1))
            data_final.append(latest_data_2)
        except Exception as exp:
            print('Exception =>',exp)
        counter+=1
            
    #newslist = [275318,275317,275316,275315,275313]
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ht_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(source=source_id,id_status=id_status,data=data_final)

@app.route("/headlinestoday_hn/cloud/stopword", methods=['GET', 'POST'])
def headlinestoday_hn_wordcloud_stopword():
    import tags.tags_mongo_db_file_model as tags_mongo_db_file_model
    mdfb_tags=tags_mongo_db_file_model.tags_mongo_db_file_model()
    
    action=0
    word=None
    status=False
    description=None
    try:
        action = int(request.args.get('action'))
    except Exception as exp:
        action=0

    try:
        word = request.args.get('word')
        #word = word.encode("utf-8")
    except Exception as exp:
        word=None

    try:
        site = request.args.get('site')
    except Exception as exp:
        site=None
        
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|headlinestoday_hn_wordcloud_stopword"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    #log+= "|utm_source=%s"%(utm_source)
    #news_id=1044367
    t1 = datetime.now()
        
    collection_name='cloud_stopword_hn'    
    if action==1 and word!=None and site=='ht':
        word = word.encode("utf-8")
        if mdfb_tags.is_record_exist(collection_name=collection_name,fieldname='data',fieldvalue=word):
            description='Already exist'
            print('No Insert...')
        else:
            #Insert Data into DB
            mdfb_tags.save_file(collection_name=collection_name,filename='stopword',data=word)
            status=True
            description='Success'
            print('Insert...')
        #word = word.decode("utf-8")    
    elif action==2 and word!=None and site=='ht':
        word = word.encode("utf-8")
        if mdfb_tags.is_record_exist(collection_name=collection_name,fieldname='data',fieldvalue=word):
            #Delete word from DB
            mdfb_tags.remove_collection_from_at_recom_updated(collection_name=collection_name,fieldname='data',fieldvalue=word)
            print('Delete...')
            status=True
            description='Success'
        else:    
            print('No Delete...')
            description='Word Does Not exist'
        #word = word.decode("utf-8")    
    elif action==3 and site=='ht':    
        #show stopword from DB
        word=mdfb_tags.get_alldata_record_from_mongodb(collection_name=collection_name,filename='stopword')
        print('Show...',word)
        word = [i.decode("utf-8") for i in word]
        #word = [(print(i),' -- ',print(type(i))) for i in word]
        #print('word...',word)
        status=True        
        description='Success'
    else:
        print('No Action...')
        status=False        
        description='No Action'
        word=None
        
    print('1.....') 
    
    
    if action==1 or action==2:
        log+='|action=%d,word=%s,site=%s,status=%s,description=%s'%(action,word,site,status,description)
    elif action==3:
        log+='|action=%d,word=word,site=%s,status=%s,description=%s'%(action,site,status,description)
    print('2.....')
    
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    print('3.....')
    log+= "|queryString=%s"%(request.query_string)
    filename="cloud_stopword_ht_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    print('4.....')
    filename = filepath + filename
    print('5.....')
    fw.log_write(filepath=filename,log=log)
    #print('6.....',word)
    print('7.....',description)
    
    if action==1 or action==2:
        word=word.decode('utf-8')

    return jsonify(status=status,stopword=word,description=description)

@app.route("/recengine/lt/getarticles", methods=['GET', 'POST'])
def lallantop_getarticles():
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    
    u=aajtak_utility.utility()
    mdfb=aajtak_mongo_db_file_model.mongo_db_file_model()
    
    t_psl = False

    #story_count=5
    #news_id='1046690'
    newsdata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdfb.get_aajtak_news_text_from_mongodb(collection_name='at_recom',fieldname='id',fieldvalue=news_id)
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id='0'
        newsdata=None
        utm_medium='Unknown'
 
    try:
        story_count=8
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|at-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid')))
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus'))) 
    
    #print(1044367')
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        data=mdfb.get_latest_news_records(collection_name='at_recom', field_name='modified', LIMIT=story_count)
    else:
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 4]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        similar_news = lda_index[lda_at[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        data=mdfb.get_aajtak_news_data_for_json(collection_name='at_recom',fieldname='id',fieldvaluelist=newslist_local)

    total_dict_data={}
    counter=1
    for row_data in data:
        dict_data={}
        dict_data['newsid']=row_data['id']
        dict_data['title']=row_data['title']
        #dict_data['uri']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        dict_data['uri']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&referral=yes&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        dict_data['mobile_image']=row_data['media']['kicker_image2']
        total_dict_data[row_data['id']]=dict_data
        newslist_local.append(row_data['id'])
        counter+=1
    
    data_final=[]
    for newsid in newslist_local:
        if len(data_final)==story_count:
            break
        try:
            data_final.append(total_dict_data[newsid])
        except Exception as exp:   
            print('Exception =>',exp)

    '''    
    data_final=[]    
    for row_data in data:
        temp_data={}
        temp_data['newsid']=row_data['id']
        temp_data['title']=row_data['title']
        temp_data['uri']=row_data['url']
        temp_data['mobile_image']=row_data['media']['kicker_image2']
        data_final.append(temp_data)
     '''   
    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_lt_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=news_id,data=data_final)

@app.route("/recengine/it/getarticles_amp", methods=['GET', 'POST'])
def indiatoday_getarticles_amp():
    import indiatoday.utility as indiatoday_utility 
    import indiatoday.mongo_db_model as indiatoday_mongo_db_model
    print('1.....')
    u=indiatoday_utility.utility()
    mdb=indiatoday_mongo_db_model.mongo_db_model()
    #mdb=indiatoday_mongo_db_model()
    #db = indiatoday_db_model.db_model() 
    story_count=8
    t=2
    p=2
    #news_id='1594854'
    text=None
    newsdata=[]
    utm_source = None
    utm_medium = None
    print('2.....')

    news_id=None
    latest_Flag=False
    t_psl = False
    
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdb.get_indiatoday_news_text_from_mongodb(collection_name='it_recom',fieldname='n_id',fieldvalue=news_id)
        text=newsdata
        print('text =>',text)
        #t = int(request.args.get('t'))
        #p = int(request.args.get('p'))
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
        #t=2
        #p=2
    print('4.....')    

    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'

    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5
        
    print('5.....')
    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|it-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = None
    lda_index = None

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdb.get_latest_news_records(collection_name='tbl_it_newsdata', field_name='publishdate', LIMIT=story_count)
    else:
        if redisFlag:
            #print('5......')  
            key_1='2-mapping-dic'
            key_2='2-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_new')))
            lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='portal_corpus_new')))
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        print('6.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        
        '''
        trendnews_mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_trend_news')))
        trendnews_lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='trend_news_corpus'))) 

        popularnews_mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='id_newsid_popular_news')))
        popularnews_lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='it_file_system',filename='popular_news_corpus'))) 
        
        trend_similar_news = trendnews_lda_index[lda_it[news_corpus]]
        trend_similar_news = sorted(enumerate(trend_similar_news[0]), key=lambda item: -item[1])
        print('7.....',trend_similar_news[:3])
        
        
        popular_similar_news = popularnews_lda_index[lda_it[news_corpus]]
        popular_similar_news = sorted(enumerate(popular_similar_news[0]), key=lambda item: -item[1])
        print('8.....',popular_similar_news[:3])
        '''
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        #print('9.....',similar_news[:3])
        #similar_news[:5]
        #story_count=8
        '''    
        newslist_local_trend=[]
        for x in trend_similar_news[:t]:
            newslist_local_trend.append(trendnews_mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(trendnews_mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local_trend=list(filter(lambda x:x!=news_id,newslist_local_trend))[:t]    
        
        print('10.....',newslist_local_trend)
        newslist_local_popular=[]
        for x in popular_similar_news[:p]:
            newslist_local_popular.append(popularnews_mapping_id_newsid_dictionary[x[0]])
            log+=',(%d-%s)'%(popularnews_mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local_popular=list(filter(lambda x:x!=news_id,newslist_local_popular))[:(t+p)]    
        print('11.....',newslist_local_popular)
        
        s_counter=1
        #newslist_local=[]
        for x in similar_news[:(story_count + t + p)]:
            if s_counter==1:
                newslist_local.append(newslist_local_trend[0])
            elif s_counter==3:
                newslist_local.append(newslist_local_popular[0])
            else:
                newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
        '''
        s_counter=1
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
            
        #newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count + t + p]    
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        #newslist_local_final=['1595355','1595354','1595339','1595350','1595044']
        #db = indiatoday_db_model.db_model()    
        #data=db.pickData_from_newsid_it(newslist=newslist_local_final) 
        data=mdb.get_indiatoday_news_data_for_json(collection_name='tbl_it_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        print('14.....data',data)
    #utm_medium='WEB'    
    
    #data[0]
    #data_final=[]
    counter=1        
    for decode_data in data[:story_count]:
        temp_data={}
        temp_data['newsid']=int(decode_data['newsid'])
        temp_data['title']=html.unescape(decode_data['title'])
        #temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        #temp_data['uri']=decode_data['amp_url']+'?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d'%(counter)
        temp_data['uri']=decode_data['amp_url']+'?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=amp&t_content=footerstrip-%d&t_psl=%s'%(counter,counter,t_psl)
        
        temp_data['mobile_image']=decode_data['mobile_image']
        newslist.append(decode_data['newsid'])
        data_final.append(temp_data)
        counter+=1
    #type(data_final)  
    #type(newslist)      
    #newslist = [275318,275317,275316,275315,275313]
    print("newslist =>",newslist)   
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_it_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=int(news_id),items=data_final)

@app.route("/recengine/ht_bt/getarticles", methods=['GET', 'POST'])
def headlinetoday_bt_getarticles():
    import tags.tags_utility as tags_utility 
    import tags.tags_mongo_db_file_model_recommnedation as tags_mongo_db_file_model_recommnedation
    import tags.tags_db_model as tags_db_model
    
    u=tags_utility.tags_utility()
    mdb=tags_mongo_db_file_model_recommnedation.tags_mongo_db_file_model_recommnedation()
    db = tags_db_model.tags_db_model() 
    story_count=5
    id=0
    text=None
    newsdata=[]
    #utm_source = request.args.get('utm_source')
    #utm_medium = request.args.get('utm_medium')
    #text = request.args.get('text')
    
    #text='Relative values: With foster family; and Niharika Up close and personal, it was a mixed year for the prime minister. There was both a wedding and a funeral in his family. While Vajpayee attended a niece&#039;s wedding reception in February, he lost his sister Urmila Mishra to cancer in May. But both the grief and the celebrations were in private. The prime ministerial public visage was a pair of Ray-Ban glasses. His own birthday found the 79-year-old in eloquent mood. Addressing a crowd of 400 party supporters, Vajpayee wryly said, &quot; Assi saal ke vyakti ko happy returns kehna bade saahas ka kaam hai (It takes some courage to wish an 80-year-old man happy returns).&quot; And typical of the paradox of his life, the prime minister had three birthday celebrations on December 25. One was with PMO officials while driving from Jaipur to Delhi. At midnight, he and his entourage stopped at a restaurant and cut the famous Alwar milk cake. The next was a public function at his official residence, followed by lunch with his foster family. The cake was pineapple, the food, his favourite Chinese, and the conversation, apolitical. At home, his day begins at 7 a.m. with tea and biscuits. Then he gets on the fitness cycle and for half an hour surfs channels while cycling. Breakfast- usually sprouts, upma or idli, toast and'
    try:
        #text = request.args.get('text')
        #news_id = int(request.args.get('newsid'))
        #id=10888803
        #id = int(request.args.get('id'))
        id=int(request.args.get('newsid'))
        newsdata=db.getTextData_bt(id=id)
        text=newsdata[0]['text']
    except Exception as exp:
        print('Exception in get news id=>',exp)
        #id=0
        #text=None

    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>30:
        story_count=30
        
    #print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ht_bt-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    #log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    
    newslist=[]
    newslist_local=[]
    data_final = []
    id_status=False
    source_id=id
    language=None
    newslist_dic={}
    try:
        source_id=[]
        source_id.append(newsdata[0]['id'])
        source_id.append(newsdata[0]['site_name'])
        source_id.append(newsdata[0]['title'])
        source_id.append(newsdata[0]['Cat_Name'])
        source_id.append(newsdata[0]['URL'])
        source_id.append(newsdata[0]['Description'])
        #source_id.append(newsdata[0]['language'])
        language=newsdata[0]['language']
        print('language =>',language)
        
        #text=list(d)[2] + ' ' + list(d)[5]
        id_status=True
    except Exception as exp:
        print('Exception in get news id=>',exp)
        text=None
        source_id=id
        id_status=False
        
    if id_status:
            if language=='en':
                mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_bt',filename='id_newsid')))
                lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_bt',filename='portal_corpus'))) 
                latest_data = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_bt',filename='latest_data'))) 
            else:
                mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_bt',filename='id_newsid')))
                lda_index = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_bt',filename='portal_corpus'))) 
                latest_data = pickle.loads(gzip.decompress(mdb.get_data_record_from_mongodb(collection_name='tags_file_system_bt',filename='latest_data'))) 
                
    response="SUCCESS"
    message="OK"
    data=""
    print('text=>',text)
    if text==None or len(text)<=10:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        
        #response="FAIL"
        #message="Text should be more than 30 characters"
        #db = db_model()
        #data = db.get_portal_Data(model="BT",LIMIT=5)
        #data_final['text']=unquote(data[0]['text'])
    else:
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
       
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            newslist_dic[mapping_id_newsid_dictionary[x[0]]]=x[1]
            log+=',(%d-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=id,newslist_local))[:story_count]    
        #for x in newslist_local:
         #   newslist.append(x)
        #db = indiatoday_db_model.db_model()    
        #data=db.pickData_from_newsid_it(newslist=newslist_local) 
    data_final=[]    
    counter=1        
    for table_id in newslist_local:
        try:
            latest_data_1=latest_data[table_id]
            latest_data_2=latest_data_1 + (str(newslist_dic[table_id]),)
            data_final.append(latest_data_2)
        except Exception as exp:
            print('Exception =>',exp)
        counter+=1
        
        #len(data_final)
            
    #newslist = [275318,275317,275316,275315,275313]
    #print("data =>",data)
    log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_ht_bt_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    #return jsonify(status=response,message=message,source_newsid=news_id,data=data_final)
    #return jsonify(status=response,message=message,source_text=text,data=data_final)
    return jsonify(source=source_id,id_status=id_status,data=data_final)


@app.route("/recengine/ltop/video/getarticles", methods=['GET', 'POST'])
def ltop_video_getarticles():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    video_count=5
    #video_id='215569'
    video_id=0
    videodata=None
    utm_source = None
    utm_medium = None
    t_psl = False
    
    try:
        video_id = request.args.get('videoid')
        videodata=mdfb.get_lallantop_video_text_from_mongodb(collection_name='ltop_recom_video',fieldname='unique_id',fieldvalue=int(video_id))
        
    except Exception as exp:
        print('Exception in get video id=>',exp)
        video_id=0
 
    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown' 
        
    try:
        video_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        video_count=5        

    if video_count>20:
        video_count=20
        
    print("source_videoid = ",video_id)
    print("video_count = ",video_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-video-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_video_id=%s|sessionid=%s'%(video_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid_video')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus_video'))) 
    mapping_id_newsid_dictionary = None
    lda_index = None
    
    #print(lda_index)
    
    #print(1044367')
    videolist_local=[]
    
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    if videodata==None or len(videodata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(videodata)
        #r=mdfb.get_latest_news_records(collection_name='ltop_recom_video',field_name='modified', LIMIT=50)
        data=mdfb.get_latest_news_records(collection_name='tbl_ltop_videodata', field_name='publishdate', LIMIT=video_count)
        #video_count=30
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='4-mapping-dic-video'
            key_2='4-lda-video'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid_video')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus_video'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)         
        
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(videodata)
        #print('1....')
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        #print('2....')
        tokens = cleaned_tokens_n.split(' ')
        #print('3....')
        cleaned_tokens = [word for word in tokens if len(word) > 3]
        #print('4....')
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        #print('5....')
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        #print('6....')
        video_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        #print('7....')
        similar_news = lda_index[lda_at[video_corpus]]
        #print('8....')
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        #print('9....')
       
        for x in similar_news[:(video_count + 1)]:
            #print('9.1....')    
            videolist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        #print('10....')    
        videolist_local=list(filter(lambda x:x!=video_id,videolist_local))[:video_count]    
        #print('10....')
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        
        videolist_local = list(map(int, videolist_local))
        print('videolist_local =>',videolist_local)
        
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_videodata',fieldname='videoid',fieldvaluelist=videolist_local)
        #print('11....')
        
        #print(data)
    total_dict_data={}
    counter=1
    videolist_local_new=[]
    print('12....')
    #utm_medium='test'    
    
    for row_data in data:
        #print(row_data['videoid'])
        dict_data={}
        
        dict_data['videoid']=row_data['videoid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['link']=row_data['uri']
        dict_data['image']=row_data['mobile_image'] + '?size=299:168'
        dict_data['duration']=row_data['fileduration']
        
        total_dict_data[row_data['videoid']]=dict_data
        videolist_local_new.append(row_data['videoid'])
        counter+=1
     
    #total_dict_data[215522]['link']    
    if latest_Flag:
        videolist_local=videolist_local_new    
        
    uniquelist = []
    for x in videolist_local:
        if x not in uniquelist:
            uniquelist.append(x)
    #print(uniquelist)
    videolist_local=uniquelist   
    
    data_final=[]
    #for videoid in videolist_local_new:
    #utm_medium='test'
    c=1
    for videoid in videolist_local:    
        if len(data_final)==video_count:
            break
        try:
            print(videoid)
            if videoid in videolist_local_new:
                #total_dict_data[videoid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                total_dict_data[videoid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d&t_source=recengine&t_medium=%s&t_content=video_player-slot-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                
                #print(c, ' => ' ,total_dict_data[videoid]['link'])
                data_final.append(total_dict_data[videoid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    #print('Final Data=>',data_final)    
            
    log+='|response=%s'%(videolist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_videoid=video_id,playlist=data_final)

@app.route("/recengine/ltop/story_to_video/getarticles", methods=['GET', 'POST'])
def ltop_story_to_video_getarticles():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    video_count=5
    #newsid='212436'
    newsid=0
    newsdata=None
    utm_source = None
    utm_medium = None
    t_psl = False
    
    try:
        newsid = request.args.get('newsid')
        newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0
 
    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown' 
        
    try:
        video_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        video_count=5        

    if video_count>20:
        video_count=20
        
    print("source_newsid = ",newsid)
    print("video_count = ",video_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-story_to_video-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    #news_id=1044367
    t1 = datetime.now()

    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid_video')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus_video'))) 
    mapping_id_newsid_dictionary = None
    lda_index = None

    
    #print(1044367')
    videolist_local=[]
    
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        #data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
        data=mdfb.get_latest_news_records(collection_name='tbl_ltop_videodata', field_name='publishdate', LIMIT=video_count)
        #video_count=30
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='4-mapping-dic-story-2-video'
            key_2='4-lda-story-2-video'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid_video')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus_video'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        #print('1....')
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        #print('2....')
        tokens = cleaned_tokens_n.split(' ')
        #print('3....')
        cleaned_tokens = [word for word in tokens if len(word) > 3]
        #print('4....')
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        #print('5....')
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        #print('6....')
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        #print('7....')
        similar_news = lda_index[lda_at[news_corpus]]
        #print('8....')
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        #print('9....')
       
        for x in similar_news[:(video_count + 1)]:
            #print('9.1....')    
            videolist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        #print('10....')    
        #videolist_local=list(filter(lambda x:x!=newsid,videolist_local))[:video_count]    
        #print('10....')
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        
        videolist_local = list(map(int, videolist_local))
        print('videolist_local =>',videolist_local)
        
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_videodata',fieldname='videoid',fieldvaluelist=videolist_local)
        #print('11....')
        
        #print(data)
    total_dict_data={}
    counter=1
    videolist_local_new=[]
    print('12....')
    #utm_medium='test'  
    
    #data[6]
    
    for row_data in data:
        #print(row_data['videoid'])
        dict_data={}
        
        dict_data['videoid']=row_data['videoid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['link']=row_data['uri']
        dict_data['image']=row_data['mobile_image'] + '?size=299:168'
        dict_data['duration']=row_data['fileduration']
        
        total_dict_data[row_data['videoid']]=dict_data
        videolist_local_new.append(row_data['videoid'])
        counter+=1
     
    #total_dict_data[215522]['link']    
    if latest_Flag:
        videolist_local=videolist_local_new    
        
    uniquelist = []
    for x in videolist_local:
        if x not in uniquelist:
            uniquelist.append(x)
    #print(uniquelist)
    videolist_local=uniquelist   
    
    data_final=[]
    #for videoid in videolist_local_new:
    #utm_medium='test'
    c=1
    for videoid in videolist_local:    
        if len(data_final)==video_count:
            break
        try:
            print(videoid)
            if videoid in videolist_local_new:
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                #total_dict_data[videoid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                
                total_dict_data[videoid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d&t_source=recengine&t_medium=%s&t_content=video_player-slot-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                #print(c, ' => ' ,total_dict_data[videoid]['link'])
                data_final.append(total_dict_data[videoid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    #print('Final Data=>',data_final)    
            
    log+='|response=%s'%(videolist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=newsid,playlist=data_final)

@app.route("/recengine/ltop/video/unpublish",methods=['GET', 'POST'])
def unpublish_video_lallantop():
    videoid=None
    model=None
    ctype=None
    key=None
    unpublishtime=None
    update_status=True
    message='SUCCESS'
    try:
        print('1.....')
        model = request.args.get('model',None)
        print('2.....',model)
        videoid = request.args.get('videoid',None)
        print('3.....',videoid)
        ctype = request.args.get('ctype',None)
        print('4.....',ctype)
        unpublishtime = request.args.get('unpublishtime',None)
        print('5.....',key)
        key = request.args.get('key',None)
        print("=====>",model,' -- ',videoid,'  --  ',ctype,' -- ',unpublishtime,' -- ',key)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        update_status=False
        message='FAIL'
        
    if model==None or videoid==None or ctype==None or unpublishtime==None:
        update_status=False
        message='FAIL'
        
    if update_status==True and key=='lallantop$r32ufr':
        update_status=True
    else:
        update_status=False
        message='FAIL'
 
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+= "|unpublish_video_lallantop"
    print("Flask Time",datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    log+= "|model=%s|videoid=%s|ctype=%s|unpublishtime=%s|key=%s|status=%s"%(model,videoid,ctype,unpublishtime,key,update_status)
    print("log=>",log)
    filename="flask_unpublish_model_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return jsonify(model=model,ctype=ctype,update_status=update_status,videoid=videoid,message=message)

@app.route("/recengine/ltop/unpublish",methods=['GET', 'POST'])
def unpublish_news_lallantop():
    newsid=None
    model=None
    ctype=None
    key=None
    unpublishtime=None
    update_status=True
    message='SUCCESS'
    try:
        print('1.....')
        model = request.args.get('model',None)
        print('2.....',model)
        newsid = request.args.get('newsid',None)
        print('3.....',newsid)
        ctype = request.args.get('ctype',None)
        print('4.....',ctype)
        unpublishtime = request.args.get('unpublishtime',None)
        print('5.....',key)
        key = request.args.get('key',None)
        print("=====>",model,' -- ',newsid,'  --  ',ctype,' -- ',unpublishtime,' -- ',key)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        update_status=False
        message='FAIL'
        
    if model==None or newsid==None or ctype==None or unpublishtime==None:
        update_status=False
        message='FAIL'
        
    if update_status==True and key=='lallantop$r32ufr':
        update_status=True
    else:
        update_status=False
        message='FAIL'
 
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+= "|unpublish_newsid_lallantop"
    print("Flask Time",datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    log+= "|model=%s|newsid=%s|ctype=%s|unpublishtime=%s|key=%s|status=%s"%(model,newsid,ctype,unpublishtime,key,update_status)
    print("log=>",log)
    filename="flask_unpublish_model_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return jsonify(model=model,ctype=ctype,update_status=update_status,newsid=newsid,message=message)

@app.route("/recengine/ltop/video/getarticles_test", methods=['GET', 'POST'])
def ltop_video_getarticles_test():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    video_count=5
    #video_id='215569'
    video_id=0
    videodata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    
    try:
        video_id = request.args.get('videoid')
        videodata=mdfb.get_lallantop_video_text_from_mongodb(collection_name='ltop_recom_video',fieldname='unique_id',fieldvalue=int(video_id))
        
    except Exception as exp:
        print('Exception in get video id=>',exp)
        video_id=0
        data=None
        
 
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    try:
        video_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        video_count=5        

    if video_count>20:
        video_count=20
        
    print("source_videoid = ",video_id)
    print("video_count = ",video_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-video-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_video_id=%s|sessionid=%s'%(video_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid_video')))
    
    #print(lda_index)
    
    #print(mapping_id_newsid_dictionary)
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus_video'))) 
    
    #print(lda_index)
    
    #print(1044367')
    videolist_local=[]
    
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    if videodata==None or len(videodata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(videodata)
        #r=mdfb.get_latest_news_records(collection_name='ltop_recom_video',field_name='modified', LIMIT=50)
        data=mdfb.get_latest_news_records(collection_name='tbl_ltop_videodata', field_name='publishdate', LIMIT=video_count)
        #video_count=30
        latest_Flag=True
    else:
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(videodata)
        #print('1....')
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        #print('2....')
        tokens = cleaned_tokens_n.split(' ')
        #print('3....')
        cleaned_tokens = [word for word in tokens if len(word) > 3]
        #print('4....')
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        #print('5....')
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        #print('6....')
        video_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        #print('7....')
        similar_news = lda_index[lda_at[video_corpus]]
        #print('8....')
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        #print('9....')
       
        for x in similar_news[:(video_count + 1)]:
            #print('9.1....')    
            videolist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        #print('10....')    
        videolist_local=list(filter(lambda x:x!=video_id,videolist_local))[:video_count]    
        #print('10....')
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        
        videolist_local = list(map(int, videolist_local))
        print('videolist_local =>',videolist_local)
        
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_videodata',fieldname='videoid',fieldvaluelist=videolist_local)
        #print('11....')
        
        #print(data)
    total_dict_data={}
    counter=1
    videolist_local_new=[]
    print('12....')
    #utm_medium='test'    
    
    for row_data in data:
        #print(row_data['videoid'])
        dict_data={}
        
        dict_data['videoid']=row_data['videoid']
        #dict_data['title']=row_data['title']
        dict_data['title']=''
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['link']=row_data['uri']
        dict_data['image']=row_data['mobile_image'] + '?size=299:168'
        dict_data['duration']=row_data['fileduration']
        
        total_dict_data[row_data['videoid']]=dict_data
        videolist_local_new.append(row_data['videoid'])
        counter+=1
     
    #total_dict_data[215522]['link']    
    if latest_Flag:
        videolist_local=videolist_local_new    
        
    uniquelist = []
    for x in videolist_local:
        if x not in uniquelist:
            uniquelist.append(x)
    #print(uniquelist)
    videolist_local=uniquelist   
    
    data_final=[]
    #for videoid in videolist_local_new:
    #utm_medium='test'
    c=1
    for videoid in videolist_local:    
        if len(data_final)==video_count:
            break
        try:
            print(videoid)
            if videoid in videolist_local_new:
                total_dict_data[videoid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                #print(c, ' => ' ,total_dict_data[videoid]['link'])
                data_final.append(total_dict_data[videoid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    #print('Final Data=>',data_final)    
            
    log+='|response=%s'%(videolist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_videoid=video_id,playlist=data_final)


@app.route("/recengine/ltop/getarticles_org_0082020", methods=['GET', 'POST'])
def ltop_getarticles_org_0082020():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    news_count=5
    #newsid='212436'
    newsid=0
    newsdata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    source_newsid = 0
    t_psl = False
    #print('1......')
    try:
        newsid = request.args.get('newsid')
        newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0
        data=None
        
    #print('2......')    
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    #print('3......')    
    try:
        news_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        news_count=5        

    if news_count>20:
        news_count=20
        
    print("source_newsid = ",newsid)
    #print("news_count = ",news_count)
    #news_corpus=u.get_newsid_corpus(news_id)
    source_newsid=newsid
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    #news_id=1044367
    t1 = datetime.now()
    print('4......')     
    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)

    #print(lda_index)
    #print(1044367')
    newslist_local=[]
    
    data_final = []
    mapping_id_newsid_dictionary=None
    lda_index=None
    
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        #data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
        data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
        #video_count=30
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='4-mapping-dic'
            key_2='4-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
            
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        #print('1....')
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        #print('2....')
        tokens = cleaned_tokens_n.split(' ')
        #print('3....')
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        #print('4....')
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        #print('5....')
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        #print('6....')
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        #print('7....')
        similar_news = lda_index[lda_at[news_corpus]]
        #print('8....')
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        #print('9....')
       
        for x in similar_news[:(news_count + 1)]:
            #print('9.1....')    
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=newsid,newslist_local))[:news_count]    
        #print('10....')    
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        
        newslist_local = list(map(int, newslist_local))
        print('newslist_local =>',newslist_local)
        
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_newsdata',fieldname='newsid',fieldvaluelist=newslist_local)
        #print('11....')
        
    total_dict_data={}
    counter=1
    newslist_local_new=[]
    print('12....')
    
    for row_data in data:
        dict_data={}
        dict_data['newsid']=row_data['newsid']
        dict_data['title']=row_data['title']
        dict_data['uri']=row_data['uri']
        dict_data['mobile_image']=row_data['mobile_image'] + '?size=133:75'
        total_dict_data[row_data['newsid']]=dict_data
        newslist_local_new.append(row_data['newsid'])
        counter+=1
     
    if latest_Flag:
        newslist_local=newslist_local_new    
        
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x)
            
    #print(uniquelist)
    newslist_local=uniquelist   
    
    data_final=[]
    c=1
    for newsid in newslist_local:    
        if len(data_final)==news_count:
            break
        try:
            print(newsid)
            if newsid in newslist_local_new:
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                
                total_dict_data[newsid]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                
                #print(c, ' => ' ,total_dict_data[videoid]['link'])
                data_final.append(total_dict_data[newsid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    log+='|response=%s'%(newslist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=source_newsid,data=data_final)

@app.route("/recengine/ltop/getarticles_uat", methods=['GET', 'POST'])
def ltop_getarticles_uat():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    news_count=5
    #newsid='212436'
    newsid=0
    newsdata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    source_newsid = 0
    
    try:
        newsid = request.args.get('newsid')
        newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
        
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0
        data=None
 
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    try:
        news_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        news_count=5        

    if news_count>20:
        news_count=20
        
    print("source_newsid = ",newsid)
    print("news_count = ",news_count)
    #news_corpus=u.get_newsid_corpus(news_id)
    source_newsid=newsid
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid')))
    
    #print(lda_index)
    
    #print(mapping_id_newsid_dictionary)
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus'))) 
    
    #print(lda_index)
    
    #print(1044367')
    newslist_local=[]
    
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        #data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
        data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
        #video_count=30
        latest_Flag=True
    else:
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        #print('1....')
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        #print('2....')
        tokens = cleaned_tokens_n.split(' ')
        #print('3....')
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        #print('4....')
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        #print('5....')
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        #print('6....')
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        #print('7....')
        similar_news = lda_index[lda_at[news_corpus]]
        #print('8....')
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        #print('9....')
       
        for x in similar_news[:(news_count + 1)]:
            #print('9.1....')    
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=newsid,newslist_local))[:news_count]    
        #print('10....')    
        #videolist_local=list(filter(lambda x:x!=newsid,videolist_local))[:video_count]    
        #print('10....')
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        
        newslist_local = list(map(int, newslist_local))
        print('newslist_local =>',newslist_local)
        
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_newsdata',fieldname='newsid',fieldvaluelist=newslist_local)
        #print('11....')
        
        #print(data)
    total_dict_data={}
    counter=1
    newslist_local_new=[]
    print('12....')
    #utm_medium='test'  
    
    #data[6]
    
    for row_data in data:
        #print(row_data['videoid'])
        dict_data={}
        
        dict_data['newsid']=row_data['newsid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['uri']=row_data['uri']
        dict_data['mobile_image']=row_data['mobile_image'] + '?size=133:75'
        
        total_dict_data[row_data['newsid']]=dict_data
        newslist_local_new.append(row_data['newsid'])
        counter+=1
     
    #total_dict_data[215522]['link']    
    if latest_Flag:
        newslist_local=newslist_local_new    
        
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x)
    #print(uniquelist)
    newslist_local=uniquelist   
    
    data_final=[]
    #for videoid in videolist_local_new:
    #utm_medium='test'
    c=1
    for newsid in newslist_local:    
        if len(data_final)==news_count:
            break
        try:
            print(newsid)
            if newsid in newslist_local_new:
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                total_dict_data[newsid]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                #print(c, ' => ' ,total_dict_data[videoid]['link'])
                data_final.append(total_dict_data[newsid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    #print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=source_newsid,data=data_final)

@app.route("/recengine/ltop/getarticles_amp", methods=['GET', 'POST'])
def ltop_getarticles_amp():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    news_count=5
    #newsid='212436'
    newsid=0
    newsdata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    source_newsid = 0
    t_psl = False
    
    try:
        newsid = request.args.get('newsid')
        newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
        
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0
        data=None
 
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    try:
        news_count=int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        news_count=5        

    if news_count>20:
        news_count=20
        
    print("source_newsid = ",newsid)
    print("news_count = ",news_count)
    #news_corpus=u.get_newsid_corpus(news_id)
    source_newsid=newsid
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-getarticles-amp"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    #news_id=1044367
    t1 = datetime.now()

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    mapping_id_newsid_dictionary = None
    lda_index = None
    
    newslist_local=[]
    
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        #data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
        data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
        #video_count=30
        latest_Flag=True
    else:
        
        if redisFlag:
            #print('5......')  
            key_1='4-mapping-dic'
            key_2='4-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        #print('1....')
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        #print('2....')
        tokens = cleaned_tokens_n.split(' ')
        #print('3....')
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        #print('4....')
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        #print('5....')
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        #print('6....')
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        #print('7....')
        similar_news = lda_index[lda_at[news_corpus]]
        #print('8....')
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        #print('9....')
       
        for x in similar_news[:(news_count + 1)]:
            #print('9.1....')    
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=newsid,newslist_local))[:news_count]    
        
        newslist_local = list(map(int, newslist_local))
        print('newslist_local =>',newslist_local)
        
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_newsdata',fieldname='newsid',fieldvaluelist=newslist_local)
        #print('11....')
        
        #print(data)
    total_dict_data={}
    counter=1
    newslist_local_new=[]
    print('12....')
    #utm_medium='test'  
    
    for row_data in data:
        #print(row_data['videoid'])
        dict_data={}
        
        dict_data['newsid']=row_data['newsid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        #dict_data['uri']=row_data['uri']
        #For AMP url 
        dict_data['uri']=row_data['amp_url']
        dict_data['mobile_image']=row_data['mobile_image'] + '?size=133:75'
        
        total_dict_data[row_data['newsid']]=dict_data
        newslist_local_new.append(row_data['newsid'])
        counter+=1
     
    #total_dict_data[215522]['link']    
    if latest_Flag:
        newslist_local=newslist_local_new    
        
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x)
    #print(uniquelist)
    newslist_local=uniquelist   
    
    data_final=[]
    #for videoid in videolist_local_new:
    #utm_medium='amp'
    c=1
    for newsid in newslist_local:    
        if len(data_final)==news_count:
            break
        try:
            print(newsid)
            if newsid in newslist_local_new:
                #total_dict_data[newsid]['uri'] += '?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d'%(c)
                total_dict_data[newsid]['uri'] += '?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=amp&t_content=footerstrip-%d&t_psl=%s'%(c,c,t_psl)
                data_final.append(total_dict_data[newsid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    #print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=source_newsid,items=data_final)


@app.route("/recengine/bt/getarticles", methods=['GET', 'POST'])
def businesstoday_getarticles():
    import businesstoday.utility as businesstoday_utility 
    import businesstoday.mongo_db_file_model as businesstoday_mongo_db_file_model
    print('1.....')
    u=businesstoday_utility.utility()
    mdfb=businesstoday_mongo_db_file_model.mongo_db_file_model()
    
    story_count=8
    #news_id='393182'
    text=None
    newsdata=[]
    utm_source = None
    utm_medium = None
    latest_Flag=False
    t_psl = False
    print('2.....')
    
    try:
        news_id = int(request.args.get('newsid'))
        newsdata=mdfb.get_businesstoday_news_text_from_mongodb(collection_name='bt_recom',fieldname='id',fieldvalue=news_id)
        text=newsdata
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
        
    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'
        
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5
        
        
    print('5.....')
    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|bt-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid_new')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus_new'))) 

    mapping_id_newsid_dictionary = None
    lda_index = None
    
    #print(lda_index)

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdfb.get_latest_news_records(collection_name='tbl_bt_newsdata', field_name='publishdate', LIMIT=story_count)
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='3-mapping-dic'
            key_2='3-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid_new')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus_new'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        #print('6.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        
        
        
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        print('9.....',similar_news[:3])
        s_counter=1
        #newslist_local=[]
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
            
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        data=mdfb.get_businesstoday_news_data_for_json(collection_name='tbl_bt_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        print('14.....data',data)
        
    total_dict_data={}
    counter=1
    #newslist=[]

    
    counter=1        
    for decode_data in data[:story_count]:
        temp_data={}
        temp_data['newsid']=int(decode_data['newsid'])
        temp_data['title']=html.unescape(decode_data['title'])
        #temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        temp_data['uri']=decode_data['uri']
        temp_data['mobile_image']=decode_data['mobile_image']
        newslist.append(decode_data['newsid'])
        print(decode_data['newsid'])
        total_dict_data[decode_data['newsid']]=temp_data
        counter+=1
    
    if latest_Flag:
        newslist_local_final=newslist 
        newslist_local=newslist
        
   
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x) 
            
    #print(uniquelist)
    newslist_local=uniquelist 

    data_final=[]
    c=1
    for n_id in newslist_local_final:    
        if len(data_final)==story_count:
            break
        try:
            print(n_id)
            if n_id in newslist_local:
                #total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                
                total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                data_final.append(total_dict_data[n_id])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)             
        
    print("newslist =>",newslist_local_final)
    #print("data =>",data)
    log+='|response=%s'%(newslist_local_final)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_bt_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=int(news_id),data=data_final)


@app.route("/recengine/bt/getarticles_amp", methods=['GET', 'POST'])
def businesstoday_getarticles_amp():
    import businesstoday.utility as businesstoday_utility 
    import businesstoday.mongo_db_file_model as businesstoday_mongo_db_file_model
    print('1.....')
    u=businesstoday_utility.utility()
    mdfb=businesstoday_mongo_db_file_model.mongo_db_file_model()
    story_count=8
    #news_id='393182'
    text=None
    newsdata=[]
    utm_source = None
    utm_medium = None
    latest_Flag=False
    t_psl = False
    print('2.....')
    
    try:
        news_id = int(request.args.get('newsid'))
        newsdata=mdfb.get_businesstoday_news_text_from_mongodb(collection_name='bt_recom',fieldname='id',fieldvalue=news_id)
        text=newsdata
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
        
    try:
        utm_source = request.args.get('utm_source')
    except Exception as exp:
        print('Exception in get utm_source =>',exp)
        utm_source='Unknown'
        
    try:
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get utm_medium =>',exp)
        utm_medium='Unknown'
        
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    print('5.....')
    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|bt-getarticles_amp"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid_new')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus_new'))) 

    mapping_id_newsid_dictionary = None
    lda_index = None
    
    #print(lda_index)

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdfb.get_latest_news_records(collection_name='tbl_bt_newsdata', field_name='publishdate', LIMIT=story_count)
        latest_Flag=True
    else:
        if redisFlag:
            #print('5......')  
            key_1='3-mapping-dic'
            key_2='3-lda'
            try:
                print('6......')  
                mapping_id_newsid_dictionary=pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index=pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
            except Exception as exp:
                print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
                mapping_id_newsid_dictionary=None
                lda_index=None
        if mapping_id_newsid_dictionary==None or mapping_id_newsid_dictionary==[] or lda_index==None or lda_index==[]:        
            print('7......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='id_newsid_new')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='portal_corpus_new'))) 
            
            if redisFlag:
                newslist_ttl = 5 * 60
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    
        print('6.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        
        
        
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        print('9.....',similar_news[:3])
        s_counter=1
        #newslist_local=[]
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
            
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        data=mdfb.get_businesstoday_news_data_for_json(collection_name='tbl_bt_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        print('14.....data',data)
        
    total_dict_data={}
    counter=1
    #newslist=[]

    
    counter=1        
    for decode_data in data[:story_count]:
        temp_data={}
        temp_data['newsid']=int(decode_data['newsid'])
        temp_data['title']=html.unescape(decode_data['title'])
        #temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        temp_data['uri']=decode_data['amp_url']
        temp_data['mobile_image']=decode_data['mobile_image']
        newslist.append(decode_data['newsid'])
        print(decode_data['newsid'])
        total_dict_data[decode_data['newsid']]=temp_data
        counter+=1
    
    if latest_Flag:
        newslist_local_final=newslist 
        newslist_local=newslist
        
   
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x) 
            
    #print(uniquelist)
    newslist_local=uniquelist 

    data_final=[]
    c=1
    for n_id in newslist_local_final:    
        if len(data_final)==story_count:
            break
        try:
            print(n_id)
            if n_id in newslist_local:
                #total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d'%(c)
                total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=amp&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=amp&t_content=footerstrip-%d&t_psl=%s'%(c,c,t_psl)
                data_final.append(total_dict_data[n_id])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)             
        
    print("newslist =>",newslist_local_final)
    #print("data =>",data)
    log+='|response=%s'%(newslist_local_final)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_bt_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=int(news_id),items=data_final)

@app.route("/recengine/ltop/getcontentdetail", methods=['GET', 'POST'])
def ltop_getcontentdetail():
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()
    
    idlist=[]
    ids=None    

    try:
        #ids='217383,217590,217538'
        ids = request.args.get('ids')
        print('1......ids=> ',ids)
        idlist=ids.split(',')
        print('2......idlist=> ',idlist)
        
        idlist = list(map(int, idlist))
        print('2.1......idlist=> ',idlist)
    except Exception as exp:
        print('Exception in get news id=>',exp)
        idlist=None
        ids=None
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-getcontentdetail"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)

    newslist=[]
   
    response="SUCCESS"
    data=""

    if idlist==None or len(idlist)<=0:
        newslist=None 
        print('3........')
        response="FAIL"
    else:
        print('4........')
        data=mdfb.get_lallantop_news_data_for_json(collection_name='ltop_recom',fieldname='unique_id',fieldvaluelist=idlist)
        counter=1
        print('data =>', data)
        for row_data in data:
            #print(row_data)
            dict_data={}
            dict_data['newsid']=row_data['unique_id']
            dict_data['title']=row_data['title']  
            #str(source, encoding='utf-8', errors = 'ignore')
            dict_data['title']=str(row_data['title'])
            dict_data['url']=row_data['url']
            dict_data['imageurl']=row_data['imageurl'] + '?size=133:75'
            dict_data['modified']=row_data['modified']
            newslist.append(dict_data)
            counter+=1
            
    if newslist==[]:
        response="FAIL"
        
    #print('newslist =>', newslist)   

    #print('type(newslist) =>', type(newslist))     
  
    #import json
    #r=json.dumps(newslist)
        
    #log+='|response=%s'%(newslist)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_getcontent_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,data=newslist)

@app.route("/recengine/rd/getarticles", methods=['GET', 'POST'])
def rdigest_getarticles():
    import rdigest.utility as rdigest_utility 
    import rdigest.mongo_db_file_model as rdigest_mongo_db_file_model
    print('1.....')
    u=rdigest_utility.utility()
    mdfb=rdigest_mongo_db_file_model.mongo_db_file_model()
    story_count=10
    #news_id=125279
    text=None
    newsdata=[]
    utm_source = request.args.get('utm_source')
    utm_medium = request.args.get('utm_medium')
    latest_Flag=False
    t_psl = False
    print('2.....')
    try:
        news_id = int(request.args.get('newsid'))
        newsdata=mdfb.get_rdigest_news_text_from_mongodb(collection_name='rd_recom',fieldname='unique_id',fieldvalue=str(news_id))
        text=newsdata
        print('text =>',text)
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
    print('4.....')    
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=10
    print('5.....')
    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|rd-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='rd_file_system',filename='id_newsid')))
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='rd_file_system',filename='portal_corpus'))) 
    
    #print(lda_index)

    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdfb.get_latest_news_records(collection_name='tbl_rd_newsdata', field_name='publishdate', LIMIT=story_count)
        latest_Flag=True
    else:
        print('6.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        print('9.....',similar_news[:3])
        s_counter=1
        #newslist_local=[]
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
        
        newslist_local = list(map(int, newslist_local))    
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        
        data=mdfb.get_rdigest_news_data_for_json(collection_name='tbl_rd_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        print('14.....data',data)

    total_dict_data={}
    counter=1
    #newslist=[]

    
    counter=1        
    for decode_data in data[:story_count]:
        temp_data={}
        temp_data['newsid']=int(decode_data['newsid'])
        temp_data['title']=html.unescape(decode_data['title'])
        #temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        temp_data['uri']=decode_data['uri']
        temp_data['mobile_image']=decode_data['mobile_image']
        newslist.append(decode_data['newsid'])
        print(decode_data['newsid'])
        total_dict_data[decode_data['newsid']]=temp_data
        counter+=1
    
    if latest_Flag:
        newslist_local_final=newslist 
        newslist_local=newslist
        
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x) 
            
    #print(uniquelist)
    newslist_local=uniquelist 

    data_final=[]
    c=1
    for n_id in newslist_local_final:    
        if len(data_final)==story_count:
            break
        try:
            print(n_id)
            if n_id in newslist_local:
                #total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                data_final.append(total_dict_data[n_id])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)             
        
    print("newslist =>",newslist_local_final)
    #print("data =>",data)
    log+='|response=%s'%(newslist_local_final)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_rd_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=int(news_id),data=data_final)

@app.route("/recengine/rd/getarticles_amp", methods=['GET', 'POST'])
def rdigest_getarticles_amp():
    import rdigest.utility as rdigest_utility 
    import rdigest.mongo_db_file_model as rdigest_mongo_db_file_model
    print('1.....')
    u=rdigest_utility.utility()
    mdfb=rdigest_mongo_db_file_model.mongo_db_file_model()
    story_count=10
    #news_id=125279
    text=None
    newsdata=[]
    utm_source = request.args.get('utm_source')
    utm_medium = request.args.get('utm_medium')
    latest_Flag=False
    t_psl = False
    print('2.....')
    try:
        news_id = int(request.args.get('newsid'))
        newsdata=mdfb.get_rdigest_news_text_from_mongodb(collection_name='rd_recom',fieldname='unique_id',fieldvalue=str(news_id))
        text=newsdata
        print('text =>',text)
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
        text=None
    print('4.....')    
    try:
        story_count = int(request.args.get('no'))
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=10
    print('5.....')
    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|rd-getarticles-amp"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=790807
    t1 = datetime.now()
    #story_count=8
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    lemma   = WordNetLemmatizer()

    mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='rd_file_system',filename='id_newsid')))
    #lda_it=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='rd_file_system',filename='portal_corpus'))) 
    #print(lda_index)
    newslist=[]
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if text==None or len(text)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
        #data=db.picklatestData_from_newsid_it(LIMIT=story_count) 
        data=mdfb.get_latest_news_records(collection_name='tbl_rd_newsdata', field_name='publishdate', LIMIT=story_count)
        latest_Flag=True
    else:
        print('6.....')
        log+= '|SUCCESS'
        log+= '|Result'
        text = text.lower()
        text = u.clean_doc(text)
        tokens = tokenizer.tokenize(text)
        cleaned_tokens = [word for word in tokens if len(word) > 2]
        stopped_tokens = [i for i in cleaned_tokens if not i in en_stop]
        #print(stopped_tokens)
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        #print(stemmed_tokens)
        lemma_tokens = [lemma.lemmatize(i) for i in stemmed_tokens]
        #print("lemma_tokens => ",lemma_tokens)
        news_corpus = [dictionary_it.doc2bow(text) for text in [lemma_tokens]]
        similar_news = lda_index[lda_it[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
        print('9.....',similar_news[:3])
        s_counter=1
        #newslist_local=[]
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            s_counter+=1
        
        newslist_local = list(map(int, newslist_local))    
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        print('12.....',newslist_local)
         
        used = set()
        newslist_local_final = [x for x in newslist_local if x not in used and (used.add(x) or True)]         
        print('13.....',newslist_local_final)
        
        data=mdfb.get_rdigest_news_data_for_json(collection_name='tbl_rd_newsdata',fieldname='newsid',fieldvaluelist=newslist_local_final)
        print('14.....data',data)

    total_dict_data={}
    counter=1
    #newslist=[]

    
    counter=1        
    for decode_data in data[:story_count]:
        temp_data={}
        temp_data['newsid']=int(decode_data['newsid'])
        temp_data['title']=html.unescape(decode_data['title'])
        #temp_data['uri']=decode_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        temp_data['uri']=decode_data['amp_url']
        temp_data['mobile_image']=decode_data['mobile_image']
        newslist.append(decode_data['newsid'])
        print(decode_data['newsid'])
        total_dict_data[decode_data['newsid']]=temp_data
        counter+=1
    
    if latest_Flag:
        newslist_local_final=newslist 
        newslist_local=newslist
        
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x) 
            
    #print(uniquelist)
    newslist_local=uniquelist 

    data_final=[]
    c=1
    utm_medium='amp'
    for n_id in newslist_local_final:    
        if len(data_final)==story_count:
            break
        try:
            print(n_id)
            if n_id in newslist_local:
                #total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                total_dict_data[n_id]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                data_final.append(total_dict_data[n_id])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)             
        
    print("newslist =>",newslist_local_final)
    #print("data =>",data)
    log+='|response=%s'%(newslist_local_final)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    #log+= "|json_request_data=%s"%(json_data)
    #log+= '|text=%s'%(text)
    filename="flask_web_application_rd_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,newsid=newslist,message=message,data=data_final)
    return jsonify(status=response,message=message,source_newsid=int(news_id),items=data_final)

@app.route('/recengine/cookie/s')
def setcookie():
    res = make_response("Setting a cookie")
    value=None
    key=None
    try:
        key=request.args.get('key')
        value = request.args.get('value')
        if value!=None and key!=None:
            res.set_cookie(key, value, max_age=60*60*24*365)        
    except Exception as exp:
        value=None
        key=None
    return res

@app.route('/recengine/cookie/g')
def getcookie():
    print('1....')
    key=None
    cookie='No cookie for this key'
    try:
        print('2....')
        key = request.args.get('key')
        print('3....')
        if key!=None:
            cookie=flask.request.cookies.get(key)
        print('4....')    
    except Exception as exp:
        print('5....')
        key=None
        cookie='No cookie for this key'
    print('6....')    
    print('cookie==>',cookie)
    headers = request.headers
    return jsonify(key=key,cookie=cookie,headers=str(headers)) 

@app.route("/recengine/rd/getarticles_test", methods=['GET', 'POST'])
def rdigest_getarticles_test():
    news_id=None
    print('2.....')
    try:
        news_id = request.args.get('newsid')
        print('3.....')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id=0
    print('4.....') 
    
    data_final = [{"newsid": 125389,"title": "The Best From The World Of Entertainment: Misbehaviour, Ludo, Sergio, After Life Season 2 And More","uri": "https://www.readersdigest.co.in/culturescape/story-the-best-from-the-world-of-entertainment-misbehaviour-darbaan-sergio-after-life-season-2-and-more-125389?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-1","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202004/misbehaviour2_1585809261_370x208.jpeg"},{"newsid": 125245,"title": "The Best From The World Of Entertainment: The Call Of The Wild, Shubh Mangal Zyada Saavdhan, Hunters, Outlander And More","uri": "https://www.readersdigest.co.in/culturescape/story-the-best-from-the-world-of-entertainment-the-call-of-the-wild-shubh-mangal-zyada-saavdhan-hunters-outlander-and-more-125245?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-2","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202001/callofthewild_1580476581_370x208.png"},{"newsid": 125386,"title": "All You Homebound People, Show Your Feet Some Tender Loving Care","uri": "https://www.readersdigest.co.in/health-wellness/story-all-you-homebound-people-show-your-feet-some-tender-loving-care-125386?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-3","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202004/img_20200330185628_1513578062_huge_1585754549_370x208.jpeg"},{"newsid": 125385,"title": "Delish Breakfast Bowls For A Healthy Start","uri": "https://www.readersdigest.co.in/better-living/story-delish-breakfast-bowls-for-a-healthy-start-125385?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-4","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202004/img_20200401140411_1555499630_huge_1585753247_370x208.jpeg"},{"newsid": 125388,"title": "The Man With A Heart Of Gold","uri": "https://www.readersdigest.co.in/true-stories/story-the-man-with-a-heart-of-gold-125388?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-5","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202004/kos_1585804870_370x208.jpeg"},{"newsid": 125387,"title": "These Super Grannies Came Out On Top Battling Coronavirus","uri": "https://www.readersdigest.co.in/true-stories/story-these-super-grannies-came-out-on-top-battling-coronavirus-125387?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-6","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202004/supergranny_1585802954_370x208.png"},{"newsid": 125382,"title": "5 Classic Crime Fiction Reads In The Times Of Quarantine","uri": "https://www.readersdigest.co.in/culturescape/story-5-classic-crime-fiction-reads-in-the-times-of-quarantine-125382?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-7","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202004/detective_1585741124_370x208.png"},{"newsid": 125379,"title": "A Note From The Editor: How Fragile We Are","uri": "https://www.readersdigest.co.in/conversations/story-how-fragile-we-are-125379?utm_source=recengine&utm_medium=web&referral=yes&utm_content=footerstrip-8","mobile_image": "https://akm-img-a-in.tosshub.com/sites/rd/resources/202004/ntsquare_1585747152_370x208.png"} ]
    
    response="SUCCESS"
    message="OK"
    return jsonify(status=response,message=message,source_newsid=int(news_id),data=data_final)

@app.route("/recengine/info/efidwq", methods=['GET', 'POST'])
def get_recengine_uid():
    print('1....')
    sp_itgd=None
    uid=None
    try:
        sp_itgd=flask.request.cookies.get('sp')
        print('Getting sp....')  
    except Exception as exp:
        print('Exception....', exp)
        sp_itgd=None

    try:
        uid=flask.request.cookies.get('uid')
        print('Getting uid...')        
    except Exception as exp:
        print('Exception....', exp)
        user_id=None

    print('2....')    
    return jsonify(sp_itgd=sp_itgd,uid=uid) 

@app.route("/recengine/remodel/ltop/process", methods=['GET', 'POST'])
def process_ltop():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()     
    
    process_Flag=False

    story_count=5
    text=None
    news_topics_distribution={}

    dictionary=None
    terms=None
    similar_news=[]
    newslist=[]
    newslist_local=[]
    
    print('1.....')

    try:
        newsid = request.args.get('newsid')
        #newsid='217448'
        text=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
    except Exception as exp:
        print('Exception in get news id=>',exp)

    print('2.....')
    
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-process-algo"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(id,'None')


    print("source_newsid = ",newsid)
    #print("story_count = ",type(story_count))

    
    if text==None or len(text)<=10:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(text)
    else:
        print('3.....')
        hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(text)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 3]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        terms=stemmed_tokens
        print('4.....')
        portal_corpus = [dictionary_at.doc2bow(stemmed_tokens)] 
        print('5.....')
        lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus')))
        print('6.....')
        print(lda_at)
        print('7.....')
        s_news = lda_index[lda_at[portal_corpus]]
        similar_news = sorted(enumerate(s_news[0]), key=lambda item: -item[1])
        print('8.....')
        mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid')))
        process_Flag=True
        print('9.....')
    #mapping_id_newsid_dictionary[0]
    uniquelist = [] 
    mapping_for_algo='News - '
    print('100......')
    #if terms is not None: 
    if process_Flag:
    #if True:
        log+='|L1=%s'%(newsid)
        print('104......')
        #for x in similar_news[:(story_count + 1)]:
        
        for x in similar_news[:30]:
            print('105......',mapping_id_newsid_dictionary[x[0]])
            
            #print(mapping_id_newsid_dictionary[x[0]], ' == ' , mapping_id_newsid_dictionary[x[0]]!=newsid , ' = Both ' ,mapping_id_newsid_dictionary[x[0]] not in uniquelist, (mapping_id_newsid_dictionary[x[0]]!=newsid) and (mapping_id_newsid_dictionary[x[0]] not in uniquelist) , ' :: uniquelist', uniquelist)
            if (mapping_id_newsid_dictionary[x[0]]!=newsid) and (mapping_id_newsid_dictionary[x[0]] not in uniquelist) and len(uniquelist)<story_count:
                uniquelist.append(mapping_id_newsid_dictionary[x[0]])
                newslist_local.append(mapping_id_newsid_dictionary[x[0]])
                #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
                mapping_for_algo += ', (%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            #print('106......')  
        
        #newslist_local=list(filter(lambda x:x!=id,newslist_local))[:story_count] 
        print('newslist_local ==>',newslist_local)
        for x in newslist_local:
            newslist.append(x)
    print('mapping_for_algo=>',mapping_for_algo)
    match_list=[]
    nomatch_list=[]
    
    dictionary=dictionary_at
    
    #len(match_list)

    for t in terms:
        temp_var = dictionary.doc2idx([t])[0]
        if temp_var>-1:
            match_list.append(t)
            #print('Match =>',t)
        else:
            nomatch_list.append(t)
            #print('No Match =>',t)
    
    match_per='100'
    nomatch_per='0'
    
    if len(terms)>0: 
        match_per= round((len(match_list)*100)/(len(terms)),2)
        nomatch_per= round((len(nomatch_list)*100)/(len(terms)),2)  
    
    match_list1=None
    match_list2=None
    match_list3=None
    match_list4=None
    match_list5=None
  
    news_counter=1
    for news_temp in newslist:
        print('R news id=>',news_temp)
        #news_temp=77611967
        try:
            #newsdata=db.getTextData_tag(id=news_temp,lang=language)
            #text=newsdata[0]['text']
            #news_temp='218598'
            text=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(news_temp))            
            
            match_list_temp=[]
            #len(terms['term'])
            
            clean_text = u.clean_doc_hindi(text)
            cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
            tokens = cleaned_tokens_n.split(' ')
            cleaned_tokens = [word for word in tokens if len(word) > 3]
            stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
            stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
            temp_terms=stemmed_tokens
            #dictionary_temp = Dictionary([temp_terms]) 
            #print(dictionary_temp)
            #len(terms)
            #match_list_temp=[]
            for t_dic in match_list:
                #print('news_counter =>',news_counter)
                #print(t_dic)
                #temp_var = dictionary_temp.doc2idx([t_dic])[0]
                #print(t_dic,' == ',temp_var)
                
                try:
                    temp_var=temp_terms.index(t_dic)
                    #print(t_dic)
                except Exception as exp:
                    temp_var=-1
                    #print("Exception in similar news =",exp,' :x=',x)
                
                if temp_var>-1:
                    match_list_temp.append(t_dic)
            if news_counter==1:
                match_list1=match_list_temp
            if news_counter==2:
                match_list2=match_list_temp
            if news_counter==3:
                match_list3=match_list_temp
            if news_counter==4:
                match_list4=match_list_temp
            if news_counter==5:
                match_list5=match_list_temp
           
            lda_at_topic=lda_at
            #local_portal_corpus=[]    
            #local_portal_corpus = [dictionary_it.doc2bow(match_list_temp)] 
            #topics_distribution  = lda_it.get_document_topics(local_portal_corpus, per_word_topics=True)
            topics_distribution  = lda_at_topic.get_document_topics([dictionary_at.doc2bow(match_list_temp)])
            news_topics_distribution[news_temp]=topics_distribution[0]

        except Exception as exp:
            print('Exception to get news Article =>',exp)
            #news_flag = False
        print('news_topics_distribution =>',news_topics_distribution)    
        news_counter +=1                        

    #all_topics  = lda_it.get_document_topics(portal_corpus, per_word_topics=True)
    #all_topics  = lda_at.get_document_topics(temp_terms)
    all_topics  = lda_at.get_document_topics(portal_corpus)
    
    document_topics=all_topics[0]
    #for doc_topics, word_topics, phi_values in all_topics:
        #print(doc_topics)
        #print("\n")
        #document_topics.append(doc_topics)
    
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_ltop_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)

    return render_template('ltop/ltop_process_algo.html',total_term=(0 if terms is None else len(terms)),term=terms,match_per=match_per,match_list=match_list,match_list_length=(0 if match_list is None else len(match_list)),nomatch_per=nomatch_per,nomatch_list=nomatch_list,nomatch_list_length=(0 if nomatch_list is None else len(nomatch_list)),match_list1=match_list1,match_list1_length=(0 if match_list1 is None else len(match_list1)),match_list2=match_list2,match_list2_length=(0 if match_list2 is None else len(match_list2)),match_list3=match_list3,match_list3_length=(0 if match_list3 is None else len(match_list3)),match_list4=match_list4,match_list4_length=(0 if match_list4 is None else len(match_list4)),match_list5=match_list5,match_list5_length=(0 if match_list5 is None else len(match_list5)), mapping_for_algo=mapping_for_algo,document_topics=document_topics,news_topics_distribution=news_topics_distribution)



@app.route("/recengine/remodel/ltop/topics", methods=['GET', 'POST'])
def ltop_topics():
    num = int(request.args.get('num'))
    model = request.args.get('model')
    
    print('num=',num)
    print('model=',model)
    
    if num<=0:
        num = 5
    elif num>1000:
        num=1000        

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|gettopics"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|model=%s"%(model)
    log+= "|queryString=%s"%(request.query_string)
    log+='|num=%s|sessionid=%s'%(num,'None')
    
    print('lda_at=',lda_at)
    
    lda_topic_list = lda_at.show_topics(num_topics=-1, num_words=num, log=False, formatted=True)        
    
    print('lda_topic_list=',lda_topic_list)
    
    
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    log+='|total_time=%s'%(d)
    filename="flask_web_application_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    
    return render_template('lda_topic.html',lda_topic_list=lda_topic_list)

@app.route("/recengine/ltop/demo/get_history_based_articles_v1", methods=['GET', 'POST'])
def ltop_history_based_articles_v1():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    import pickle  
    import gzip   
    import boto3
    from boto3.dynamodb.conditions import Key
    import pandas as pd
    import requests      
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()
    
    count=10
    newsid=None
    newsdata=None
    utm_medium = None
    user_id=None
        
    try:
        newsid = request.args.get('newsid')
    except Exception as exp:
        print('Exception in get video id=>',exp)
        newsid=0
    
    all_flag = 0
    
    try:
        all_flag = int(request.args.get('all'))
    except Exception as exp:
        print('Exception in get all=>',exp)
        all_flag=0
        
    try:
        #Temp
        #user_id='01f6aaac-9c37-440d-8973-6ad4ca7203a4'
        user_id = request.args.get('user_id')
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        
    try:
        count = int(request.args.get('count'))
    except Exception as exp:
        print('Exception in get video id=>',exp)
        count=10    
        
    print('1......')
        
    newslist = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='newslist'))) 
    
    if count>15:
        count=15

    print('2......')
        
    print("source_newsid = ",newsid)
    print("count = ",count)
    print('newslist =>', newslist)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-historybased-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    #news_id=1044367
    t1 = datetime.now()

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    
    print('3......')
    
    #Temporary
    #user_id='01f6aaac-9c37-440d-8973-6ad4ca7203a4'
    
    TABLE_NAME = "itgd_cs_model_info_prod"
    #dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1",aws_access_key_id='AKIAJ6K7K5FZMBPE4DYA', aws_secret_access_key='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')
    dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
    table = dynamodb.Table(TABLE_NAME)
    
    print('4......')
    
    final_id=user_id
    module_exists=False
    
    try:
        response = table.get_item(
            Key={
                'final_id': final_id,
                'site_id': 4
            }
        )
    except:
        print('Exception')
 
    try:
        res=response['Item']    
        if res!=None:
            module_exists=True
        print('User exist in Model')
    except:
        print('User does not exist in Model')    
 
    print('5......')
    
    json_d={}
    user_interection={}
    if module_exists:
        TABLE_NAME = "itgd_cs_interaction_data_prod"
        #TABLE_NAME = "itgd_cs_user_interaction_prod"
        # Creating the DynamoDB Table Resource
        #dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1",aws_access_key_id='AKIAJ6K7K5FZMBPE4DYA', aws_secret_access_key='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')
        dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
        table = dynamodb.Table(TABLE_NAME)
        
        print('6......')


        #Temparory
        final_id=user_id
        #final_id='f92bc5d9-ef60-4b1a-8b12-245da1944cfa'
        response = table.query(
         IndexName="final_id-index",
          KeyConditionExpression=Key('final_id').eq(final_id)
        )
        #response['Items'] 
        
        print('7......')

        print('U1...')
        df_r = pd.DataFrame(response['Items'],columns=['event','ist_tstamp','final_id','newsid'])   
        print('U2...')
        df_r=df_r[df_r['event']=='page_view']
        df_r=df_r[['ist_tstamp','final_id','newsid']]
        df_r=df_r.rename(columns={'ist_tstamp': 'tstamp','final_id': 'user_id','news': 'newsid'})
        print('U3...')
        user_viewed_news=df_r
        user_viewed_news=user_viewed_news[['tstamp','newsid']]
        n_list=user_viewed_news['newsid'].tolist()
        #========User Interaction Start 
        print('U4...',n_list)
        ids = ','.join(str(e) for e in n_list)
        print('U4.1...', ids)
        lallantop_get_news_url="https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=%s"%(ids)
        print('lallantop_get_news_url =>', lallantop_get_news_url)
        
        #Temp
        #lallantop_get_news_url = "https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=1197339,258193,257031,1199095,256974,257031,256687,256827,256974,256459,1197400,1196735,257431,1196702,258226,257031,1197853,256974,258074,256663,257431,258074,1197880,258074,1196702,1196228,256786,256936,255552,257431,258074,253087,256265,258074,258226,256976,258193,1198351,256774,1196702,258361,256723,256786,1197383,1196702,1197383,1198470,256265,1198849,258074,255552,256723,1196702,1198013,1198472,1197339,1197400,257031,1196353,258300,256057,257031,255552,1196702,258074,256827,1196357,1196735,1196702,1198159,256499,256974,256459,1196228,256687,258226,257305,21755,258226,1198844,256057,256687,258300,21755,258226,258361,256723,258300,1196702,1196702,258193,256974,1196702,1196702,256687,257031,256723,256499,1196353,258171,257431,1196702,258300,258226,256723,256976,256936,256293,258361,253087,257087,258226,258300,256936,253087,1197222,256936,1198004,1197222,1196357,1196702,256774,258361,258193,256974,255552,256723,1198630,256663,1196702,257323,258226,258226,1198105,256293"
        print('U5...')
        r = requests.get(lallantop_get_news_url)
        
        #r.text
        lallantop_news_response = json.loads(r.text)
        lallantop_df=pd.DataFrame(lallantop_news_response['data'])
        print('U6...')
        
        if lallantop_df.shape[0]>0:
            print('U7...')
            lallantop_df=lallantop_df[['newsid','title','url']]
            user_viewed_news['newsid'] = user_viewed_news['newsid'].astype(int)
            user_interected_news = user_viewed_news.merge(lallantop_df, on='newsid')
            print('U8...')
            #user_interected_news = lallantop_df
            user_interected_news=user_interected_news.sort_values(by=['tstamp'], ascending=[False])
            print('U9...')
            #user_interected_news
            #user_interected_news=user_interected_news.drop_duplicates(subset=['tstamp','user_id','newsid'])
            #user_interected_news=user_interected_news.drop_duplicates(subset=['tstamp','newsid'])
            if all_flag==0:
                user_interected_news = user_interected_news.groupby(['newsid']).agg({'tstamp':'last','title':'last','url':'last'}).sort_values(by=['tstamp'], ascending=[False]).reset_index()
            user_interected_news = user_interected_news.head(30) 
            #user_interected_news.style.format({'url': make_clickable})
        else:
            print('No Data Avaialble')    
            
        print('U10...')
        
        #Temp    
        #newslist = ['115531','125307','126168','126327','131743','13677','138181','139833','149492','149688','151568','166715','17174','19338','20408','20760','21220','21315','214261','226172','230542','241401','246104','246543','247669','248494','249205','250380','250601','251516','251542','251555','251589','251606','251607','251612','251648','251650','251659','251676','251694','251699','251706','251725','251735','251738','251743','251751','251754','251756','251757','251763','251799','251821','251827','251834','251835','251852','251877','251881','251884','251884','251735','251888','251896','251897','251901','251905','251912','251922','251926','251933','251939','251945','251948','251951','251960','251974','251976','251979','251985','251997','252010','252012','252015','252023','252024','252035','252036','252045','252049','252057','252058','252059','252070','252090','252096','252103','252119','252123','252127','252137','252139','252147','252152','252153','252196','252201','252203','252211','252217','252218','252225','252238','252239','252245','252253','252257','252260','252290','252293','252305','252309','252312','252335','252354','252358','252362','252380','252384','252396','252408','252422','252434','252436','252438','252438','252509','252461','252469','252474','252487','252489','252492','252494','252506','252512','252526','252528','252548','252552','252553','252559','252592','252597','252601','252605','252611','252623','252625','252635','252641','252643','252648','252669','252684','252691','252725','252729','252738','252742','252744','252750','252779','252788','252789','252791','252792','252803','252807','252807','252810','252841','252846','252858','252860','252861','252863','252872','252880','252882','252891','252905','252912','252927','252928','252932','252942','252948','252952','252958','252959','252962','252973','252983','252993','253000','253006','253018','253020','253022','253030','253031','253037','253038','253047','253050','253059','253062','253064','253070','253072','253087','253091','253114','253117','253134','253145','253147','253150','253155','253160','253161','253164','253165','253168','253184','253192','253195','253197','253209','253233','253247','253255','253269','253274','253274','253269','253275','253280','253290','253293','253304','253309','253312','253325','253327','253331','253355','253356','253359','253370','253371','253383','253387','253411','253422','253427','253429','253430','253434','253446','253452','253464','253469','253476','253478','253479','253480','253498','253534','253544','253629','253630','253639','253644','253645','253654','253659','253660','253671','253693','253696','253699','253703','253705','253710','253718','253721','253723','253734','253757','253770','253776','253779','253781','253807','253269','253817','253818','253821','253824','253826','253828','253838','253841','253849','253861','253867','253873','253887','253898','253903','253923','253926','253930','253936','253945','253949','253966','253973','253975','253976','253985','253987','253997','254003','254015','254032','254048','254050','254051','254055','254067','254073','254076','254082','254092','254107','254113','254114','254123','254133','254141','254146','254148','254161','254166','254168','254185','254196','254198','254217','254218','254231','254238','254248','254256','254277','254279','254282','254284','254301','254302','254324','254327','254336','254338','254351','254353','254356','254358','254370','254374','254382','254389','254392','254394','254395','254400','254405','254445','254448','254451','254454','254460','254485','254491','254502','254505','254506','254548','254558','254563','254568','254570','254575','254585','254588','254590','254604','254610','254614','254615','254624','254640','254645','254653','254663','254665','254693','254694','254701','254705','254706','254707','254715','254722','254725','254732','254736','254752','254753','254786','254809','254817','254823','254828','254837','254841','254845','254854','254864','254868','254877','254880','254881','254889','254894','254909','254923','254931','254932','254936','254946','254948','254949','254959','254961','254980','254983','254993','255016','255032','255042','255059','255064','255066','255068','255073','255075','255077','255087','255088','255095','255104','255106','255109','255111','255139','255149','255156','255167','255169','255175','255192','255203','255205','255206','255214','255224','255228','255229','255266','255268','255271','255272','255273','255282','255297','255313','255314','255315','255342','255352','255353','255359','255381','255385','255390','255395','255398','255414','255422','255423','255456','255458','255478','255487','255499','255502','255522','255538','255547','255553','255563','255569','255581','255583','255588','255605','41494','47538','47958','53849','74318','74416','75567','75654','75953','76132','77268','80500','95975']
        newslist=list(map(int, newslist))   
        newslist_set=set(newslist)  
        user_viewed_news=df_r[['tstamp','newsid']]
        n_list=user_viewed_news['newsid'].tolist()
        ids = ','.join(str(e) for e in n_list)
        interacted_newsid=set(list(map(int, ids.split(','))))
        non_interacted_news_list = newslist_set- interacted_newsid -set([newsid])
        
        #user_id='086b2d0c-62da-4fbb-9f09-7f1653a402f5'
        data={}
        data['user_id']=user_id
        data['newsid']=newsid
        data['count']=count
        data['newslist']=list(non_interacted_news_list)    
        
        url='https://142dro4haa.execute-api.ap-southeast-1.amazonaws.com/prod/lallantop-recengine'
        headers = {"Content-Type": "application/json","Accept": "application/json"}
        response = requests.post(url, json=data, headers=headers)
        responseBody=response.text     
        
        res=json.loads(responseBody)
        json_d={}
        
        if res!=None:
            news_list_with_estimator = pd.DataFrame.from_records(res)
            
            n_list=[]
            for response_data in res:
                n_list.append(response_data['newsid'])
                
            uniquelist = []
            for x in n_list:
                if x not in uniquelist:
                    uniquelist.append(x) 
            
            #uniquelist=[217449,217387,214087,217749]
            
            data=mdfb.get_lallantop_news_data_for_json(collection_name='ltop_recom',fieldname='unique_id',fieldvaluelist=uniquelist)
                
            ids = ','.join(str(e) for e in n_list)
            lallantop_get_news_url="https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=%s"%(ids)
            
            #lallantop_get_news_url="https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=254640,254722,255563,255499,255167"
            
            #import requests
            #import pandas as pd
            #print(lallantop_get_news_url)
            print('8.....')
            r = requests.get(lallantop_get_news_url)
            print('9.....')
            lallantop_news_response = json.loads(r.text)
            print('10.....')
            lallantop_df=pd.DataFrame(lallantop_news_response['data'])
            print('11.....')
            #lallantop_df = lallantop_df.drop_duplicates(subset=['newsid'])
            lallantop_df = lallantop_df.groupby(['newsid']).agg({'title':'last','url':'last','modified':'last'}).reset_index()
            print('12.....')
            lallantop_df=lallantop_df[['newsid','title','url']]
            print('13.....')
            final_recommended_news=news_list_with_estimator.merge(lallantop_df, on='newsid')    
            print('14.....')
            final_recommended_news=final_recommended_news.sort_values(['est'], ascending=[False])
            print('15.....', type(final_recommended_news))
            json_d = final_recommended_news.to_json(orient='records')
            print('16.....', json_d)
            json_d = json.loads(json_d)
            print('17.....', json_d)
        else:
            print('No Response')    
            json_d={}
    else:
        print('Module Does Not Exists')
        json_d=[]
        user_interection=[]
    user_interection={}
    try:
        print('18.....')
        user_interection = user_interected_news.to_json(orient='records')
        print('19.....', type(user_interected_news))
        user_interection = json.loads(user_interection)
        print('20.....')
    except:
        user_interection=[]
        print('21.....Exception user_interection')
        
            
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,message=message,source_videoid=video_id,playlist=data_final)
    #return jsonify(status='OK',recommnedation=json_d,user_interection=user_interection)    
    return render_template('ltop/ltop_history_rec_demo.html',user_id=user_id,newsid=newsid,count=count,user_interection=user_interection,recommnedation=json_d)



@app.route("/recengine/ltop/demo/get_history_based_articles1", methods=['GET', 'POST'])
def ltop_history_based_articles1():
     recommnedation = [{"est":4.729060781,"newsid":256794,"title":"\u0915\u094d\u092f\u0942\u0902 \u0928\u0947\u0936\u0928\u0932 \u092a\u0947\u0902\u0936\u0928 \u0938\u094d\u0915\u0940\u092e \u0938\u0947 \u0928 \u0938\u0930\u0915\u093e\u0930\u0940 \u0915\u0930\u094d\u092e\u091a\u093e\u0930\u0940 \u0916\u093c\u0941\u0936 \u0939\u0948\u0902, \u0928 \u092c\u093e\u0915\u093c\u0940 \u0932\u094b\u0917 \u0907\u0938\u0938\u0947 \u091c\u0941\u0921\u093c \u0930\u0939\u0947 \u0939\u0948\u0902?","url":"https://www.thelallantop.com/bherant/basics-of-nps-national-pension-scheme-and-difference-with-ops-old-pension-scheme-and-gpf-general-provident-fund/"},{"est":4.6449231297,"newsid":256791,"title":"\u092e\u093e\u0909\u0902\u091f\u092c\u0947\u091f\u0928 \u092f\u094b\u091c\u0928\u093e, \u091c\u093f\u0938\u0928\u0947 \u092d\u093e\u0930\u0924 \u0915\u0947 \u0926\u094b \u091f\u0941\u0915\u095c\u0947 \u0915\u0930 \u0926\u093f\u090f","url":"https://www.thelallantop.com/bherant/mountbatten-plan-which-divided-india-into-two-parts/"},{"est":4.639846666,"newsid":22013,"title":"\u0935\u0938\u0940\u092e \u0905\u0915\u0930\u092e: \u091c\u093f\u0938\u0947 \u0916\u0947\u0932\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0938\u094d\u092a\u0947\u092f\u0930 \u092a\u093e\u0930\u094d\u091f\u094d\u0938 \u091a\u093e\u0939\u093f\u090f \u0925\u0947","url":"https://www.thelallantop.com/bherant/legendary-cricketer-from-pakistan-and-the-master-of-swing-bowling-wasim-akrams-birthday/"},{"est":4.6150762541,"newsid":257263,"title":"\u0915\u093e\u0902\u0917\u094d\u0930\u0947\u0938 \u0928\u0947 \u092e\u094b\u0926\u0940 \u0915\u0947 \u0916\u093f\u0932\u093e\u092b \u092e\u094b\u0926\u0940 \u0915\u094b \u0939\u0940 \u0932\u093e \u0916\u0921\u093c\u093e \u0915\u093f\u092f\u093e","url":"https://www.thelallantop.com/jhamajham/twitter-trends-modiexposesmodi-and-points-out-all-the-contradictions-claims-of-the-pm-before-and-after-he-assumed-office/"},{"est":4.5457057957,"newsid":257346,"title":"\u0932\u093e\u0916\u094b\u0902 \u0930\u0941\u092a\u090f \u092e\u0947\u0902 \u092c\u0947\u0921 \u0915\u0940 \u092c\u094d\u0932\u0948\u0915-\u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0902\u0917 \u0915\u0930 \u0930\u0939\u0947 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0905\u0938\u094d\u092a\u0924\u093e\u0932\u094b\u0902 \u092a\u0930 \u092d\u0921\u093c\u0915\u0947 \u0915\u0947\u091c\u0930\u0940\u0935\u093e\u0932, \u0926\u0940 \u091a\u0947\u0924\u093e\u0935\u0928\u0940","url":"https://www.thelallantop.com/news/delhi-cm-arvind-kejriwal-warns-private-hospitals-accuses-them-of-black-marketing-of-beds/"},{"est":4.5278353972,"newsid":104098,"title":"13 \u0938\u093e\u0932 \u092e\u0947\u0902 \u0938\u093f\u0930\u094d\u092b 32 \u092e\u0948\u091a \u0916\u0947\u0932\u0928\u0947 \u0935\u093e\u0932\u093e \u0916\u093f\u0932\u093e\u095c\u0940, \u091c\u094b \u0920\u0940\u0915 \u0938\u0947 \u0916\u0947\u0932\u0924\u093e \u0924\u094b \u0907\u0902\u0921\u093f\u092f\u093e \u0915\u094b \u0927\u094b\u0928\u0940 \u0928 \u092e\u093f\u0932\u0924\u093e","url":"https://www.thelallantop.com/bherant/dinesh-karthik-who-debuted-before-mahendra-singh-dhoni-but-failed-to-perform-on-big-stage/"},{"est":4.5201272455,"newsid":257582,"title":"\u0938\u093e\u0902\u0938 \u092e\u0947\u0902 \u0926\u093f\u0915\u094d\u0915\u0924 \u0915\u0947 \u092c\u093e\u0926 \u092d\u0940 \u092a\u0941\u0932\u093f\u0938\u0935\u093e\u0932\u0947 \u0915\u094b \u0907\u0932\u093e\u091c \u0928\u0939\u0940\u0902 \u092e\u093f\u0932\u093e, \u0905\u092b\u0938\u0930\u094b\u0902 \u0928\u0947 \u092a\u0930\u093f\u0935\u093e\u0930 \u092a\u0930 \u0939\u0940 \u0926\u094b\u0937 \u092e\u0922\u093c \u0926\u093f\u092f\u093e","url":"https://www.thelallantop.com/news/mumbai-police-constable-suspect-of-corona-virus-was-not-getting-bed-in-hospital-his-sister-pleads-on-viral-video/"},{"est":4.5201133313,"newsid":256902,"title":"\u0905\u092e\u0947\u0930\u093f\u0915\u0940 \u0928\u0938\u094d\u0932\u0935\u093e\u0926 \u092a\u0930 \u0906\u0935\u093e\u095b \u0909\u0920\u093e\u0928\u0947 \u0935\u093e\u0932\u0947 \u092c\u0949\u0932\u0940\u0935\u0941\u0921 \u0938\u0947\u0932\u0947\u092c\u094d\u0930\u093f\u091f\u0940\u095b \u0915\u094b \u0905\u092d\u092f \u0926\u0947\u0913\u0932 \u0928\u0947 \u0906\u0908\u0928\u093e \u0926\u093f\u0916\u093e \u0926\u093f\u092f\u093e","url":"https://www.thelallantop.com/news/abhay-deol-calls-out-indian-celebrities-talking-about-black-lives-matter-about-ignoring-issues-back-home/"},{"est":4.4907669155,"newsid":257436,"title":"\u0917\u0930\u094d\u092d\u0935\u0924\u0940 \u0928\u0947 13 \u0918\u0902\u091f\u0947 \u0924\u0915 \u0906\u0920 \u0905\u0938\u094d\u092a\u0924\u093e\u0932\u094b\u0902 \u0915\u0947 \u091a\u0915\u094d\u0915\u0930 \u0932\u0917\u093e\u090f, \u0915\u093f\u0938\u0940 \u0928\u0947 \u092d\u0930\u094d\u0924\u0940 \u0928\u0939\u0940\u0902 \u0915\u093f\u092f\u093e, \u092e\u094c\u0924 \u0939\u094b \u0917\u0908","url":"https://www.thelallantop.com/news/pregnant-woman-dies-after-allegedly-denied-treatment-by-8-government-and-private-hospitals-due-to-non-availability-of-beds/"},{"est":4.4845585847,"newsid":257634,"title":"\u0915\u093e\u0930\u0917\u093f\u0932 \u0935\u0949\u0930 \u0915\u0947 \u0926\u0930\u092e\u094d\u092f\u093e\u0928 \u091c\u092c \u092a\u093e\u0915\u093f\u0938\u094d\u0924\u093e\u0928 \u0915\u0940 \u0938\u0902\u0938\u0926 \u092e\u0947\u0902 \u0938\u091a\u093f\u0928 \u0915\u0947 \u0906\u0909\u091f \u0939\u094b\u0928\u0947 \u0915\u0940 \u0938\u0942\u091a\u0928\u093e \u0926\u0940 \u0917\u0908!","url":"https://www.thelallantop.com/kisse/story-of-india-and-pakistan-1999-world-cup-super-six-match-during-kargil-war/"}],
     user_interection = [{"newsid":257105,"tstamp":"2020-06-06 11:21:38.879000","title":"\u090f\u0915\u094d\u091f\u0930 \u0930\u094b\u0928\u093f\u0924 \u0930\u0949\u092f \u0928\u0947 \u092c\u0924\u093e\u092f\u093e, \u0932\u0949\u0915\u0921\u093e\u0909\u0928 \u092e\u0947\u0902 \u0918\u0930 \u0915\u0940 \u091a\u0940\u091c\u093c\u0947\u0902 \u092c\u0947\u091a\u0928\u0940 \u092a\u0921\u093c \u0930\u0939\u0940 \u0939\u0948\u0902","url":"https://www.thelallantop.com/news/actor-ronit-roy-says-not-earned-a-single-penny-since-january-and-selling-things-to-support-staff/"},{"newsid":249243,"tstamp":"2020-06-05 12:57:37.559000","title":"\u091c\u092c \u090b\u0937\u093f \u0915\u092a\u0942\u0930 \u0928\u0947 \u0905\u092a\u0928\u0940 \u092a\u0939\u0932\u0940 \u092b\u093f\u0932\u094d\u092e \u0915\u093e \u092c\u0926\u0932\u093e \u0930\u093e\u091c\u0947\u0936 \u0916\u0928\u094d\u0928\u093e \u0938\u0947 \u0906\u0916\u093f\u0930 \u092e\u0947\u0902 \u0932\u093f\u092f\u093e","url":"https://www.thelallantop.com/bherant/rajesh-khanna-was-supposed-to-star-in-rishi-kapoors-debut-film-bobby-later-rishi-casted-him-in-his-directorial-debut-aa-ab-laut-chalen/"}]
     print('1.....2...3')
     user_id='123456789'
     newsid='12345'
     count=10
     
     #user_interection = user_interection.to_json(orient='records')
     #recommnedation = recommnedation.to_json(orient='records')
     #recommnedation=[]
     #user_interection=[]
     #return jsonify(status='OK',recommnedation=recommnedation,user_interection=user_interection)    
     return render_template('ltop/ltop_history_rec_demo.html',user_id=user_id,newsid=newsid,count=count,user_interection=user_interection,recommnedation=recommnedation) 


@app.route("/recengine/ltop/demo/get_history_based_articles", methods=['GET', 'POST'])
def ltop_history_based_articles():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    import pickle  
    import gzip   
    import boto3
    from boto3.dynamodb.conditions import Key
    import pandas as pd
    import requests      
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()
    
    count=10
    newsid=None
    newsdata=None
    utm_medium = None
    user_id=None
    sp=None
        
    try:
        newsid = request.args.get('newsid')
    except Exception as exp:
        print('Exception in get newsid=>',exp)
        newsid=0
    
    all_flag = 0
    
    try:
        all_flag = int(request.args.get('all'))
    except Exception as exp:
        print('Exception in get allflag=>',exp)
        all_flag=0
        
    try:
        #Temp
        #user_id='01f6aaac-9c37-440d-8973-6ad4ca7203a4'
        user_id = request.args.get('user_id')
    except Exception as exp:
        print('Exception in get user-id =>',exp)
        user_id = None


    if user_id==None:        
        try:
            user_id=flask.request.cookies.get('sp')
            print('Getting sp....')  
        except Exception as exp:
            print('Exception....', exp)
            user_id=None
        
    try:
        count = int(request.args.get('count'))
    except Exception as exp:
        print('Exception in get count=>',exp)
        count=10    
        
    print('1......')
        
    newslist = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='newslist'))) 
    
    if count>15:
        count=15

    print('2......')
        
    print("source_newsid = ",newsid)
    print("count = ",count)
    print('newslist Length=>', len(newslist))

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-historybased-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    #news_id=1044367
    t1 = datetime.now()

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    
    print('3......user_id=', user_id)
    
    module_exists=True
    json_d={}
    user_interection={}
    if module_exists:
        TABLE_NAME = "itgd_cs_interaction_data_prod"
        #TABLE_NAME = "itgd_cs_user_interaction_prod"
        # Creating the DynamoDB Table Resource
        #dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1",aws_access_key_id='AKIAJ6K7K5FZMBPE4DYA', aws_secret_access_key='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')
        dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
        table = dynamodb.Table(TABLE_NAME)
        
        print('6......')

        #final_id='f92bc5d9-ef60-4b1a-8b12-245da1944cfa'
        response = table.query(
         IndexName="final_id-index",
          KeyConditionExpression=Key('final_id').eq(user_id)
        )
        #response['Items'] 
        
        print('U1...')
        df_r = pd.DataFrame(response['Items'],columns=['event','ist_tstamp','final_id','newsid','site_id'])   
        print('U2...')
        df_r=df_r[df_r['event']=='page_view']
        df_r=df_r[['ist_tstamp','final_id','newsid','site_id']]
        df_r=df_r.rename(columns={'ist_tstamp': 'tstamp','final_id': 'user_id','news': 'newsid'})
        print('U3...')
        df_r=df_r[df_r.site_id==4]
        user_viewed_news=df_r
        user_viewed_news=user_viewed_news[['tstamp','newsid']]
        n_list=user_viewed_news['newsid'].tolist()
        
        #========User Interaction Start 
       
        #============================
        lallantop_news_response=[]
        #n_list='217383,217590,217538,217693,217504,217736'
        #n_list=[217383, 217590, 217538, 217693, 217504, 217736]
        #n_list=n_list.split(',')
        n_list = list(map(int, n_list))
        
        if n_list!=None and len(n_list)>0:
            data=mdfb.get_lallantop_news_data_for_json(collection_name='ltop_recom',fieldname='unique_id',fieldvaluelist=n_list)
            #newslist=[]
            counter=1
            print('data =>', data)
            for row_data in data:
                print(row_data)
                dict_data={}
                dict_data['newsid']=row_data['unique_id']
                dict_data['title']=row_data['title']  
                #str(source, encoding='utf-8', errors = 'ignore')
                dict_data['title']=str(row_data['title'])
                dict_data['url']=row_data['url']
                dict_data['imageurl']=row_data['imageurl'] + '?size=133:75'
                dict_data['modified']=row_data['modified']
                lallantop_news_response.append(dict_data)
                counter+=1     
                
       
        #Temp
        #lallantop_get_news_url = "https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=1197339,258193,257031,1199095,256974,257031,256687,256827,256974,256459,1197400,1196735,257431,1196702,258226,257031,1197853,256974,258074,256663,257431,258074,1197880,258074,1196702,1196228,256786,256936,255552,257431,258074,253087,256265,258074,258226,256976,258193,1198351,256774,1196702,258361,256723,256786,1197383,1196702,1197383,1198470,256265,1198849,258074,255552,256723,1196702,1198013,1198472,1197339,1197400,257031,1196353,258300,256057,257031,255552,1196702,258074,256827,1196357,1196735,1196702,1198159,256499,256974,256459,1196228,256687,258226,257305,21755,258226,1198844,256057,256687,258300,21755,258226,258361,256723,258300,1196702,1196702,258193,256974,1196702,1196702,256687,257031,256723,256499,1196353,258171,257431,1196702,258300,258226,256723,256976,256936,256293,258361,253087,257087,258226,258300,256936,253087,1197222,256936,1198004,1197222,1196357,1196702,256774,258361,258193,256974,255552,256723,1198630,256663,1196702,257323,258226,258226,1198105,256293"
        print('U5...')
        #r = requests.get(lallantop_get_news_url)
        
        #r.text
        #lallantop_news_response = json.loads(r.text)
        #lallantop_df=pd.DataFrame(lallantop_news_response['data'])
        lallantop_df=pd.DataFrame(lallantop_news_response)
        print('U6...')
        
        data_flag=False
        if lallantop_df.shape[0]>0:
            print('U7...')
            lallantop_df=lallantop_df[['newsid','title','url']]
            user_viewed_news['newsid'] = user_viewed_news['newsid'].astype(int)
            user_interected_news = user_viewed_news.merge(lallantop_df, on='newsid')
            print('U8...')
            #user_interected_news = lallantop_df
            user_interected_news=user_interected_news.sort_values(by=['tstamp'], ascending=[False])
            print('U9...')
            #user_interected_news
            #user_interected_news=user_interected_news.drop_duplicates(subset=['tstamp','user_id','newsid'])
            #user_interected_news=user_interected_news.drop_duplicates(subset=['tstamp','newsid'])
            if all_flag==0:
                user_interected_news = user_interected_news.groupby(['newsid']).agg({'tstamp':'last','title':'last','url':'last'}).sort_values(by=['tstamp'], ascending=[False]).reset_index()
            user_interected_news = user_interected_news.head(30) 
            #user_interected_news.style.format({'url': make_clickable})
            data_flag=True
        else:
            print('No Data Avaialble')    
            
        print('U10...')
        
        #Temp    
        #newslist = ['115531','125307','126168','126327','131743','13677','138181','139833','149492','149688','151568','166715','17174','19338','20408','20760','21220','21315','214261','226172','230542','241401','246104','246543','247669','248494','249205','250380','250601','251516','251542','251555','251589','251606','251607','251612','251648','251650','251659','251676','251694','251699','251706','251725','251735','251738','251743','251751','251754','251756','251757','251763','251799','251821','251827','251834','251835','251852','251877','251881','251884','251884','251735','251888','251896','251897','251901','251905','251912','251922','251926','251933','251939','251945','251948','251951','251960','251974','251976','251979','251985','251997','252010','252012','252015','252023','252024','252035','252036','252045','252049','252057','252058','252059','252070','252090','252096','252103','252119','252123','252127','252137','252139','252147','252152','252153','252196','252201','252203','252211','252217','252218','252225','252238','252239','252245','252253','252257','252260','252290','252293','252305','252309','252312','252335','252354','252358','252362','252380','252384','252396','252408','252422','252434','252436','252438','252438','252509','252461','252469','252474','252487','252489','252492','252494','252506','252512','252526','252528','252548','252552','252553','252559','252592','252597','252601','252605','252611','252623','252625','252635','252641','252643','252648','252669','252684','252691','252725','252729','252738','252742','252744','252750','252779','252788','252789','252791','252792','252803','252807','252807','252810','252841','252846','252858','252860','252861','252863','252872','252880','252882','252891','252905','252912','252927','252928','252932','252942','252948','252952','252958','252959','252962','252973','252983','252993','253000','253006','253018','253020','253022','253030','253031','253037','253038','253047','253050','253059','253062','253064','253070','253072','253087','253091','253114','253117','253134','253145','253147','253150','253155','253160','253161','253164','253165','253168','253184','253192','253195','253197','253209','253233','253247','253255','253269','253274','253274','253269','253275','253280','253290','253293','253304','253309','253312','253325','253327','253331','253355','253356','253359','253370','253371','253383','253387','253411','253422','253427','253429','253430','253434','253446','253452','253464','253469','253476','253478','253479','253480','253498','253534','253544','253629','253630','253639','253644','253645','253654','253659','253660','253671','253693','253696','253699','253703','253705','253710','253718','253721','253723','253734','253757','253770','253776','253779','253781','253807','253269','253817','253818','253821','253824','253826','253828','253838','253841','253849','253861','253867','253873','253887','253898','253903','253923','253926','253930','253936','253945','253949','253966','253973','253975','253976','253985','253987','253997','254003','254015','254032','254048','254050','254051','254055','254067','254073','254076','254082','254092','254107','254113','254114','254123','254133','254141','254146','254148','254161','254166','254168','254185','254196','254198','254217','254218','254231','254238','254248','254256','254277','254279','254282','254284','254301','254302','254324','254327','254336','254338','254351','254353','254356','254358','254370','254374','254382','254389','254392','254394','254395','254400','254405','254445','254448','254451','254454','254460','254485','254491','254502','254505','254506','254548','254558','254563','254568','254570','254575','254585','254588','254590','254604','254610','254614','254615','254624','254640','254645','254653','254663','254665','254693','254694','254701','254705','254706','254707','254715','254722','254725','254732','254736','254752','254753','254786','254809','254817','254823','254828','254837','254841','254845','254854','254864','254868','254877','254880','254881','254889','254894','254909','254923','254931','254932','254936','254946','254948','254949','254959','254961','254980','254983','254993','255016','255032','255042','255059','255064','255066','255068','255073','255075','255077','255087','255088','255095','255104','255106','255109','255111','255139','255149','255156','255167','255169','255175','255192','255203','255205','255206','255214','255224','255228','255229','255266','255268','255271','255272','255273','255282','255297','255313','255314','255315','255342','255352','255353','255359','255381','255385','255390','255395','255398','255414','255422','255423','255456','255458','255478','255487','255499','255502','255522','255538','255547','255553','255563','255569','255581','255583','255588','255605','41494','47538','47958','53849','74318','74416','75567','75654','75953','76132','77268','80500','95975']
        if data_flag:
            print('U11...')
            newslist=list(map(int, newslist))   
            newslist_set=set(newslist)  
            user_viewed_news=df_r[['tstamp','newsid']]
            n_list=user_viewed_news['newsid'].tolist()
            ids = ','.join(str(e) for e in n_list)
            interacted_newsid=set(list(map(int, ids.split(','))))
            non_interacted_news_list = newslist_set- interacted_newsid -set([newsid])
            print('U12...')
            #non_interacted_news_list=newslist
            #user_id='f68f7418-086b-48c3-8cf7-f048d6ed0a5d'
            #count=10
            data={}
            data['user_id']=user_id
            data['count']=count
            data['newslist']=list(non_interacted_news_list)    
            
            print('U12.1...',user_id,count)
            
            url='https://142dro4haa.execute-api.ap-southeast-1.amazonaws.com/prod/lallantop-recengine'
            headers = {"Content-Type": "application/json","Accept": "application/json"}
            response = requests.post(url, json=data, headers=headers)
            responseBody=response.text     
            res=json.loads(responseBody)
            print('U12.2...res= ',res)
            json_d={}
            print('U13...')
            
            if res!=None:
                print('U14...')
                news_list_with_estimator = pd.DataFrame.from_records(res)
                n_list=[]
                for response_data in res:
                    n_list.append(response_data['newsid'])
                    
                uniquelist = []
                for x in n_list:
                    if x not in uniquelist:
                        uniquelist.append(x) 
                
                #uniquelist=[217449,217387,214087,217749]
                uniquelist = list(map(int, uniquelist))
                
                lallantop_news_response=[]
                if uniquelist!=None and len(uniquelist)>0:
                    data=mdfb.get_lallantop_news_data_for_json(collection_name='ltop_recom',fieldname='unique_id',fieldvaluelist=uniquelist)
                    #newslist=[]
                    counter=1
                    print('data =>', data)
                    
                    for row_data in data:
                        print(row_data)
                        dict_data={}
                        dict_data['newsid']=row_data['unique_id']
                        dict_data['title']=row_data['title']  
                        #str(source, encoding='utf-8', errors = 'ignore')
                        dict_data['title']=str(row_data['title'])
                        dict_data['url']=row_data['url']
                        dict_data['imageurl']=row_data['imageurl'] + '?size=133:75'
                        dict_data['modified']=row_data['modified']
                        lallantop_news_response.append(dict_data)
                        counter+=1     
                
                lallantop_df=pd.DataFrame(lallantop_news_response)
                print('11.....')
                #lallantop_df = lallantop_df.drop_duplicates(subset=['newsid'])
                lallantop_df = lallantop_df.groupby(['newsid']).agg({'title':'last','url':'last','modified':'last'}).reset_index()
                print('12.....')
                lallantop_df=lallantop_df[['newsid','title','url','modified']]
                print('13.....')
                final_recommended_news=news_list_with_estimator.merge(lallantop_df, on='newsid')    
                print('14.....')
                final_recommended_news=final_recommended_news.sort_values(['est'], ascending=[False])
                print('15.....', type(final_recommended_news))
                json_d = final_recommended_news.to_json(orient='records')
                print('16.....', json_d)
                json_d = json.loads(json_d)
                print('17.....', json_d)
        else:
            print('No Response')    
            json_d={}
    else:
        print('Module Does Not Exists')
        json_d=[]
        user_interection=[]
    user_interection={}
    try:
        print('18.....')
        user_interection = user_interected_news.to_json(orient='records')
        #print('19.....', type(user_interected_news))
        user_interection = json.loads(user_interection)
        print('20.....')
    except:
        user_interection=[]
        print('21.....Exception user_interection')
            
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
 
    return render_template('ltop/ltop_history_rec_demo.html',user_id=user_id,newsid=newsid,count=count,user_interection=user_interection,recommnedation=json_d)



@app.route("/recengine/ltop/demo/get_history_based_articles_all", methods=['GET', 'POST'])
def ltop_history_based_articles_all():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    import pickle  
    import gzip   
    import boto3
    from boto3.dynamodb.conditions import Key
    import pandas as pd
    import requests      
    
    u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()
    
    count=10
    newsid=None
    newsdata=None
    utm_medium = None
    user_id=None
    sp=None
        
    try:
        newsid = request.args.get('newsid')
    except Exception as exp:
        print('Exception in get newsid=>',exp)
        newsid=0
    
    all_flag = 0
    
    try:
        all_flag = int(request.args.get('all'))
    except Exception as exp:
        print('Exception in get allflag=>',exp)
        all_flag=0
        
    try:
        #Temp
        #user_id='01f6aaac-9c37-440d-8973-6ad4ca7203a4'
        user_id = request.args.get('user_id')
    except Exception as exp:
        print('Exception in get user-id =>',exp)
        user_id = None


    if user_id==None:        
        try:
            user_id=flask.request.cookies.get('sp')
            print('Getting sp....')  
        except Exception as exp:
            print('Exception....', exp)
            user_id=None
        
    try:
        count = int(request.args.get('count'))
    except Exception as exp:
        print('Exception in get count=>',exp)
        count=10    
        
    print('1......')
        
    newslist = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='newslist'))) 
    
    if count>15:
        count=15

    print('2......')
        
    print("source_newsid = ",newsid)
    print("count = ",count)
    print('newslist =>', newslist)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-historybased-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    #news_id=1044367
    t1 = datetime.now()

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    
    print('3......')
    
    #Temporary
    #user_id='01f6aaac-9c37-440d-8973-6ad4ca7203a4'
    
    TABLE_NAME = "itgd_cs_model_info_prod"
    #dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1",aws_access_key_id='AKIAJ6K7K5FZMBPE4DYA', aws_secret_access_key='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')
    dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
    table = dynamodb.Table(TABLE_NAME)
    
    print('4......')
    
    final_id=user_id
    module_exists=False
    
    try:
        response = table.get_item(
            Key={
                'final_id': final_id,
                'site_id': 4
            }
        )
    except:
        print('Exception')
 
    try:
        res=response['Item']    
        if res!=None:
            module_exists=True
        print('User exist in Model')
    except:
        print('User does not exist in Model')    
 
    print('5......')
    
    json_d={}
    user_interection={}
    if module_exists:
        TABLE_NAME = "itgd_cs_interaction_data_prod"
        #TABLE_NAME = "itgd_cs_user_interaction_prod"
        # Creating the DynamoDB Table Resource
        #dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1",aws_access_key_id='AKIAJ6K7K5FZMBPE4DYA', aws_secret_access_key='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')
        dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
        table = dynamodb.Table(TABLE_NAME)
        
        print('6......')


        #Temparory
        final_id=user_id
        #final_id='f92bc5d9-ef60-4b1a-8b12-245da1944cfa'
        response = table.query(
         IndexName="final_id-index",
          KeyConditionExpression=Key('final_id').eq(final_id)
        )
        #response['Items'] 
        
        print('7......')

        print('U1...')
        df_r = pd.DataFrame(response['Items'],columns=['event','ist_tstamp','final_id','newsid','site_id'])   
        print('U2...')
        df_r=df_r[df_r['event']=='page_view']
        df_r=df_r[['ist_tstamp','final_id','newsid','site_id']]
        df_r=df_r.rename(columns={'ist_tstamp': 'tstamp','final_id': 'user_id','news': 'newsid'})
        print('U3...')
        #df_r=df_r[df_r.site_id==4]
        user_viewed_news=df_r
        user_viewed_news=user_viewed_news[['tstamp','newsid','site_id']]
        n_list=user_viewed_news['newsid'].tolist()
        
        #========User Interaction Start 
       
        #============================
        lallantop_news_response=[]
        #n_list='217383,217590,217538,217693,217504,217736'
        #n_list=[217383, 217590, 217538, 217693, 217504, 217736]
        #n_list=n_list.split(',')
        n_list = list(map(int, n_list))
        
        if n_list!=None and len(n_list)>0:
            user_viewed_news=df_r
            lallantop_news_response=[]
            
            d={}
            d[1]='AajTak'
            d[2]='IndiaToday'
            d[3]='BusinessToday'
            d[4]='LallanTop'
            
            
            for ind in user_viewed_news.index: 
                #print(df_r['tstamp'][ind], df_r['newsid'][ind], df_r['site_id'][ind]) 
                #print(ind) 
                dict_data={}
                dict_data['newsid']=user_viewed_news['newsid'][ind]
                dict_data['title']=user_viewed_news['site_id'][ind]
                dict_data['url']=d[user_viewed_news['site_id'][ind]]
                dict_data['tstamp']=user_viewed_news['tstamp'][ind]
                lallantop_news_response.append(dict_data)
                    
            print('lallantop_news_response =>', lallantop_news_response)      
                
        #============================
        
        #Temp
        #lallantop_get_news_url = "https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=1197339,258193,257031,1199095,256974,257031,256687,256827,256974,256459,1197400,1196735,257431,1196702,258226,257031,1197853,256974,258074,256663,257431,258074,1197880,258074,1196702,1196228,256786,256936,255552,257431,258074,253087,256265,258074,258226,256976,258193,1198351,256774,1196702,258361,256723,256786,1197383,1196702,1197383,1198470,256265,1198849,258074,255552,256723,1196702,1198013,1198472,1197339,1197400,257031,1196353,258300,256057,257031,255552,1196702,258074,256827,1196357,1196735,1196702,1198159,256499,256974,256459,1196228,256687,258226,257305,21755,258226,1198844,256057,256687,258300,21755,258226,258361,256723,258300,1196702,1196702,258193,256974,1196702,1196702,256687,257031,256723,256499,1196353,258171,257431,1196702,258300,258226,256723,256976,256936,256293,258361,253087,257087,258226,258300,256936,253087,1197222,256936,1198004,1197222,1196357,1196702,256774,258361,258193,256974,255552,256723,1198630,256663,1196702,257323,258226,258226,1198105,256293"
        print('U5...')
        #r = requests.get(lallantop_get_news_url)
        
        #r.text
        #lallantop_news_response = json.loads(r.text)
        #lallantop_df=pd.DataFrame(lallantop_news_response['data'])
        lallantop_df=pd.DataFrame(lallantop_news_response)
        print('U6...')
        
        if lallantop_df.shape[0]>0:
            print('U7...')
            lallantop_df=lallantop_df[['newsid','title','url','tstamp']]
            #user_viewed_news['newsid'] = user_viewed_news['newsid'].astype(int)
            #user_interected_news = user_viewed_news.merge(lallantop_df, on='newsid')
            #print('U8...')
            #user_interected_news = lallantop_df
            user_interected_news=lallantop_df.sort_values(by=['tstamp'], ascending=[False])
            print('U9...')
            #user_interected_news
            #user_interected_news=user_interected_news.drop_duplicates(subset=['tstamp','user_id','newsid'])
            #user_interected_news=user_interected_news.drop_duplicates(subset=['tstamp','newsid'])
            #if all_flag==0:
                #user_interected_news = user_interected_news.groupby(['newsid']).agg({'tstamp':'last','title':'last','url':'last'}).sort_values(by=['tstamp'], ascending=[False]).reset_index()
            user_interected_news = user_interected_news.head(100) 
            #user_interected_news.style.format({'url': make_clickable})
        else:
            print('No Data Avaialble')    
            
        print('U10...')
        
        #Temp    
        #newslist = ['115531','125307','126168','126327','131743','13677','138181','139833','149492','149688','151568','166715','17174','19338','20408','20760','21220','21315','214261','226172','230542','241401','246104','246543','247669','248494','249205','250380','250601','251516','251542','251555','251589','251606','251607','251612','251648','251650','251659','251676','251694','251699','251706','251725','251735','251738','251743','251751','251754','251756','251757','251763','251799','251821','251827','251834','251835','251852','251877','251881','251884','251884','251735','251888','251896','251897','251901','251905','251912','251922','251926','251933','251939','251945','251948','251951','251960','251974','251976','251979','251985','251997','252010','252012','252015','252023','252024','252035','252036','252045','252049','252057','252058','252059','252070','252090','252096','252103','252119','252123','252127','252137','252139','252147','252152','252153','252196','252201','252203','252211','252217','252218','252225','252238','252239','252245','252253','252257','252260','252290','252293','252305','252309','252312','252335','252354','252358','252362','252380','252384','252396','252408','252422','252434','252436','252438','252438','252509','252461','252469','252474','252487','252489','252492','252494','252506','252512','252526','252528','252548','252552','252553','252559','252592','252597','252601','252605','252611','252623','252625','252635','252641','252643','252648','252669','252684','252691','252725','252729','252738','252742','252744','252750','252779','252788','252789','252791','252792','252803','252807','252807','252810','252841','252846','252858','252860','252861','252863','252872','252880','252882','252891','252905','252912','252927','252928','252932','252942','252948','252952','252958','252959','252962','252973','252983','252993','253000','253006','253018','253020','253022','253030','253031','253037','253038','253047','253050','253059','253062','253064','253070','253072','253087','253091','253114','253117','253134','253145','253147','253150','253155','253160','253161','253164','253165','253168','253184','253192','253195','253197','253209','253233','253247','253255','253269','253274','253274','253269','253275','253280','253290','253293','253304','253309','253312','253325','253327','253331','253355','253356','253359','253370','253371','253383','253387','253411','253422','253427','253429','253430','253434','253446','253452','253464','253469','253476','253478','253479','253480','253498','253534','253544','253629','253630','253639','253644','253645','253654','253659','253660','253671','253693','253696','253699','253703','253705','253710','253718','253721','253723','253734','253757','253770','253776','253779','253781','253807','253269','253817','253818','253821','253824','253826','253828','253838','253841','253849','253861','253867','253873','253887','253898','253903','253923','253926','253930','253936','253945','253949','253966','253973','253975','253976','253985','253987','253997','254003','254015','254032','254048','254050','254051','254055','254067','254073','254076','254082','254092','254107','254113','254114','254123','254133','254141','254146','254148','254161','254166','254168','254185','254196','254198','254217','254218','254231','254238','254248','254256','254277','254279','254282','254284','254301','254302','254324','254327','254336','254338','254351','254353','254356','254358','254370','254374','254382','254389','254392','254394','254395','254400','254405','254445','254448','254451','254454','254460','254485','254491','254502','254505','254506','254548','254558','254563','254568','254570','254575','254585','254588','254590','254604','254610','254614','254615','254624','254640','254645','254653','254663','254665','254693','254694','254701','254705','254706','254707','254715','254722','254725','254732','254736','254752','254753','254786','254809','254817','254823','254828','254837','254841','254845','254854','254864','254868','254877','254880','254881','254889','254894','254909','254923','254931','254932','254936','254946','254948','254949','254959','254961','254980','254983','254993','255016','255032','255042','255059','255064','255066','255068','255073','255075','255077','255087','255088','255095','255104','255106','255109','255111','255139','255149','255156','255167','255169','255175','255192','255203','255205','255206','255214','255224','255228','255229','255266','255268','255271','255272','255273','255282','255297','255313','255314','255315','255342','255352','255353','255359','255381','255385','255390','255395','255398','255414','255422','255423','255456','255458','255478','255487','255499','255502','255522','255538','255547','255553','255563','255569','255581','255583','255588','255605','41494','47538','47958','53849','74318','74416','75567','75654','75953','76132','77268','80500','95975']
        newslist=list(map(int, newslist))   
        newslist_set=set(newslist)  
        user_viewed_news=df_r[['tstamp','newsid']]
        n_list=user_viewed_news['newsid'].tolist()
        ids = ','.join(str(e) for e in n_list)
        interacted_newsid=set(list(map(int, ids.split(','))))
        non_interacted_news_list = newslist_set- interacted_newsid -set([newsid])
        #non_interacted_news_list=newslist
        #user_id='f68f7418-086b-48c3-8cf7-f048d6ed0a5d'
        #count=10
        print('U11...')
        data={}
        data['user_id']=user_id
        #data['newsid']=newsid
        data['count']=count
        data['newslist']=list(non_interacted_news_list)    
        
        url='https://142dro4haa.execute-api.ap-southeast-1.amazonaws.com/prod/lallantop-recengine'
        headers = {"Content-Type": "application/json","Accept": "application/json"}
        response = requests.post(url, json=data, headers=headers)
        responseBody=response.text     
        
        res=json.loads(responseBody)
        json_d={}
        print('U12...')
        if res!=None:
            news_list_with_estimator = pd.DataFrame.from_records(res)
            
            n_list=[]
            for response_data in res:
                n_list.append(response_data['newsid'])
                
            uniquelist = []
            for x in n_list:
                if x not in uniquelist:
                    uniquelist.append(x) 
            
            #uniquelist=[217449,217387,214087,217749]
            
            #=====================
            uniquelist = list(map(int, uniquelist))
            
            lallantop_news_response=[]
            if uniquelist!=None and len(uniquelist)>0:
                data=mdfb.get_lallantop_news_data_for_json(collection_name='ltop_recom',fieldname='unique_id',fieldvaluelist=uniquelist)
                #newslist=[]
                counter=1
                print('data =>', data)
                
                for row_data in data:
                    print(row_data)
                    dict_data={}
                    dict_data['newsid']=row_data['unique_id']
                    dict_data['title']=row_data['title']  
                    #str(source, encoding='utf-8', errors = 'ignore')
                    dict_data['title']=str(row_data['title'])
                    dict_data['url']=row_data['url']
                    dict_data['imageurl']=row_data['imageurl'] + '?size=133:75'
                    dict_data['modified']=row_data['modified']
                    lallantop_news_response.append(dict_data)
                    counter+=1     
            
            lallantop_df=pd.DataFrame(lallantop_news_response)
            #=======================
            
            '''            
            data=mdfb.get_lallantop_news_data_for_json(collection_name='ltop_recom',fieldname='unique_id',fieldvaluelist=uniquelist)
                
            ids = ','.join(str(e) for e in n_list)
            lallantop_get_news_url="https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=%s"%(ids)
            
            #lallantop_get_news_url="https://recengine.intoday.in/recengine/ltop/getcontentdetail?ids=254640,254722,255563,255499,255167"
            
            #import requests
            #import pandas as pd
            #print(lallantop_get_news_url)
            print('8.....')
            r = requests.get(lallantop_get_news_url)
            print('9.....')
            lallantop_news_response = json.loads(r.text)
            print('10.....')
            lallantop_df=pd.DataFrame(lallantop_news_response['data'])
            '''
            print('11.....')
            #lallantop_df = lallantop_df.drop_duplicates(subset=['newsid'])
            lallantop_df = lallantop_df.groupby(['newsid']).agg({'title':'last','url':'last','modified':'last'}).reset_index()
            print('12.....')
            lallantop_df=lallantop_df[['newsid','title','url','modified']]
            print('13.....')
            final_recommended_news=news_list_with_estimator.merge(lallantop_df, on='newsid')    
            print('14.....')
            final_recommended_news=final_recommended_news.sort_values(['est'], ascending=[False])
            print('15.....', type(final_recommended_news))
            json_d = final_recommended_news.to_json(orient='records')
            print('16.....', json_d)
            json_d = json.loads(json_d)
            print('17.....', json_d)
        else:
            print('No Response')    
            json_d={}
    else:
        print('Module Does Not Exists')
        json_d=[]
        user_interection=[]
    user_interection={}
    try:
        print('18.....')
        user_interection = user_interected_news.to_json(orient='records')
        print('19.....', type(user_interected_news))
        user_interection = json.loads(user_interection)
        print('20.....')
    except:
        user_interection=[]
        print('21.....Exception user_interection')
        
            
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    #return jsonify(status=response,message=message,source_videoid=video_id,playlist=data_final)
    #return jsonify(status='OK',recommnedation=json_d,user_interection=user_interection)    
    return render_template('ltop/ltop_history_rec_demo.html',user_id=user_id,newsid=newsid,count=count,user_interection=user_interection,recommnedation=json_d)


@app.route("/recengine/at/gettest", methods=['GET', 'POST'])
def aajtak_getarticles_test():
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    import time
    
    u=aajtak_utility.utility()
    mdfb=aajtak_mongo_db_file_model.mongo_db_file_model()
    
    data_flag = True
    status='OK'
    data=None
    try:
        news_id = request.args.get('newsid')
    except Exception as exp:
        news_id='0'
        data_flag = False
        
    try:
        data1=mdfb.get_aajtak_news_data_for_json(collection_name='at_recom',fieldname='id',fieldvaluelist=[news_id])
        data=data1[0]['url']
    except:
        status='NOK'
        data=None
    time.sleep(30)    
    return jsonify(status=status,data=data)

@app.route("/recengine/at/getarticles_temp", methods=['GET', 'POST'])
def aajtak_getarticles_temp():
    import aajtak.utility as aajtak_utility 
    import aajtak.mongo_db_file_model as aajtak_mongo_db_file_model
    
    u=aajtak_utility.utility()
    mdfb=aajtak_mongo_db_file_model.mongo_db_file_model()

    story_count=10
    #news_id='1046690'
    newsdata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    
    try:
        news_id = request.args.get('newsid')
        newsdata=mdfb.get_aajtak_news_text_from_mongodb(collection_name='at_recom',fieldname='id',fieldvalue=news_id)
        utm_medium = request.args.get('utm_medium')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        news_id='0'
        newsdata=None
        utm_medium='Unknown'
 
    try:
        story_count=10
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        story_count=5

    if story_count>15:
        story_count=15
        
    print("source_newsid = ",news_id)
    print("story_count = ",story_count)
    #news_corpus=u.get_newsid_corpus(news_id)

    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|at-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(news_id,'None')
    #news_id=1044367
    t1 = datetime.now()
    #story_count=8
    #tokenizer = RegexpTokenizer(r'\w+')

    #hi_stop = get_stop_words('hi')
    hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
    #print(dictionary)
    
    model_path = '/home/itgd/model/'
    
    print('Pick file from Local Path')
    mapping_id_newsid_dictionary=pickle.loads(gzip.decompress(np.load(model_path + 'mapping_id_newsid_dictionary.npy')))
    lda_index=pickle.loads(gzip.decompress(np.load(model_path + 'lda_index.npy')))
    
    #mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='id_newsid')))
    #lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='at_file_system',filename='portal_corpus'))) 
    
    #print(1044367')
    newslist_local=[]
    data_final = []
    
    response="SUCCESS"
    message="OK"
    data=""
    if newsdata==None or len(newsdata)<=30:
        print("No Source Data........")
        log+= '|FAIL'
        log+= '|No Process,text=%s'%(newsdata)
        data=mdfb.get_latest_news_records(collection_name='at_recom', field_name='modified', LIMIT=story_count)
    else:
        print('Similar data.....')
        log+= '|SUCCESS'
        log+= '|Result'
        clean_text = u.clean_doc_hindi(newsdata)
        cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
        tokens = cleaned_tokens_n.split(' ')
        cleaned_tokens = [word for word in tokens if len(word) > 4]
        stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
        stemmed_tokens = [u.generate_stem_words(i) for i in stopped_tokens]
        news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
        similar_news = lda_index[lda_at[news_corpus]]
        similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
       
        for x in similar_news[:(story_count + 1)]:
            newslist_local.append(mapping_id_newsid_dictionary[x[0]])
            log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
        newslist_local=list(filter(lambda x:x!=news_id,newslist_local))[:story_count]    
        
        #tempprary - newslist -need to be commented
        #newslist_local=['1044388','1044382','1044376']
        data=mdfb.get_aajtak_news_data_for_json(collection_name='at_recom',fieldname='id',fieldvaluelist=newslist_local)

    total_dict_data={}
    counter=1
    for row_data in data:
        dict_data={}
        dict_data['newsid']=row_data['id']
        dict_data['title']=row_data['title']
        dict_data['uri']=row_data['url']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,counter)
        dict_data['mobile_image']=row_data['media']['kicker_image2']
        total_dict_data[row_data['id']]=dict_data
        newslist_local.append(row_data['id'])
        counter+=1
    
    data_final=[]
    for newsid in newslist_local:
        if len(data_final)==story_count:
            break
        try:
            data_final.append(total_dict_data[newsid])
        except Exception as exp:   
            print('Exception =>',exp)

    '''    
    data_final=[]    
    for row_data in data:
        temp_data={}
        temp_data['newsid']=row_data['id']
        temp_data['title']=row_data['title']
        temp_data['uri']=row_data['url']
        temp_data['mobile_image']=row_data['media']['kicker_image2']
        data_final.append(temp_data)
     '''   
    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_at_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status=response,message=message,source_newsid=news_id,data=data_final)

@app.route("/recengine/ltop/getarticles_uid", methods=['GET', 'POST'])
def ltop_getarticles_uid():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    
    ltop_u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    import boto3
    from boto3.dynamodb.conditions import Key
    import requests     

    news_count=5
    #newsid='212436'
    newsid=0
    newsdata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    source_newsid = 0
    uid=None
    min_story_count=3
    t_psl=False
    try:
        uid = request.args.get('uid',None)
        print('uid =>', uid)
    except Exception as exp:
        print('Exception in get uid=>',exp)
        uid = None

    try:
        newsid = request.args.get('newsid')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0
 
    try:
        utm_medium = request.args.get('utm_medium','Unknown')
        print('utm_medium =>', utm_medium)
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    try:
        news_count=int(request.args.get('no',5))
        print('news_count =>', news_count)
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        news_count=5        

    if news_count>20:
        news_count=20
        
    print("source_newsid = ",newsid)
    print("news_count = ",news_count)
    #news_corpus=u.get_newsid_corpus(news_id)
    source_newsid=newsid
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    log+='|request_uid=%s'%(uid)
    #news_id=1044367
    
    #p1_ttl = 5 * 60
    p2_ttl = 5 * 60
    p3_ttl = 5 * 60
    newslist_ttl = 5 * 60
    
    t1 = datetime.now()
    uid_exists_in_model=False
    dynamodb=None
    interaction_item = None
    newslist_local=[]


    data_final = []
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    
    if uid!=None:
        #Temporary
        #uid='065643db-6669-4346-95bd-016e3fc03ceb'
        #uid = '00c6aa23-c794-40e4-86dd-b8cc6d305625'
        site_id=str(4)
           
        #if uid_exists_in_model:
        u_interaction_list=[]
        print('UID Found')
        try:
            print('U 1 ...')
           #Check recommended personalized data available in in Redis Cache....
            key = site_id + '-p3-' + uid
            p3_processflag = False
            #Get this user intrection list from Redis Cache
            try:
                print('U 3 ...')
                newslist_local = json.loads(ltop_rh.get_data_from_cache(key=key))
                print('newslist_local => ', len(newslist_local))
                
                #print(list[:(news_count + 1)])                    
                if newslist_local==None or newslist_local==[]:
                    p3_processflag=False
                    newslist_local=[]
                else:
                    p3_processflag=True
                    uid_exists_in_model=True
            except:
                print('U 4 ...')
                newslist_local=[]
                p3_processflag=False
                
            print('2....Final recommended data =>', newslist_local, '  p3_processflag=>', p3_processflag)                
            
            if p3_processflag==False:
                print('U 5 ...')
                #key for get user interaction data
                key = site_id + '-p2-' + uid
                try:
                    u_interaction_list = json.loads(ltop_rh.get_data_from_cache(key=key))

                    if u_interaction_list==None or u_interaction_list==[]:
                        u_interaction_list=[]
                    else:
                        if len(u_interaction_list)>=min_story_count:
                            uid_exists_in_model=True    
                except:
                    u_interaction_list=[]

                print('3....u_interaction_list length=>', len(u_interaction_list))                
                if u_interaction_list==[] or u_interaction_list==None:
                    try:
                        print('U 5.1...')
                        dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
                        response = dynamodb.Table("itgd_cs_interaction_data_prod").query(IndexName="final_id-index",KeyConditionExpression=Key('final_id').eq(uid))
                        print('U 5.2...')
                        interaction_item = response['Items'] 
                        print('U 5.3...')
                        #interaction_list=list(map(int, list(map(lambda x:x['newsid'], interaction_item))))
                        interaction_list=list(map(int, list(map(lambda x: x['newsid'] , list(filter(lambda x: x['site_id']==4, interaction_item))))))
                        print('U 5.4...', interaction_list)
                        u_interaction_list = []
                        
                        #if len(interaction_list)<min_story_count:
                            
                        u = [x for x in interaction_list if x not in u_interaction_list and u_interaction_list.append(x)]
                        print('3....u_interaction_list at setting time in cache=>', len(u_interaction_list))
                        key = site_id + '-p2-' + uid
                        set_flag = ltop_rh.set_data_in_cache(key=key, data=str(u_interaction_list), ttl=p2_ttl)    
                        print('user intrected news set in Redis Cache =>', set_flag)
                        if len(u_interaction_list)>=min_story_count:
                            uid_exists_in_model=True    
                    except Exception as e:
                        print('U 6.2 ...',e)
                        print('Exception to set user list for key', key)
                        uid_exists_in_model=False
                
                if uid_exists_in_model:
                    u_interaction_list = set(u_interaction_list)
                    print('U 7 ...')
                    
                    key = site_id + '-newslist'
                    #Getting news list from Cache
                    newslist=[]
                    try:
                        print('U 8 ...')
                        newslist = json.loads(ltop_rh.get_data_from_cache(key=key))
                        if newslist==None or newslist==[]:
                            newslist=[]
                    except:
                        print('U 9 ...')
                        newslist=[]
                        
                    if newslist==None or newslist==[]:
                        print('U 10 ...')
                        newslist = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='newslist'))) 
                        set_flag = ltop_rh.set_data_in_cache(key=key, data=str(newslist), ttl=newslist_ttl)    
                        print('newslist set in Redis Cache =>', set_flag)                    
                        
                    print('U 11 ...')
                    newslist=set(list(map(int, newslist))) 
                    non_interacted_news_list = newslist - u_interaction_list - set([newsid])
                    print('U 12 ...non_interacted_news_list=',len(non_interacted_news_list))
                    data={}
                    fix_count=20
                    data['user_id']=uid
                    data['count']=fix_count
                    data['newslist']=list(non_interacted_news_list)  
                    
                    #print('data=>',data)
                    #data=json.dumps(data)
                    responseBody=None
                    try:
                        print('U 13 ...')
                        url='https://142dro4haa.execute-api.ap-southeast-1.amazonaws.com/prod/lallantop-recengine'
                        print('U 14 ...')
                        headers = {"Content-Type": "application/json","Accept": "application/json"}
                        print('U 15 ...',headers)
                        response = requests.post(url, json=data, headers=headers)
                        print('U 16 ...',response)
                        responseBody=response.text
                        print('U 17 ...',responseBody)
                    except:
                        print('U 18 ...')
                        print('Exception =>')
                    res = json.loads(responseBody) 
                    print('U 19 ...',res)
                    newslist_local = list(map(lambda x:x['newsid'], res))
                    print('U 20 ...',newslist_local)
                    key = site_id + '-p3-' + uid
                    set_flag = ltop_rh.set_data_in_cache(key=key, data=str(newslist_local), ttl=p3_ttl)    
                    print('Recommennded news set in Redis Cache =>', set_flag)  
                    uid_exists_in_model=True 
                    t_psl=True
                    
            else:
                print('U 21 ...')
                newslist_local = list(set(newslist_local) - set([newsid]))
                t_psl=True
        except:
            print('U 22 ...')
            uid_exists_in_model=False
            interaction_item = None
            t_psl=False
            print('No uid exist in User interaction')
        #else:
        #    uid_exists_in_model=False
            
    print('uid_exists_in_model =>',uid_exists_in_model)    
    if uid_exists_in_model==False:
        t_psl=False
        try:
            newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
        except:
            newsdata=None
        #hi_stop = get_stop_words('hi')
        hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
        #print(dictionary)
        mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid')))
        lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus'))) 
        if newsdata==None or len(newsdata)<=30:
            print("No Source Data........")
            log+= '|FAIL'
            log+= '|No Process,text=%s'%(newsdata)
            #data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
            data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
            #video_count=30
            latest_Flag=True
        else:
            print('Similar data.....')
            log+= '|SUCCESS'
            log+= '|Result'
            
            print('E1....')
            print('Length = >', len(newsdata))
            clean_text = ltop_u.clean_doc_hindi(newsdata)
            print('E2....', clean_text)
            #print('1....')
            cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
            #print('2....')
            tokens = cleaned_tokens_n.split(' ')
            #print('3....')
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            #print('4....')
            stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
            #print('5....')
            stemmed_tokens = [ltop_u.generate_stem_words(i) for i in stopped_tokens]
            #print('6....')
            news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
            #print('7....')
            similar_news = lda_index[lda_at[news_corpus]]
            #print('8....')
            similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
            #print('9....')
           
            for x in similar_news[:(news_count + 1)]:
                #print('9.1....')    
                newslist_local.append(mapping_id_newsid_dictionary[x[0]])
                log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            newslist_local=list(filter(lambda x:x!=newsid,newslist_local))[:news_count]    
    
    if latest_Flag==False:        
        newslist_local = list(map(int, newslist_local))
        print('newslist_local 1=>',newslist_local)
        if uid_exists_in_model:
            random.shuffle(newslist_local)
        print('newslist_local 2=>',newslist_local)
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_newsdata',fieldname='newsid',fieldvaluelist=newslist_local)

    total_dict_data={}
    counter=1
    newslist_local_new=[]
    print('12....')
    
    for row_data in data:
        #print(row_data['videoid'])
        dict_data={}
        
        dict_data['newsid']=row_data['newsid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['uri']=row_data['uri']
        dict_data['mobile_image']=row_data['mobile_image'] + '?size=133:75'
        
        total_dict_data[row_data['newsid']]=dict_data
        newslist_local_new.append(row_data['newsid'])
        counter+=1
     
    #total_dict_data[215522]['link']    
    if latest_Flag:
        newslist_local=newslist_local_new    
        
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x)
    #print(uniquelist)
    newslist_local=uniquelist   
    
    data_final=[]
    #for videoid in videolist_local_new:
    #utm_medium='test'
    c=1
    for newsid in newslist_local:    
        if len(data_final)==news_count:
            break
        try:
            print(newsid)
            if newsid in newslist_local_new:
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                total_dict_data[newsid]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                #print(c, ' => ' ,total_dict_data[videoid]['link'])
                data_final.append(total_dict_data[newsid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status='Ok',message=message,source_newsid=source_newsid,data=data_final)

@app.route("/recengine/ltop/getarticles", methods=['GET', 'POST'])
def ltop_getarticles():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    #u=lallantop_utility.utility()
    
    ltop_u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    import boto3
    from boto3.dynamodb.conditions import Key
    #import requests     

    news_count=10
    #newsid='217674'
    newsid=0
    newsdata=None
    utm_source = None
    utm_medium = None
    source_newsid = 0
    uid=None
    min_story_count=5
    t_psl=False

    try:
        uid = request.args.get('uid',None)
        print('uid =>', uid)
    except Exception as exp:
        print('Exception in get uid=>',exp)
        uid = None
        
    try:
        utm_source = request.args.get('utm_source',None)
        print('uid =>', utm_source)
    except Exception as exp:
        print('Exception in get utm_source=>',exp)
        utm_source = 'Unknown'        

    try:
        newsid = request.args.get('newsid')
        if newsid.isdigit()==False:
            newsid=0    
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0
 
    try:
        utm_medium = request.args.get('utm_medium','Unknown')
        print('utm_medium =>', utm_medium)
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    try:
        news_count=int(request.args.get('no',10))
        print('news_count =>', news_count)
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        news_count=10        

    if news_count>20:
        news_count=20
        
    print("source_newsid = ",newsid)
    print("news_count = ",news_count)
    #news_corpus=u.get_newsid_corpus(news_id)
    source_newsid=newsid
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    log+='|request_uid=%s'%(uid)
    #newsid='212421'
    
    p1_ttl = 15 * 60
    p2_ttl = 60 * 60
    p3_ttl = 15 * 60
    #p4_ttl = 4 * 24 * 60 *60
    #Temp
    p4_ttl = 2 * 24 * 60 *60
    newslist_ttl = 10 * 60
    
    
    
    #p1_ttl = 2 * 60
    #p2_ttl = 1 * 60
    #p3_ttl = 1 * 60
    #p4_ttl = 1 * 60
    
    t1 = datetime.now()
    dynamodb=None
    interaction_item = None
    newslist_local=[]
    
    #This is for Lallantop =4
    site_id=str(4)
    
    newsrc_Flag=False
    uidrc_Flag=False
    history_db_Flag=False

    news_rc_list=None
    u_history_list=[]
    u_history_for_save=[]
    history_rc=[]

    data_final = []
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    
    print('1.....')
    
    mapping_id_newsid_dictionary = None
    lda_index = None  
    
    if redisFlag:
    #if 2==3:    
        #print('5......')  
        key_1=site_id + '-mapping-dic'
        key_2=site_id + '-lda'
        try:
            print('2......')  
            if ltop_rh.exists_key(key=key_1)==1 and ltop_rh.exists_key(key=key_2)==1:
                print('2.1......') 
                mapping_id_newsid_dictionary = pickle.loads(ltop_rh.get_data_from_cache(key=key_1))
                lda_index = pickle.loads(ltop_rh.get_data_from_cache(key=key_2))
        except Exception as exp:
            print('Exception in get mapping_id_newsid_dictionary and lda_index=>',exp)
            mapping_id_newsid_dictionary=None
            lda_index=None
            
        if mapping_id_newsid_dictionary==None or lda_index==None:        
            print('3......')  
            mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid')))
            lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus'))) 
            
            if redisFlag:
                set_flag = ltop_rh.set_data_in_cache(key=key_1, data=pickle.dumps(mapping_id_newsid_dictionary), ttl=newslist_ttl)    
                print('mapping_id_newsid_dictionary set in Redis Cache =>', set_flag)            
                set_flag = ltop_rh.set_data_in_cache(key=key_2, data=pickle.dumps(lda_index), ttl=newslist_ttl)    
                print('lda_index set in Redis Cache =>', set_flag)    

    def recommended_newsarray_hindi(newsdata=None,newsid=0):
        rc_news_array=[] 
        try:
            #hi_stop = get_stop_words('hi')
            hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
            clean_text = ltop_u.clean_doc_hindi(newsdata)
            cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
            tokens = cleaned_tokens_n.split(' ')
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
            stemmed_tokens = [ltop_u.generate_stem_words(i) for i in stopped_tokens]
            news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
            similar_news = lda_index[lda_at[news_corpus]]
            similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
            for x in similar_news[:(news_count + 1)]:
                rc_news_array.append(mapping_id_newsid_dictionary[x[0]])
                #log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            rc_news_array = list(filter(lambda x:x!=newsid,rc_news_array))  
            uniq = []
            [uniq.append(x) for x in rc_news_array if x not in uniq]
            rc_news_array = uniq 
        except:
            rc_news_array=[] 
        return rc_news_array   
    
    #newsid='83271'    
    if newsid!=None and int(newsid)>0:
        print('4.....')
        key = site_id + '-p1-' + newsid
        try:
            
            if ltop_rh.exists_key(key=key)==1:
                print('5.....')
                try:
                    news_rc_list = pickle.loads(ltop_rh.get_data_from_cache(key=key))
                    print('5.1   news_rc_list => ', news_rc_list)
                    news_rc_list = list(filter(lambda x:x!=newsid,news_rc_list)) 
                    newsrc_Flag=True
                    print('news_rc_list => ', news_rc_list)
                except:
                    print('Exception in get recommnedation news for passing newsid')
                    newsrc_Flag=False
            
            if newsrc_Flag==False:
                print('6.....')
                try:
                    newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
                except:
                    newsdata=None
                    print('6.1.....')
 
                if newsdata!=None:
                    news_rc_list = recommended_newsarray_hindi(newsdata=newsdata,newsid=newsid)
                    newsrc_Flag=True
                    print('6.2.....news_rc_list=>', news_rc_list)
                    if len(news_rc_list)>0:
                        ltop_rh.set_data_in_cache(key=key, data=pickle.dumps(news_rc_list), ttl=p1_ttl) 
        except:
            print('7 ...exp')
            newsrc_Flag=False

    min_story_Flag=False
    if uid!=None and uid!='':
        print('8 ....')
        #Temporary
        #uid='04b024e2-8285-4a83-b8cc-7706587fc1cc'

        key = site_id + '-p4-' + uid
        try:
            if ltop_rh.exists_key(key=key)==1:
                t_psl=True
                u_history_list = pickle.loads(ltop_rh.get_data_from_cache(key=key))
                print('8.1 ....u_history_list=> ', u_history_list)
                u_history_for_save = u_history_list
                u_history_list = list(reversed(u_history_list))
                uidrc_Flag=True
                if u_history_list!=None and len(u_history_list)>=min_story_count:
                    print('8.2 ....')
                    min_story_Flag=True    
        except:
            u_history_list=[]
            u_history_for_save = []
            print('8.3 ....')
        
        if min_story_Flag==False:
            print('8.4 ....')
            u_interaction_list=[]
            key = site_id + '-p2-' + uid
            try:
                if ltop_rh.exists_key(key=key)==1:
                    print('9 ....')
                    try:
                        u_interaction_list = pickle.loads(ltop_rh.get_data_from_cache(key=key))
                        history_db_Flag=True
                        uidrc_Flag=True
                    except:
                        print('Exception')
                        #history_db_Flag=True
                        #uidrc_Flag=True
                else:
                    try:
                        #import boto3
                        dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
                        #dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1", aws_access_key_id="AKIAJBYGBQ4PH5FQQSBQ", aws_secret_access_key= "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")
                        response = dynamodb.Table("itgd_cs_interaction_data_prod").query(IndexName="final_id-index",KeyConditionExpression=Key('final_id').eq(uid))
                        print('11 ....')
                        interaction_item = response['Items'] 
                        print('11.1 ....interaction_item=>', interaction_item)
                        if len(interaction_item)>0:
                            history_db_Flag=True
                            uidrc_Flag=True
                            interaction_item = sorted(interaction_item, key = lambda i: i['ist_tstamp'],reverse=True)[:20]
                            print('12 ....interaction_item=>', interaction_item)
                            #interaction_list=list(map(int, list(map(lambda x: x['newsid'] , list(filter(lambda x: x['site_id']==int(site_id), interaction_item))))))
                            interaction_item=list(map(str, list(map(lambda x: x['newsid'] , list(filter(lambda x: x['site_id']==int(site_id), interaction_item))))))
                            
                            print('13 ....', interaction_item)
                            
                            if len(interaction_item)>0:
                                u_interaction_list = []
                                [x for x in interaction_item if x not in u_interaction_list and u_interaction_list.append(x)]
                                
                                print('14 ...unique .len(u_interaction_list)', len(u_interaction_list))
                                key = site_id + '-p2-' + uid
                                set_flag = ltop_rh.set_data_in_cache(key=key, data=pickle.dumps(u_interaction_list), ttl=p2_ttl) 
                                #set_flag = ltop_rh.set_data_in_cache(key=key, data=str(u_interaction_list), ttl=p2_ttl)    
                            print('user interacted set  =>', set_flag)
                    except Exception as e:
                        history_db_Flag=False
                        print('Except=>',e)
            except:
                u_interaction_list=[]
                print('15 ....')        
                
        if history_db_Flag:   
            t_psl=True
            u_history_list.extend(u_interaction_list)   
            print('15.1 ....u_history_list =>', u_history_list) 
        
        if uidrc_Flag:                 
            try:
                print('16 ....')
                if len(u_history_list)>0:
                    u_history_list_unique = []
                    [x for x in u_history_list if x not in u_history_list_unique and u_history_list_unique.append(x)]
                
                    u_history_list=u_history_list_unique
                    

                u_history_list = list(filter(lambda x: x not in set([newsid]), list(map(str, u_history_list))))
                
                history_stack_length=10
                u_history_list=u_history_list[:history_stack_length]
                
                print('16.05.....u_history_list(with db records) =>',u_history_list)
                
                #For process
                #u_history_list=u_history_list[-(history_stack_length -1):]

                print('16.1 ....')
                counter=0
                #for nid in reversed(u_history_list):
                for nid in u_history_list:
                    news_rc_temp_list=[]
                    key = site_id + '-p1-' + str(nid)
                    counter += 1
                    try:
                        print('17 ....', counter,' => ', nid)
                        rc_content_flag=False
                        if ltop_rh.exists_key(key=key)==1:
                            print('18 ....in Redis exists', nid)
                            try:
                                news_rc_temp_list = pickle.loads(ltop_rh.get_data_from_cache(key=key))
                                print('18.1 ....news_rc_temp_list=>', news_rc_temp_list)
                                rc_content_flag=True
                            except:
                                print('18.2......Exception')
                                rc_content_flag=False
                        #else:
                        if rc_content_flag==False:
                            print('19.....')
                            #Process from DB only first 5 news Rest will check only from Cache
                            if counter<=5:
                                try:
                                    newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(nid))
                                except:
                                    newsdata=None
                                    print('19.1.....exp')
                                    
                                if newsdata!=None:
                                    news_rc_temp_list = recommended_newsarray_hindi(newsdata=newsdata,newsid=nid)
                                    if len(news_rc_temp_list)>0:
                                        ltop_rh.set_data_in_cache(key=key, data=pickle.dumps(news_rc_temp_list), ttl=p1_ttl) 

                        if len(news_rc_temp_list)>0:
                            #newsrc_Flag=True
                            history_rc.extend(news_rc_temp_list)
                            print('20.....len(history_rc)', len(history_rc))
                            print('news_rc_temp_list => ', news_rc_temp_list)
                    except:
                        print('21 ...exp')
                        
            except Exception as e:
                print('22 ...',e)
                print('Exception to set user list for key', key)
            
            try:
                #news_rc_list[2:]
                subtract_newslist=[]
                if newsrc_Flag:
                    history_rc.extend(news_rc_list[2:])
                    print('22.1.....history_rc=>', history_rc)
                    subtract_newslist = news_rc_list[:2]
                    print('22.2.....subtract_newslist=>', subtract_newslist)
                    
                subtract_newslist.append(newsid)
                print('22.3.....subtract_newslist=>', subtract_newslist)    
                #history_rc = list(set(history_rc) - set([newsid]) -  set(news_rc_list[:2]))
                history_rc = list(filter(lambda x: x not in set(subtract_newslist), history_rc))
                print('22.4.....history_rc=>=>', history_rc)
                h_uniq = []
                [h_uniq.append(x) for x in history_rc if x not in h_uniq]
                history_rc = h_uniq      
            
                if history_rc!=None and len(history_rc)<8:
                    uidrc_Flag=False
                    print('22.5.....')
            except Exception as e:    
                print('22.6.....')
                print('Exception to set user list for key', key)
   
    try:
        u_history_for_save = list(filter(lambda x: x not in set([newsid]), u_history_for_save))
        
        #u_history_for_save = list(set(history_rc) - set(newsid))
        u_history_for_save = u_history_for_save[-(news_count - 1):]
        u_history_for_save.append(newsid)
        print('23.....u_history_for_save=>', u_history_for_save)
     
        if uid!=None and len(u_history_for_save)>0:
            print('24.....u_history_for_save=>',u_history_for_save)
            key = site_id + '-p4-' + uid
            try:
                set_flag = ltop_rh.set_data_in_cache(key=key, data=pickle.dumps(u_history_for_save), ttl=p4_ttl) 
                print('user history set  =>', set_flag)
            except:
                print('Error to set History')
    except Exception as e: 
        print('24.1.....')
    
    final_rc_array = []
    try:
        if newsrc_Flag==True and uidrc_Flag==True:
            print('25.....')
            final_rc_array.extend(news_rc_list[:2])
            random.shuffle(history_rc)
            final_rc_array.extend(history_rc[:(news_count - 2)])
        elif newsrc_Flag==True and uidrc_Flag==False:  
            print('26.....')
            final_rc_array=news_rc_list[:news_count]
        elif newsrc_Flag==False and uidrc_Flag==True:    
            print('27.....')
            random.shuffle(history_rc)
            final_rc_array = history_rc[:news_count]
        else:
            print('28.....')
            latest_Flag=True
    except Exception as e:
        print('28.1.....') 
        latest_Flag=True
        
    if latest_Flag==False:        
        print('29.....')
        final_rc_array = list(map(int, final_rc_array))
        print('final_rc_array=>',final_rc_array)
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_newsdata',fieldname='newsid',fieldvaluelist=final_rc_array)
    else:
        print('30.....')
        data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
    
    counter=1
    newslist_local_new=[]
    
    #create a Dictionary for all news content to maintain sequence
    total_dict_data={}
    for row_data in data:
        dict_data={}
        dict_data['newsid']=row_data['newsid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['uri']=row_data['uri']
        dict_data['mobile_image']=row_data['mobile_image'] + '?size=133:75'
        total_dict_data[row_data['newsid']]=dict_data
        newslist_local_new.append(row_data['newsid'])
        counter+=1
   
    #For maintaning sequence of News
    if latest_Flag:
        print('31.....')
        newslist_local=newslist_local_new    
    else:   
        print('32.....')
        newslist_local=final_rc_array   
    
    data_final=[]
    newslist=[]        
    
    counter=1     
    for rc_newsid in newslist_local[:news_count]:
        temp_data=total_dict_data[rc_newsid]
        temp_data['uri']=temp_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,counter,utm_medium,counter,t_psl)
        newslist.append(rc_newsid)
        data_final.append(temp_data)
        counter+=1              

    print('Final Data Length=>',len(data_final))   
    
    print('newslist=>',newslist) 
            
    log+='|response=%s'%(newslist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status='Ok',message=message,source_newsid=source_newsid,data=data_final)

@app.route("/recengine/ltop/getarticles_20082020", methods=['GET', 'POST'])
def ltop_getarticles_20082020():
    import lallantop.utility as lallantop_utility 
    import lallantop.mongo_db_file_model as lallantop_mongo_db_file_model
    
    u=lallantop_utility.utility()
    
    ltop_u=lallantop_utility.utility()
    mdfb=lallantop_mongo_db_file_model.mongo_db_file_model()

    import boto3
    from boto3.dynamodb.conditions import Key
    import requests     

    news_count=5
    #newsid='212436'
    newsid=0
    newsdata=None
    utm_source = request.args.get('utm_source')
    utm_medium = None
    source_newsid = 0
    uid=None
    min_story_count=3
    t_psl=False
    try:
        uid = request.args.get('uid',None)
        print('uid =>', uid)
    except Exception as exp:
        print('Exception in get uid=>',exp)
        uid = None

    try:
        newsid = request.args.get('newsid')
    except Exception as exp:
        print('Exception in get news id=>',exp)
        newsid=0
 
    try:
        utm_medium = request.args.get('utm_medium','Unknown')
        print('utm_medium =>', utm_medium)
    except Exception as exp:
        print('Exception in get utm_medium Count =>',exp)
        utm_medium='Unknown'
        
    try:
        news_count=int(request.args.get('no',5))
        print('news_count =>', news_count)
    except Exception as exp:
        print('Exception in get Story Count =>',exp)
        news_count=5        

    if news_count>20:
        news_count=20
        
    print("source_newsid = ",newsid)
    print("news_count = ",news_count)
    #news_corpus=u.get_newsid_corpus(news_id)
    source_newsid=newsid
    log=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
    log+="|ltop-getarticles"
    log+= "|method=%s"%(request.method)
    log+= "|source_ip=%s"%(request.remote_addr)
    log+= "|utm_source=%s"%(utm_source)
    log+= "|queryString=%s"%(request.query_string)
    log+='|request_news_id=%s|sessionid=%s'%(newsid,'None')
    log+='|request_uid=%s'%(uid)
    #news_id=1044367
    
    #p1_ttl = 5 * 60
    p2_ttl = 5 * 60
    p3_ttl = 5 * 60
    newslist_ttl = 5 * 60
    
    t1 = datetime.now()
    uid_exists_in_model=False
    dynamodb=None
    interaction_item = None
    newslist_local=[]


    data_final = []
    response="SUCCESS"
    message="OK"
    data=""
    latest_Flag=False
    
    if uid!=None:
        #Temporary
        #uid='065643db-6669-4346-95bd-016e3fc03ceb'
        #uid = '00c6aa23-c794-40e4-86dd-b8cc6d305625'
        site_id=str(4)
           
        #if uid_exists_in_model:
        u_interaction_list=[]
        print('UID Found')
        try:
            print('U 1 ...')
           #Check recommended personalized data available in in Redis Cache....
            key = site_id + '-p3-' + uid
            p3_processflag = False
            #Get this user intrection list from Redis Cache
            try:
                print('U 3 ...')
                newslist_local = json.loads(ltop_rh.get_data_from_cache(key=key))
                print('newslist_local => ', len(newslist_local))
                
                #print(list[:(news_count + 1)])                    
                if newslist_local==None or newslist_local==[]:
                    p3_processflag=False
                    newslist_local=[]
                else:
                    p3_processflag=True
                    uid_exists_in_model=True
            except:
                print('U 4 ...')
                newslist_local=[]
                p3_processflag=False
                
            print('2....Final recommended data =>', newslist_local, '  p3_processflag=>', p3_processflag)                
            
            if p3_processflag==False:
                print('U 5 ...')
                #key for get user interaction data
                key = site_id + '-p2-' + uid
                try:
                    u_interaction_list = json.loads(ltop_rh.get_data_from_cache(key=key))

                    if u_interaction_list==None or u_interaction_list==[]:
                        u_interaction_list=[]
                    else:
                        if len(u_interaction_list)>=min_story_count:
                            uid_exists_in_model=True    
                except:
                    u_interaction_list=[]

                print('3....u_interaction_list length=>', len(u_interaction_list))                
                if u_interaction_list==[] or u_interaction_list==None:
                    try:
                        print('U 5.1...')
                        dynamodb = boto3.resource('dynamodb', region_name="ap-southeast-1")
                        response = dynamodb.Table("itgd_cs_interaction_data_prod").query(IndexName="final_id-index",KeyConditionExpression=Key('final_id').eq(uid))
                        print('U 5.2...')
                        interaction_item = response['Items'] 
                        print('U 5.3...')
                        #interaction_list=list(map(int, list(map(lambda x:x['newsid'], interaction_item))))
                        interaction_list=list(map(int, list(map(lambda x: x['newsid'] , list(filter(lambda x: x['site_id']==4, interaction_item))))))
                        print('U 5.4...', interaction_list)
                        u_interaction_list = []
                        
                        #if len(interaction_list)<min_story_count:
                            
                        u = [x for x in interaction_list if x not in u_interaction_list and u_interaction_list.append(x)]
                        print('3....u_interaction_list at setting time in cache=>', len(u_interaction_list))
                        key = site_id + '-p2-' + uid
                        set_flag = ltop_rh.set_data_in_cache(key=key, data=str(u_interaction_list), ttl=p2_ttl)    
                        print('user intrected news set in Redis Cache =>', set_flag)
                        if len(u_interaction_list)>=min_story_count:
                            uid_exists_in_model=True    
                    except Exception as e:
                        print('U 6.2 ...',e)
                        print('Exception to set user list for key', key)
                        uid_exists_in_model=False
                
                if uid_exists_in_model:
                    u_interaction_list = set(u_interaction_list)
                    print('U 7 ...')
                    
                    key = site_id + '-newslist'
                    #Getting news list from Cache
                    newslist=[]
                    try:
                        print('U 8 ...')
                        newslist = json.loads(ltop_rh.get_data_from_cache(key=key))
                        if newslist==None or newslist==[]:
                            newslist=[]
                    except:
                        print('U 9 ...')
                        newslist=[]
                        
                    if newslist==None or newslist==[]:
                        print('U 10 ...')
                        newslist = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='newslist'))) 
                        set_flag = ltop_rh.set_data_in_cache(key=key, data=str(newslist), ttl=newslist_ttl)    
                        print('newslist set in Redis Cache =>', set_flag)                    
                        
                    print('U 11 ...')
                    newslist=set(list(map(int, newslist))) 
                    non_interacted_news_list = newslist - u_interaction_list - set([newsid])
                    print('U 12 ...non_interacted_news_list=',len(non_interacted_news_list))
                    data={}
                    fix_count=20
                    data['user_id']=uid
                    data['count']=fix_count
                    data['newslist']=list(non_interacted_news_list)  
                    
                    #print('data=>',data)
                    #data=json.dumps(data)
                    responseBody=None
                    try:
                        print('U 13 ...')
                        url='https://142dro4haa.execute-api.ap-southeast-1.amazonaws.com/prod/lallantop-recengine'
                        print('U 14 ...')
                        headers = {"Content-Type": "application/json","Accept": "application/json"}
                        print('U 15 ...',headers)
                        response = requests.post(url, json=data, headers=headers)
                        print('U 16 ...',response)
                        responseBody=response.text
                        print('U 17 ...',responseBody)
                    except:
                        print('U 18 ...')
                        print('Exception =>')
                    res = json.loads(responseBody) 
                    print('U 19 ...',res)
                    newslist_local = list(map(lambda x:x['newsid'], res))
                    print('U 20 ...',newslist_local)
                    key = site_id + '-p3-' + uid
                    set_flag = ltop_rh.set_data_in_cache(key=key, data=str(newslist_local), ttl=p3_ttl)    
                    print('Recommennded news set in Redis Cache =>', set_flag)  
                    uid_exists_in_model=True 
                    t_psl=True
                    
            else:
                print('U 21 ...')
                newslist_local = list(set(newslist_local) - set([newsid]))
                t_psl=True
        except:
            print('U 22 ...')
            uid_exists_in_model=False
            interaction_item = None
            t_psl=False
            print('No uid exist in User interaction')
        #else:
        #    uid_exists_in_model=False
            
    print('uid_exists_in_model =>',uid_exists_in_model)    
    if uid_exists_in_model==False:
        t_psl=False
        try:
            newsdata=mdfb.get_lallantop_news_text_from_mongodb(collection_name='ltop_recom',fieldname='unique_id',fieldvalue=int(newsid))
        except:
            newsdata=None
        #hi_stop = get_stop_words('hi')
        hi_stop=['अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'अंदर', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर', 'करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नके', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वग़ैरह', 'वर्ग', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'संग', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने']
        #print(dictionary)
        mapping_id_newsid_dictionary = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='id_newsid')))
        lda_index = pickle.loads(gzip.decompress(mdfb.get_data_record_from_mongodb(collection_name='ltop_file_system',filename='portal_corpus'))) 
        if newsdata==None or len(newsdata)<=30:
            print("No Source Data........")
            log+= '|FAIL'
            log+= '|No Process,text=%s'%(newsdata)
            #data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
            data=mdfb.get_latest_news_records(collection_name='tbl_ltop_newsdata', field_name='publishdate', LIMIT=news_count)
            #video_count=30
            latest_Flag=True
        else:
            print('Similar data.....')
            log+= '|SUCCESS'
            log+= '|Result'
            
            print('E1....')
            print('Length = >', len(newsdata))
            clean_text = ltop_u.clean_doc_hindi(newsdata)
            print('E2....', clean_text)
            #print('1....')
            cleaned_tokens_n = re.sub('[0-9a-zA-Z]+', '', clean_text)
            #print('2....')
            tokens = cleaned_tokens_n.split(' ')
            #print('3....')
            cleaned_tokens = [word for word in tokens if len(word) > 2]
            #print('4....')
            stopped_tokens = [i for i in cleaned_tokens if not i in hi_stop]
            #print('5....')
            stemmed_tokens = [ltop_u.generate_stem_words(i) for i in stopped_tokens]
            #print('6....')
            news_corpus = [dictionary_at.doc2bow(text) for text in [stemmed_tokens]]
            #print('7....')
            similar_news = lda_index[lda_at[news_corpus]]
            #print('8....')
            similar_news = sorted(enumerate(similar_news[0]), key=lambda item: -item[1])
            #print('9....')
           
            for x in similar_news[:(news_count + 1)]:
                #print('9.1....')    
                newslist_local.append(mapping_id_newsid_dictionary[x[0]])
                log+=',(%s-%s)'%(mapping_id_newsid_dictionary[x[0]],x[1])
            newslist_local=list(filter(lambda x:x!=newsid,newslist_local))[:news_count]    
    
    if latest_Flag==False:        
        newslist_local = list(map(int, newslist_local))
        print('newslist_local 1=>',newslist_local)
        if uid_exists_in_model:
            random.shuffle(newslist_local)
        print('newslist_local 2=>',newslist_local)
        data=mdfb.get_lallantop_news_data_for_json(collection_name='tbl_ltop_newsdata',fieldname='newsid',fieldvaluelist=newslist_local)

    total_dict_data={}
    counter=1
    newslist_local_new=[]
    print('12....')
    
    for row_data in data:
        #print(row_data['videoid'])
        dict_data={}
        
        dict_data['newsid']=row_data['newsid']
        dict_data['title']=row_data['title']
        #dict_data['link']=row_data['uri']+'?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,counter)
        dict_data['uri']=row_data['uri']
        dict_data['mobile_image']=row_data['mobile_image'] + '?size=133:75'
        
        total_dict_data[row_data['newsid']]=dict_data
        newslist_local_new.append(row_data['newsid'])
        counter+=1
     
    #total_dict_data[215522]['link']    
    if latest_Flag:
        newslist_local=newslist_local_new    
        
    uniquelist = []
    for x in newslist_local:
        if x not in uniquelist:
            uniquelist.append(x)
    #print(uniquelist)
    newslist_local=uniquelist   
    
    data_final=[]
    #for videoid in videolist_local_new:
    #utm_medium='test'
    c=1
    for newsid in newslist_local:    
        if len(data_final)==news_count:
            break
        try:
            print(newsid)
            if newsid in newslist_local_new:
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=video_player-slot-%d'%(utm_medium,c)
                #total_dict_data[newsid]['link'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d'%(utm_medium,c)
                total_dict_data[newsid]['uri'] += '?utm_source=recengine&utm_medium=%s&referral=yes&utm_content=footerstrip-%d&t_source=recengine&t_medium=%s&t_content=footerstrip-%d&t_psl=%s'%(utm_medium,c,utm_medium,c,t_psl)
                #print(c, ' => ' ,total_dict_data[videoid]['link'])
                data_final.append(total_dict_data[newsid])
            c+=1    
        except Exception as exp:   
            print('Exception =>',exp)

    print('Final Data=>',data_final)    
            
    log+='|response=%s'%(newslist_local_new)
    log+='|finish_time=%s'%(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f'))
    t2 = datetime.now()
    d=t2-t1
    log+='|total_time=%s'%(d)
    log+= "|queryString=%s"%(request.query_string)
    filename="flask_web_application_ltop_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') + ".txt"
    filename = filepath + filename
    fw.log_write(filepath=filename,log=log)
    return jsonify(status='Ok',message=message,source_newsid=source_newsid,data=data_final)



if __name__ == "__main__":
    '''
    fw=file_writer()
    config = configparser.ConfigParser()
    config.read_file(open('config.properties'))

    #filepath="E:/ITG/logs/data_transfer/"
    filepath=config.get('LOGPATH', 'data_transfer_logpath')
    model_path=config.get('FILEPATH', 'model_path')
    print('Start loading ....')
    print('1....')
    t1=datetime.now()
    #lda_model=mdb.get_data_record_from_mongodb(collection_name='bt_file_system',filename='lda_model')
    #lda = pickle.loads(gzip.decompress(lda_model))
    lda=mdb.load_latest_version_file_data_in_gridfs(filename='lda_model')
    lda_it=mdb_it.load_latest_version_file_data_in_gridfs(filename='lda_model_it')
    lda_at=mdfb_at.load_latest_version_file_data_in_gridfs(filename='lda_model_at')
    print('Business Today lda Model=>',lda)
    print('India Today lda Model=>',lda_it)
    print('Aaj Tak lda Model=>',lda_at)
    
    t2=datetime.now()
    d=t2-t1
    print("Total Time = %s "%(d))
    print('Start for take request.......')
    '''
    #app.run(threaded=True, debug=False)
    #app.run()
    app.run(host="0.0.0.0", port=int("5000"), threaded=False, debug=False)















