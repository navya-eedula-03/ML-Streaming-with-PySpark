from pyspark import SparkContext
from pyspark.ml.pipeline import Pipeline
from pyspark.rdd import PipelinedRDD
from pyspark.sql.types import ArrayType, StringType
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from functools import reduce
import  pyspark.sql.functions as F
import json,math
import numpy as np
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from pyspark.ml.feature import HashingTF, RegexTokenizer, StringIndexer, Tokenizer, StopWordsRemover,IDF
from nltk.stem.snowball import SnowballStemmer
import joblib
import sys
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
import csv 
from sklearn.cluster import MiniBatchKMeans

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score,recall_score


def vectrize(x_test):
    cv = CountVectorizer(max_features = 2500)
    x = cv.fit_transform(x_test).toarray()
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x



def createDataFrame(rdd):
    try:
        # creating the partial dataframe
        temp = spark.createDataFrame(rdd)
        oldColumns = temp.schema.names
        newColumns = ["Label", "Tweet"]
        temp= reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), temp)
        

        #doing preprocessing
        df_clean=temp.dropna(subset=['Tweet'])
        df_select_clean = (df_clean.withColumn("Tweet", F.regexp_replace("Tweet", r"[@#&][A-Za-z0-9-]+", " "))
                       .withColumn("Tweet", F.regexp_replace("Tweet", r"\w+://\S+", " "))
                       .withColumn("Tweet", F.regexp_replace("Tweet", r"[^A-Za-z]", " "))
                       .withColumn("Tweet", F.regexp_replace("Tweet", r"\s+", " "))
                       .withColumn("Tweet", F.lower(F.col("Tweet")))
                       .withColumn("Tweet", F.trim(F.col("Tweet")))
                      ) 

        # printing the final DataFrame
        tokenizer = Tokenizer(inputCol='Tweet', outputCol='words_token')
        df_words_token = tokenizer.transform(df_select_clean)
        # t1 = tokenizer.transform(test_df)
        # Remove stop words
        remover = StopWordsRemover(inputCol='words_token', outputCol='filtered')
        df_words_no_stopw = remover.transform(df_words_token)
        # t2 = remover.transform(t1)

        hashtf = HashingTF(numFeatures=2500, inputCol="filtered", outputCol='tf')
        label_stringIdx = StringIndexer(inputCol = "Label", outputCol = "target")
        
        pipeline = Pipeline(stages=[hashtf,label_stringIdx])
        pipelineFit = pipeline.fit(df_words_no_stopw)
        train_df = pipelineFit.transform(df_words_no_stopw)
        # testd=pipelineFit.transform(t2)
        train_df.show(5)
        # train_df,testd=train_test_split(train_df,test_size=0.2)
        trainx=np.array(train_df.select("tf").collect())
        trainy=np.array(train_df.select("target").collect())
        d, x, y = trainx.shape
        trainx=trainx.reshape((d,x*y))
        # print(trainx.shape)
        # print("y",trainy.shape)
        trainx,test_x,trainy,test_y=train_test_split(trainx, trainy, test_size=0.2, random_state=42)
        # test_x=np.array(testd.select("tf").collect())
        # test_y=np.array(testd.select("target").collect())
        # d, x, y = test_x.shape
        # test_x=test_x.reshape((d,x*y))
        global maxfsc,batchsize,inc
        try:
            model=joblib.load("KC_"+str(batchsize)+'.pkl')
            model.partial_fit(trainx)
            joblib.dump(model,"KC_"+str(batchsize)+'.pkl')

        except Exception as e: 
            print("error in traning:",e)

        try:

        
            y_pred = model.predict(test_x)
            # print(y_pred)
            inc+=1
            fsc=f1_score(test_y,y_pred)
            cm = confusion_matrix(test_y, y_pred)
            print(cm)
            
            print('--------------------------',inc,'------------------------')    
            print("f1 score",fsc)
            
            rmse=mean_squared_error(test_y, y_pred)
            acc=accuracy_score(test_y, y_pred)
            rsc=recall_score(test_y, y_pred, average=None)[0]
            psc=precision_score(test_y, y_pred, average=None)[0]
            data=[inc,fsc,maxfsc,acc,psc,rsc,batchsize,rmse]
            if(fsc>maxfsc):
                maxfsc=fsc
                joblib.dump(model,"KC_bestfsc_"+str(batchsize)+'.pkl')
                facp=open('./KC_best_stats_'+str(batchsize)+'.txt','w+')
                facp.write("best f1:"+str(maxfsc)+"\nbatchsize:"+str(batchsize)+'\ncm:'+str(list(cm))+'\ninc:'+str(inc)+'\nprecsion'+str(psc)+'\nrecall'+str(rsc))
                facp.close()
            with open('./KC_stats_'+str(batchsize)+'.csv','a+') as fp:
                writer=csv.writer(fp)
                writer.writerow(data)
            facp=open('./KC_stats_'+str(batchsize)+'.txt','w+')
            facp.write("current f1:"+str(fsc)+"\nbest f1:"+str(maxfsc)+"\nbatchsize:"+str(batchsize)+'\ncm:'+str(list(cm))+'\ninc:'+str(inc))
            facp.close()


        except Exception as e: 
            print("error in accuracy:",e)



    except Exception as e: 
        print(e)



# getting the spark context
sc = SparkContext("local[2]", "Sentiment")
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
ssc = StreamingContext(sc, 5)

# making sore
sc.setLogLevel("ERROR")
spark.sparkContext.setLogLevel("ERROR")



batchsize=2048

classifier =  MiniBatchKMeans(n_clusters=2,batch_size=batchsize)


joblib.dump(classifier,"KC_"+str(batchsize)+".pkl")
maxfsc=0
inc=0
header=['iter','f1','maxfc','acc','precision','recall','batchsize','rmse']
with open('./KC_stats_'+str(batchsize)+'.csv','a+') as fp:
    writer=csv.writer(fp)
    writer.writerow(header)



# getting the streaming contents
lines = ssc.socketTextStream('localhost',6100)
words = lines.flatMap(lambda line : json.loads(str(line)))
words=words.map(lambda x:x.split(',',1))

# function to create a data frame from the streamed data
words.foreachRDD(createDataFrame)


# starting the stream
ssc.start()
ssc.awaitTermination()

ssc.stop(stopSparkContext=True, stopGraceFully=True)
