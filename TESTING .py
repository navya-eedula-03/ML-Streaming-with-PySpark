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

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score,recall_score



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
        tdf = pipelineFit.transform(df_words_no_stopw)
        # testd=pipelineFit.transform(t2)
        tdf.show(5)
        # tdf,testd=train_test_split(tdf,test_size=0.2)
        testx=np.array(tdf.select("tf").collect())
        testy=np.array(tdf.select("target").collect())
        d, x, y = testx.shape
        testx=testx.reshape((d,x*y))
        
        # global maxfsc,batchsize,inc
        try:
            model=joblib.load('PAC_3000.pkl')
            y_pred = model.predict(testx)
            # print(y_pred)
            # inc+=1
            fsc=f1_score(testy,y_pred)
            cm = confusion_matrix(testy, y_pred)
            print(cm)
            print("f1 score",fsc)
            rmse=mean_squared_error(testy, y_pred)
            acc=accuracy_score(testy, y_pred)
            rsc=recall_score(testy, y_pred, average=None)[0]
            psc=precision_score(testy, y_pred, average=None)[0]
            # data=[inc,fsc,maxfsc,acc,psc,rsc,batchsize,rmse]


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


