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
from sklearn.linear_model import SGDClassifier
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
from nltk.stem.snowball import SnowballStemmer
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

        # Remove stop words
        remover = StopWordsRemover(inputCol='words_token', outputCol='filtered')
        df_words_no_stopw = remover.transform(df_words_token)

        #Stemming data
        stemmer = SnowballStemmer(language='english')
        stemmer_udf = F.udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
        df_stemmed = df_words_no_stopw.withColumn("word_stemmed", stemmer_udf("filtered")).select('*')

        #Hashing using HashingTF
        hashtf = HashingTF(numFeatures=2500, inputCol="word_stemmed", outputCol='tf')
        label_stringIdx = StringIndexer(inputCol = "Label", outputCol = "target")
        
        #Pipelining and transforming using the dtages of hashingTF
        pipeline = Pipeline(stages=[hashtf,label_stringIdx])
        
        #Pipeline fitting using the clean data
        pipelineFit = pipeline.fit(df_stemmed)
        train_df = pipelineFit.transform(df_stemmed)

        #converting to numpy array and reshaping it according to the dimensions
        trainx=np.array(train_df.select("tf").collect())
        trainy=np.array(train_df.select("target").collect())
        d, x, y = trainx.shape
        trainx=trainx.reshape((d,x*y))

        #splitting train pre-processed data into train and test
        trainx,test_x,trainy,test_y=train_test_split(trainx, trainy, test_size=0.2, random_state=42)

        global maxfsc,batchsize,inc
        try:
            #creating pkl files to use the best fit trained model
            model=joblib.load("SGDC_"+str(batchsize)+'.pkl')

            #partial fitting the model
            model.partial_fit(trainx, trainy,classes=np.unique(trainy))
            joblib.dump(model,"SGDC_"+str(batchsize)+'.pkl')

        except Exception as e: 
            print("error in traning:",e)

        try:
            #predicting the sentiment
            y_pred = model.predict(test_x)

            inc+=1

            #finding the performance metrics
            fsc=f1_score(test_y,y_pred)
            cm = confusion_matrix(test_y, y_pred)
            print(cm)
            
            print('--------------------------',inc,'------------------------')    
            print("f1 score",fsc)
            
            #finding rmse,accuracy,recall and precision
            rmse=mean_squared_error(test_y, y_pred)
            acc=accuracy_score(test_y, y_pred)
            rsc=recall_score(test_y, y_pred, average=None)[0]
            psc=precision_score(test_y, y_pred, average=None)[0]
            data=[inc,fsc,maxfsc,acc,psc,rsc,batchsize,rmse]
            if(fsc>maxfsc):
                maxfsc=fsc
                joblib.dump(model,"SGDC_bestfsc_"+str(batchsize)+'.pkl')
                facp=open('./SGDC_best_stats_'+str(batchsize)+'.txt','w+')
                facp.write("best f1:"+str(maxfsc)+"\nbatchsize:"+str(batchsize)+'\ncm:'+str(list(cm))+'\ninc:'+str(inc)+'\nprecsion'+str(psc)+'\nrecall'+str(rsc))
                facp.close()
            with open('./SGDC_stats_'+str(batchsize)+'.csv','a+') as fp:
                writer=csv.writer(fp)
                writer.writerow(data)
            facp=open('./SGDC_stats_'+str(batchsize)+'.txt','w+')
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


# creating a classifier
classifier = SGDClassifier()

batchsize=sys.argv[1]
joblib.dump(classifier,"SGDC_"+str(batchsize)+".pkl")
maxfsc=0
inc=0
header=['iter','f1','maxfc','acc','precision','recall','batchsize','rmse']
with open('./SGDC_stats_'+str(batchsize)+'.csv','a+') as fp:
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