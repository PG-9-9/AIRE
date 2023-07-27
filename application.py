import pyrebase
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from sklearn import preprocessing,metrics
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,LassoLars,Ridge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests
from pandas_datareader import data
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template, request
import time
from datetime import datetime, timedelta
import json

#twitter imports
import tweepy
from textblob import TextBlob
import nltk
from yahoo_fin import stock_info as si 
import re
import string

application = Flask(__name__, template_folder = 'templates', static_folder='static')
risk_score = 0

# pip list --format=freeze >requirements.txt
model,accuracy,bs_error,squared_error,rms_error=0,0,0,0,0
savemetric=[]
stockdata=pd.DataFrame()       

_EXCHANGE_LIST = ['nyse', 'nasdaq', 'amex']

#twitter api credentials
#loading the twitter API credentials
consumerKey = ""
consumerSecret = ""
accessToken = ""
accessTokenSecret = ""

authenicate = tweepy.OAuthHandler(consumerKey, consumerSecret)
        #set the access token and access secret token
authenicate.set_access_token(accessToken, accessTokenSecret)
        #create the api object
api = tweepy.API(authenicate, wait_on_rate_limit=True)

headers = {
    'authority': 'api.nasdaq.com',
    'accept': 'application/json, text/plain, */*',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
    'origin': 'https://www.nasdaq.com',
    'sec-fetch-site': 'same-site',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'referer': 'https://www.nasdaq.com/',
    'accept-language': 'en-US,en;q=0.9',
}

_SECTORS_LIST = set(['Consumer Non-Durables', 'Capital Goods', 'Health Care',
                     'Energy', 'Technology', 'Basic Industries', 'Finance',
                     'Consumer Services', 'Public Utilities', 'Miscellaneous',
                     'Consumer Durables', 'Transportation'])

def params(exchange):
    return (
        ('letter', '0'),
        ('exchange', exchange),
        ('render', 'download'),
    )

params = (
    ('tableonly', 'true'),
    ('limit', '25'),
    ('offset', '0'),
    ('download', 'true'),
)

############################## sentiment analysis start ######################################
#getting tweets
def getTweets(stock_name):
    N_DAYS_AGO = 14

    today = datetime.now()    
    n_days_ago = today - timedelta(days=N_DAYS_AGO)
    n_days_ago = n_days_ago.strftime('%Y-%m-%d')
    search_term = '#' + stock_name + ' -filter:retweets'
    #create a cursor object
    tweets = tweepy.Cursor(api.search, q=search_term, lang='en', since=n_days_ago, tweet_mode='extended').items(100)
    #store the tweets in a variable
    all_tweets = [tweet.full_text for tweet in tweets]
    return all_tweets

#cleaning tweets
def clean_twt(twt):
    if twt == 0:
        return '0'
    else:
        twt = re.sub('#[A-Za-z0]+','',twt) #Removing any string with a hastag
        twt = re.sub('\\n','',twt) #removing the \n
        twt = re.sub('https?:\/\/\S+','', twt) #removes any hyperlinks
        twt = "".join([char for char in twt if char not in string.punctuation]) #removing punctuation
        twt = re.sub('[0-9]+','',twt) #removing numbers
        return twt

#finding subjectivity and polarity to the columns
#function to get subjectivity
def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity

#function to get polarity
def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity

#function to get the sentiment outcome
def getSentiment(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'

#function to calculate percentage of positive
def cal_percentage(positive_score):
    positive_percentage = positive_score / 100
    return positive_percentage

############################## sentiment analysis end ######################################

def daily_routine():
        ##################################Daily Recommendationt############################
        tickers = si.tickers_sp500()
        tickers=tickers[0:40] #for testing 
        recommendations = []
        
        for ticker in tickers:
                lhs_url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/'
                rhs_url = '?formatted=true&crumb=swg7qs5y9UP&lang=en-US&region=US&' \
                          'modules=upgradeDowngradeHistory,recommendationTrend,' \
                          'financialData,earningsHistory,earningsTrend,industryTrend&' \
                          'corsDomain=finance.yahoo.com'
                          
                url =  lhs_url + ticker + rhs_url
                r = requests.get(url)
                if not r.ok:
                    recommendation = 6
                try:
                    result = r.json()['quoteSummary']['result'][0]
                    recommendation =result['financialData']['recommendationMean']['fmt']
                except:
                    recommendation = 6
                
                recommendations.append(recommendation)
                
                
        dataframe = pd.DataFrame(list(zip(tickers, recommendations)), columns =['Company', 'Recommendations']) 
        dataframe = dataframe.set_index('Company')
        
        
        dataframe.insert(0,"Tickers",tickers)
        df_list=dataframe.values.T.tolist()
        
        company_df = pd.read_csv('static/companylist.csv')
        ctickers=company_df.iloc[:,0].values
        rtickers=dataframe.iloc[:,0].values    
        
        final_arr=[]
        outer_index=0
        
        for each_ticker in rtickers:
            inner_index=0
            for stock_names in ctickers:
                
                if each_ticker==stock_names:
                    names=company_df.iloc[inner_index,1]
                    rs=dataframe.iloc[outer_index,1]
                    new_entry={'Stock Ticker':each_ticker,'Stock Names':names,'Recommendation Score':rs}
                    final_arr.append(new_entry.copy())
                inner_index+=1
            outer_index+=1
            
        Daily_Recommendation=pd.DataFrame(final_arr)
        Daily_Recommendation.to_csv('static/Dashboard_Rec.csv')
        ##################################Recommendation Stock Collection -1 Start############################
        models=[]
        evaluation_metric=np.array([])
        actual_date = dt.date.today()
        historic_date=pd.DataFrame()
        eval_metric=[]
        stock_size_counter=[]
        past_date = actual_date - dt.timedelta(days=365 * 3)
        actual_date = actual_date.strftime("%Y-%m-%d")
        past_date = past_date.strftime("%Y-%m-%d")
        filtered_tickers = get_tickers_filtered(mktcap_min=100000, mktcap_max=10000000)
        filtered_tickers=list(dict.fromkeys(filtered_tickers))
        total_tickers=len(filtered_tickers)
        model_error=[]
        add_metric=np.array([])
        final_stock_tickers=[]
        models=["Stepwise","Svm","Lasso","Ridge","Boosted","forest"]
        ##################################Recommendation Stock Collection -1 END############################

        ############################## sentiment analysis start ######################################
        #assigning tickers
        tickers = filtered_tickers
        #create the authentication object

        #getting tweets from twitter
        all_stocks_df = pd.DataFrame()

        filler = 0
        #loop over the tickers to send to function
        for t in tickers:
            tweets_one_stock = getTweets(t)
            all_stocks_df[t] = tweets_one_stock + [filler]*(len(all_stocks_df.index) - len(tweets_one_stock))

        #cleaning tweets
        all_stocks_cleaned_df = pd.DataFrame()

        #looping through the tickers
        for tick in tickers:
            all_stocks_cleaned_df[tick] = all_stocks_df[tick].apply(clean_twt)
        
        all_stocks_polarity_df = pd.DataFrame()
        all_stocks_subjectivity_df = pd.DataFrame()

        #looping through dataframe and adding it to new dataframe
        for ticka in tickers:
            all_stocks_polarity_df[ticka] = all_stocks_cleaned_df[ticka].apply(getPolarity)
            all_stocks_subjectivity_df[ticka] = all_stocks_cleaned_df[ticka].apply(getSubjectivity)

        all_sentiment_df = pd.DataFrame()

        for tickb in tickers:
            all_sentiment_df[tickb] = all_stocks_polarity_df[tickb].apply(getSentiment)
            

        all_sentiment_scores_df = pd.DataFrame()

        #get percentage of positive for every stock
        #get all sentiment value counts
        for tickc in tickers:
            all_sentiment_scores_df[tickc] = all_sentiment_df[tickc].value_counts()
            
        #change nan values to 0
        all_sentiment_scores_df = all_sentiment_scores_df.fillna(0)

        temp_scores_df = pd.DataFrame()

        for tickd in tickers:
            temp_scores_df[tickd] = all_sentiment_scores_df[tickd].apply(cal_percentage)
            
        positive_score = []
        for ticke in tickers:
            positive_score.append(temp_scores_df[ticke][1])

        print("Hey Roobesh! It's done! Check out the temp_scores_df")

        temp_scores_df.to_csv('static/Sentiment_scores.csv')

        #final sentiment filtering
        #reading the saved sentiment scores
        sentiment_score_df = pd.read_csv('static/Sentiment_scores.csv')

        #removing the unnecessary column
        sentiment_score_df = sentiment_score_df.drop('Unnamed: 0', axis = 1)
        ############################## sentiment analysis end ######################################

        # models.append(out)
        for model in models:
            count=0
            model_name=model
            predicted_df=np.array([])
            historic_data=np.array([])
            for stock_symbol in filtered_tickers:
                count+=1
                
                if count<=total_tickers:
                    prev_data=pd.DataFrame()
                    print(' {} {}/{} '.format(stock_symbol,count,total_tickers))    
                    dataframe = get_stock_data(stock_symbol, past_date, actual_date)
                    prev_data=dataframe.copy()
                    prev_dates=dataframe.copy()
                    # moving data index
                    prev_dates=prev_dates.reset_index()
                    prev_dates=prev_dates['Date']
                    historic_date=prev_dates
                    
                    prev_data=prev_data['Close'].values.tolist()
                    
                    prev_data=np.array([prev_data]).T
                    stock_size_counter.append(prev_data.size)
                    if (len(stock_size_counter)==1 or stock_size_counter[-2]==stock_size_counter[-1]):
                        
                        historic_data=np.hstack([historic_data,prev_data]) if historic_data.size else prev_data
                        (dataframe, forecast_out,pred_arr,date_field,metrics_df) = stock_forecasting(dataframe,model,stock_symbol)
                        
                        df_out_metric=metric_calculation(metrics_df,count,total_tickers,model)
                        if(count==total_tickers):
                            add_metric=np.hstack([add_metric,np.array(df_out_metric.values.tolist())]) if add_metric.size else np.array(df_out_metric.values.tolist())   
                        pred_values=np.array([pred_arr]).T
                     
                        final_stock_tickers.append(stock_symbol)
                        metric_values=np.array([metrics_df]).T
                        evaluation_metric=np.hstack([evaluation_metric,metric_values]) if evaluation_metric.size else metric_values
                        predicted_df=np.hstack([predicted_df,pred_values]) if predicted_df.size else pred_values
                   
            final_dataframe=pd.DataFrame(predicted_df)
            historic_dataframe=pd.DataFrame(historic_data)
            final_stock_tickers=list(dict.fromkeys(final_stock_tickers))
            historic_dataframe.columns=final_stock_tickers
            historic_dataframe.insert(0,"Date",historic_date.values.tolist())
            historic_dataframe['Date']=pd.to_datetime(historic_dataframe['Date'])

            historic_dataframe=historic_dataframe.drop(len(historic_dataframe)-1)
            
            final_dataframe.columns=final_stock_tickers
            final_dataframe.insert(0,"Date",date_field)
            historic_dataframe=pd.concat([historic_dataframe,final_dataframe])
            historic_dataframe.to_csv("static/"+model_name+"Dataset.csv") 
           
        add_df=pd.DataFrame(add_metric)
        add_df.to_csv("static/Final_Metric.csv")
        return "Scheduler Call Successfull"



sched = BackgroundScheduler(daemon=True)
sched.add_job(daily_routine,'interval',minutes=1440)
sched.start()


def user_risk_calculator():
    #firebase configuration information
    firebaseConfig = {
    "apiKey": "",
    "authDomain": "",
    "databaseURL": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "measurementId": ""
    };

    #connecting to firebase
    firebase = pyrebase.initialize_app(firebaseConfig)

    db = firebase.database()

    #retrive data
    users = db.child("User_info").get()


    #iterate throught the information to store values into dataframe
    user_values_list = []
    #iterate throught the users
    for user in users.each():
        user_values_list.append(user.val())

    user_information = pd.DataFrame(user_values_list)

    #------------------------Calculating risk score and tiers---------------------

    #converting blank values to nan
    user_information = user_information.replace(r'^\s*$', np.nan, regex=True)


    #converting nan to 0
    user_information = user_information.fillna(0)


    #function to calculate risk score
    def calculate_risk_score(user_ref):
        user_row = user_information.loc[user_information['userid'] == user_ref]
        user_row = user_row.iloc[:,8:19]
        
        #converting string values in numbers
        user_row = user_row.astype(float)
        
        #getting mean values
        user_mean = user_row.mean(axis = 1)
        user_mean = round(user_mean, 2)
        user_mean = user_mean.values
        user_mean = user_mean[0]
        return user_mean
        

    #iterating the dataframe to call function 
    user_score_tot = []    
    for user_id in user_information['userid']:
        user_score = calculate_risk_score(user_id)
        user_score_tot.append(user_score)

    #adding list to dataframe
    user_information['risk_score'] = user_score_tot

    #tier assigning function
    def assign_tier(score):
        if score > 0 and score <= 0.40:
            return 'A'
        elif score > 0.40 and score <= 0.60:
            return 'B'
        elif score > 0.60 and score <= 0.80:
            return 'C'
        else:
            return 'D'
        
    #calling function to assign tiers
    tier_list = []
    for score in user_information['risk_score']:
        tier_val = assign_tier(score)
        tier_list.append(tier_val)
        
    #adding tier list to dataframe
    user_information['tier'] = tier_list

    #converting all values in tables to string
    user_information = user_information.drop('risk_score', axis = 1)
    user_score_tot_str = [str(n) for n in user_score_tot]
    user_information['risk_score'] = user_score_tot_str

    #--------------------Sending data to firebase---------------------------------

    for i in range(len(user_information)):
        user_score = user_information.loc[i, 'risk_score']
        user_tier = user_information.loc[i, 'tier']
        user_id = user_information.loc[i, 'userid']
        data = {"risk_score" : user_score, "tier": user_tier}
        db.child("User_info").child(user_id).update(data)
    return "firebase successful"

def get_tickers_filtered(mktcap_min=None, mktcap_max=None, sectors=None):
    tickers_list = []
    for exchange in _EXCHANGE_LIST:
        tickers_list.extend(
            __exchange2list_filtered(exchange, mktcap_min=mktcap_min, mktcap_max=mktcap_max, sectors=sectors))
    return tickers_list                                       

def __exchange2df(exchange):
    # response = requests.get('https://old.nasdaq.com/screening/companies-by-name.aspx', headers=headers, params=params(exchange))
    # data = io.StringIO(response.text)
    # df = pd.read_csv(data, sep=",")
    r = requests.get('https://api.nasdaq.com/api/screener/stocks', headers=headers, params=params)
    data = r.json()['data']
    df = pd.DataFrame(data['rows'], columns=data['headers'])
    return df
                          
def __exchange2list_filtered(exchange, mktcap_min=None, mktcap_max=None, sectors=None):
    df = __exchange2df(exchange)
    # df = df.dropna(subset={'MarketCap'})
    df = df.dropna(subset={'marketCap'})
    df = df[~df['symbol'].str.contains("\.|\^")]

    if sectors is not None:
        if isinstance(sectors, str):
            sectors = [sectors]
        if not _SECTORS_LIST.issuperset(set(sectors)):
            raise ValueError('Some sectors included are invalid')
        sector_filter = df['sector'].apply(lambda x: x in sectors)
        df = df[sector_filter]

    def cust_filter(mkt_cap):
        if 'M' in mkt_cap:
            return float(mkt_cap[1:-1])
        elif 'B' in mkt_cap:
            return float(mkt_cap[1:-1]) * 1000
        elif mkt_cap == '':
            return 0.0
        else:
            return float(mkt_cap[1:]) / 1e6

    df['marketCap'] = df['marketCap'].apply(cust_filter)
    if mktcap_min is not None:
        df = df[df['marketCap'] > mktcap_min]
    if mktcap_max is not None:
        df = df[df['marketCap'] < mktcap_max]
    return df['symbol'].tolist()
                 
def store_metric(metrics_df):
    global metric_frames
    metric_frames.append(metrics_df)
    
def get_stock_data(symbol, from_date, to_date):
    data = yf.download(symbol, start=from_date, end=to_date)
    df = pd.DataFrame(data=data)
    global stockdata
    # stockdata['Dates']
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
    df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HighLoad', 'Change', 'Volume']]
    # if symbol=="WFC":
    #     print(df)
    return df

def stock_forecasting(df,modelname,stock_symbol):

    forecast_col = 'Close'
    # forecast_out = int(math.ceil(0.01*len(df)))
    forecast_out =1

    # print("Value=",forecast_out)
    df['Label'] = df[[forecast_col]].shift(-forecast_out)
    # df['Label'] = df[[forecast_col]]

    X = np.array(df.drop(['Label'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)
    y = np.array(df['Label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if modelname=='Stepwise':
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    elif modelname=='Svm':
        clf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    elif modelname=='Lasso':
        clf =LassoLars(alpha=.1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
    
    elif modelname=='Ridge':
        clf =Ridge(alpha=.1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    elif modelname=='Boosted':
        clf =GradientBoostingRegressor(random_state=0)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
    else:
        clf = RandomForestRegressor(n_estimators=20, random_state=0)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
    
    forecast = clf.predict(X_forecast)
    metrics_df=[]
    test_predict=clf.predict(X_test)

    bs_error=metrics.mean_absolute_error(y_test,test_predict ) 
    squared_error=metrics.mean_squared_error(y_test,test_predict)
    rms_error=np.sqrt(metrics.mean_squared_error(y_test,test_predict))
    
    metrics_df.append(model)
    metrics_df.append(stock_symbol)
    metrics_df.append(accuracy)
    metrics_df.append(bs_error)
    metrics_df.append(squared_error) 
    metrics_df.append(rms_error)
   
    df['Prediction'] = np.nan
    last_date = df.iloc[-1].name
    # print(last_date)
    last_date = dt.datetime.strptime(str(last_date), "%Y-%m-%d %H:%M:%S")
    pred_arr=[]
    date_field=[]
    for pred in forecast:
        pred_arr.append(pred)
        last_date += dt.timedelta(days=1)
        df.loc[last_date.strftime("%Y-%m-%d")] = [np.nan for _ in range(len(df.columns) - 1)] + [pred]
        # df.loc[last_date.strftime("%Y-%m-%d")] = [date_field.append(last_date.strftime("%Y-%m-%d")) for _ in range(len(df.columns) - 1)] + [pred]
        date_field.append(last_date.strftime("%Y-%m-%d"))

    return df, forecast_out,pred_arr,date_field,metrics_df
#Evaluation Metric generation
# change the model name to generate the specific model's evaluation metric
def metric_calculation(metrics_df,count,total_tickers,model_name):
    global model,accuracy,bs_error,squared_error,rms_error
    global savemetric
    model=metrics_df[0]
    if count<total_tickers:
        accuracy+=metrics_df[2]   
        bs_error+=metrics_df[3] 
        squared_error+=metrics_df[4]
        rms_error+=metrics_df[5]
        
    elif count==total_tickers:
        accuracy+=metrics_df[2]   
        bs_error+=metrics_df[3] 
        squared_error+=metrics_df[4]
        rms_error+=metrics_df[5]
        
        accuracy=round(accuracy/total_tickers,4)
        bs_error=round(bs_error/total_tickers,4)
        rms_error=round(rms_error/total_tickers,4)
        squared_error=round(squared_error/total_tickers,4)
        
        savemetric.append(model)
        savemetric.append(accuracy)
        savemetric.append(bs_error)
        savemetric.append(squared_error)
        savemetric.append(rms_error)
        df=pd.DataFrame(savemetric)
        df_out=df.copy() 
        model,accuracy,bs_error,squared_error,rms_error =0,0,0,0,0
        savemetric=[]
        return df_out
   
def choose_rf(dataframe,rf):
    
    final_model=""
    dataframe_copy=dataframe.copy()
    list_of_models=dataframe.iloc[[0]].values.tolist()
    
    #dropping model names
    dataframe_copy.columns=list_of_models    
    accuracy=dataframe_copy.iloc[[1]]
    rms_error=dataframe_copy.iloc[[4]]
    dataframe_copy.drop(0,axis=0,inplace=True)
    # dataframe_copy= dataframe_copy.sort_values(dataframe_copy.columns[1],ascending=False)
    if(rf>0 and rf<=0.40):
        
        #get the model from the indexes 
        accuracy=accuracy.set_index(pd.Index(["Accuracy"]))
        accuracy=accuracy.sort_values(by="Accuracy",ascending=False,axis=1)
        model_to_choose=accuracy.iloc[0].index.values
        final_model=str(model_to_choose[0])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')


    if (rf>0.40 and rf<=0.60):
        
        #get the model from the indexes 
        accuracy=accuracy.set_index(pd.Index(["Accuracy"]))
        accuracy=accuracy.sort_values(by="Accuracy",ascending=False,axis=1)
        model_to_choose=accuracy.iloc[0].index.values
        final_model=str(model_to_choose[1])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')

    
    if (rf>0.60 and rf<=0.80):
        
        #get the model from the indexes 
        rms_error=rms_error.set_index(pd.Index(["RMS"]))
        rms_error=rms_error.sort_values(by="RMS",ascending=False,axis=1)
        model_to_choose=rms_error.iloc[0].index.values
        final_model=str(model_to_choose[0])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')

    if (rf>0.80 and rf<=1.00):
        
        #get the model from the indexes 
        rms_error=rms_error.set_index(pd.Index(["RMS"]))
        rms_error=rms_error.sort_values(by="RMS",ascending=False,axis=1)
        model_to_choose=rms_error.iloc[0].index.values
        final_model=str(model_to_choose[1])
        final_model=final_model.replace("'",'')
        final_model=final_model.replace("(",'')
        final_model=final_model.replace(")",'')
        final_model=final_model.replace(",",'')

        
    # dataset_to_open=pd.read_csv("static/"+final_model+"Dataset.csv")
    return final_model



def stock_model():

        global risk_score
        rf=float(risk_score)
        Final_metric=pd.read_csv("static/Final_Metric.csv")
        Final_metric=Final_metric.drop(["Unnamed: 0"],axis=1)
        #choose rf needs rf ,and evaluation metric as input
        #return model name  to work on 
        model_name=choose_rf(Final_metric, rf)

           
        df_org=pd.read_csv("static/"+model_name+"Dataset.csv")
        df_org=df_org.drop(["Unnamed: 0"],axis=1)


        # # df=historic_dataframe.copy()
        # #Pass on to the recommendation function
        if (rf>=0.40):
            rf = 0.10
        if (rf>0.40 and rf<=0.60):
            rf = 0.20
        if (rf>0.60 and rf<=0.80):
            rf = 0.30
        if (rf>0.80 and rf<=1.00):
            rf = 0.40
        recommendation_array=recommendation_model(df_org, rf)
        # # return dataframe,risk_factor 
        # #recommendation_array = recommendation_array.iloc[:,0]
        recommendation_array = recommendation_array.reset_index()
        recommendation_array = recommendation_array.iloc[:,1:-1]
        recommendation_array = recommendation_array.to_json(orient = 'records')
        return recommendation_array
       
def recommendation_model(df,rf):
        # df=historic_dataframe.copy()
        date_df=df.reset_index()
        date_df=df['Date'].values.tolist()

        stock=df.iloc[0,:]
        stock=stock.reset_index()
        stock=stock[['index']]
        stock=stock.drop([0,1]).values
        stock=stock.flatten()
        df=df.drop(df.columns[0],axis=1)
        df=df[stock] 

        cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
        corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
# df=pd.DataFrame(date_df,columns=stock,)
        df.insert(0,"Date",date_df)
        df['Date']=pd.to_datetime(df['Date'])
        df=df.set_index("Date")
#Calculate Yearly index 
        ind_er = df.resample('Y').last().pct_change().mean()
#Calculating volatility and Returns 
        ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

# print(ann_sd)
        assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
        assets.columns = ['Returns', 'Volatility']
# print(assets)
        p_ret = [] # Define an empty array for portfolio returns
        p_vol = [] # Define an empty array for portfolio volatility
        p_weights = [] # Define an empty array for asset weights
        num_assets = len(df.columns)
        num_portfolios = 100

        for portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights = weights/np.sum(weights)
            p_weights.append(weights)
            returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
            p_ret.append(returns)
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
            sd = np.sqrt(var) # Daily standard deviation
            ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
            p_vol.append(ann_sd)
        data = {'Returns':p_ret, 'Volatility':p_vol}

        for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
            data[symbol] = [w[counter] for w in p_weights]
        portfolios  = pd.DataFrame(data)
# print(portfolios.head())
        min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
        # rf = 0.01 # risk factor
        
        optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
        # print(optimal_risky_port)
        
        top_opr=optimal_risky_port.to_frame()
        returns=round(top_opr.iloc[0,0],3)*100
        volatility=round(top_opr.iloc[1,0],3)*100
        top_opr=top_opr.drop(['Returns','Volatility'])
        top_opr_copy=top_opr.copy()
        top_opr_copy=top_opr.reset_index()
        
        val=len(top_opr_copy)
        for i in range(0,val):
            top_opr_copy.iloc[i,1]= top_opr_copy.iloc[i,1]*100
            top_opr_copy.iloc[i,1]=round(top_opr_copy.iloc[i,1],3)
            i+=1

        top_opr_copy=top_opr_copy.sort_values(top_opr_copy.columns[1],ascending=False)
        #need to append sentimental analysis
        # top_opr_copy.to_csv("static/tempfinal.csv")
        top_opr_copy=top_opr_copy.reset_index()
        # temp_final=pd.read_csv("static/tempfinal.csv")
        temp_final=top_opr_copy
        sentiment=pd.read_csv("static/Sentiment_scores.csv")
        sentiment=sentiment.T
        sentiment=sentiment.reset_index()
       
        temp_stock_tickers=temp_final.iloc[:,1].values.tolist()
        company_list=pd.read_csv("static/companylist.csv")
        company_list=company_list[company_list['Symbol'].isin(temp_stock_tickers)]
        # for each_stock in recommendation_tickers:
        #     s_counter=0
        #     for sentiment_stock in sentiment_tickers:
        final_arr=[]
        for i in range(0,len(temp_final)):
           
           for j in range(0,len(sentiment)):
             
               if temp_final.iloc[i,1]==sentiment.iloc[j,0]:
                   ticker=temp_final.iloc[i,1]
                   # print("Hi")
                   recomendation_score=temp_final.iloc[i,2]
                   neutral_score=sentiment.iloc[j,1]
                   positive_score=sentiment.iloc[j,2]
                   negative_score=sentiment.iloc[j,3]
                   flag=0
                   for k in range(0,len(company_list) ):

                       if temp_final.iloc[i,1] == company_list.iloc[k,0]:
                           cname=company_list.iloc[k,1]
                           flag=1
                           break
                       else :
                           cname="Not Available"
                
                   new_entry={'Stock_Ticker':ticker,
                              'Organization':cname,
                              'Recommended_Perc':str(recomendation_score),
                              'Neutral_Mentality':str(neutral_score),
                              'Positive_Mentality':str(positive_score),
                              'Negative_Mentality':str(negative_score)}
                   break
                  
                          
    
               elif  temp_final.iloc[i,1]!=sentiment.iloc[j,0]:
                   ticker=temp_final.iloc[i,1]
                   recomendation_score=temp_final.iloc[i,2]
                   neutral_score="0"
                   positive_score="0"
                   negative_score="0"
                   for k in range(0,len(company_list) ):
                      if temp_final.iloc[i,1] == company_list.iloc[k,0]:
                           cname=company_list.iloc[k,1]
                           flag=1
                           break
                      else :
                           cname="Not Available" 
                   new_entry={'Stock_Ticker':ticker,
                              'Organization':cname,
                              'Recommended_Perc':str(recomendation_score),
                              'Neutral_Mentality':str(neutral_score),
                              'Positive_Mentality':str(positive_score),
                              'Negative_Mentality':str(negative_score)}

               
           final_arr.append(new_entry.copy())
        Final_Recommendation=pd.DataFrame(final_arr)

                   
 
        

        return Final_Recommendation


@application.route('/')
def index():
    return render_template('index.html')

@application.route('/signup.html')
def signup():
    return render_template('signup.html')

@application.route('/signin.html', methods=["GET", "POST"])
def signin():
    global risk_score 
    if(request.method == "POST"):
        risk_score = request.form['risk_score']
        message = {'message': 'hello'}

        return message
    return render_template('signin.html')

@application.route('/userInfoForm.html')
def userInfoForm():
    return render_template('userInfoForm.html')
  
@application.route('/dashboard.html')
def dashboard():
    user_risk_calculator()
    return render_template('dashboard.html')

@application.route('/similar_users.html')
def similar_users():
    return render_template('similar_users.html')

@application.route('/recommendations.html')
def recommendations():
    df = stock_model()
    return render_template('recommendations.html', data = df)


if __name__ == '__main__':
    application.run(debug=True)
