import pandas as pd
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

myzip=ZipFile('episodes-sample.zip')
f=myzip.open('episodes-sample.csv')
data=pd.read_csv(f)
df_use=data.iloc[:10000,:]


tfidf_vector = TfidfVectorizer(stop_words='english')
df_use['description'] = df_use['description'].fillna('')

tfidf_matrix=tfidf_vector.fit_transform(df_use['description'])

sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)

indices=pd.Series(df_use.index,index=df_use['title']).drop_duplicates()

def content_based_recommender(title,sim_scores=sim_matrix):
    idx = indices[title]
    
    sim_scores=list(enumerate(sim_matrix[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    
    podcast_indices=[i[0] for i in sim_scores]
    
    podcast_title=df_use['title'].iloc[podcast_indices]
    recommendation_data=pd.DataFrame(columns=['title'])
    
    recommendation_data['title']=podcast_title
    
    recommendation_data=recommendation_data.to_dict('records')
    
    return recommendation_data

a=content_based_recommender('Auto Harp')
print(a)
