import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autoint import AutoIntModel, predict_model

# Streamlit 애플리케이션 실행 명령: streamlit run show_st.py 

@st.cache_resource
def load_data():
    '''
    앱에서 필요한 데이터를 불러오는 함수
    - 사용자, 영화, 평점 데이터 로드
    - 저장된 모델 및 관련 객체 불러오기
    '''
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = f"{project_path}/{data_dir_nm}"
    model_path = f"{project_path}/{model_dir_nm}"
    field_dims = np.load(f'{data_path}/field_dims.npy')
    dropout = 0.4
    embed_dim = 16
    
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    users_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')
    
    model = AutoIntModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
                         l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001)
    
    # 모델 입력을 numpy array로 변환
    input_data = np.array([[0 for _ in range(len(field_dims))]], dtype=np.int32)
    model(input_data)  # 모델을 초기화
    model.load_weights(f'{model_path}/autoInt_model.weights.h5')
    
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
    return users_df, movies_df, ratings_df, model, label_encoders

def get_user_seen_movies(ratings_df):
    '''사용자가 시청한 영화 목록을 반환'''
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seen_dict(movies_df, users_df, user_seen_movies):
    '''사용자가 보지 않은 영화 목록을 반환'''
    unique_movies = movies_df['movie_id'].unique()
    user_non_seen_dict = {}
    for user in users_df['user_id'].unique():
        seen_movies = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
        user_non_seen_dict[user] = list(set(unique_movies) - set(seen_movies))
    return user_non_seen_dict

def get_user_info(user_id, users_df):
    '''사용자 정보를 반환'''
    return users_df[users_df['user_id'] == user_id]

def get_user_past_interactions(user_id, ratings_df, movies_df):
    '''사용자가 4점 이상 준 영화 반환'''
    return ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')

def get_recom(user, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders):
    '''추천 영화 목록을 반환'''
    user_non_seen_movie = user_non_seen_dict.get(user)
    user_id_list = [user] * len(user_non_seen_movie)
    r_decade = str(r_year - (r_year % 10)) + 's'
    
    user_non_seen_movie = pd.merge(pd.DataFrame({'movie_id': user_non_seen_movie}), movies_df, on='movie_id')
    user_info = pd.merge(pd.DataFrame({'user_id': user_id_list}), users_df, on='user_id')
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade
    
    merge_data = pd.concat([user_non_seen_movie, user_info], axis=1).fillna('no')
    merge_data = merge_data[['user_id', 'movie_id', 'movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1', 'genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']]
    
    for col, le in label_encoders.items():
        merge_data[col] = le.transform(merge_data[col])
    
    recom_top = predict_model(model, merge_data)
    origin_m_id = label_encoders['movie_id'].inverse_transform([r[0] for r in recom_top])
    return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# 데이터 로드
users_df, movies_df, ratings_df, model, label_encoders = load_data()
user_seen_movies = get_user_seen_movies(ratings_df)
user_non_seen_dict = get_user_non_seen_dict(movies_df, users_df, user_seen_movies)

# Streamlit UI 구성
st.title("영화 추천 시스템")
st.header("사용자 정보를 입력하세요")
user_id = st.number_input("사용자 ID", min_value=users_df['user_id'].min(), max_value=users_df['user_id'].max(), value=users_df['user_id'].min())
r_year = st.number_input("추천 연도", min_value=ratings_df['rating_year'].min(), max_value=ratings_df['rating_year'].max(), value=ratings_df['rating_year'].min())
r_month = st.number_input("추천 월", min_value=ratings_df['rating_month'].min(), max_value=ratings_df['rating_month'].max(), value=ratings_df['rating_month'].min())

if st.button("추천 영화 보기"):
    st.subheader("사용자 정보")
    st.dataframe(get_user_info(user_id, users_df))
    
    st.subheader("과거 시청 이력 (평점 4점 이상)")
    st.dataframe(get_user_past_interactions(user_id, ratings_df, movies_df))
    
    st.subheader("추천 영화")
    st.dataframe(get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders))
