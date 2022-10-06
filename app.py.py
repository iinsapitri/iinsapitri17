import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt

st.write("""
# KneeMRI Detection App

MRI (Magnetic Resonance Imaging) lutut. tes pencitraan diagnostik non 
invasif yang dilakukan untuk menganalisis kondisi berbagai bagian lutut, 
seperti jaringan, tendon, sendi, otot, tulang, dan ligamen.

Aplikasi ini mendeteksi kemungkinan terjadinya cedera Robekan Anterior Cruciate Ligament (ACL) 
yang biasa dialami oleh atlet papan atas dalam sepak bola atau bola basket. 
Cedera ini juga kerap timbul pada usia 40 tahun keatas.

""")

url_dataset = f'<a href="kneeMRI.csv">Download Dataset CSV File</a>'
st.markdown(url_dataset, unsafe_allow_html=True)

def user_input_features() :
    kneeLR = st.sidebar.selectbox('Knee', ('Left', 'Right'))
    roiX = st.sidebar.slider('RoiX', 71,146)
    roiY = st.sidebar.slider('RoiY', 22,184)
    roiZ = st.sidebar.slider('RoiZ', 9,22)
    roiHeight = st.sidebar.slider('RoiHeight', 62,124)
    roiWidth = st.sidebar.slider('RoiWidth', 61,136)
    roiDepth  = st.sidebar.slider('RoiDepth ', 2,6)
    kneeLR01 = 1
    if(kneeLR == 'Left') :
        kneeLR01 = 0
    data = {'kneeLR':[kneeLR01], 
    'roiX':[roiX],
    'roiY':[roiY],
    'roiZ':[roiZ],
    'roiHeight':[roiHeight],
    'roiWidth':[roiWidth],
    'roiDepth':[roiDepth],}

    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

knee_raw = pd.read_csv('kneeMRI.csv')
knee_raw.fillna(0, inplace=True)
knee = knee_raw.drop(columns=['aclDiagnosis'])
df = pd.concat([input_df, knee],axis=0)

df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)

features = ['kneeLR', 'roiX', 'roiY', 'roiZ', 'roiHeight',
       'roiWidth', 'roiDepth']

df = df[features]


st.subheader('User Input features')
st.write(df)

load_clf = pickle.load(open('kneeMRI_clf.pkl', 'rb'))
detection = load_clf.predict(df)
detection_proba = load_clf.predict_proba(df)
knee_labels = np.array(['Healthy', 'Partially injured', 'Completely ruptured'])
st.subheader('Detection')
st.write(knee_labels[detection])
st.subheader('Detection Probability')
df_prob = pd.DataFrame(data=detection_proba, index=['Probability'], columns=knee_labels)
st.write(df_prob)