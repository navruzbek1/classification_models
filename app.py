import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform
#plotly vazifasi grafikniu chiroyli qilib chizib berish
plt = platform.system()
if plt=='Linux':pathlib.WindowsPath = pathlib.PosixPath
st.title("Obyektni klassifikatsiya qiladigan dastur![Pineapple,Orange,Aplle,Peach]")


#rasmni joylash
file = st.file_uploader("Rasm yuklash",type=['png','jpeg','gif','svg','jpg'])
if file:
    img = PILImage.create(file)
    #model
    model = load_learner("fruits.pkl")
    #predict

    model.predict(img)

    #predict

    pred,pred_id,prob = model.predict(img)
    st.image(file)
    st.success(pred)
    st.info(f'Ehtimollik: {prob[pred_id]*100:.1f}%')
    
    
    #plotling visualization qismini tahlab olamiz
    fig = px.bar(x=prob*100,y=model.dls.vocab)
    st.plotly_chart(fig)
