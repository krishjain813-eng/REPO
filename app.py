
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

clf = pickle.load(open("clf.pkl","rb"))
reg = pickle.load(open("reg.pkl","rb"))

st.title("Fintech Advanced Dashboard")

data = pd.read_csv("dataset.csv")

st.subheader("Dataset Snapshot")
st.write(data.head())

# charts
st.subheader("Revenue Distribution")
fig = px.histogram(data, x="revenue")
st.plotly_chart(fig)

st.subheader("Interest Distribution")
fig2 = px.pie(data, names="interest")
st.plotly_chart(fig2)

# feature importance
importances = clf.feature_importances_
features = data.drop(["interest","loan_size","cluster"], axis=1).columns
imp_df = pd.DataFrame({"feature":features,"importance":importances})

fig3 = px.bar(imp_df, x="feature", y="importance")
st.plotly_chart(fig3)

# upload
uploaded = st.file_uploader("Upload new customer data")

if uploaded:
    df_new = pd.read_csv(uploaded)
    pred = clf.predict(df_new)
    prob = clf.predict_proba(df_new)[:,1]
    loan = reg.predict(df_new)

    df_new["prediction"] = pred
    df_new["probability"] = prob
    df_new["loan"] = loan

    st.write(df_new)
