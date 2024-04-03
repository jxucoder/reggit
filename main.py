import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from openai import OpenAI
import pandas as pd
import altair as alt


st.set_page_config(layout="wide")


@st.cache_resource
def create_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets['GOOGLE_APPLICATION_CREDENTIALS_REGGIT']
    )
    client = bigquery.Client(credentials=credentials)
    return client


@st.cache_resource
def create_openai_client():
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    return client


@st.cache_data(ttl='1h')
def run_query(query):
    client = create_client()
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [dict(row) for row in rows_raw]
    return rows


@st.cache_data
def openai_chat(question):
    openai_client = create_openai_client()
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model="gpt-4",
    )
    return chat_completion.choices[0].message.content


st.sidebar.title("Reggit")
document_id = st.sidebar.selectbox(label="Document ID", options=['FINCEN-2024-0009-0001'])
rows = run_query(f"""
    WITH RankedComments AS (
        SELECT 
            cs.*,
            cg.*,
            ROW_NUMBER() OVER(PARTITION BY cs.commentId ORDER BY cs.batch_updated_time DESC) as rn
        FROM `reggit.regulations_gov.comments` cs
        LEFT JOIN `reggit.regulations_gov.comments_genai` cg 
            ON cs.commentId = cg.commentId
        WHERE cs.commentOnDocumentId='{document_id}' AND prompt_version = 'v5'
    )
    SELECT *
    FROM RankedComments
    WHERE rn = 1;
""")
df = pd.DataFrame.from_records(rows)
document_title = df['document_title'].iloc[0]
st.sidebar.markdown(f"**Document**: [{df['document_title'].iloc[0]}](https://www.regulations.gov/document/{document_id})")
df.sort_values(['posted_date'], inplace=True, ascending=False)


start_time = st.sidebar.slider(
    "Comments Posted After",
    value=df['posted_date'].min().date(),
    min_value=df['posted_date'].min().date(),
    max_value=df['posted_date'].max().date(),
    format="MM/DD/YY")


df_filtered = df[pd.to_datetime(df['posted_date']) >= pd.to_datetime(start_time)]

series = df_filtered.sentiment
df_counts = series.value_counts().reset_index()
df_counts.columns = ['Sentiment', 'Counts']

# Sorting values for better visualization
df_counts = df_counts.sort_values(by='Counts', ascending=True)
color_scale = alt.Scale(domain=['oppose', 'support'],
                        range=['red', 'green'])
# Create a horizontal bar chart using Altair with narrower bars
chart = alt.Chart(df_counts).mark_bar(size=20).encode(  # Adjust size here
    x='Counts',
    y=alt.Y('Sentiment', sort='-x'),  # Sort bars by Counts
    color=alt.Color('Sentiment', scale=color_scale)
)
st.altair_chart(chart, use_container_width=True)



col_left, col_right = st.columns(2)

for col, direction, title in [(col_left, 'oppose', '❌'), (col_right, 'support', '✅')]:
    with col:
        st.markdown(f"<h1 style='text-align: center;'> {title} </h1>", unsafe_allow_html=True)
        for i, row in df_filtered[df_filtered.sentiment == direction].iterrows():
            # Create an expander for each piece of text. Note: `expanded=True` makes it open by default.
            with st.expander(f"{row['title']}", expanded=True):
                # Render the HTML content inside the expander
                st.markdown(row['comment'], unsafe_allow_html=True)

support_comments = '\n --- \n'.join(df[df.sentiment == "support"].comment.values)
oppose_comments = '\n --- \n'.join(df[df.sentiment == "oppose"].comment.values)

insights = openai_chat(f"""
Your job is to analyze public comments posted related to Regulations.gov document: {document_title}. 
Please answer
(1) what are the main arguments for opposing it?
(2) what are the main arguments for supporting it?
(3) any concrete ideas and suggestions?

Before answering, do the following:
(1) Read the supporting views below:
{support_comments}
(2) Read the oppose views below:
{oppose_comments}
""")

with st.sidebar:
    with st.spinner('Wait for Insights to be generated...'):
        with st.expander("Insights", expanded=True):
            # Render the HTML content inside the expander
            st.markdown(insights, unsafe_allow_html=True)
