import pandas as pd
import streamlit as st

from rag.langchain import inspect_vector_store
from rag.langchain import get_vector_store_info 

st.set_page_config(
    page_title="Knowledge Base",
    page_icon="🧠",
)

st.title("Knowledge Base")

st.subheader("Visualiser les informations contenues dans la base de connaissances")

st.table(pd.DataFrame.from_dict(get_vector_store_info(), orient='index').transpose())

docs_df = inspect_vector_store(100)
st.dataframe(docs_df, width=1000, use_container_width=False, hide_index=True,
             column_config={
                 'id': None,
                 'insert_date': st.column_config.DatetimeColumn('Insert Date', format="YYYY-MM-D")
                 }
            )
