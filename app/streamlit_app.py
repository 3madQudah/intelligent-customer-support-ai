import streamlit as st
from pipeline import run_pipeline

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Intelligent Customer Support AI")
st.markdown("""
This system uses **BERT + GPT (LoRA)** to automatically  
classify customer feedback and generate support responses.
""")

review = st.text_area(
    "📝 Enter Customer Review",
    height=150,
    placeholder="e.g. I was charged twice for my order..."
)

if st.button("🚀 Generate Response"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing review..."):
            category, response = run_pipeline(review)

        st.success("Analysis Complete!")

        st.subheader("🏷 Predicted Category")
        st.write(category)

        st.subheader("🤖 Support Response")
        st.write(response)