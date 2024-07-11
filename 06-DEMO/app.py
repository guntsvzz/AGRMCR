import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Configuration", "Chat Interface"])

if page == "Configuration":
    st.title("Configuration Page")
    st.header("Select a Domain")
    domain = st.selectbox("Domain", ["Beauty", "CDs", "Cellphones", "Clothing"])
    st.write(f"You selected the domain: {domain}")

elif page == "Chat Interface":
    st.title("Chat Interface")
    st.header("Enter your message below:")
    
    user_input = st.text_input("Your Message:")
    if user_input:
        st.write(f"You said: {user_input}")
