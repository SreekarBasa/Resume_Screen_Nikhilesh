import streamlit as st
import pickle
import re
import nltk

from PIL import Image

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))


def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def main():

    st.set_page_config(page_title="AI Resume Analyzer", page_icon=":file_pdf:")
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #001f3f;
            color: #ffffff;
            padding: 20px;
            font-family: 'Arial', sans-serif;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            border-radius: 4px;
            transition: all 0.4s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .stButton > button::before {
            content: "";
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300%;
            height: 300%;
            background: rgba(255, 255, 255, 0.15);
            transition: all 0.75s ease;
            border-radius: 50%;
            z-index: 0;
            transform: translate(-50%, -50%) scale(0);
        }
        .stButton > button:hover::before {
            transform: translate(-50%, -50%) scale(1);
        }
        .stButton > button:hover {
            background-color: #0056b3;
            color: white;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
            transform: scale(1.05);
        }
        .stTitle {
            color: #ffffff;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            margin-top: 17px
        }
        .stSubtitle {
            font-size: 18px;
            color: #cccccc;
        }
        .stSuccess {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 4px;
        }
        .stMarkdown {
            color: #ffffff;
        }
        .push-button-3d {
            box-shadow: 0 4px #c1a23c;
            color: #5e4800;
            background-color: #ffd95e;
            text-transform: uppercase;
            padding: 10px 20px;
            border-radius: 5px;
            transition: all .2s ease;
            font-weight: 900;
            cursor: pointer;
            letter-spacing: 1px;
        }
        .push-button-3d:active {
            box-shadow: 0 1px #c1a23c;
            transform: translateY(3px);
        }
        .uploadFileText {
            color: #ffffff;
            margin-top: 50px;
            margin-bottom: 5px;            
        }        
        </style>
        """,
        unsafe_allow_html=True
    )

    

    # Header section with logo
    col1, col2 = st.columns([2, 8])
    with col1:
        img = Image.open('./Images/robot_icon.png')
        img = img.resize((100, 100))
        st.image(img)
    with col2:
        st.markdown("<div class='stTitle'>AI Resume Analyzer</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='stSubtitle'>
            This application is designed to assist you in categorizing resumes based on keywords. 
            Upload your resume file (TXT or PDF format) and it will predict the most relevant category. 
            This can be a helpful tool for initial resume screening during the recruitment process, 
            saving you time and effort in sorting through a large number of applications.
        </div>
    """, unsafe_allow_html=True)

    # File upload section
    st.markdown('<div class="uploadFileText">Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader('', type=['txt', 'pdf'], key="resume_upload")

    #st.markdown('<button class="push-button-3d">Analyze Resume</button>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner('Analyzing your resume...'):
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')

            cleaned_resume = clean_resume(resume_text)
            input_features = tfidfd.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            # Result section with enhanced styling, spacing, and visual elements
            st.markdown(f"<div class='stSuccess'>This resume is a strong fit for the <b>{category_name}</b> role!</div>", unsafe_allow_html=True)
            st.markdown(f"<b>Unleash your potential as a {category_name}!</b>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
