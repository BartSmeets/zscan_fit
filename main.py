import streamlit as st

st.title('Z-Scan')
st.markdown('''This project has been part of my thesis for the degree of [Master of Science in Nanoscience, Nanotechnology and Nanoengineering](https://www.kuleuven.be/programmes/master-nanoscience-nanotechnology-nanoengineering) at KU Leuven. 
            Read the [thesis report](/docs/MasterThesis_BartSmeets_final.pdf) for methodology of the measurment and documentation of the code.''')

st.header('Pages', anchor=False)
st.page_link('pages/1_Normalise_Data.py', label='Normalise Data')
st.page_link('pages/2_Beam_Profile.py', label='Beam Profile Fitting')
st.page_link('pages/3_Z-Scan_Fit.py', label='Z-Scan Fit')