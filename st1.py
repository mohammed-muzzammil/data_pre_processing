import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import base64
from io import BytesIO
import matplotlib.pyplot as plt




# Headings

st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("<h1 style='text-align: center; color: Light gray;'>Data Preprocessing App</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Data import</h3>", unsafe_allow_html=True)












    
    
    
    
    
    
    
    # All Functions
def mvt_mean(df):
        
    x = df.iloc[:,0:].values
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imp.fit(x[:,1:])
    
    x[:,1:] = imputer.transform(x[:,1:])
    df=pd.DataFrame(x)
    st.sidebar.write("Processed file")
    st.dataframe(df) 
    st.line_chart(df)
    return df
        
        
        
        
        
    
def get_table_download_link_csv(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframes
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'



def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer)
    writer.save()
    processed_data = output.getvalue()
    return processed_data



def get_table_download_link_xlsx(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="dataprep.xlsx">Download xlsx file</a>' # decode b'abc' => abc



    
    
    
        
    
        
        
        

    
    
    

    
                
                
            
                    
# MVT Options 


def mvt_options(df):
    
    optionm=("Mean","Median","Mode","K NN Imputer")
    mvt_selection=st.sidebar.radio('Choose a Missing Value Treatment Method',optionm)
    if mvt_selection == 'Mean':
        st.sidebar.write('you selected mean')
        if st.sidebar.button('Process Mean'):
            df = pd.read_csv(r'C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit\temp.csv')
            df=mvt_mean(df)
            df.to_csv(r'C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit\temp.csv', index=False)
            return df
            #st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            
    elif mvt_selection == 'Mode':
        st.sidebar.write('You selected mode')
        if st.sidebar.button('Process Mode'):
            st.sidebar.write('Pressed')
            
            
            
    elif mvt_selection == 'Median':
        st.sidebar.write('You selected Median')
        if st.sidebar.button('Process Median'):
            st.sidebar.write('Pressed')
                
                
                
                
    elif mvt_selection == 'K NN Imputer':
        st.sidebar.write('You selected K NN Imputer')
        if st.sidebar.button('Process K NN'):
            st.sidebar.write('Pressed')
            
            
            

# Outliers Function


def outlier_function():
    st.sidebar.write('you selected outlier treatment')
    df = pd.read_csv(r'C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit\temp.csv')
    
    if st.sidebar.button('Process Outliers'):
        st.sidebar.write('Pressed')
        return 0
        
        
    
    

    
# feature Scaling options



def fso(df):
    
    fs_option=("Standard Scalar","Min Max Scalar", "Max Absolute Scalar" , "Robust Scalar")
    fs_selection=st.sidebar.radio('Choose a Feature Scaling Method',fs_option)
    
    
    if fs_selection == 'Standard Scalar':
        st.sidebar.write('you selected Standard Scalar')
        if st.sidebar.button('Process SS'):
            st.sidebar.write('Pressed')
            return 0
            
            
    elif fs_selection == 'Min Max Scalar':
        st.sidebar.write('you selected min max')
        if st.sidebar.button('Process mm'):
            st.sidebar.write('Pressed')
            
            
    elif fs_selection == 'Max Absolute Scalar':
        st.sidebar.write('You selected max absolute')
        if st.sidebar.button('Process Ma'):
            st.sidebar.write('Pressed')
            
            
    elif fs_selection == 'Robust Scalar':
        st.sidebar.write('You selected Robust Scalar')
        if st.sidebar.button('Process rs'):
            st.sidebar.write('Pressed')
    
    
    
    
    
    
        
    
    
        
    
    
    
    
            
            
    
            
            
            
            
            
            
# File Upload

def file_upload():
    
    f_option=('.Xlsx','.Csv','Oracle')
    f_select=st.sidebar.radio('Choose a file type',f_option)

    if f_select == '.Xlsx':
    
        uploaded_file = st.sidebar.file_uploader("Choose a file", type="xlsx")

        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.dataframe(df)
            return df
        
    elif f_select == '.Csv':
        uploaded_file = st.sidebar.file_uploader("Choose a file", type="csv")
    
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            return df
        

        
    
        

# Data export

def data_export(df):
    
        
    st.sidebar.markdown("<h3 style='text-align: left; color: black;'>Data Export</h3>", unsafe_allow_html=True)
    fd_option=('.Xlsx','.Csv','Oracle')
    fd_select=st.sidebar.radio('Choose a file type to download',fd_option)
    
    if fd_select == '.Csv':
        if st.sidebar.button('Download Csv'):
            df = pd.read_csv(r'C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit\temp.csv')
            
            st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
            return 0
            
    elif fd_select == '.Xlsx':
        if st.sidebar.button('Download Xlsx'):
            df = pd.read_csv(r'C:\Users\MOHAMMED MUZZAMMIL\Desktop\streamlit\temp.csv')
            st.sidebar.markdown(get_table_download_link_xlsx(df), unsafe_allow_html=True)
            return 0

            
    elif fd_select == 'Oracle':
        if st.sidebar.button('Download Oracle'):
            st.sidebar.write('Oracle')
            
            
            

# Give main options


def main_option():
    
    option=('Missing Value Treatment', 'Outlier Treatment', 'Feature Scaling')
    
    option_select = st.sidebar.radio('What would you like to do?',option)
    
    return option_select
    
    
    
                        

def main():
    df=file_upload()
        
    m_option = main_option()
    
    if m_option == 'Missing Value Treatment':
        df=mvt_options(df)
        
        
    elif m_option == 'Outlier Treatment':
        outlier_function()
        
    elif m_option == 'Feature Scaling':
        fso(df)
        
        
    data_export(df)
    


main()
        
        

        
    
    
    
            
            
            
            
            
            
            
            
            
            

    
    
    
    
    
    
    

# BOT   


#st.sidebar.title("Need Help")

#get_text is a simple function to get user input from text_input
#def get_text():
 #   input_text = st.sidebar.text_input("You: ","So, what's in your mind")
  #  return input_text
    
    


    
#if st.sidebar.button('Initialize bot'):
 #   st.sidebar.title("Your bot is ready to talk to you")
  #  st.sidebar.title("""
   #         Help Bot  
    #    Just paste here the error you are getting 
     #       """)
    #user_input = get_text()
    #st.sidebar.write('Hello')*/

#if True:
 #   st.sidebar.text_area("Bot: Hello")
    
    
    
    
    
    
        
        
    
    
    
    
    
    