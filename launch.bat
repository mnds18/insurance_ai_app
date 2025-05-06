@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set STREAMLIT_FILE_WATCHER_TYPE=none
streamlit run ui/app.py
