run_api:
	uvicorn api.main:app --reload

run_streamlit:
	streamlit run interface_inspiration.py
  streamlit run interface_pretty.py
