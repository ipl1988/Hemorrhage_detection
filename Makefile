run_api:
	uvicorn api.main:app --reload

run_streamlit:
	streamlit run interface.py
