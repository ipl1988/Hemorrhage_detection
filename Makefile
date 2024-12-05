run_api:
	uvicorn api.main:app --reload

run_streamlit:
	streamlit run interface_v2.py
