import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DATA_URL = 'https://raw.githubusercontent.com/starlight-coders/Course_Work_Data_Presentation/refs/heads/main/TB_dr_surveillance_2026-02-17.csv'
    DATA_CACHE_FILE = 'data/tb_data_cache.csv'
    DEBUG = os.environ.get('FLASK_DEBUG', True)
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))