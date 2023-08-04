"""routes.py


ROTAS
--------------------------------------------------------------------------------
Aqui estão as rotas da API e a especificação das funções que serão executadas 
quando cada rota for acessada.
"""
from app import app
from flask import jsonify

# Home
@app.route('/')
def index():
    return 'API em Flask funcionando!'