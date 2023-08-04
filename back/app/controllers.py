"""controllers.py


CONTROLADORES
--------------------------------------------------------------------------------
Contém funções que interagem com as rotas definidas em routes.py e fazem a 
lógica de negócios da aplicação.
"""
from app import app
from app.models import User
from flask import jsonify

def get_user_by_id(user_id):
    user = User(46876, 'Danilo')
    if user:
        return jsonify({'id': user.idx, 'username': user.name})
    else:
        return jsonify({'message': 'Usuário não encontrado'}), 404
