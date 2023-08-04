"""settings.py


CONFIGURAÇÔES
--------------------------------------------------------------------------------
Configurações gerais da aplicação, como as configurações do banco de dados, 
chaves secretas, etc.
"""

import os

# Configurações do banco de dados
basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Outras configurações
SECRET_KEY = 'sua_chave_secreta_aqui'
DEBUG = True
