"""services.py


SERVIÇOS
--------------------------------------------------------------------------------
Funções lógicas da aplicação. Esse arquivo serve para manter o código do 
controller mais enxuto.
"""

# Exemplo de função de serviço que pode ser usada no controller
def is_username_available(username):
    db = ['Danilo', 'Adriana', 'Raissa']
    return username in db