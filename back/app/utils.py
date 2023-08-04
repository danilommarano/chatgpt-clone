"""utils.py


ÚTEIS
--------------------------------------------------------------------------------
Funções utilitárias, que podem ser usadas em várias partes do código.
"""

# Exemplo de função utilitária
def generate_random_string(length=10):
    import string
    import random
    choices = random.choices(string.ascii_letters + string.digits, k=length)
    return ''.join(choices)
