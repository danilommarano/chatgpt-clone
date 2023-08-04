"""models.py


MODELOS
--------------------------------------------------------------------------------
Contém a definição dos modelos de dados, que representam as tabelas do banco de
dados ou objetos são usados na API.
"""

class User():

    def __init__(self, idx:int, name:str) -> None:
        self.idx = idx
        self.name = name

    def __repr__(self) -> str:
        return f'<User {self.idx}, {self.name}'