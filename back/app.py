from app import app
from config import settings

if __name__ == '__main__':

    # Inicia a aplicação Flask
    app.run(host='0.0.0.0', port=5000, debug=settings.DEBUG)