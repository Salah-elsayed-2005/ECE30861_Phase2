from mangum import Mangum
from src.api.routes import app

handler = Mangum(app)
