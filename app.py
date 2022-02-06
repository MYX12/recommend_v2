from operator import imod
from flask import *
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app) 
        
@app.route('/podcast', methods=['GET'])
def recommend_podcast():
    res = recommendation.content_based_recommender(request.args.get('title'))
    return jsonify(res)

if __name__=='__main__':
    app.run(port = 5000, debug = True)