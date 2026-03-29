from flask import Flask, jsonify
from flask_cors import CORS  # Import this
from main import get_Cordinates

app = Flask(__name__)
CORS(app)  # This line enables CORS for all routes

@app.route('/api/data')
def get_data():
    try:
        raw_coords = get_Cordinates()
        return jsonify({
            "status": "success",
            "data": raw_coords
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)