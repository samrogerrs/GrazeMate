from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_status():
    data = {
        "status": {
            "pitch": -10.0,
            "roll": 0.0,
            "rollPitchControlMode": "ANGLE",
            "rollPitchCoordinateSystem": "BODY",
            "verticalControlMode": "VELOCITY",
            "verticalThrottle": 0.0,
            "yaw": 0.0,
            "yawControlMode": "ANGULAR_VELOCITY"
        }
    }
    return jsonify(data)  # Flask will automatically convert it to valid JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Change port if needed
