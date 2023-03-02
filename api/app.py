from flask import Flask
from flask import jsonify, request
from flask_cors import CORS
import torch
import numpy as np

import numpy as np
from api.game_func import init_board, shift_down, shift_left, shift_right, shift_up
from game.brain import FFN


app = Flask(__name__)
CORS(app)

@app.route("/init", methods=["GET"])
def init_game(start_point=0):
    board = init_board()
    board = board.tolist()

    response = jsonify([board, start_point])
    response.status = 200
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/down", methods=["POST"])
def down():
    board = [[int(i) for i in row] for row in request.json["board"]]

    board, points = shift_down(np.array(board))
    response = jsonify({
        "board": board.tolist(),
        "points": int(points)
    })
    response.status = 200
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/up", methods=["POST"])
def up():
    board = [[int(i) for i in row] for row in request.json["board"]]

    board, points = shift_up(np.array(board))
    response = jsonify({
        "board": board.tolist(),
        "points": int(points)
    })
    response.status = 200
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/right", methods=["POST"])
def right():
    board = [[int(i) for i in row] for row in request.json["board"]]

    board, points = shift_right(np.array(board))
    response = jsonify({
        "board": board.tolist(),
        "points": int(points)
    })
    response.status = 200
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/left", methods=["POST"])
def left():
    board = [[int(i) for i in row] for row in request.json["board"]]

    board, points = shift_left(np.array(board))
    response = jsonify({
        "board": board.tolist(),
        "points": int(points)
    })
    response.status = 200
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/model", methods=["POST"])
def get_model_sugestion():
    model = FFN(16, 4, hidden_size=100)
    model.load_state_dict(torch.load("model_checkpoints/2023_02_26_20_22.pt"))

    board = [[int(i) for i in row] for row in request.json["board"]]
    board_tensor = torch.tensor(np.array(board).flatten().tolist(), dtype=torch.float32).unsqueeze(dim=0)
    q_values = model(board_tensor)[0].tolist()

    response = jsonify({
        "q_values": q_values
    })
    response.status = 200
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)