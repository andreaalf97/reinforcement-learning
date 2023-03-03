default_backgrounds = {
    "2": "#dad9bb",
    "4": "#e6e38e",
    "8": "#f49236",
    "16": "#f44d36",
    "32": "#ff1f00",
    "64": "#fff800",
    "128": "#afffff",
    "256": "#619aff",
    "512": "#e485f1",
    "1024": "#b5ffc0",
    "2048": "#00ff26"
}

function read_cached_board(){
    tmp = localStorage['board'].split('|');
    board = [];
    for (let i = 0; i < tmp.length; i++){
        board.push(
            tmp[i].split(',')
        );
    }
    return board
}

function update(board, points, board_id, points_id){
    document.getElementById(points_id).innerHTML = "[" + localStorage['n_moves'] + "] " + points

    var board_html = document.getElementById(board_id)

    rows = board_html.getElementsByTagName("tr")

    for (let i = 0; i < board.length; i++){
        row = rows[i];
        for (let j = 0; j < board[i].length; j++){
            var element = row.getElementsByTagName("td")[j]
            if (board[i][j] != 0){
                element.innerHTML = board[i][j];
            }
            else {
                element.innerHTML = "";
            }
            bg_color = default_backgrounds[board[i][j]] ?? "white";
            element.setAttribute("style", "background-color: " + bg_color);
            
        }
    }

}

function init(){
    fetch("http://127.0.0.1:5000/init")
    .then(x => x.json())
    .then(json => {
        var board = json[0];
        var points = json[1];

        localStorage['n_moves'] = 0

        localStorage['board'] = board.join('|');
        localStorage['points'] = points;
        update(board, points, "game", "points_counter");
    });
}

function get_suggestion(){
    board = read_cached_board()

    url = "http://127.0.0.1:5000/model"

    request = {
        method: "POST",
        body: JSON.stringify({"board": board}),
        headers: {
            'Content-Type': "application/json",
        },
    };

    fetch(url, request)
    .then(x => x.json())
    .then(json => {
        div = document.getElementById("model_suggestion")
        div.innerHTML = ""
        let actions = ["Right", "Left", "Up", "Down"]

        array = []
        for (let i = 0; i < json["q_values"].length; i++){
            array.push(parseFloat(json["q_values"][i]))
        };
        min = array[0];
        min_i = 0;
        max = array[0];
        max_i = 0;
        for (let i = 1; i < array.length; i++){
            if (array[i] < min){
                min = array[i];
                min_i = i;
            }
            if (array[i] > max){
                max = array[i];
                max_i = i;
            }
        };
        for (let i = 0; i < array.length; i++){
            array[i] = (array[i] - min) / (max -  min) * 100
        };
        console.log(array)

        for (let i = 0; i < array.length; i++){
            if (i == max_i){
                div.innerHTML += '<span style="color: green;">' + actions[i] + ": " + array[i].toFixed(2) + "</span>" + "<br/>"
            }
            else {
                div.innerHTML += actions[i] + ": " + array[i].toFixed(2) + "<br/>" 
            }
        }
    });
}

function move(direction){
    board = read_cached_board()

    url = "http://127.0.0.1:5000/" + direction

    request = {
        method: "POST",
        body: JSON.stringify({"board": board}),
        headers: {
            'Content-Type': "application/json",
        },
    };

    fetch(url, request)
    .then(x => x.json())
    .then(json => {
        console.log(json)
        var board = json["board"];
        var points = json["points"];

        if (points == -1){
            document.getElementById("end-game-message").setAttribute("style", "display: flex;")
            document.onkeydown = null
        }

        localStorage['board'] = board.join('|');
        localStorage['points'] = parseInt(localStorage['points']) + parseInt(points);

        localStorage['n_moves'] = parseInt(localStorage['n_moves']) + 1

        update(board, parseInt(localStorage['points']), "game", "points_counter")
        get_suggestion()
    });
}

document.onkeydown = checkKey;

function checkKey(e) {

    e = e || window.event;

    if (e.keyCode == '38') {
        move("up");
    }
    else if (e.keyCode == '40') {
        move("down");
    }
    else if (e.keyCode == '37') {
        move("left");
    }
    else if (e.keyCode == '39') {
        move("right");
    }
    else if (e.keyCode == '32') {
        get_suggestion();
    }
}