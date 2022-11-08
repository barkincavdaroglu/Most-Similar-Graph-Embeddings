from flask import Flask, jsonify, request

from find_most_similar import find_most_similar_timeline

app = Flask(__name__)

@app.route('/get_next_state', methods=['POST'])
def get_next_state():
    prev_timeline_filenames = ["sample_data/1.txt", "sample_data/2.txt"]
    graph_snapshot = request.get_json()['snapshot']
    most_similar_timeline, snapshot_key = find_most_similar_timeline(prev_timeline_filenames, graph_snapshot, "string")

    data = open(prev_timeline_filenames[most_similar_timeline],'r').read().splitlines()

    time = 0
    graph_to_return = ""

    for line in data:
        if len(line) >= 1 and time == snapshot_key + 1:
            graph_to_return += line + "\n"

        elif len(line) == 0: # another graph
            if time == snapshot_key + 1:
                return graph_to_return
            time += 1

    return jsonify(snapshot=graph_to_return)