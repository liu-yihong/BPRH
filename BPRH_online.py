from flask import Flask
from flask import request, jsonify
from bprH_gpu import bprH
import random
import pandas as pd

app = Flask(__name__)

bprh_model = bprH()
bprh_model.load("test_online.pkl")

ranking_list = bprh_model.train_data[bprh_model.train_data.Action == "P"].groupby("ItemID").count().UserID.copy()
ranking_list.sort_values(inplace=True, ascending=False)
ranking_list = ranking_list.index.to_list()

item_list = bprh_model.item_original_id_list.copy()


@app.route('/init_pop', methods=['GET'])
def model_poprank():
    # at first stage, we use Popularity ranking to decide an offer set
    if 'topk' in request.args:
        topk = int(request.args['topk'])
        topk_ranking_list = ranking_list[:topk]
        topk_ranking_list_original_id = [item_list[item_idx] for item_idx in topk_ranking_list]
        topk_ranking_dict = dict()
        topk_ranking_dict['isError'] = False
        topk_ranking_dict['MSG'] = 'Good'
        for i in range(topk):
            topk_ranking_dict["Item" + str(i)] = topk_ranking_list_original_id[i]
        return jsonify(topk_ranking_dict)
    else:
        return jsonify({"isError": True,
                        'MSG': "Top K not provided. Please specify it"})


@app.route('/init_rand', methods=['GET'])
def model_rand():
    # at first stage, we randomly decide an offer set
    if 'topk' in request.args:
        topk = int(request.args['topk'])
        random_item_idx = random.sample(range(len(item_list)), topk)
        random_ranking_list_original_id = [item_list[item_idx] for item_idx in random_item_idx]
        random_ranking_dict = dict()
        random_ranking_dict['isError'] = False
        random_ranking_dict['MSG'] = 'Good'
        for i in range(topk):
            random_ranking_dict["Item" + str(i)] = random_ranking_list_original_id[i]
        return jsonify(random_ranking_dict)
    else:
        return jsonify({"isError": True,
                        'MSG': "Top K not provided. Please specify it"})


@app.route('/user_online_update', methods=['POST'])
def model_user_update():
    if ('UserID' not in request.args) | ('ItemID' not in request.args) | ('Action1' not in request.args) | (
            'MaxIter' not in request.args) | ('topk' not in request.args) | ('TrainReference' not in request.args):
        return jsonify({"isError": True,
                        'MSG': "Wrong Data Format"})
    u_id = request.args['UserID']
    i_id = request.args['ItemID']
    first_action = request.args['Action1']
    maxIter = int(request.args['MaxIter'])
    topk = int(request.args['topk'])
    train_as_reference = True if request.args['TrainReference'] == 'True' else False
    if 'Action2' in request.args:
        second_action = request.args['Action2']
        new_data = pd.DataFrame([[u_id, i_id, first_action],
                                 [u_id, i_id, second_action]], columns=['UserID', 'ItemID', 'Action'])
    else:
        new_data = pd.DataFrame([[u_id, i_id, first_action]], columns=['UserID', 'ItemID', 'Action'])

    next_offerset = bprh_model.online_updating_user(new_user_with_data=new_data,
                                                    max_iteration=maxIter,
                                                    topK=topk,
                                                    input_data_as_reference=train_as_reference)

    bprh_ranking_dict = dict()
    bprh_ranking_dict['isError'] = False
    bprh_ranking_dict['MSG'] = 'Good'
    for i in range(topk):
        bprh_ranking_dict["Item" + str(i)] = next_offerset[i]
    return jsonify(bprh_ranking_dict)


app.run(host='0.0.0.0')