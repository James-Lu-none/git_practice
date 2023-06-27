
import numpy as np
import os
import pickle

def data_augmentation(game_data):
    extend_data = []
    print("original:{}".format(len(game_data)))
    for game in game_data:
        for i in [1, 2, 3, 4]:
            extend_data.append(np.flipud(np.rot90(game, i)))
            extend_data.append(np.rot90(game, i))
    print("after aug:{}".format(len(extend_data)))
    return extend_data


def data_trans(game_state, last_n):
    data_trans3 = [] #每一手的組合
    for dim1 in range(-1, int(np.max(game_state))):
        #每一手的局勢
        data_trans = np.full((4+last_n,15,15),0)
        if int(np.max(game_state)) % 2 == 0: #每一手最後贏家
            whowin = 1
        else:
            whowin = -1
        if dim1 % 2 == 0:
            whowin = whowin*-1
            data_trans[0] = [[1] * 15 for _ in range(15)]  #第一個特徵_roll to play(black or white)
        if dim1 > 0 :
            data_trans[2] = data_trans3[-1][0][3] #第三個特徵繼承_p1state
            data_trans[3] = data_trans3[-1][0][2] #第四個特徵繼承_p2state
        if dim1 >= 0 :
            dim3 = np.where(game_state == dim1) #第幾個位置(x,y)
            data_trans[3][dim3[0][0]][dim3[1][0]] = 1 #第四個特徵更新_p2state
            for _ in range(last_n): #第4+n個特徵_last_n_move
                if dim1-_ >= 0:
                    dim4 = np.where(game_state == dim1-_) #第幾個位置(x,y)
                    data_trans[4+_][dim4[0][0]][dim4[1][0]] = 1
           
        data_trans2 = [[0] * 15 for _ in range(15)] #每一手的下一手機率
        if dim1 != int(np.max(game_state)):
            P = 1 #給定的機率
            dim_P = np.where(game_state == dim1+1)
            data_trans2[dim_P[0][0]][dim_P[1][0]] = P
        data_trans[1] = np.full((15, 15), 1)-data_trans[2]-data_trans[3] #第二個特徵available
        data_trans3.append((data_trans, np.array(data_trans2), whowin))
    
    data_trans3 = data_trans3[int(np.max(game_state))-3:]
    # if (len(data_trans3) < 10):
    #     print("dropped game has length", len(data_trans3))
    # else:
    #     data_final.append(data_trans3)
    return data_trans3
def save_trans_data(root_path,data,source,last_n):
    
    new_folder_dir=os.path.join(root_path,f'training_data_v{last_n}')
    os.makedirs(new_folder_dir, exist_ok=True)

    print("data Generated", len(data))
    file_path = os.path.join(root_path,f'training_data_v{last_n}',source)
    print("saving data to ",file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
