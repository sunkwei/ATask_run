'''
函数名： dec_mute    <- 调用此函数进行静音片段检测
功能：筛查并剔除静音片段
输入：音频段（字词对应的音频），起始时间戳，结束时间戳
输出：
    输出内容：非静音片段的起始时间戳，结束时间戳
    输出格式：[(start_time),(end_time)]

'''



import numpy as np
# 滑动平滑
def np_move_avg(a,n,mode="same"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))
# 计算前1/3音量值的均值
def cal_mean_audio(audio):
    audio = np.array(audio)
    audio_up = audio[audio>0]
    sort_d = np.sort(audio_up)
    keep = sort_d[-int(len(sort_d) * 0.3):]
    # print("15个值:",keep)
    return float(round(np.mean(keep),5))

def dec_mute(audio, start_stamp, end_stamp):
    # 1.计算片段（10ms）音量、输入片段的音量序列
    num = len(audio)//160 # 输入音频共有num个10ms片段
    Seg_10ms_Vol = [] # 存放整个输入音频的序列，每个序列为10ms片段的音量平滑最大值
    for i in range(num):
        sigle_10ms = abs(audio[i*160:(i+1)*160]) 
        temp = max(np_move_avg(sigle_10ms, 3))
        Seg_10ms_Vol.append(temp)
    # 2.计算基准音量（输入片段最大多个值的均值）
    Max_Volume = cal_mean_audio(Seg_10ms_Vol)
    Min_Mute_Vol = min(Seg_10ms_Vol)
    # 3.筛选静音片段
    mute_config = [] # 存放静音片段标记
    # 3.1 根据10ms片段音量与基准音量的关系，初步判断是否为静音片段
    for vol in Seg_10ms_Vol:
        Vol_jump = Max_Volume / (vol + 0.00001)
        if Vol_jump >= 8:
            mute_config.append(1)
        elif vol < max(0.00305, 1.3*Min_Mute_Vol):
            mute_config.append(1)
        else:
            mute_config.append(0)
    # print("初步静音片段:", mute_config)
    # 3.2 根据音量波动，进一步判断是否为静音片段
    mute_dex = np.where(np.array(mute_config) == 1)[0] # 获取所有静音片段的索引
    if len(mute_dex):
        for i in range(1, len(mute_dex)):
            Max_Vol = max(Seg_10ms_Vol[mute_dex[i]-1:mute_dex[i]+1])
            Min_Vol = min(Seg_10ms_Vol[mute_dex[i]-1:mute_dex[i]+1])
            rl = Max_Vol/(Min_Vol+0.00001)
            if rl > min((1.1*Max_Volume)/(6*Seg_10ms_Vol[mute_dex[i]]+0.00001), 3):
                mute_config[mute_dex[i]] = 0
        # print("静音片段:", mute_config)

    # --------------------------------------------------------------------------- #
    # 3.3 整理静音片段，如果静音片段位于中部（两侧均有非静音片段）则不记该静音片段  
        # 3.3.1 单独处理首尾静音片段
        if mute_config[0] == 1 and mute_config[1] == 0: mute_config[0] = 0
        if mute_config[-1] == 1 and mute_config[-2] == 0: mute_config[-1] = 0
        # 3.3.2 如果静音片段位于中部（两侧均有非静音片段）则不记该静音片段
        mute_dex1 = np.where(np.array(mute_config) == 1)[0] # 获取所有静音片段的索引
        for i in range(0, len(mute_dex1)):
            if 0 in mute_config[:mute_dex1[i]] and 0 in mute_config[mute_dex1[i]:]:
                mute_config[mute_dex1[i]] = 0
    # --------------------------------------------------------------------------- #
    #             
        # 3.4 统计非静音片段返回时间戳
        mute_dex = np.where(np.array(mute_config) == 0)[0] # 获取所有非静音片段的索引
        n = 0
        p = []
        p.append(mute_dex[0]*0.01+start_stamp)
        while n < len(mute_dex)-1:
            if mute_dex[n] == mute_dex[n+1]-1:
                n += 1
            else:
                p.append(((mute_dex[n]+1)*0.01+start_stamp))
                p.append((mute_dex[n+1]*0.01+start_stamp))
                n += 1
        p.append((mute_dex[-1]+1)*0.01+start_stamp)  
        stamp = []
        for d in range(1,len(p),2)  :
            stamp.append((p[d-1],p[d]))
        return stamp
    # 没有需要剔除的静音片段
    else:
        return [(start_stamp, end_stamp)]

def crop_str_time(wav, time_list):
    # 微调字时间戳
    crop_time_list = []
    for t in time_list:
        
        s = t[0] / 1000
        t = t[1] / 1000

        part_audio = wav[int(s*16000):int(t*16000)]
        del_stamp = dec_mute(part_audio, s, t)
        new_s = round(del_stamp[0][0], 2)
        new_e = round(del_stamp[0][1], 2)
        if new_e - new_s < 0.5:
            new_e = min(new_s + 0.5, t)

        crop_time_list.append([new_s, new_e])
    
    return crop_time_list

if __name__ == '__main__':
    import soundfile

    sample_path = '/var/data_old/home/kylinchen/Documents/zk_pro/基于0.8s非静音片段的声纹分割算法/静音片段检测/726.wav'
    audio, rate = soundfile.read(sample_path)
    res = open('726_testmute.txt', 'w')
    with open('/var/data_old/home/kylinchen/Documents/zk_pro/基于0.8s非静音片段的声纹分割算法/静音片段检测/726_文字asr.txt', 'r') as f:
        cont = f.readlines()
    for line in cont:
        line= line.strip()
        data = line.split('\t')
        start_t = float(data[0])
        end_t = float(data[1])
        audio_part = audio[int(start_t*rate):int(end_t*rate)]
        mute_res = dec_mute(audio_part, start_t, end_t)
        for r in mute_res:
            res.write(str(r[0]) + '\t' + str(r[1]) + '\n')
    res.close()
    print('down!')
        
        

    

