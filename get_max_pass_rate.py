import json

def get_max_pass_rate(json_file, result_file):
    # json_file = 
    with open(json_file, "r") as f:
        pr = json.load(f)
    keys = list(pr.keys())  
    cut_pr = [(keys[k], pr[keys[k]]+pr[keys[k+1]]+pr[keys[k+2]] + pr[keys[k+3]]) for k in range(len(keys[:-4])) if pr[keys[k]] > 0.945 and pr[keys[k+1]] > 0.945 and pr[keys[k+2]] > 0.945 and pr[keys[k+3]] > 0.945]   # 3个一组的pr
    sorted_pr = sorted(cut_pr, key=lambda k : k[1], reverse=True)
    my_list = [(k[0], pr[keys[keys.index(k[0])]], pr[keys[keys.index(k[0])+1]], pr[keys[keys.index(k[0])+2]], pr[keys[keys.index(k[0])+3]]) for k in sorted_pr]
    with open(result_file, "w") as f:
        json.dump(my_list, f, indent=4)
    # print(min(cut_pr, keys=lambda k : k[1], reverse=True))
get_max_pass_rate("trained_models/DFCAN-SISR_ER/pass_rate.json", "DFCAN.json")
get_max_pass_rate("/root/autodl-tmp/MoE-DFCAN-pytorch/trained_models/MoE_DFCAN-SISR_MixedData/pass_rate.json", "MoE.json")
pass