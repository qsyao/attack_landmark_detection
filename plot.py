import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse

x = (1, 20, 50, 99, 150, 200, 250, 299, 400, 600, 750, 999)
# x = (1, 20, 50, 99, 150, 200, 250, 299)

def calculate_epsilon(attack, defend):
    temp = list()
    for item in attack:
        item = [i[1] for i in item]
        temp.append(np.mean(item))
    print("Attack Mean {}".format(np.mean(temp)))

    temp = list()
    for item in attack:
        item = [i[1] for i in item]
        temp.append(np.median(item))
    print("Attack median {}".format(np.median(temp)))

    temp = list()
    for item in defend:
        item = [i[1] for i in item]
        temp.append(np.mean(item))
    print("Defend Mean {}".format(np.mean(temp)))

    temp = list()
    for item in defend:
        item = [i[1] for i in item]
        temp.append(np.median(item))
    print("Defend median {}".format(np.median(temp)))

    temp = list()
    for item in attack:
        temp += [i[1] for i in item]
    rate = np.sum(np.array(temp) < 4) / len(temp)
    print("Attack 4mm rate {}".format(rate))

    temp = list()
    for item in defend:
        temp += [i[1] for i in item]
    rate = np.sum(np.array(temp) < 4) / len(temp)
    print("Defend 4mm rate {}".format(rate))

def calculate(input):
    temp = list()
    for item in input:
        item = [i[1] for i in item]
        temp.append(np.mean(item))
    return np.mean(temp)

    temp = list()   
    for item in input:
        item = [i[1] for i in item]
        temp += item
    rate = np.sum(np.array(temp) < 4) / len(temp)   
    return rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a cgan Xray network")
    parser.add_argument("--tag", default='ours', help="position of the output dir")
    parser.add_argument("--iteration", default='299', help="position of the output dir")
    args = parser.parse_args()

    iteration = int(args.iteration)
    mode = 1 if args.tag != 'TIFGSM' else 0

    with open(args.tag + '/dict_attack.pkl', 'rb') as f:
        dict_attack = pickle.load(f)

    with open(args.tag + '/dict_defend.pkl', 'rb') as f:
        dict_defend = pickle.load(f)
    
    calculate_epsilon(dict_attack[mode][iteration], dict_defend[mode][iteration])

    # with open('I1' + '/dict_attack.pkl', 'rb') as f:
    #     dict_attack = pickle.load(f)

    # with open('I2' + '/dict_defend.pkl', 'rb') as f:
    #     dict_defend = pickle.load(f)
    
    # for string in ['I2', 'I3', 'I4', 'I5']:
    #     with open(string + '/dict_attack.pkl', 'rb') as f:
    #         temp_dict_attack = pickle.load(f)
    #         for iteration in x:
    #             dict_attack[1][iteration].extend(temp_dict_attack[1][iteration])

    #     with open(string + '/dict_defend.pkl', 'rb') as f:
    #         temp_dict_defend = pickle.load(f)
    #         for iteration in x:
    #             dict_defend[1][iteration].extend(temp_dict_defend[1][iteration])

    # for iteration in x:
    #     print("\nIteration {} ------:".format(iteration))
    #     calculate_epsilon(dict_attack[1][iteration], dict_defend[1][iteration])
    


    failure_case = dict()
    for i in range(19):
        failure_case[i] = list()
    for item in dict_attack[1][299]:
        for landmark in item:
            failure_case[landmark[0]].append(landmark[1])
    for i in range(19):
        if len(failure_case[i]) != 0:
            failure_case[i] = np.mean(failure_case[i])
    
    with open('distance.pkl', 'rb') as f:
        distance_list = pickle.load(f)
    with open('top5.pkl', 'rb') as f:
        mean_list = pickle.load(f)
    failure_case = list(failure_case.values())
    distance_list = list(distance_list.values())
    mean_list = list(mean_list.values())
    for i in range(19):
        distance_list[i] *= 3 / 10
        mean_list[i] *= 3 / 10
    ids = list(range(19))
    import ipdb; ipdb.set_trace()

    import csv
    with open('failure.csv', 'w') as f:
        writer =  csv.writer(f)
        writer.writerow(ids)
        writer.writerow(failure_case)
        writer.writerow(distance_list)
        writer.writerow(mean_list)

    # Mean Radial Error (mm)
    # Minimum Distance (mm)
    # ID of the Landmarks


    # with open('ours' + '/dict_attack.pkl', 'rb') as f:
    #     ours_dict_attack = pickle.load(f)

    # with open('ours' + '/dict_defend.pkl', 'rb') as f:
    #     ours_dict_defend = pickle.load(f)

    # with open('TIFGSM' + '/dict_attack.pkl', 'rb') as f:
    #     theirs_dict_attack = pickle.load(f)

    # with open('TIFGSM' + '/dict_defend.pkl', 'rb') as f:
    #     theirs_dict_defend = pickle.load(f)

    # # calculate_epsilon(dict_attack[mode][iteration], dict_defend[mode][iteration])

    # ours_targeted = list()
    # theirs_targeted = list()
    # ours_stationary = list()
    # theirs_stationary = list()
    # for i in x:
    #     ours_targeted.append(calculate(ours_dict_attack[1][i]))
    #     ours_stationary.append(calculate(ours_dict_defend[1][i]))
    #     theirs_targeted.append(calculate(theirs_dict_attack[0][i]))
    #     theirs_stationary.append(calculate(theirs_dict_defend[0][i]))
    # import ipdb; ipdb.set_trace()

    # plt.plot(x, theirs_targeted, 'b-', label="(Targeted) Targeted I-FGSM")
    # plt.plot(x, ours_targeted, 'g-', label="(Targeted) Adaptive Targeted I-FGSM")
    # plt.plot(x, theirs_stationary, 'b--', label="(Stationary) Targeted I-FGSM")
    # plt.plot(x, ours_stationary, 'g--', label="(Stationary) Adaptive Targeted I-FGSM")
    # plt.legend(fontsize=13)
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # # plt.title('Median RE between pro')
    # plt.xlabel("Iteration", fontsize=13)
    # plt.ylabel("MRE (mm)", fontsize=13)
    # plt.savefig("test.jpg")