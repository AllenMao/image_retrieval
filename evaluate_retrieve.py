
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return "sequences of unequal length!"
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))



file_object = open("train_hashcode.txt")
file_lines = file_object.readlines()

train_hashcode_gt_list = []
for line in file_lines:
    tmp = line.strip().split()
    hash_code = tmp[:-1]
    gt = tmp[-1]
    train_hashcode_gt_list.append([hash_code, gt])

#print train_hashcode_gt_list[0]

file_object = open("test_hashcode.txt")
file_lines = file_object.readlines()

acc_cnt = 0
for line in file_lines:
    candidate_dict = {}
    tmp = line.strip().split()
    test_hash_code = tmp[:-1]
    test_gt = tmp[-1]
    
    for i in range(len(train_hashcode_gt_list)):
        train_hash_code = train_hashcode_gt_list[i][0]
        train_gt = train_hashcode_gt_list[i][1]

        if hamming_distance(test_hash_code, train_hash_code) <= 2:
            if not candidate_dict.has_key(train_gt):
                candidate_dict[train_gt] = 1
            else: candidate_dict[train_gt] += 1

    candidate_dict_sorted = sorted(candidate_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse = True)
    if test_gt == candidate_dict_sorted[0][0]:
        acc_cnt += 1

print 'accuracy:', acc_cnt * 1.0 / len(file_lines)