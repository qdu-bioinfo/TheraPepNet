def get_data():
    peptide_seq_dict = {}
    label = []
    label_index = 0
    peptide_index = 0
    with open('data.fa', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[-2]
                label_temp = int(values)
                label.append(label_temp)
                label_index += 1
            else:
                seq = line[:-1]
                peptide_seq_dict[peptide_index] = seq
                peptide_index += 1

    keys_to_remove = []
    for key, value in peptide_seq_dict.items():
        if len(value) < 7:
            keys_to_remove.append(key)

    for idx in sorted(keys_to_remove, reverse=True):
        del peptide_seq_dict[idx]
    label = [label[i] for i in range(len(label)) if i not in keys_to_remove]

    return peptide_seq_dict, label
