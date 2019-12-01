import re

train_loss = {}
validation_loss = {}

number_regex = r'(-?[0-9]+.?[0-9]*e?E?.?[0-9]*)'

with open('transformer-training-logs.txt') as f:
    lines = f.readlines()
    for line in lines:
        if "loss" in line:
            if "global_step" in line:
                # evaluation
                found = re.search('global_step = {}, loss = {},'.format(number_regex, number_regex), line)
                if found:
                    step = int(found.group(1))
                    loss = float(found.group(2))
                    validation_loss[step] = loss
            else:
                # training
                found = re.search('loss = {}, step = {} '.format(number_regex, number_regex), line)
                if found:
                    loss = float(found.group(1))
                    step = int(found.group(2))
                    train_loss[step] = loss


print("train_loss:", sorted(train_loss.items()))
print("validation_loss:", sorted(validation_loss.items()))
