import sys
with open('vqa_train_text_recognition_OLD.csv', 'r', encoding='latin-1') as f:
	csv = f.readlines()

with open('vqa_train_text_recognition_NEW.csv', 'w', encoding='utf-8') as f:
	f.writelines(csv)

with open('vqa_train_text_recognition_NEW.csv', 'r') as f:
        csv = f.readlines()

new = []
for line in csv:
    sp = line.split(';')
    head = sp[:-1]
    trail = sp[-1].replace(',','')
    new_line = ';'.join(head) + trail
    new.append(new_line)

with open('vqa_train_text_recognition.csv', 'w') as f:
    f.writelines(new)
