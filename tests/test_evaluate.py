import evaluate 
import torch

precision_recall = evaluate.combine(['precision', 'recall'])
accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')

predictions = torch.tensor([1, 2, 3, 0])
labels = torch.tensor([1, 3, 4, 0])
acc = accuracy.compute(predictions=predictions, 
                       references=labels)
f_score = f1.compute(predictions=predictions, 
                     references=labels, 
                     average='macro')
p_r = precision_recall.compute(predictions=predictions, 
                               references=labels, 
                               average='macro', 
                               zero_division=0)

results = {**acc, **f_score, **p_r}
print(results)
