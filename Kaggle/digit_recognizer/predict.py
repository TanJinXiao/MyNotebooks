import numpy as np
from sqlalchemy import false
import torch
import pandas as pd
from torch.utils.data import DataLoader

def predict(model, test_iter: DataLoader, device):
    model.to(device)
    predicts = None
    with torch.no_grad():
        for x, _ in test_iter:
            x = x.to(device)
            # print(torch.typename(x))
            output = model(x).to('cpu')
            if predicts == None:
                predicts = torch.argmax(output, dim=1)
            else:
                predicts = torch.cat([predicts, torch.argmax(output, dim=1)])
    sample_df = pd.read_csv('./datasets/sample_submission.csv')
    sample_df['Label'] = predicts.to('cpu').numpy()
    sample_df.to_csv('./datasets/result.csv', index=False)