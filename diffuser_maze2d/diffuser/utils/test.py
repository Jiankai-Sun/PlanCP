import torch

# cal_scores:  torch.Size([1, 384, 6])
cal_scores = torch.randn(1, 2, 6)
cal_scores.requires_grad_()
print('cal_scores: ', cal_scores)
n = cal_scores.shape[1]
alpha = 0.1  # 1-alpha is the desired coverage
print(n, alpha, torch.ceil(torch.Tensor([(n + 1) * (1 - alpha) / n])))
# Get the score quantile
# qhat:  torch.Size([1])
qhat = torch.quantile(cal_scores,
                      torch.ceil(torch.Tensor([(n) * (1 - alpha) / n]).to(cal_scores.device)),
                      interpolation='higher', dim=1).contiguous()
# cal_labels.shape:  torch.Size([32, 384, 6])
# cal_scores:  torch.Size([32, 384, 6])
# qhat:  torch.Size([1, 32, 6])
# Deploy (output=lower and upper adjusted quantiles)
# prediction_sets = [val_lower - qhat, val_upper + qhat]
cp_loss = 2 * qhat.mean()
cp_loss.backward()
print('cal_scores.grad: ', cal_scores.grad)