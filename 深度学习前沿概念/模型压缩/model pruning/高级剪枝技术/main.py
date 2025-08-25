# 基于梯度的剪枝
class GradientBasedPruner:
    def __init__(self, model):
        self.model = model
        self.gradients = {}
    
    def compute_weight_importance(self, train_loader, criterion):
        """基于梯度的权重重要性计算"""
        self.model.train()
        
        # 注册梯度钩子
        handles = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                handle = param.register_hook(
                    lambda grad, name=name: self._store_gradient(grad, name)
                )
                handles.append(handle)
        
        # 计算梯度
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            self.model.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
        
        # 移除钩子
        for handle in handles:
            handle.remove()
    
    def _store_gradient(self, grad, name):
        if name not in self.gradients:
            self.gradients[name] = grad.abs().clone()
        else:
            self.gradients[name] += grad.abs().clone()
    
    def gradient_based_pruning(self, pruning_rate=0.5):
        """基于梯度的剪枝"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and name in self.gradients:
                importance = self.gradients[name]
                threshold = torch.quantile(importance, pruning_rate)
                
                mask = (importance > threshold).float()
                param.data *= mask

# L1正则化剪枝
def l1_regularization_pruning(model, train_loader, test_loader, lambda_l1=0.001, epochs=5):
    """使用L1正则化进行剪枝"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 添加L1正则化
            l1_penalty = 0
            for param in model.parameters():
                l1_penalty += param.abs().sum()
            
            total_loss_with_l1 = loss + lambda_l1 * l1_penalty
            total_loss_with_l1.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 剪枝小权重
        with torch.no_grad():
            for param in model.parameters():
                param.data[param.abs() < 0.01] = 0  # 剪枝小权重
        
        test_acc = evaluate_model(model, test_loader)
        print(f'L1 Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {test_acc:.2f}%')
    
    return model