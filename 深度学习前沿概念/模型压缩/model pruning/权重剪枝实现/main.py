# 权重剪枝类
class WeightPruner:
    def __init__(self, model):
        self.model = model
        self.masks = {}  # 存储剪枝掩码
        self.original_state = deepcopy(model.state_dict())
    
    def magnitude_pruning(self, pruning_rate=0.5):
        """
        基于权重大小的剪枝
        pruning_rate: 剪枝比例 (0-1)
        """
        self.masks.clear()
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:  # 只剪枝权重，不剪枝偏置
                # 计算阈值
                weights = param.data.abs().clone()
                threshold = torch.quantile(weights, pruning_rate)
                
                # 创建掩码
                mask = (weights > threshold).float()
                self.masks[name] = mask
                
                # 应用剪枝
                param.data *= mask
    
    def global_magnitude_pruning(self, pruning_rate=0.5):
        """
        全局权重大小剪枝（所有层使用相同的阈值）
        """
        self.masks.clear()
        all_weights = []
        
        # 收集所有权重
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                all_weights.append(param.data.abs().view(-1))
        
        if not all_weights:
            return
        
        all_weights = torch.cat(all_weights)
        global_threshold = torch.quantile(all_weights, pruning_rate)
        
        # 应用全局剪枝
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                mask = (param.data.abs() > global_threshold).float()
                self.masks[name] = mask
                param.data *= mask
    
    def apply_masks(self):
        """应用存储的掩码"""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
    
    def calculate_sparsity(self):
        """计算模型稀疏度"""
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
        
        if total_params == 0:
            return 0
        
        sparsity = 100. * zero_params / total_params
        return sparsity
    
    def count_parameters(self):
        """统计参数数量"""
        total_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total_params += param.numel()
        return total_params
    
    def restore_original(self):
        """恢复原始模型"""
        self.model.load_state_dict(self.original_state)
        self.masks.clear()

# 迭代式剪枝
def iterative_pruning(model, train_loader, test_loader, target_sparsity=0.9, steps=5):
    """
    迭代式剪枝：逐步增加剪枝比例
    """
    pruner = WeightPruner(model)
    original_acc = evaluate_model(model, test_loader)
    original_params = pruner.count_parameters()
    
    results = {
        'sparsity': [0],
        'accuracy': [original_acc],
        'params': [original_params]
    }
    
    print(f"原始模型: 参数 {original_params:,}, 准确率 {original_acc:.2f}%")
    
    for step in range(1, steps + 1):
        # 计算当前目标稀疏度
        current_target = target_sparsity * (step / steps)
        
        # 剪枝
        pruner.magnitude_pruning(current_target)
        sparsity = pruner.calculate_sparsity()
        
        # 微调
        print(f"\n步骤 {step}: 稀疏度 {sparsity:.2f}%")
        print("进行微调...")
        train_model(model, train_loader, test_loader, epochs=3, lr=0.0001)
        
        # 评估
        acc = evaluate_model(model, test_loader)
        params = pruner.count_parameters()
        
        results['sparsity'].append(sparsity)
        results['accuracy'].append(acc)
        results['params'].append(params)
        
        print(f"剪枝后: 参数 {params:,}, 准确率 {acc:.2f}%")
    
    return results