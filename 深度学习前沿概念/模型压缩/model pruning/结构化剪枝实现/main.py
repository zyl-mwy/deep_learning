# 结构化剪枝（通道剪枝）
class ChannelPruner:
    def __init__(self, model):
        self.model = model
        self.importance_scores = {}
    
    def calculate_channel_importance(self, train_loader, criterion):
        """
        计算通道重要性（基于L1范数）
        """
        self.model.eval()
        self.importance_scores.clear()
        
        # 初始化重要性分数
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.importance_scores[name] = torch.zeros(module.out_channels)
        
        # 计算每个通道的L1范数
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(device)
                _ = self.model(data)
                
                # 更新重要性分数
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) and hasattr(module, 'output_feature'):
                        # 使用输出的绝对值之和作为重要性指标
                        importance = module.output_feature.abs().sum(dim=(0, 2, 3))
                        self.importance_scores[name] += importance.cpu()
    
    def prune_channels(self, pruning_rate=0.3):
        """
        剪枝最不重要的通道
        """
        new_model = deepcopy(self.model)
        
        for name, module in new_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.importance_scores:
                importance = self.importance_scores[name]
                num_channels = len(importance)
                num_to_prune = int(pruning_rate * num_channels)
                
                if num_to_prune > 0:
                    # 找到最不重要的通道
                    _, indices = torch.topk(importance, num_to_prune, largest=False)
                    
                    # 创建新的卷积层（减少输出通道）
                    new_out_channels = num_channels - num_to_prune
                    new_conv = nn.Conv2d(
                        module.in_channels,
                        new_out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                        module.bias is not None
                    )
                    
                    # 复制重要的权重
                    important_indices = torch.tensor([i for i in range(num_channels) if i not in indices])
                    new_conv.weight.data = module.weight.data[important_indices]
                    if module.bias is not None:
                        new_conv.bias.data = module.bias.data[important_indices]
                    
                    # 替换模块
                    parent_name = name.rsplit('.', 1)[0]
                    parent_module = dict(new_model.named_modules())[parent_name]
                    setattr(parent_module, name.split('.')[-1], new_conv)
        
        return new_model

# 注册前向钩子来捕获输出
def register_forward_hooks(model):
    hooks = []
    
    def hook_fn(module, input, output):
        module.output_feature = output.detach()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    return hooks