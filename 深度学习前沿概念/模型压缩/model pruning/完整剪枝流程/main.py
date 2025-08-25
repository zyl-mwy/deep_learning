def complete_pruning_pipeline():
    """完整的模型剪枝流程"""
    # 1. 准备数据
    train_loader, test_loader = get_data_loaders()
    
    # 2. 训练原始模型
    print("训练原始模型...")
    model = SimpleCNN()
    train_history = train_model(model, train_loader, test_loader, epochs=10)
    original_acc = evaluate_model(model, test_loader)
    print(f"原始模型准确率: {original_acc:.2f}%")
    
    # 3. 权重剪枝
    print("\n开始权重剪枝...")
    pruner = WeightPruner(model)
    
    # 一次性剪枝
    pruner.magnitude_pruning(pruning_rate=0.5)
    sparsity = pruner.calculate_sparsity()
    pruned_acc = evaluate_model(model, test_loader)
    print(f"一次性剪枝后 - 稀疏度: {sparsity:.2f}%, 准确率: {pruned_acc:.2f}%")
    
    # 恢复并尝试迭代剪枝
    pruner.restore_original()
    iterative_results = iterative_pruning(model, train_loader, test_loader, 
                                        target_sparsity=0.8, steps=4)
    
    # 4. 结构化剪枝（可选）
    print("\n尝试结构化剪枝...")
    # 注册前向钩子
    hooks = register_forward_hooks(model)
    
    # 计算通道重要性
    channel_pruner = ChannelPruner(model)
    channel_pruner.calculate_channel_importance(train_loader, nn.CrossEntropyLoss())
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 执行通道剪枝
    pruned_model = channel_pruner.prune_channels(pruning_rate=0.3)
    pruned_model_acc = evaluate_model(pruned_model, test_loader)
    print(f"通道剪枝后准确率: {pruned_model_acc:.2f}%")
    
    return {
        'original_acc': original_acc,
        'weight_pruned_acc': pruned_acc,
        'iterative_results': iterative_results,
        'channel_pruned_acc': pruned_model_acc
    }

# 可视化结果
def visualize_pruning_results(results):
    """可视化剪枝结果"""
    plt.figure(figsize=(15, 10))
    
    # 准确率对比
    plt.subplot(2, 2, 1)
    models = ['Original', 'Weight Pruned', 'Channel Pruned']
    accuracies = [results['original_acc'], 
                 results['weight_pruned_acc'], 
                 results['channel_pruned_acc']]
    
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # 迭代剪枝结果
    if 'iterative_results' in results:
        iterative = results['iterative_results']
        plt.subplot(2, 2, 2)
        plt.plot(iterative['sparsity'], iterative['accuracy'], 'o-', linewidth=2)
        plt.title('Iterative Pruning Results')
        plt.xlabel('Sparsity (%)')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        # 参数数量变化
        plt.subplot(2, 2, 3)
        plt.plot(iterative['sparsity'], iterative['params'], 's-', linewidth=2, color='orange')
        plt.title('Parameter Count vs Sparsity')
        plt.xlabel('Sparsity (%)')
        plt.ylabel('Number of Parameters')
        plt.grid(True)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('pruning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 模型统计工具
def model_statistics(model):
    """打印模型统计信息"""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            if 'weight' in name:
                zero_params += (param == 0).sum().item()
    
    sparsity = 100. * zero_params / total_params if total_params > 0 else 0
    
    print(f"总参数: {total_params:,}")
    print(f"零值参数: {zero_params:,}")
    print(f"稀疏度: {sparsity:.2f}%")
    print(f"压缩比: {total_params/(total_params - zero_params):.2f}x")
    
    return total_params, zero_params, sparsity

# 主函数
def main():
    print("开始模型剪枝实验...")
    
    # 运行完整的剪枝流程
    results = complete_pruning_pipeline()
    
    # 可视化结果
    visualize_pruning_results(results)
    
    print("\n实验完成!")
    print(f"原始模型准确率: {results['original_acc']:.2f}%")
    print(f"权重剪枝后准确率: {results['weight_pruned_acc']:.2f}%")
    print(f"通道剪枝后准确率: {results['channel_pruned_acc']:.2f}%")

if __name__ == "__main__":
    main()