import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义模型架构
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 教师模型（更大更复杂）
class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 学生模型（更小更轻量）
class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 监督知识蒸馏训练器
class SupervisedKnowledgeDistillation:
    def __init__(self, teacher, student, device, 
                 temperature=4.0, alpha=0.7, 
                 teacher_weight=1.0, student_weight=1.0):
        """
        初始化监督知识蒸馏
        
        Args:
            teacher: 教师模型
            student: 学生模型
            device: 训练设备
            temperature: 温度参数，控制软标签的平滑度
            alpha: 蒸馏损失权重 (0-1)
            teacher_weight: 教师模型权重（用于加权平均）
            student_weight: 学生模型权重（用于加权平均）
        """
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_weight = teacher_weight
        self.student_weight = student_weight
        
        # 优化器和学习率调度器
        self.student_optimizer = optim.Adam(
            self.student.parameters(), 
            lr=0.001, 
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.student_optimizer, 
            step_size=10, 
            gamma=0.5
        )
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        
    def distillation_loss(self, teacher_logits, student_logits):
        """
        计算知识蒸馏损失
        
        Args:
            teacher_logits: 教师模型的原始输出
            student_logits: 学生模型的原始输出
            
        Returns:
            distillation_loss: 蒸馏损失值
        """
        # 使用温度参数软化概率分布
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # 计算KL散度损失
        distillation_loss = F.kl_div(
            soft_student, 
            soft_teacher, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss
    
    def weighted_ensemble_loss(self, teacher_logits, student_logits, labels):
        """
        加权集成损失函数
        
        Args:
            teacher_logits: 教师模型输出
            student_logits: 学生模型输出
            labels: 真实标签
            
        Returns:
            ensemble_loss: 加权集成损失
        """
        # 计算教师和学生的概率
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.softmax(student_logits, dim=1)
        
        # 加权平均概率
        ensemble_probs = (self.teacher_weight * teacher_probs + 
                         self.student_weight * student_probs) / (self.teacher_weight + self.student_weight)
        
        # 计算集成损失
        ensemble_loss = self.ce_loss(ensemble_probs, labels)
        
        return ensemble_loss
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            avg_loss: 平均损失
            student_acc: 学生模型准确率
            kd_loss_avg: 平均蒸馏损失
            ce_loss_avg: 平均交叉熵损失
        """
        self.teacher.eval()  # 教师模型设为评估模式
        self.student.train()  # 学生模型设为训练模式
        
        total_loss = 0
        kd_loss_total = 0
        ce_loss_total = 0
        student_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 教师模型前向传播（不计算梯度）
            with torch.no_grad():
                teacher_logits = self.teacher(data)
            
            # 学生模型前向传播
            student_logits = self.student(data)
            
            # 计算各种损失
            kd_loss = self.distillation_loss(teacher_logits, student_logits)
            ce_loss = self.ce_loss(student_logits, target)
            ensemble_loss = self.weighted_ensemble_loss(teacher_logits, student_logits, target)
            
            # 组合损失
            total_batch_loss = (self.alpha * kd_loss + 
                              (1 - self.alpha) * ce_loss + 
                              0.1 * ensemble_loss)  # 集成损失权重较小
            
            # 反向传播和优化
            self.student_optimizer.zero_grad()
            total_batch_loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.student_optimizer.step()
            
            # 统计信息
            total_loss += total_batch_loss.item()
            kd_loss_total += kd_loss.item()
            ce_loss_total += ce_loss.item()
            
            # 计算准确率
            _, predicted = torch.max(student_logits, 1)
            student_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Total Loss: {total_batch_loss.item():.4f}, '
                      f'KD Loss: {kd_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}')
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均值
        avg_loss = total_loss / len(train_loader)
        kd_loss_avg = kd_loss_total / len(train_loader)
        ce_loss_avg = ce_loss_total / len(train_loader)
        student_acc = 100. * student_correct / total_samples
        
        return avg_loss, student_acc, kd_loss_avg, ce_loss_avg
    
    def evaluate(self, test_loader, model_type='student'):
        """
        评估模型性能
        
        Args:
            test_loader: 测试数据加载器
            model_type: 要评估的模型类型 ('student' 或 'teacher')
            
        Returns:
            accuracy: 模型准确率
        """
        if model_type == 'student':
            model = self.student
        else:
            model = self.teacher
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def get_model_complexity(self):
        """获取模型复杂度信息"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        teacher_params = count_parameters(self.teacher)
        student_params = count_parameters(self.student)
        
        return {
            'teacher_params': teacher_params,
            'student_params': student_params,
            'compression_ratio': teacher_params / student_params,
            'params_reduced': teacher_params - student_params
        }

# 数据加载函数
def get_data_loaders(batch_size=128):
    """获取MNIST数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                  transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, 
                                 transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# 训练教师模型
def train_teacher_model(train_loader, test_loader, epochs=15):
    """训练教师模型"""
    teacher = TeacherModel().to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0
    history = {'train_loss': [], 'test_acc': []}
    
    print("训练教师模型中...")
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = teacher(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估
        teacher.eval()
        test_acc = evaluate_model(teacher, test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(teacher.state_dict(), 'best_teacher.pth')
        
        history['train_loss'].append(total_loss / len(train_loader))
        history['test_acc'].append(test_acc)
        
        print(f'Teacher Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, '
              f'Test Acc: {test_acc:.2f}%')
    
    return teacher, history

def evaluate_model(model, test_loader):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100. * correct / total

# 训练监督知识蒸馏
def train_supervised_distillation():
    """训练监督知识蒸馏"""
    # 获取数据
    train_loader, test_loader = get_data_loaders()
    
    # 训练或加载教师模型
    print("准备教师模型...")
    teacher_model, teacher_history = train_teacher_model(train_loader, test_loader, epochs=15)
    teacher_model.load_state_dict(torch.load('best_teacher.pth'))
    teacher_acc = evaluate_model(teacher_model, test_loader)
    print(f"教师模型最终测试准确率: {teacher_acc:.2f}%")
    
    # 初始化学生模型
    student_model = StudentModel()
    
    # 初始化蒸馏训练器
    distiller = SupervisedKnowledgeDistillation(
        teacher_model, student_model, device,
        temperature=4.0, alpha=0.7,
        teacher_weight=1.0, student_weight=1.0
    )
    
    # 训练记录
    history = {
        'total_loss': [], 'kd_loss': [], 'ce_loss': [],
        'train_acc': [], 'test_acc': [], 'teacher_acc': []
    }
    
    # 训练循环
    epochs = 20
    print("\n开始监督知识蒸馏训练...")
    for epoch in range(epochs):
        total_loss, train_acc, kd_loss, ce_loss = distiller.train_epoch(train_loader)
        test_acc = distiller.evaluate(test_loader, 'student')
        
        # 记录结果
        history['total_loss'].append(total_loss)
        history['kd_loss'].append(kd_loss)
        history['ce_loss'].append(ce_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['teacher_acc'].append(teacher_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Total Loss: {total_loss:.4f} (KD: {kd_loss:.4f}, CE: {ce_loss:.4f})')
        print(f'  Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        print('-' * 50)
    
    return history, distiller

# 可视化结果
def plot_supervised_results(history, distiller):
    """可视化训练结果"""
    plt.figure(figsize=(18, 12))
    
    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['total_loss'], label='Total Loss', marker='o', linewidth=2, color='red')
    plt.plot(history['kd_loss'], label='KD Loss', marker='s', linestyle='--', color='blue')
    plt.plot(history['ce_loss'], label='CE Loss', marker='^', linestyle='--', color='green')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='Student Train', marker='o', linewidth=2, color='orange')
    plt.plot(history['test_acc'], label='Student Test', marker='s', linewidth=2, color='purple')
    plt.plot([history['teacher_acc'][0]] * len(history['test_acc']), 
             label='Teacher', linestyle='--', linewidth=2, color='red')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 损失组成比例
    plt.subplot(2, 3, 3)
    final_kd_ratio = history['kd_loss'][-1] / history['total_loss'][-1]
    final_ce_ratio = history['ce_loss'][-1] / history['total_loss'][-1]
    
    labels = ['KD Loss', 'CE Loss']
    sizes = [final_kd_ratio, final_ce_ratio]
    colors = ['lightblue', 'lightgreen']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Final Loss Composition')
    
    # 模型复杂度对比
    plt.subplot(2, 3, 4)
    complexity = distiller.get_model_complexity()
    
    models = ['Teacher', 'Student']
    params = [complexity['teacher_params'], complexity['student_params']]
    colors = ['lightcoral', 'lightblue']
    
    bars = plt.bar(models, params, color=colors, alpha=0.7)
    plt.title('Model Complexity (Parameters)')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')
    
    for bar, param in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{param:,}', ha='center', va='bottom')
    
    # 最终性能对比
    plt.subplot(2, 3, 5)
    final_teacher_acc = history['teacher_acc'][-1]
    final_student_acc = history['test_acc'][-1]
    
    models = ['Teacher', 'Student']
    accuracies = [final_teacher_acc, final_student_acc]
    colors = ['lightcoral', 'lightblue']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.title('Final Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # 压缩效果
    plt.subplot(2, 3, 6)
    compression_ratio = complexity['compression_ratio']
    params_reduced = complexity['params_reduced']
    
    metrics = ['Compression Ratio', 'Params Reduced']
    values = [compression_ratio, params_reduced / 1000]  # 转换为千参数
    colors = ['lightgreen', 'lightyellow']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Model Compression Effect')
    plt.ylabel('Value')
    
    for bar, value in zip(bars, values):
        if 'Ratio' in metrics[bars.index(bar)]:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}x', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value:.0f}K', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('supervised_distillation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
def main():
    print("开始监督知识蒸馏训练...")
    
    # 训练监督知识蒸馏
    history, distiller = train_supervised_distillation()
    
    # 获取最终性能
    final_teacher_acc = history['teacher_acc'][-1]
    final_student_acc = history['test_acc'][-1]
    complexity = distiller.get_model_complexity()
    
    print(f"\n最终结果:")
    print(f"教师模型准确率: {final_teacher_acc:.2f}%")
    print(f"学生模型准确率: {final_student_acc:.2f}%")
    print(f"参数压缩比例: {complexity['compression_ratio']:.1f}x")
    print(f"参数减少量: {complexity['params_reduced']:,}")
    
    # 可视化结果
    plot_supervised_results(history, distiller)
    
    # 保存模型
    torch.save(distiller.student.state_dict(), 'supervised_student_model.pth')
    print("学生模型已保存")

if __name__ == "__main__":
    main()