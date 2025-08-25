import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义教师模型（较大的CNN模型）
class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 定义学生模型（较小的CNN模型）
class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 知识蒸馏损失函数
class DistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # 软化教师输出
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # 软化学生输出
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # 计算蒸馏损失
        distillation_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 计算学生分类损失
        student_loss = self.ce_loss(student_logits, labels)
        
        # 组合损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss

# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, model_type="Model"):
    model.to(device)
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # 评估测试集
        test_accuracy = evaluate_model(model, test_loader)
        train_accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'{model_type} - Epoch: {epoch+1}/{epochs}, '
              f'Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}%')
    
    return train_losses, test_accuracies

# 离线知识蒸馏训练
def offline_knowledge_distillation(teacher, student, train_loader, test_loader, 
                                 temperature=5.0, alpha=0.7, epochs=10, lr=0.001):
    """
    离线知识蒸馏训练过程
    teacher: 预训练好的教师模型
    student: 待训练的学生模型
    """
    # 设置教师模型为评估模式
    teacher.eval()
    teacher.to(device)
    student.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = DistillationLoss(temperature, alpha)
    
    train_losses = []
    test_accuracies = []
    
    print("开始离线知识蒸馏训练...")
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 教师模型前向传播（不计算梯度）
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # 学生模型前向传播
            student_logits = student(data)
            
            # 计算蒸馏损失
            loss = criterion(student_logits, teacher_logits, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = student_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # 评估学生模型
        test_accuracy = evaluate_model(student, test_loader)
        train_accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Distillation - Epoch: {epoch+1}/{epochs}, '
              f'Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}%')
    
    return train_losses, test_accuracies

# 模型评估函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# 数据加载和预处理
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 使用MNIST数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 可视化结果
def plot_results(teacher_history, student_history, distillation_history):
    plt.figure(figsize=(15, 10))
    
    # 准确率对比
    plt.subplot(2, 2, 1)
    plt.plot(teacher_history['test_acc'], label='Teacher', marker='o', linewidth=2)
    plt.plot(student_history['test_acc'], label='Student (No Distill)', marker='s', linewidth=2)
    plt.plot(distillation_history['test_acc'], label='Student (Distill)', marker='^', linewidth=2)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 损失对比
    plt.subplot(2, 2, 2)
    plt.plot(teacher_history['train_loss'], label='Teacher', marker='o', linewidth=2)
    plt.plot(student_history['train_loss'], label='Student (No Distill)', marker='s', linewidth=2)
    plt.plot(distillation_history['train_loss'], label='Student (Distill)', marker='^', linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 最终性能对比
    plt.subplot(2, 2, 3)
    models = ['Teacher', 'Student (No Distill)', 'Student (Distill)']
    accuracies = [
        teacher_history['test_acc'][-1],
        student_history['test_acc'][-1],
        distillation_history['test_acc'][-1]
    ]
    plt.bar(models, accuracies, color=['blue', 'red', 'green'])
    plt.title('Final Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # 参数量对比
    plt.subplot(2, 2, 4)
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    param_ratios = [teacher_params, student_params, student_params]
    plt.bar(models, param_ratios, color=['blue', 'red', 'green'])
    plt.title('Number of Parameters')
    plt.ylabel('Parameters')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('offline_distillation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
def main():
    # 超参数设置
    teacher_epochs = 10
    student_epochs = 15
    distillation_epochs = 10
    temperature = 5.0
    alpha = 0.7
    batch_size = 128
    
    # 获取数据
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 初始化模型
    global teacher_model, student_model
    teacher_model = TeacherModel()
    student_model = StudentModel()
    
    print(f"教师模型参数量: {count_parameters(teacher_model):,}")
    print(f"学生模型参数量: {count_parameters(student_model):,}")
    print(f"压缩比例: {count_parameters(teacher_model)/count_parameters(student_model):.2f}x")
    
    # 1. 训练教师模型
    print("\n" + "="*50)
    print("训练教师模型...")
    print("="*50)
    
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    teacher_criterion = nn.CrossEntropyLoss()
    
    teacher_train_loss, teacher_test_acc = train_model(
        teacher_model, train_loader, test_loader, 
        teacher_criterion, teacher_optimizer, teacher_epochs, "Teacher"
    )
    
    # 2. 训练学生模型（无蒸馏，作为基线）
    print("\n" + "="*50)
    print("训练学生模型（无知识蒸馏）...")
    print("="*50)
    
    student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    student_criterion = nn.CrossEntropyLoss()
    
    student_train_loss, student_test_acc = train_model(
        student_model, train_loader, test_loader,
        student_criterion, student_optimizer, student_epochs, "Student (No Distill)"
    )
    
    # 3. 离线知识蒸馏
    print("\n" + "="*50)
    print("进行离线知识蒸馏...")
    print("="*50)
    
    # 重新初始化学生模型
    distilled_student = StudentModel()
    distillation_train_loss, distillation_test_acc = offline_knowledge_distillation(
        teacher_model, distilled_student, train_loader, test_loader,
        temperature, alpha, distillation_epochs
    )
    
    # 最终评估
    print("\n" + "="*50)
    print("最终性能评估")
    print("="*50)
    
    teacher_acc = evaluate_model(teacher_model, test_loader)
    student_acc = evaluate_model(student_model, test_loader)
    distilled_acc = evaluate_model(distilled_student, test_loader)
    
    print(f"教师模型测试准确率: {teacher_acc:.2f}%")
    print(f"学生模型测试准确率（无蒸馏）: {student_acc:.2f}%")
    print(f"学生模型测试准确率（有蒸馏）: {distilled_acc:.2f}%")
    print(f"蒸馏带来的提升: {distilled_acc - student_acc:.2f}%")
    
    # 保存结果
    results = {
        'teacher': {
            'train_loss': teacher_train_loss,
            'test_acc': teacher_test_acc,
            'final_acc': teacher_acc
        },
        'student_no_distill': {
            'train_loss': student_train_loss,
            'test_acc': student_test_acc,
            'final_acc': student_acc
        },
        'student_distill': {
            'train_loss': distillation_train_loss,
            'test_acc': distillation_test_acc,
            'final_acc': distilled_acc
        }
    }
    
    # 可视化结果
    plot_results(
        results['teacher'],
        results['student_no_distill'],
        results['student_distill']
    )
    
    return results

if __name__ == "__main__":
    start_time = time.time()
    results = main()
    end_time = time.time()
    
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
    
    # 保存模型（可选）
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    torch.save(student_model.state_dict(), 'student_model_distilled.pth')