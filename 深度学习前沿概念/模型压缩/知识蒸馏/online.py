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
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
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

# 教师模型（更深更宽）
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

# 学生模型（更轻量）
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

# 在线知识蒸馏训练器
class OnlineKnowledgeDistillation:
    def __init__(self, teacher, student, device, temperature=3.0, alpha=0.5):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        # 使用不同的优化器
        self.teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=0.001, weight_decay=1e-4)
        self.student_optimizer = optim.Adam(self.student.parameters(), lr=0.001, weight_decay=1e-4)
        
        self.teacher_scheduler = optim.lr_scheduler.StepLR(self.teacher_optimizer, step_size=10, gamma=0.5)
        self.student_scheduler = optim.lr_scheduler.StepLR(self.student_optimizer, step_size=10, gamma=0.5)
        
    def kl_divergence(self, p_logits, q_logits):
        """计算KL散度损失"""
        p = F.softmax(p_logits / self.temperature, dim=1)
        q = F.log_softmax(q_logits / self.temperature, dim=1)
        return F.kl_div(q, p, reduction='batchmean') * (self.temperature ** 2)
    
    def mutual_distillation_loss(self, teacher_logits, student_logits, labels):
        """相互蒸馏损失"""
        # 各自的分类损失
        teacher_ce = F.cross_entropy(teacher_logits, labels)
        student_ce = F.cross_entropy(student_logits, labels)
        
        # 相互蒸馏损失
        teacher_to_student = self.kl_divergence(teacher_logits, student_logits)
        student_to_teacher = self.kl_divergence(student_logits, teacher_logits)
        
        # 总损失
        total_loss = (teacher_ce + student_ce + 
                     self.alpha * (teacher_to_student + student_to_teacher))
        
        return total_loss, teacher_ce, student_ce, teacher_to_student, student_to_teacher
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.teacher.train()
        self.student.train()
        
        total_loss = 0
        teacher_acc = 0
        student_acc = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            teacher_logits = self.teacher(data)
            student_logits = self.student(data)
            
            # 计算损失
            loss, teacher_ce, student_ce, t2s, s2t = self.mutual_distillation_loss(
                teacher_logits, student_logits, target
            )
            
            # 反向传播
            self.teacher_optimizer.zero_grad()
            self.student_optimizer.zero_grad()
            
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.teacher_optimizer.step()
            self.student_optimizer.step()
            
            # 计算准确率
            _, teacher_pred = torch.max(teacher_logits, 1)
            _, student_pred = torch.max(student_logits, 1)
            
            teacher_acc += (teacher_pred == target).sum().item()
            student_acc += (student_pred == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Loss: {loss.item():.4f}, '
                      f'Teacher CE: {teacher_ce.item():.4f}, Student CE: {student_ce.item():.4f}')
        
        # 更新学习率
        self.teacher_scheduler.step()
        self.student_scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        teacher_acc = 100. * teacher_acc / total_samples
        student_acc = 100. * student_acc / total_samples
        
        return avg_loss, teacher_acc, student_acc
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.teacher.eval()
        self.student.eval()
        
        teacher_correct = 0
        student_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                teacher_outputs = self.teacher(data)
                student_outputs = self.student(data)
                
                _, teacher_pred = torch.max(teacher_outputs.data, 1)
                _, student_pred = torch.max(student_outputs.data, 1)
                
                teacher_correct += (teacher_pred == target).sum().item()
                student_correct += (student_pred == target).sum().item()
                total += target.size(0)
        
        teacher_acc = 100. * teacher_correct / total
        student_acc = 100. * student_correct / total
        
        return teacher_acc, student_acc

# 数据加载
def get_data_loaders(batch_size=128):
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

# 训练函数
def train_online_distillation():
    # 初始化模型
    teacher_model = TeacherModel()
    student_model = StudentModel()
    
    # 数据加载
    train_loader, test_loader = get_data_loaders()
    
    # 初始化蒸馏训练器
    distiller = OnlineKnowledgeDistillation(
        teacher_model, student_model, device,
        temperature=3.0, alpha=0.7
    )
    
    # 训练记录
    history = {
        'train_loss': [], 'teacher_acc': [], 'student_acc': [],
        'test_teacher_acc': [], 'test_student_acc': []
    }
    
    # 训练循环
    epochs = 15
    for epoch in range(epochs):
        # 训练一个epoch
        train_loss, train_teacher_acc, train_student_acc = distiller.train_epoch(train_loader)
        
        # 评估
        test_teacher_acc, test_student_acc = distiller.evaluate(test_loader)
        
        # 记录结果
        history['train_loss'].append(train_loss)
        history['teacher_acc'].append(train_teacher_acc)
        history['student_acc'].append(train_student_acc)
        history['test_teacher_acc'].append(test_teacher_acc)
        history['test_student_acc'].append(test_student_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Train Acc - Teacher: {train_teacher_acc:.2f}%, Student: {train_student_acc:.2f}%')
        print(f'  Test Acc - Teacher: {test_teacher_acc:.2f}%, Student: {test_student_acc:.2f}%')
        print('-' * 50)
    
    return history, distiller.teacher, distiller.student

# 可视化结果
def plot_results(history):
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 训练准确率
    plt.subplot(2, 2, 2)
    plt.plot(history['teacher_acc'], label='Teacher Train Acc', marker='s')
    plt.plot(history['student_acc'], label='Student Train Acc', marker='^')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 测试准确率
    plt.subplot(2, 2, 3)
    plt.plot(history['test_teacher_acc'], label='Teacher Test Acc', marker='s')
    plt.plot(history['test_student_acc'], label='Student Test Acc', marker='^')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 模型对比
    plt.subplot(2, 2, 4)
    final_teacher_acc = history['test_teacher_acc'][-1]
    final_student_acc = history['test_student_acc'][-1]
    
    models = ['Teacher', 'Student']
    accuracies = [final_teacher_acc, final_student_acc]
    colors = ['lightblue', 'lightgreen']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.title('Final Model Performance')
    plt.ylabel('Accuracy (%)')
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('online_distillation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 模型参数统计
def model_statistics(teacher, student):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    compression_ratio = teacher_params / student_params
    
    print(f"\n模型参数统计:")
    print(f"教师模型参数: {teacher_params:,}")
    print(f"学生模型参数: {student_params:,}")
    print(f"压缩比例: {compression_ratio:.2f}x")
    print(f"参数减少: {(teacher_params - student_params):,}")

# 主函数
def main():
    print("开始在线知识蒸馏训练...")
    
    # 训练模型
    history, teacher_model, student_model = train_online_distillation()
    
    # 可视化结果
    plot_results(history)
    
    # 模型统计
    model_statistics(teacher_model, student_model)
    
    # 保存模型
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    torch.save(student_model.state_dict(), 'student_model.pth')
    print("模型已保存")

if __name__ == "__main__":
    main()