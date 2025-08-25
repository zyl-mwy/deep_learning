import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义教师模型（较大的模型）
class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 定义学生模型（较小的模型）
class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 800)
        self.fc2 = nn.Linear(800, num_classes)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 知识蒸馏损失函数
class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # 计算蒸馏损失（KL散度）
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 计算学生模型的交叉熵损失
        student_loss = self.ce_loss(student_logits, labels)
        
        # 组合损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return total_loss

# 训练教师模型
def train_teacher(model, train_loader, test_loader, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 评估
        accuracy = evaluate(model, test_loader)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accuracies.append(accuracy)
        
        print(f'教师模型 - Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies

# 使用知识蒸馏训练学生模型
def train_student_with_distillation(teacher, student, train_loader, test_loader, 
                                  temperature=5.0, alpha=0.7, epochs=10):
    teacher.eval()  # 教师模型设为评估模式
    student.to(device)
    
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    criterion = DistillationLoss(temperature, alpha)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            student_logits = student(data)
            loss = criterion(student_logits, teacher_logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 评估
        accuracy = evaluate(student, test_loader)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accuracies.append(accuracy)
        
        print(f'学生模型 - Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies

# 评估函数
def evaluate(model, test_loader):
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
    
    accuracy = 100 * correct / total
    return accuracy

# 数据加载和预处理
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 主函数
def main():
    # 超参数
    temperature = 5.0
    alpha = 0.7
    teacher_epochs = 5
    student_epochs = 10
    
    # 获取数据
    train_loader, test_loader = get_data_loaders()
    
    # 初始化模型
    teacher_model = TeacherModel()
    student_model = StudentModel()
    
    print("训练教师模型...")
    teacher_loss, teacher_acc = train_teacher(teacher_model, train_loader, test_loader, teacher_epochs)
    
    print("\n使用知识蒸馏训练学生模型...")
    student_loss, student_acc = train_student_with_distillation(
        teacher_model, student_model, train_loader, test_loader, 
        temperature, alpha, student_epochs
    )
    
    # 比较性能
    final_teacher_acc = evaluate(teacher_model, test_loader)
    final_student_acc = evaluate(student_model, test_loader)
    
    print(f"\n最终结果:")
    print(f"教师模型测试准确率: {final_teacher_acc:.2f}%")
    print(f"学生模型测试准确率: {final_student_acc:.2f}%")
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(teacher_acc, label='Teacher', marker='o')
    plt.plot(student_acc, label='Student', marker='s')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(teacher_loss, label='Teacher', marker='o')
    plt.plot(student_loss, label='Student', marker='s')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('knowledge_distillation_results.png')
    plt.show()

if __name__ == "__main__":
    main()