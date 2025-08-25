import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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

# 教师模型（更大）
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

# 学生模型（更小）
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

# 半监督知识蒸馏训练器
class SemiSupervisedDistillation:
    def __init__(self, teacher, student, device, 
                 temperature=3.0, alpha=0.7, lambda_u=0.5, confidence_threshold=0.9):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha  # 有标签数据中蒸馏损失的权重
        self.lambda_u = lambda_u  # 无标签数据损失的权重
        self.confidence_threshold = confidence_threshold
        
        self.student_optimizer = optim.Adam(self.student.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.student_optimizer, step_size=10, gamma=0.5)
    
    def kl_divergence(self, p_logits, q_logits, temperature=None):
        """计算KL散度损失"""
        if temperature is None:
            temperature = self.temperature
        p = F.softmax(p_logits / temperature, dim=1)
        q = F.log_softmax(q_logits / temperature, dim=1)
        return F.kl_div(q, p, reduction='batchmean') * (temperature ** 2)
    
    def generate_pseudo_labels(self, unlabeled_data):
        """为无标签数据生成伪标签"""
        self.teacher.eval()
        with torch.no_grad():
            logits = self.teacher(unlabeled_data)
            probabilities = F.softmax(logits, dim=1)
            max_probs, pseudo_labels = torch.max(probabilities, dim=1)
            
            # 置信度过滤
            confident_mask = max_probs > self.confidence_threshold
            confident_data = unlabeled_data[confident_mask]
            confident_labels = pseudo_labels[confident_mask]
            confident_probs = max_probs[confident_mask]
            
        return confident_data, confident_labels, confident_probs
    
    def train_epoch(self, labeled_loader, unlabeled_loader):
        """训练一个epoch"""
        self.teacher.eval()  # 教师模型固定
        self.student.train()
        
        total_loss = 0
        labeled_loss_total = 0
        unlabeled_loss_total = 0
        student_acc = 0
        total_samples = 0
        used_unlabeled = 0
        
        # 处理有标签数据
        for (labeled_data, labels), unlabeled_data in zip(labeled_loader, unlabeled_loader):
            labeled_data, labels = labeled_data.to(self.device), labels.to(self.device)
            unlabeled_data = unlabeled_data[0].to(self.device)  # 无标签数据没有标签
            
            # 为无标签数据生成伪标签
            confident_data, pseudo_labels, confident_probs = self.generate_pseudo_labels(unlabeled_data)
            used_unlabeled += len(confident_data)
            
            # 有标签数据的前向传播
            with torch.no_grad():
                teacher_labeled_logits = self.teacher(labeled_data)
            student_labeled_logits = self.student(labeled_data)
            
            # 有标签数据的损失
            supervised_ce = F.cross_entropy(student_labeled_logits, labels)
            supervised_kd = self.kl_divergence(teacher_labeled_logits, student_labeled_logits)
            supervised_loss = (1 - self.alpha) * supervised_ce + self.alpha * supervised_kd
            
            # 无标签数据的损失（如果有高置信度样本）
            unlabeled_loss = 0
            if len(confident_data) > 0:
                with torch.no_grad():
                    teacher_unlabeled_logits = self.teacher(confident_data)
                student_unlabeled_logits = self.student(confident_data)
                
                # 使用伪标签的交叉熵损失
                unlabeled_ce = F.cross_entropy(student_unlabeled_logits, pseudo_labels)
                
                # 蒸馏损失
                unlabeled_kd = self.kl_divergence(teacher_unlabeled_logits, student_unlabeled_logits)
                
                # 加权组合
                unlabeled_loss = (1 - self.alpha) * unlabeled_ce + self.alpha * unlabeled_kd
                
                # 根据置信度加权
                confidence_weights = confident_probs / confident_probs.sum()
                unlabeled_loss = (unlabeled_loss * confidence_weights).sum()
            
            # 总损失
            total_batch_loss = supervised_loss + self.lambda_u * unlabeled_loss
            
            # 反向传播
            self.student_optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.student_optimizer.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            labeled_loss_total += supervised_loss.item()
            unlabeled_loss_total += unlabeled_loss.item() if len(confident_data) > 0 else 0
            
            # 计算准确率
            _, preds = torch.max(student_labeled_logits, 1)
            student_acc += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        # 更新学习率
        self.scheduler.step()
        
        avg_loss = total_loss / len(labeled_loader)
        avg_labeled_loss = labeled_loss_total / len(labeled_loader)
        avg_unlabeled_loss = unlabeled_loss_total / max(1, used_unlabeled)
        student_acc = 100. * student_acc / total_samples
        utilization_rate = 100. * used_unlabeled / (len(unlabeled_loader) * unlabeled_loader.batch_size)
        
        return avg_loss, avg_labeled_loss, avg_unlabeled_loss, student_acc, utilization_rate
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.student.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.student(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

# 数据准备函数
def prepare_semi_supervised_data(labeled_ratio=0.1, batch_size=128):
    """准备半监督学习数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载完整训练集
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # 分割有标签和无标签数据
    indices = list(range(len(full_train_dataset)))
    labeled_indices, unlabeled_indices = train_test_split(
        indices, train_size=labeled_ratio, random_state=42, stratify=full_train_dataset.targets
    )
    
    # 创建有标签数据集
    labeled_dataset = Subset(full_train_dataset, labeled_indices)
    
    # 创建无标签数据集（移除标签）
    class UnlabeledDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            data, _ = self.dataset[self.indices[idx]]  # 忽略标签
            return data,
    
    unlabeled_dataset = UnlabeledDataset(full_train_dataset, unlabeled_indices)
    
    # 测试集
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # 创建数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"有标签样本: {len(labeled_dataset)}")
    print(f"无标签样本: {len(unlabeled_dataset)}")
    print(f"测试样本: {len(test_dataset)}")
    
    return labeled_loader, unlabeled_loader, test_loader

# 先训练教师模型
def train_teacher_model(train_loader, test_loader, epochs=10):
    """训练教师模型"""
    teacher = TeacherModel().to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
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
        acc = evaluate_model(teacher, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(teacher.state_dict(), 'best_teacher.pth')
        
        print(f'Teacher Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
    
    return teacher

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

# 训练函数
def train_semi_supervised_distillation():
    # 准备数据（10%有标签，90%无标签）
    labeled_loader, unlabeled_loader, test_loader = prepare_semi_supervised_data(
        labeled_ratio=0.1, batch_size=128
    )
    
    # 训练教师模型（使用所有有标签数据）
    print("训练教师模型...")
    teacher_model = train_teacher_model(labeled_loader, test_loader, epochs=15)
    
    # 加载最佳教师模型
    teacher_model.load_state_dict(torch.load('best_teacher.pth'))
    teacher_acc = evaluate_model(teacher_model, test_loader)
    print(f"教师模型测试准确率: {teacher_acc:.2f}%")
    
    # 初始化学生模型
    student_model = StudentModel()
    
    # 初始化半监督蒸馏器
    distiller = SemiSupervisedDistillation(
        teacher_model, student_model, device,
        temperature=3.0, alpha=0.7, lambda_u=0.5, confidence_threshold=0.95
    )
    
    # 训练记录
    history = {
        'total_loss': [], 'labeled_loss': [], 'unlabeled_loss': [],
        'train_acc': [], 'test_acc': [], 'utilization_rate': []
    }
    
    # 训练循环
    epochs = 20
    for epoch in range(epochs):
        total_loss, labeled_loss, unlabeled_loss, train_acc, util_rate = distiller.train_epoch(
            labeled_loader, unlabeled_loader
        )
        
        test_acc = distiller.evaluate(test_loader)
        
        # 记录结果
        history['total_loss'].append(total_loss)
        history['labeled_loss'].append(labeled_loss)
        history['unlabeled_loss'].append(unlabeled_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['utilization_rate'].append(util_rate)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Total Loss: {total_loss:.4f} (Labeled: {labeled_loss:.4f}, Unlabeled: {unlabeled_loss:.4f})')
        print(f'  Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        print(f'  Unlabeled Utilization: {util_rate:.2f}%')
        print('-' * 60)
    
    return history, student_model

# 可视化结果
def plot_semi_supervised_results(history):
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['total_loss'], label='Total Loss', marker='o', linewidth=2)
    plt.plot(history['labeled_loss'], label='Labeled Loss', marker='s', linestyle='--')
    plt.plot(history['unlabeled_loss'], label='Unlabeled Loss', marker='^', linestyle='--')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    plt.plot(history['test_acc'], label='Test Accuracy', marker='s', linewidth=2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 无标签数据利用率
    plt.subplot(2, 2, 3)
    plt.plot(history['utilization_rate'], label='Utilization Rate', marker='o', color='purple', linewidth=2)
    plt.title('Unlabeled Data Utilization')
    plt.xlabel('Epoch')
    plt.ylabel('Utilization Rate (%)')
    plt.legend()
    plt.grid(True)
    
    # 最终性能对比
    plt.subplot(2, 2, 4)
    final_test_acc = history['test_acc'][-1]
    utilization = history['utilization_rate'][-1]
    
    metrics = ['Test Accuracy', 'Utilization Rate']
    values = [final_test_acc, utilization]
    colors = ['lightblue', 'lightgreen']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Final Performance')
    plt.ylabel('Value (%)')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('semi_supervised_distillation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
def main():
    print("开始半监督知识蒸馏训练...")
    
    # 训练模型
    history, student_model = train_semi_supervised_distillation()
    
    # 可视化结果
    plot_semi_supervised_results(history)
    
    # 保存学生模型
    torch.save(student_model.state_dict(), 'semi_supervised_student.pth')
    print("学生模型已保存")

if __name__ == "__main__":
    main()