"""
Module Visualization cho phân tích dữ liệu bệnh tim mạch
Sử dụng Matplotlib, Seaborn và Plotly để tạo các biểu đồ trực quan
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'


class CardioVisualizer:
    """Class tạo các biểu đồ phân tích dữ liệu bệnh tim mạch"""
    
    def __init__(self, output_dir: str = "outputs/plots"):
        """
        Args:
            output_dir: Thư mục lưu các biểu đồ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Màu sắc cho target variable
        self.cardio_colors = {0: '#2ecc71', 1: '#e74c3c'}  # Xanh: không bệnh, Đỏ: có bệnh
        self.palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
    def plot_target_distribution(self, df: pd.DataFrame, save: bool = True):
        """
        Biểu đồ phân bố target variable (cardio)
        
        Args:
            df: Pandas DataFrame
            save: Lưu biểu đồ hay không
        """
        logger.info("Vẽ biểu đồ phân bố target variable...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        cardio_counts = df['cardio'].value_counts().sort_index()
        axes[0].bar(
            ['Không bệnh', 'Có bệnh'],
            cardio_counts.values,
            color=[self.cardio_colors[0], self.cardio_colors[1]],
            alpha=0.8,
            edgecolor='black'
        )
        axes[0].set_title('Phân bố Bệnh Tim mạch', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Số lượng bệnh nhân')
        
        # Thêm số liệu lên cột
        for i, v in enumerate(cardio_counts.values):
            axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
        
        # Pie chart
        axes[1].pie(
            cardio_counts.values,
            labels=['Không bệnh', 'Có bệnh'],
            colors=[self.cardio_colors[0], self.cardio_colors[1]],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0, 0.1)
        )
        axes[1].set_title('Tỷ lệ Bệnh Tim mạch', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Đã lưu biểu đồ tại {self.output_dir / 'target_distribution.png'}")
        
        return fig
    
    def plot_age_analysis(self, df: pd.DataFrame, save: bool = True):
        """
        Phân tích về tuổi
        
        Args:
            df: Pandas DataFrame
            save: Lưu biểu đồ hay không
        """
        logger.info("Vẽ biểu đồ phân tích tuổi...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram phân bố tuổi
        axes[0, 0].hist(df[df['cardio'] == 0]['age_years'], bins=30, alpha=0.6, 
                        label='Không bệnh', color=self.cardio_colors[0], edgecolor='black')
        axes[0, 0].hist(df[df['cardio'] == 1]['age_years'], bins=30, alpha=0.6, 
                        label='Có bệnh', color=self.cardio_colors[1], edgecolor='black')
        axes[0, 0].set_xlabel('Tuổi (năm)')
        axes[0, 0].set_ylabel('Số lượng')
        axes[0, 0].set_title('Phân bố Tuổi theo Tình trạng Bệnh', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot tuổi theo cardio
        sns.boxplot(data=df, x='cardio', y='age_years', palette=[self.cardio_colors[0], self.cardio_colors[1]], ax=axes[0, 1])
        axes[0, 1].set_xticklabels(['Không bệnh', 'Có bệnh'])
        axes[0, 1].set_xlabel('Tình trạng')
        axes[0, 1].set_ylabel('Tuổi (năm)')
        axes[0, 1].set_title('Phân bố Tuổi qua Box Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Phân bố theo nhóm tuổi
        age_cardio = df.groupby(['age_group', 'cardio']).size().unstack(fill_value=0)
        age_cardio.plot(kind='bar', ax=axes[1, 0], color=[self.cardio_colors[0], self.cardio_colors[1]], 
                        edgecolor='black', alpha=0.8)
        axes[1, 0].set_xlabel('Nhóm tuổi')
        axes[1, 0].set_ylabel('Số lượng')
        axes[1, 0].set_title('Số lượng Bệnh nhân theo Nhóm tuổi', fontweight='bold')
        axes[1, 0].legend(['Không bệnh', 'Có bệnh'])
        axes[1, 0].tick_params(axis='x', rotation=0)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Tỷ lệ mắc bệnh theo nhóm tuổi
        age_rate = df.groupby('age_group')['cardio'].mean() * 100
        axes[1, 1].bar(age_rate.index, age_rate.values, color=self.palette[3], 
                       edgecolor='black', alpha=0.8)
        axes[1, 1].set_xlabel('Nhóm tuổi')
        axes[1, 1].set_ylabel('Tỷ lệ mắc bệnh (%)')
        axes[1, 1].set_title('Tỷ lệ Mắc bệnh theo Nhóm tuổi', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Thêm giá trị lên cột
        for i, v in enumerate(age_rate.values):
            axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'age_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Đã lưu biểu đồ tại {self.output_dir / 'age_analysis.png'}")
        
        return fig
    
    def plot_bmi_analysis(self, df: pd.DataFrame, save: bool = True):
        """
        Phân tích về BMI
        
        Args:
            df: Pandas DataFrame
            save: Lưu biểu đồ hay không
        """
        logger.info("Vẽ biểu đồ phân tích BMI...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot BMI vs Age
        for cardio_val in [0, 1]:
            data = df[df['cardio'] == cardio_val]
            axes[0, 0].scatter(data['age_years'], data['bmi'], 
                              alpha=0.3, s=20, 
                              color=self.cardio_colors[cardio_val],
                              label='Có bệnh' if cardio_val == 1 else 'Không bệnh')
        axes[0, 0].set_xlabel('Tuổi (năm)')
        axes[0, 0].set_ylabel('BMI')
        axes[0, 0].set_title('BMI vs Tuổi', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot BMI theo cardio
        sns.boxplot(data=df, x='cardio', y='bmi', palette=[self.cardio_colors[0], self.cardio_colors[1]], ax=axes[0, 1])
        axes[0, 1].set_xticklabels(['Không bệnh', 'Có bệnh'])
        axes[0, 1].set_xlabel('Tình trạng')
        axes[0, 1].set_ylabel('BMI')
        axes[0, 1].set_title('Phân bố BMI qua Box Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Phân bố theo nhóm BMI
        bmi_order = ['Thiếu cân', 'Bình thường', 'Thừa cân', 'Béo phì']
        bmi_cardio = df.groupby(['bmi_category', 'cardio']).size().unstack(fill_value=0)
        bmi_cardio = bmi_cardio.reindex(bmi_order, fill_value=0)
        bmi_cardio.plot(kind='bar', ax=axes[1, 0], color=[self.cardio_colors[0], self.cardio_colors[1]], 
                        edgecolor='black', alpha=0.8)
        axes[1, 0].set_xlabel('Nhóm BMI')
        axes[1, 0].set_ylabel('Số lượng')
        axes[1, 0].set_title('Số lượng Bệnh nhân theo Nhóm BMI', fontweight='bold')
        axes[1, 0].legend(['Không bệnh', 'Có bệnh'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Tỷ lệ mắc bệnh theo nhóm BMI
        bmi_rate = df.groupby('bmi_category')['cardio'].mean() * 100
        bmi_rate = bmi_rate.reindex(bmi_order)
        axes[1, 1].bar(range(len(bmi_rate)), bmi_rate.values, color=self.palette[2], 
                       edgecolor='black', alpha=0.8)
        axes[1, 1].set_xticks(range(len(bmi_rate)))
        axes[1, 1].set_xticklabels(bmi_rate.index, rotation=45)
        axes[1, 1].set_xlabel('Nhóm BMI')
        axes[1, 1].set_ylabel('Tỷ lệ mắc bệnh (%)')
        axes[1, 1].set_title('Tỷ lệ Mắc bệnh theo Nhóm BMI', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Thêm giá trị lên cột
        for i, v in enumerate(bmi_rate.values):
            axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'bmi_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Đã lưu biểu đồ tại {self.output_dir / 'bmi_analysis.png'}")
        
        return fig
    
    def plot_blood_pressure_analysis(self, df: pd.DataFrame, save: bool = True):
        """
        Phân tích về huyết áp
        
        Args:
            df: Pandas DataFrame
            save: Lưu biểu đồ hay không
        """
        logger.info("Vẽ biểu đồ phân tích huyết áp...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot ap_hi vs ap_lo
        for cardio_val in [0, 1]:
            data = df[df['cardio'] == cardio_val]
            axes[0, 0].scatter(data['ap_hi'], data['ap_lo'], 
                              alpha=0.3, s=20, 
                              color=self.cardio_colors[cardio_val],
                              label='Có bệnh' if cardio_val == 1 else 'Không bệnh')
        axes[0, 0].set_xlabel('Huyết áp tâm thu (ap_hi)')
        axes[0, 0].set_ylabel('Huyết áp tâm trương (ap_lo)')
        axes[0, 0].set_title('Huyết áp Tâm thu vs Tâm trương', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot huyết áp tâm thu
        sns.boxplot(data=df, x='cardio', y='ap_hi', palette=[self.cardio_colors[0], self.cardio_colors[1]], ax=axes[0, 1])
        axes[0, 1].set_xticklabels(['Không bệnh', 'Có bệnh'])
        axes[0, 1].set_xlabel('Tình trạng')
        axes[0, 1].set_ylabel('Huyết áp tâm thu')
        axes[0, 1].set_title('Phân bố Huyết áp Tâm thu', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot huyết áp tâm trương
        sns.boxplot(data=df, x='cardio', y='ap_lo', palette=[self.cardio_colors[0], self.cardio_colors[1]], ax=axes[1, 0])
        axes[1, 0].set_xticklabels(['Không bệnh', 'Có bệnh'])
        axes[1, 0].set_xlabel('Tình trạng')
        axes[1, 0].set_ylabel('Huyết áp tâm trương')
        axes[1, 0].set_title('Phân bố Huyết áp Tâm trương', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Box plot pulse pressure
        sns.boxplot(data=df, x='cardio', y='pulse_pressure', palette=[self.cardio_colors[0], self.cardio_colors[1]], ax=axes[1, 1])
        axes[1, 1].set_xticklabels(['Không bệnh', 'Có bệnh'])
        axes[1, 1].set_xlabel('Tình trạng')
        axes[1, 1].set_ylabel('Pulse Pressure')
        axes[1, 1].set_title('Phân bố Pulse Pressure', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'blood_pressure_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Đã lưu biểu đồ tại {self.output_dir / 'blood_pressure_analysis.png'}")
        
        return fig
    
    def plot_risk_factors(self, df: pd.DataFrame, save: bool = True):
        """
        Phân tích các yếu tố nguy cơ
        
        Args:
            df: Pandas DataFrame
            save: Lưu biểu đồ hay không
        """
        logger.info("Vẽ biểu đồ phân tích yếu tố nguy cơ...")
        
        risk_factors = ['smoke', 'alco', 'cholesterol', 'gluc', 'active']
        risk_labels = {
            'smoke': 'Hút thuốc',
            'alco': 'Uống rượu',
            'cholesterol': 'Cholesterol',
            'gluc': 'Glucose',
            'active': 'Hoạt động thể chất'
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, factor in enumerate(risk_factors):
            # Tính tỷ lệ mắc bệnh cho mỗi level
            rate = df.groupby(factor)['cardio'].mean() * 100
            
            axes[idx].bar(rate.index, rate.values, color=self.palette, 
                         edgecolor='black', alpha=0.8)
            axes[idx].set_xlabel('Mức độ')
            axes[idx].set_ylabel('Tỷ lệ mắc bệnh (%)')
            axes[idx].set_title(f'Tỷ lệ mắc bệnh - {risk_labels[factor]}', fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Thêm giá trị lên cột
            for i, v in enumerate(rate.values):
                axes[idx].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Ẩn subplot thừa
        axes[-1].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'risk_factors.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Đã lưu biểu đồ tại {self.output_dir / 'risk_factors.png'}")
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save: bool = True):
        """
        Vẽ correlation heatmap
        
        Args:
            df: Pandas DataFrame
            save: Lưu biểu đồ hay không
        """
        logger.info("Vẽ correlation heatmap...")
        
        # Chọn các cột numeric
        numeric_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 
                       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 
                       'pulse_pressure', 'cardio']
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Ma trận Correlation giữa các Features', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Đã lưu biểu đồ tại {self.output_dir / 'correlation_heatmap.png'}")
        
        return fig
    
    def plot_gender_analysis(self, df: pd.DataFrame, save: bool = True):
        """
        Phân tích theo giới tính
        
        Args:
            df: Pandas DataFrame
            save: Lưu biểu đồ hay không
        """
        logger.info("Vẽ biểu đồ phân tích giới tính...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Phân bố theo giới tính
        gender_cardio = df.groupby(['gender', 'cardio']).size().unstack(fill_value=0)
        gender_cardio.index = ['Nữ', 'Nam']
        gender_cardio.plot(kind='bar', ax=axes[0], color=[self.cardio_colors[0], self.cardio_colors[1]], 
                          edgecolor='black', alpha=0.8)
        axes[0].set_xlabel('Giới tính')
        axes[0].set_ylabel('Số lượng')
        axes[0].set_title('Số lượng Bệnh nhân theo Giới tính', fontweight='bold')
        axes[0].legend(['Không bệnh', 'Có bệnh'])
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. Tỷ lệ mắc bệnh theo giới tính
        gender_rate = df.groupby('gender')['cardio'].mean() * 100
        gender_labels = ['Nữ', 'Nam']
        axes[1].bar(gender_labels, gender_rate.values, color=[self.palette[0], self.palette[1]], 
                   edgecolor='black', alpha=0.8)
        axes[1].set_xlabel('Giới tính')
        axes[1].set_ylabel('Tỷ lệ mắc bệnh (%)')
        axes[1].set_title('Tỷ lệ Mắc bệnh theo Giới tính', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Thêm giá trị lên cột
        for i, v in enumerate(gender_rate.values):
            axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'gender_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Đã lưu biểu đồ tại {self.output_dir / 'gender_analysis.png'}")
        
        return fig
    
    def generate_all_plots(self, df: pd.DataFrame):
        """
        Tạo tất cả các biểu đồ
        
        Args:
            df: Pandas DataFrame
        """
        logger.info("=" * 60)
        logger.info("TẠO TẤT CẢ CÁC BIỂU ĐỒ PHÂN TÍCH")
        logger.info("=" * 60)
        
        self.plot_target_distribution(df)
        self.plot_age_analysis(df)
        self.plot_bmi_analysis(df)
        self.plot_blood_pressure_analysis(df)
        self.plot_risk_factors(df)
        self.plot_correlation_heatmap(df)
        self.plot_gender_analysis(df)
        
        plt.close('all')  # Đóng tất cả figures để giải phóng bộ nhớ
        
        logger.info("=" * 60)
        logger.info(f"✓ ĐÃ TẠO TẤT CẢ BIỂU ĐỒ TẠI: {self.output_dir}")
        logger.info("=" * 60)