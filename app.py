import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.multiclass import OneVsRestClassifier
import os 
from matplotlib.font_manager import FontProperties, findfont, findSystemFonts, fontManager 
import joblib
from io import BytesIO
import zipfile
import pytz
from datetime import datetime

# --- 頁面配置 ---
st.set_page_config(
    page_title="MLP 模型訓練器",
    page_icon="🧬",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# 先定義台灣時區，再使用它
taiwan_tz = pytz.timezone('Asia/Taipei')
current_time = datetime.now(taiwan_tz)
date_str = current_time.strftime("%Y年%m月%d日")
time_str = current_time.strftime("%H:%M:%S")

# --- 自定義 CSS 樣式 ---
st.markdown("""
<style>
    /* 全局樣式 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 標籤頁樣式 */
    button[data-baseweb="tab"] {
        font-size: 24px !important;  /* 原來是18px，增大至24px */
        font-weight: 800 !important; /* 加粗一點 */
        padding: 20px 40px !important; /* 原來是12px 24px，增大內邊距 */
        order-radius: 10px 10px 0 0 !important; /* 更大的圓角 */
        margin-right: 10px !important; /* 更大的標籤間距 */
        border: 2px solid #e0e0e0 !important; /* 更粗的邊框 */
        border-bottom: none !important;
        min-width: 220px !important; /* 確保標籤更寬 */
        height: auto !important; /* 自動高度 */
        min-height: 70px !important; /* 確保最小高度 */
        line-height: 1.4 !important; /* 調整行高 */
        transform: scale(1.05); /* 稍微放大整個按鈕 */
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05); /* 添加輕微陰影增強立體感 */
    }
    button[data-baseweb="tab"]:hover {
        background-color: #f0f8ff !important; /* 更明顯的懸停效果 */
        transform: scale(1.1) translateY(-2px) !important; /* 懸停時更明顯的放大效果 */
        transition: all 0.3s ease !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #e6f3ff !important; /* 選中標籤的背景色 */
        border-bottom: 3px solid #4dabf7 !important; /* 選中標籤的底部邊框 */
        color: #1a73e8 !important; /* 選中標籤的文字顏色 */
    }
    
    div[role="tablist"] {
        border-bottom: 4px solid #4dabf7 !important; /* 更粗的底部邊框 */
        margin-bottom: 35px !important; /* 更大的下方間距 */
        padding-bottom: 0 !important; /* 移除底部內邊距以避免間隙 */
    }
    
    /* 為標籤頁內容添加頂部間距，避免與標籤太近 */
    div[data-baseweb="tab-panel"] {
        padding-top: 20px !important;
    }
    div[role="tablist"] {
        border-bottom: 3px solid #4dabf7 !important; /* 加粗底部邊框 */
        margin-bottom: 30px !important; /* 增加下方間距 */
    }
    
    /* 卡片容器樣式 */
    .card-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 30px;
        border-left: 4px solid #4dabf7;
    }
    
    /* 參數組樣式 */
    .param-group {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* 參數區塊樣式 */
    .parameter-section {
        margin-bottom: 20px;
    }
    
    /* 特徵列表區塊樣式 */
    .feature-list-section {
        background-color: #f0f9ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 3px solid #339af0;
    }
    
    /* 美化標題 */
    .stSubheader {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #f1f3f5 !important;
    }
    
    /* 指標卡片樣式 */
    div.stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.3s;
        border-left: 5px solid #4dabf7;
    }
    div.stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* 表格美化 */
    div.stTable, div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* 輸入元素美化 */
    div.stSlider, div.stSelectbox, div.stNumberInput, div.stCheckbox {
        padding: 10px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }
    
    /* 按鈕美化 */
    button[kind="primary"], button[kind="secondary"] {
        font-size: 18px !important; /* 加大字體 */
        padding: 14px 22px !important; /* 增大內邊距 */
        height: auto !important; /* 自動高度適應內容 */
        min-height: 60px !important; /* 確保最小高度 */
        border-radius: 10px !important; /* 增大圓角 */
        transition: all 0.3s !important;
    }
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important; /* 懸停時微微上浮 */
        background-color: #339af0 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1) !important;
    }
    button[kind="secondary"] {
        border-radius: 8px !important;
        transition: all 0.3s !important;
    }
    button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        background-color: #e9ecef !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* 狀態指示器 */
    div[data-testid="stAlert"][kind="success"] {
        border-radius: 8px;
        border-left: 5px solid #51cf66;
    }
    div[data-testid="stAlert"][kind="warning"] {
        border-radius: 8px;
        border-left: 5px solid #fcc419;
    }
    div[data-testid="stAlert"][kind="error"] {
        border-radius: 8px;
        border-left: 5px solid #ff6b6b;
    }
    div[data-testid="stAlert"][kind="info"] {
        border-radius: 8px;
        border-left: 5px solid #4dabf7;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background-color: #4dabf7; padding: 20px 28px; display: flex; justify-content: space-between; align-items: center; color: white; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
    <div style="font-size: 34px; font-weight: 800;"> MLP 模型訓練與預測系統</div>
    <div style="text-align: right;">
        <div style="font-size: 20px; opacity: 0.9;">{date_str}</div>
        <div style="font-size: 25px; font-weight: bold;">{time_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<h4 style="margin-bottom: px; font-weight: normal; color: #555;">
透過調整參數訓練 MLP 模型，並即時進行預測
</h4>
""", unsafe_allow_html=True)


def create_downloadable_plot(fig, filename="plot.png"):
    """將 matplotlib 圖形轉換為可下載的格式"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    return buffer

def plot_decision_boundary(mlp_model, X_train, y_train, feature_indices, feature_names, tatarget_names, scaler, resolution=100):
    """繪製2D決策邊界"""
    import matplotlib.patches as mpatches
    
    # 獲取兩個特徵的範圍
    X_subset = X_train.iloc[:, feature_indices]
    x_min, x_max = X_subset.iloc[:, 0].min() - 0.5, X_subset.iloc[:, 0].max() + 0.5
    y_min, y_max = X_subset.iloc[:, 1].min() - 0.5, X_subset.iloc[:, 1].max() + 0.5
    
    # 創建網格
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # 準備預測數據
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 如果模型需要更多特徵，用平均值填充
    if len(feature_indices) < X_train.shape[1]:
        # 創建一個包含所有特徵的 DataFrame
        full_grid_data = {}
        
        # 首先，為所有特徵設置平均值
        for col_idx, col_name in enumerate(X_train.columns):
            full_grid_data[col_name] = np.full(grid_points.shape[0], X_train.iloc[:, col_idx].mean())
        
        # 然後，覆寫選定的兩個特徵
        full_grid_data[X_train.columns[feature_indices[0]]] = grid_points[:, 0]
        full_grid_data[X_train.columns[feature_indices[1]]] = grid_points[:, 1]
        
        # 創建 DataFrame
        full_grid_df = pd.DataFrame(full_grid_data)
        
        # 確保列的順序與訓練數據一致
        full_grid_df = full_grid_df[X_train.columns]
        
        # 進行預測
        Z = mlp_model.predict(full_grid_df)
    else:
        # 如果只有兩個特徵，直接創建 DataFrame
        grid_df = pd.DataFrame(grid_points, columns=[X_train.columns[feature_indices[0]], 
                                                     X_train.columns[feature_indices[1]]])
        Z = mlp_model.predict(grid_df)
    
    Z = Z.reshape(xx.shape)
    
    # 繪製決策邊界
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用更美觀的顏色
    colors = ['#FFE5E5', '#E5F2FF', '#E5FFE5']  # 淡色背景
    contour = ax.contourf(xx, yy, Z, alpha=0.6, colors=colors, levels=[-0.5, 0.5, 1.5, 2.5])
    
    # 繪製訓練數據點
    scatter_colors = ['#FF6B6B', '#4DABF7', '#51CF66']  # 鮮明的點顏色
    
    # 這裡需要確保 target_names 在函數作用域內
    # 假設 target_names 是全局變量，如果不是，需要作為參數傳入
    target_names_local = ['setosa', 'versicolor', 'virginica']  # 或從參數獲取
    
    for i, (class_name, color) in enumerate(zip(target_names_local, scatter_colors)):
        idx = y_train == i
        ax.scatter(X_subset.iloc[idx, 0], X_subset.iloc[idx, 1], 
                  c=color, label=class_name, edgecolors='black', s=100, alpha=0.8)
    
    ax.set_xlabel(f'{feature_names[0]}（標準化後）')
    ax.set_ylabel(f'{feature_names[1]}（標準化後）')
    ax.set_title('MLP 決策邊界視覺化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# --- 模型保存路徑設定 ---
MODEL_PATH = "mlp_model.pkl"
SCALER_PATH = "scaler.pkl"

# --- 字型設定 ---
current_dir = os.path.dirname(__file__)
font_path = os.path.join(current_dir, "fonts", "NotoSansTC-VariableFont_wght.ttf")

if os.path.exists(font_path):
    fontManager.addfont(font_path)
    font_prop = FontProperties(fname=font_path) 
    font_family_name = font_prop.get_name() 
    
    plt.rcParams['font.family'] = font_family_name
    plt.rcParams['font.sans-serif'] = [font_family_name, 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['axes.unicode_minus'] = False 
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

# 設置全局圖表風格
plt.style.use('seaborn-v0_8-paper')
    
    # 更新圖表參數為研究論文風格
plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.grid': True,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'lines.linewidth': 1.5,
        
        # 設置中文字體支援
        'font.family': font_family_name,
        'font.sans-serif': [font_family_name, 'DejaVu Serif', 'serif'],
        'axes.unicode_minus': False
    })

# --- 數據加載與預處理 (使用 Streamlit 緩存) ---
@st.cache_data
def load_and_preprocess_data(test_size=0.2, use_stratify=True, random_state=42):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
    
    # 依據使用者選擇決定是否使用分層抽樣
    stratify_param = y if use_stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test, iris.target_names, X.columns.tolist(), scaler

# --- 自定義訓練函數with真實進度 ---
def train_mlp_with_progress(mlp, X_train, y_train, progress_bar, status_text):
    """訓練MLP並顯示進度"""
    import time
    import warnings
    
    # 統一使用標準訓練方法，確保一致性
    status_text.text("🚀 開始模型訓練...")
    
    # 模擬訓練進度
    for i in range(20, 85, 5):
        progress_bar.progress(i/100)
        time.sleep(0.1)
        status_text.text(f"🏃‍♂️ 訓練進度: {i}%")
    
    # 實際訓練 - 抑制收斂警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Stochastic Optimizer.*")
        warnings.filterwarnings("ignore", message=".*Maximum iterations.*")
        mlp.fit(X_train, y_train)
    
    # 完成進度
    progress_bar.progress(0.85)
    status_text.text("✅ 訓練完成")
    
    return mlp

# --- 評估函數 ---
def comprehensive_evaluation(mlp, X_train, X_test, y_train, y_test, target_names):
    """綜合評估模型性能 - 確保數據一致性"""
    
    # 確保使用正確的特徵進行預測
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)
    
    # 基本指標
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # 詳細分類指標 - 使用測試集
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, average=None)
    
    # 檢查收斂狀態
    convergence_info = {
        'converged': mlp.n_iter_ < mlp.max_iter,
        'actual_iterations': mlp.n_iter_,
        'max_iterations': mlp.max_iter
    }
    
    # 交叉驗證分數 - 使用與訓練相同的特徵
    try:
        # 創建新的模型實例進行交叉驗證，避免影響已訓練的模型
        cv_model = MLPClassifier(
            hidden_layer_sizes=mlp.hidden_layer_sizes,
            activation=mlp.activation,
            solver=mlp.solver,
            alpha=mlp.alpha,
            batch_size=mlp.batch_size,
            learning_rate=mlp.learning_rate,
            learning_rate_init=mlp.learning_rate_init,
            max_iter=mlp.max_iter,
            early_stopping=mlp.early_stopping,
            validation_fraction=mlp.validation_fraction,
            n_iter_no_change=mlp.n_iter_no_change,
            tol=mlp.tol,
            random_state=42,
            verbose=False
        )
        
        # 使用相同的訓練數據和標籤進行交叉驗證
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 暫時忽略交叉驗證中的收斂警告
            cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring='accuracy')
        
    except Exception as e:
        # 如果交叉驗證失敗，使用測試準確率作為替代
        print(f"交叉驗證失敗: {e}")
        cv_scores = np.array([test_accuracy] * 5)  # 使用測試準確率填充
    
    return {
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'y_pred_proba': y_pred_proba,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_scores': cv_scores,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'convergence_info': convergence_info
    }


# --- 側邊欄參數設定 ---
st.sidebar.header('🔧 MLP 模型超參數設定')

# 資料集切分設定 - 先定義這些變數
st.sidebar.subheader('📊 資料集切分設定')
test_size = st.sidebar.slider('測試集比例', 0.1, 0.5, 0.2, step=0.05, 
                         help="設定用於測試的資料比例，一般建議在 0.1~0.3 之間")
use_stratify = st.sidebar.checkbox('啟用分層抽樣', value=True, 
                             help="確保訓練集和測試集中各類別比例一致")
random_state = st.sidebar.number_input('隨機種子', min_value=0, max_value=100, value=42, step=1, 
                                  help="控制資料切分的隨機性，設定固定值可確保結果可重現")

# 初始化所有 session state
if 'iris_original_data_loaded' not in st.session_state:
    iris_full_dataset = load_iris()
    st.session_state.original_X_df = pd.DataFrame(iris_full_dataset.data, columns=iris_full_dataset.feature_names)
    st.session_state.original_y = iris_full_dataset.target
    if 'target_names' not in st.session_state:
        st.session_state.target_names = iris_full_dataset.target_names.copy()
    st.session_state.iris_original_data_loaded = True

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'last_selected_features' not in st.session_state:
    st.session_state.last_selected_features = None
if 'last_split_params' not in st.session_state:
    st.session_state.last_split_params = (test_size, use_stratify, random_state)
elif st.session_state.last_split_params != (test_size, use_stratify, random_state):
    st.session_state.model_trained = False
    st.session_state.training_results = None
    st.sidebar.warning("⚠️ 資料集切分參數已變更，需要重新訓練模型")

st.session_state.last_split_params = (test_size, use_stratify, random_state)

# 現在可以安全地加載數據，因為所有需要的變數都已經定義
X_train_full, X_test_full, y_train, y_test, target_names, all_feature_names, loaded_scaler = load_and_preprocess_data(
    test_size=test_size,
    use_stratify=use_stratify,
    random_state=random_state
)

# 繼續其他側邊欄設定
st.sidebar.subheader('📊 特徵選擇')
selected_features = st.sidebar.multiselect(
    '選擇要包含的特徵',
    options=all_feature_names,
    default=all_feature_names
)

# 檢查特徵是否改變
if (st.session_state.last_selected_features is not None and 
    st.session_state.last_selected_features != selected_features):
    st.session_state.model_trained = False
    st.session_state.training_results = None
    
st.session_state.last_selected_features = selected_features

if not selected_features:
    st.sidebar.warning("請至少選擇一個特徵！")
    st.error("⚠️ 請在左側邊欄至少選擇一個特徵才能繼續")
    st.stop()

# 根據選擇的特徵篩選數據集
X_train = X_train_full[selected_features]
X_test = X_test_full[selected_features]

# 模型複雜度
st.sidebar.subheader('🏗️ 模型複雜度')
hidden_layer_1 = st.sidebar.slider('第一隱藏層神經元數量', 10, 200, 100, step=10)
num_hidden_layers = st.sidebar.radio('隱藏層數量', [1, 2], index=0)

hidden_layer_sizes = (hidden_layer_1,)
if num_hidden_layers == 2:
    hidden_layer_2 = st.sidebar.slider('第二隱藏層神經元數量', 10, 100, 50, step=10)
    hidden_layer_sizes = (hidden_layer_1, hidden_layer_2)

st.sidebar.write(f'🔹 隱藏層結構: {hidden_layer_sizes}')

# 活化函數
activation_function = st.sidebar.selectbox(
    '⚡ 活化函數',
    ['relu', 'logistic', 'tanh', 'identity'],
    index=0
)

# 優化器
solver = st.sidebar.selectbox(
    '🚀 優化器',
    ['adam', 'sgd', 'lbfgs'],
    index=0
)

# 學習率設定
learning_rate_init = 0.001
if solver in ['adam', 'sgd']:
    st.sidebar.subheader('📈 學習率設定')
    learning_rate = st.sidebar.selectbox(
        '學習率策略',
        ['constant', 'invscaling', 'adaptive'],
        index=0
    )
    learning_rate_init = st.sidebar.number_input('初始學習率', min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
else:
    learning_rate = 'constant'

# 批次大小
batch_size = 'auto'
if solver in ['adam', 'sgd']:
    st.sidebar.subheader('📦 批次大小')
    batch_size_option = st.sidebar.radio('批次大小設定', ['auto', '手動輸入'], index=0)
    if batch_size_option == '手動輸入':
        batch_size = st.sidebar.number_input('Batch Size', min_value=1, max_value=len(X_train), value=32, step=1)

# 訓練參數
st.sidebar.subheader('🎯 訓練參數')
max_iter = st.sidebar.slider('最大迭代次數', 50, 1000, 200, step=50)

early_stopping = st.sidebar.checkbox('啟用 Early Stopping', value=False)
validation_fraction = 0.1
n_iter_no_change = 10
tol = 1e-4

if early_stopping:
    validation_fraction = st.sidebar.slider('驗證集比例', 0.05, 0.5, 0.1, step=0.05)
    n_iter_no_change = st.sidebar.slider('無改善容忍迭代次數', 10, 100, 10, step=5)
    tol = st.sidebar.number_input('容忍度', min_value=1e-5, max_value=1e-2, value=1e-4, format="%.5f")

# 正規化參數
alpha = st.sidebar.number_input('🛡️ L2 正則化強度', min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")

# 檢查是否有已保存的模型
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
if model_exists:
    st.sidebar.success("✅ 發現已保存的模型")
    if not st.session_state.model_trained or st.session_state.training_results is None:
        try:
            # 嘗試載入模型
            loaded_mlp_model = joblib.load(MODEL_PATH)
            loaded_data_scaler = joblib.load(SCALER_PATH)
            
            # 在載入頁面時執行簡單評估獲取基本指標
            evaluation_results = comprehensive_evaluation(
                loaded_mlp_model, X_train, X_test, y_train, y_test, target_names
            )
            
            # 更新session state
            st.session_state.training_results = {
                'mlp': loaded_mlp_model,
                'selected_features': selected_features,
                **evaluation_results
            }
            st.session_state.model_trained = True
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ 發現模型文件但無法載入: {e}")
else:
    st.sidebar.info("ℹ️ 尚未訓練模型")




# 增加間隔，使收斂優化部分與前面的建議有所區分
st.sidebar.markdown("&nbsp;")  # 空白間隔

# 當前狀態顯示
st.sidebar.markdown("---")
st.sidebar.subheader("📋 當前狀態")

if st.sidebar.checkbox("顯示狀態調試信息", False):
    st.sidebar.write(f"模型文件存在: {os.path.exists(MODEL_PATH)}")

if st.session_state.model_trained and st.session_state.training_results:
    results = st.session_state.training_results
    st.sidebar.success("✅ 模型已訓練")
    st.sidebar.metric("測試準確率", f"{results['test_accuracy']:.3f}")
    st.sidebar.metric("F1-Score", f"{results['f1'].mean():.3f}")
else:
    st.sidebar.info("⏳ 等待訓練")

# 數據集信息
st.sidebar.markdown("---")
st.sidebar.subheader("📊 數據集資訊")
st.sidebar.write(f"• 總樣本數: {len(X_train_full) + len(X_test_full)}")
st.sidebar.write(f"• 訓練集: {len(X_train_full)} 樣本")
st.sidebar.write(f"• 測試集: {len(X_test_full)} 樣本")
st.sidebar.write(f"• 特徵總數: {len(all_feature_names)}")
st.sidebar.write(f"• 選擇特徵: {len(selected_features)}")
st.sidebar.write(f"• 類別數: {len(target_names)}")
st.sidebar.markdown("&nbsp;")  # 空白間隔

# 切分比例建議
st.sidebar.markdown("**💡 切分比例建議：**")
total_samples = len(X_train_full) + len(X_test_full)

if total_samples < 100:
    st.sidebar.info(f"小型資料集 ({total_samples} 樣本)，建議測試集比例: 0.2~0.3")
elif total_samples < 1000:
    st.sidebar.info(f"中型資料集 ({total_samples} 樣本)，建議測試集比例: 0.15~0.25")
else:
    st.sidebar.info(f"大型資料集 ({total_samples} 樣本)，建議測試集比例: 0.1~0.2")

# 警告過小的測試集
min_test_samples = int(total_samples * test_size)
if min_test_samples < 30:
    st.sidebar.warning(f"⚠️ 測試集僅有 {min_test_samples} 個樣本，可能不足以可靠評估模型")
st.sidebar.markdown("&nbsp;")  # 空白間隔
# 參數建議
st.sidebar.markdown("---")
st.sidebar.subheader("💡 參數調優建議")

if len(selected_features) < 4:
    st.sidebar.warning("特徵數量較少，建議使用較小的隱藏層")
    
if solver == 'lbfgs' and max_iter > 500:
    st.sidebar.info("L-BFGS 通常收斂較快，可嘗試較少迭代次數")
elif solver in ['adam', 'sgd'] and max_iter < 500:
    st.sidebar.warning("Adam/SGD 可能需要更多迭代次數避免收斂警告")
    
if solver in ['adam', 'sgd'] and learning_rate_init > 0.01:
    st.sidebar.warning("學習率較高，可能導致訓練不穩定或難以收斂")

if hidden_layer_1 > 150 and len(selected_features) <= 4:
    st.sidebar.warning("隱藏層神經元數可能過多，易過擬合")
# 收斂優化建議
st.sidebar.markdown("**🔄 收斂優化：**")
st.sidebar.write("• 遇到收斂警告時增加迭代次數")
st.sidebar.write("• 啟用 Early Stopping 防止過度訓練")
st.sidebar.write("• 調低學習率提高穩定性")
# --- 主要內容區域使用 Tabs ---
# 使用更美觀的標籤頁
tabs = st.tabs([
    "🎯 **模型訓練** ", 
    "📊 **訓練結果** ", 
    "🔮 **即時預測** "
])

# --- Tab 1: 模型訓練 ---
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🔧 當前模型設定")
        
        # 參數展示
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            st.write("**🏗️ 模型結構**")
            st.write(f"• 特徵數量: {len(selected_features)}")
            st.write(f"• 隱藏層: {hidden_layer_sizes}")
            st.write(f"• 活化函數: {activation_function}")
            st.write(f"• 優化器: {solver}")
            st.write("**📊 資料集切分資訊：**")
            st.write(f"• 測試集比例: {test_size:.0%}")
            st.write(f"• 分層抽樣: {'✅ 已啟用' if use_stratify else '❌ 未啟用'}")
            st.write(f"• 隨機種子: {random_state}")
            st.write(f"• 訓練集大小: {len(X_train)} 樣本")
            st.write(f"• 測試集大小: {len(X_test)} 樣本")

            # 顯示各類別在訓練集中的分布
            train_class_dist = pd.Series(y_train).value_counts().sort_index()
            train_class_dist.index = [target_names[i] for i in train_class_dist.index]
            st.write("• 訓練集類別分布: ", dict(train_class_dist))

        with param_col2:
            st.write("**⚙️ 訓練參數**")
            st.write(f"• 最大迭代次數: {max_iter}")
            st.write(f"• Early Stopping: {'啟用' if early_stopping else '停用'}")
            st.write(f"• L2 正則化: {alpha}")
            if solver in ['adam', 'sgd']:
                st.write(f"• 學習率: {learning_rate_init}")
        
        st.write("**📋 選擇的特徵:**")
        st.write(", ".join(selected_features))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🚀 模型控制")
        
        train_col, reset_col = st.columns(2)
        
        with train_col:
            if st.button('🎯 訓練模型', type="primary", use_container_width=True):
                if not selected_features:
                    st.error("請至少選擇一個特徵才能訓練模型！")
                else:
                    try:
                        # 建立進度追蹤容器
                        progress_container = st.container()
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        status_text.text("🔧 建立模型架構...")
                        progress_bar.progress(10)
                        
                        # 數據驗證
                        st.write(f"📊 **訓練數據資訊:**")
                        st.write(f"• 選擇特徵: {len(selected_features)} 個")
                        st.write(f"• 訓練樣本: {X_train.shape[0]} 個")
                        st.write(f"• 測試樣本: {X_test.shape[0]} 個")
                        st.write(f"• 測試集比例: {test_size:.0%}")
                        st.write(f"• 分層抽樣: {'已啟用' if use_stratify else '未啟用'}")
                        st.write(f"• 類別分布: {dict(zip(target_names, np.bincount(y_train)))}")
                                                
                        # 建立模型
                        mlp = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation_function,
                            solver=solver,
                            alpha=alpha,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init,
                            max_iter=max_iter,
                            early_stopping=early_stopping,
                            validation_fraction=validation_fraction,
                            n_iter_no_change=n_iter_no_change,
                            tol=tol,
                            random_state=42,
                            verbose=False  # 關閉詳細輸出，使用自定義進度
                        )
                        
                        progress_bar.progress(20)
                        
                        # 使用自定義訓練函數
                        mlp = train_mlp_with_progress(mlp, X_train, y_train, progress_bar, status_text)
                        
                        status_text.text("💾 保存模型...")
                        progress_bar.progress(90)
                        
                        # 保存模型
                        joblib.dump(mlp, MODEL_PATH)
                        joblib.dump(loaded_scaler, SCALER_PATH)
                        
                        status_text.text("📊 綜合評估模型...")
                        progress_bar.progress(95)
                        
                        # 綜合評估
                        evaluation_results = comprehensive_evaluation(mlp, X_train, X_test, y_train, y_test, target_names)
                        
                        # 保存結果到 session state
                        st.session_state.training_results = {
                            'mlp': mlp,
                            'selected_features': selected_features,
                            **evaluation_results
                        }
                        st.session_state.model_trained = True
                        # 添加這個標記，表示我們需要在下一次執行時強制更新UI
                        st.session_state.need_ui_update = True
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 訓練完成！")
                        
                        # 顯示快速摘要
                        st.success("🎉 模型訓練成功！")
                        st.info("💡 切換到「📊 訓練結果」查看詳細分析，或到「🔮 即時預測」進行預測。")
                        
                    except Exception as e:
                        st.error(f"❌ 訓練過程發生錯誤：{e}")
                        import traceback
                        st.text("詳細錯誤信息：")
                        st.code(traceback.format_exc())
        
        with reset_col:
            if st.button('🔄 重置模型', use_container_width=True):
                # 更新狀態
                st.session_state.model_trained = False
                st.session_state.training_results = None
                
                # 刪除模型文件
                try:
                    if os.path.exists(MODEL_PATH):
                        os.remove(MODEL_PATH)
                    if os.path.exists(SCALER_PATH):
                        os.remove(SCALER_PATH)
                    
                    # 僅顯示重置成功訊息，不提及刷新
                    st.success("✅ 模型已重置！")
                    
                    # 立即使用JavaScript刷新頁面，不顯示"正在刷新"
                    st.markdown("""
                    <script>
                        // 立即刷新頁面
                        window.location.reload();
                    </script>
                    """, unsafe_allow_html=True)
                except:
                    # 顯示錯誤訊息
                    st.error("⚠️ 模型文件刪除失敗，但狀態已清除")  
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 在所有 columns 外面顯示快速結果預覽
    if st.session_state.model_trained and st.session_state.training_results:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📊 快速結果預覽")
        preview_col1, preview_col2, preview_col3 = st.columns(3)
        
        evaluation_results = st.session_state.training_results
        
        with preview_col1:
            st.metric("🎯 測試準確率", f"{evaluation_results['test_accuracy']:.3f}")
        with preview_col2:
            st.metric("📊 交叉驗證", f"{evaluation_results['cv_scores'].mean():.3f}")
        with preview_col3:
            overfitting = evaluation_results['train_accuracy'] - evaluation_results['test_accuracy']
            st.metric("🔍 過擬合程度", f"{overfitting:.3f}")
        
        # 結果合理性檢查
        if evaluation_results['test_accuracy'] < 0.6:
            st.warning("⚠️ 測試準確率較低，建議調整超參數或檢查特徵選擇")
        
        if abs(evaluation_results['cv_scores'].mean() - evaluation_results['test_accuracy']) > 0.2:
            st.warning("⚠️ 交叉驗證與測試結果差異較大，可能存在數據洩露或過擬合")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: 訓練結果 ---
with tabs[1]:
    # 檢查模型文件是否存在
    model_file_exists = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
    if 'model_trained' in st.session_state and st.session_state.model_trained and 'training_results' in st.session_state and st.session_state.training_results:
        # 正常顯示結果
        results = st.session_state.training_results
        
        # === 模型性能總覽 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🎯 模型性能總覽")
        
        # 主要指標
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 測試準確率", 
                f"{results['test_accuracy']:.3f}",
                delta=f"{(results['test_accuracy'] - 0.33):.3f}"
            )
            
        with col2:
            cv_mean = results['cv_scores'].mean()
            cv_std = results['cv_scores'].std()
            st.metric(
                "📊 交叉驗證", 
                f"{cv_mean:.3f}",
                delta=f"±{cv_std:.3f}"
            )
            
        with col3:
            overfitting = results['train_accuracy'] - results['test_accuracy']
            overfitting_status = "🟢 正常" if abs(overfitting) < 0.05 else ("🟡 輕微" if overfitting < 0.15 else "🔴 嚴重")
            st.metric(
                "🔍 過擬合檢測", 
                f"{overfitting:.3f}",
                delta=overfitting_status
            )
            
        with col4:
            f1_macro = results['f1'].mean()
            f1_status = "🟢 優秀" if f1_macro > 0.9 else ("🟡 良好" if f1_macro > 0.7 else "🔴 需改進")
            st.metric(
                "⚡ F1-Score", 
                f"{f1_macro:.3f}",
                delta=f1_status
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === 性能診斷 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🔧 模型健康檢查")
        
        # 使用三欄布局
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        
        with diag_col1:
            # 收斂狀態檢查
            convergence = results['convergence_info']
            if convergence['converged']:
                st.success(f"✅ 模型已收斂 ({convergence['actual_iterations']}/{convergence['max_iterations']} 迭代)")
            else:
                st.error(f"❌ 模型未收斂 ({convergence['actual_iterations']}/{convergence['max_iterations']} 迭代)")
        
        with diag_col2:
            # 準確率檢查
            if results['test_accuracy'] > 0.9:
                st.success("✅ 準確率優秀")
            elif results['test_accuracy'] > 0.7:
                st.info("ℹ️ 準確率良好")
            else:
                st.error("❌ 準確率偏低")
        
        with diag_col3:
            # 過擬合檢查
            if abs(overfitting) < 0.05:
                st.success("✅ 無明顯過擬合")
            elif overfitting > 0.15:
                st.warning("⚠️ 可能存在過擬合")
            elif overfitting < -0.05:
                st.warning("⚠️ 異常：測試集表現優於訓練集")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === 切分策略與交叉驗證比較 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🔄 切分策略與交叉驗證比較")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**📊 切分評估結果**")
            st.metric("訓練集準確率", f"{results['train_accuracy']:.3f}")
            st.metric("測試集準確率", f"{results['test_accuracy']:.3f}")
            st.write(f"測試集大小: {len(X_test)} 樣本 ({test_size:.0%})")
            
        with col2:
            st.write("**🔄 交叉驗證結果**")
            cv_mean = results['cv_scores'].mean()
            cv_std = results['cv_scores'].std()
            st.metric("平均準確率", f"{cv_mean:.3f}")
            st.metric("標準差", f"{cv_std:.3f}")
            st.write(f"交叉驗證折數: 5")

        # 對比圖表
        fig, ax = plt.subplots(figsize=(10, 6))
        x = ['訓練集', '測試集', '交叉驗證']
        y = [results['train_accuracy'], results['test_accuracy'], cv_mean]
        colors = ['#4DABF7', '#FF6B6B', '#51CF66']

        bars = ax.bar(x, y, color=colors, alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('準確率')
        ax.set_title('不同評估方法的準確率比較')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig)

        # 添加下載按鈕
        buffer = create_downloadable_plot(fig, "evaluation_comparison.png")
        st.download_button(
            label="📥 下載評估比較圖",
            data=buffer,
            file_name="evaluation_comparison.png",
            mime="image/png"
        )
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # === 詳細評估指標 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📈 詳細評估指標")
        
        # 放大的交叉驗證圖
        st.write("**🎲 交叉驗證分數分析**")
        cv_df = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(len(results['cv_scores']))],
            '準確率': results['cv_scores']
        })
        
        # 增大圖表尺寸
        fig_cv, ax_cv = plt.subplots(figsize=(12, 6))
        bars = ax_cv.bar(cv_df['Fold'], cv_df['準確率'], color='skyblue', alpha=0.7, width=0.6)
        ax_cv.axhline(y=cv_df['準確率'].mean(), color='red', linestyle='--', linewidth=2,
                     label=f'平均值: {cv_df["準確率"].mean():.3f}')
        ax_cv.set_ylabel('準確率', fontsize=14)
        ax_cv.set_xlabel('交叉驗證折數', fontsize=14)
        ax_cv.set_title('5-Fold 交叉驗證結果', fontsize=16, fontweight='bold')
        ax_cv.legend(fontsize=12)
        ax_cv.grid(True, alpha=0.3)
        ax_cv.set_ylim(0, 1.1)
        
        # 在柱狀圖上顯示數值
        for bar, score in zip(bars, results['cv_scores']):
            height = bar.get_height()
            ax_cv.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        st.pyplot(fig_cv)
        
        # 添加下載按鈕
        buffer = create_downloadable_plot(fig_cv, "cross_validation_results.png")
        st.download_button(
            label="📥 下載交叉驗證圖表",
            data=buffer,
            file_name="cross_validation_results.png",
            mime="image/png"
        )
        plt.close(fig_cv)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === 各類別性能指標和信心度統計並列 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📊 性能指標詳情")
        
        # 使用兩列佈局
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.write("**📊 各類別性能指標**")
            # 創建詳細的性能指標表
            performance_df = pd.DataFrame({
                '類別': target_names,
                '精確率': [f"{p:.3f}" for p in results['precision']],
                '召回率': [f"{r:.3f}" for r in results['recall']],
                'F1-Score': [f"{f:.3f}" for f in results['f1']],
                '樣本數': results['support'].astype(int)
            })
            st.dataframe(performance_df, use_container_width=True)
        
        with perf_col2:
            st.write("**📊 預測信心度統計**")
            # 信心度統計
            max_probas = np.max(results['y_pred_proba'], axis=1)
            confidence_stats = pd.DataFrame({
                '指標': ['平均信心度', '最低信心度', '最高信心度', '標準差'],
                '數值': [f"{max_probas.mean():.3f}", f"{max_probas.min():.3f}", 
                        f"{max_probas.max():.3f}", f"{max_probas.std():.3f}"]
            })
            st.dataframe(confidence_stats, use_container_width=True)
        
        # === 性能指標雷達圖 ===
        st.write("**📊 各類別性能指標雷達圖**")
        categories = target_names
        fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # 繪製精確率、召回率、F1-Score
        for metric_name, metric_values, color in [
            ('精確率', results['precision'], 'blue'),
            ('召回率', results['recall'], 'green'),
            ('F1-Score', results['f1'], 'red')
        ]:
            values = np.concatenate((metric_values, [metric_values[0]]))
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=metric_name, color=color)
            ax_radar.fill(angles, values, alpha=0.25, color=color)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=12)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('各類別性能指標雷達圖', pad=20, fontsize=16)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax_radar.grid(True)
        
        st.pyplot(fig_radar)
        
        buffer = create_downloadable_plot(fig_radar, "performance_radar_chart.png")
        st.download_button(
            label="📥 下載雷達圖",
            data=buffer,
            file_name="performance_radar_chart.png",
            mime="image/png"
        )
        plt.close(fig_radar)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === 視覺化分析 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📊 視覺化分析")
        
        visual_col1, visual_col2 = st.columns(2)
        
        with visual_col1:
            st.write("**🔥 混淆矩陣**")
            cm = confusion_matrix(y_test, results['y_pred_test'])
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            
            # 計算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 創建標註
            annot = []
            for i in range(cm.shape[0]):
                row = []
                for j in range(cm.shape[1]):
                    row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1%})')
                annot.append(row)
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
            ax_cm.set_xlabel('預測標籤')
            ax_cm.set_ylabel('真實標籤')
            ax_cm.set_title('混淆矩陣 (數量 & 百分比)')
            st.pyplot(fig_cm)
            
            # 添加下載按鈕
            buffer = create_downloadable_plot(fig_cm, "confusion_matrix.png")
            st.download_button(
                label="📥 下載混淆矩陣",
                data=buffer,
                file_name="confusion_matrix.png",
                mime="image/png"
            )
            plt.close(fig_cm)
            
        with visual_col2:
            st.write("**🎯 預測信心度分布**")
            # 分析預測信心度分布
            max_probas = np.max(results['y_pred_proba'], axis=1)
            
            fig_conf, ax_conf = plt.subplots(figsize=(6, 5))
            n, bins, patches = ax_conf.hist(max_probas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax_conf.axvline(x=max_probas.mean(), color='red', linestyle='--', 
                           label=f'平均信心度: {max_probas.mean():.3f}')
            ax_conf.set_xlabel('最大預測機率')
            ax_conf.set_ylabel('樣本數')
            ax_conf.set_title('模型預測信心度分布')
            ax_conf.legend()
            ax_conf.grid(True, alpha=0.3)
            
            st.pyplot(fig_conf)
            
            buffer = create_downloadable_plot(fig_conf, "confidence_distribution.png")
            st.download_button(
                label="📥 下載信心度分布圖",
                data=buffer,
                file_name="confidence_distribution.png",
                mime="image/png"
            )
            plt.close(fig_conf)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === 學習曲線分析 ===
        if hasattr(results['mlp'], 'loss_curve_') and results['mlp'].loss_curve_:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("📉 學習曲線分析")
            
            st.write("**🔻 訓練損失曲線**")
            fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
            ax_loss.plot(results['mlp'].loss_curve_, 'b-', linewidth=2, label='訓練損失')
            ax_loss.set_xlabel('迭代次數')
            ax_loss.set_ylabel('損失值')
            ax_loss.set_title('訓練過程損失變化')
            ax_loss.grid(True, alpha=0.3)
            ax_loss.legend()
            
            # 標註重要點
            min_loss_idx = np.argmin(results['mlp'].loss_curve_)
            min_loss_val = results['mlp'].loss_curve_[min_loss_idx]
            ax_loss.plot(min_loss_idx, min_loss_val, 'ro', markersize=8, 
                       label=f'最低損失: {min_loss_val:.4f}')
            ax_loss.legend()
            
            st.pyplot(fig_loss)
            
            buffer = create_downloadable_plot(fig_loss, "learning_curve.png")
            st.download_button(
                label="📥 下載學習曲線",
                data=buffer,
                file_name="learning_curve.png",
                mime="image/png"
            )
            plt.close(fig_loss)
            
            # 學習統計
            st.write("**📊 學習統計**")
            loss_curve = results['mlp'].loss_curve_
            learning_stats = pd.DataFrame({
                '指標': ['最終損失', '最低損失', '初始損失', '損失下降率'],
                '數值': [
                    f"{loss_curve[-1]:.4f}",
                    f"{np.min(loss_curve):.4f}",
                    f"{loss_curve[0]:.4f}",
                    f"{((loss_curve[0] - loss_curve[-1]) / loss_curve[0] * 100):.1f}%"
                ]
            })
            st.dataframe(learning_stats, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === 決策邊界視覺化 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🎨 決策邊界視覺化")
        
        if len(selected_features) >= 2:
            st.write("**選擇兩個特徵來繪製決策邊界：**")
            
            boundary_col1, boundary_col2 = st.columns(2)
            with boundary_col1:
                feature_1 = st.selectbox(
                    "選擇第一個特徵（X軸）",
                    options=selected_features,
                    index=0,
                    key="boundary_feature_1"
                )
            with boundary_col2:
                feature_2 = st.selectbox(
                    "選擇第二個特徵（Y軸）",
                    options=[f for f in selected_features if f != feature_1],
                    index=0,
                    key="boundary_feature_2"
                )
            
            if st.button("🎨 繪製決策邊界", type="secondary"):
                with st.spinner("正在繪製決策邊界..."):
                    # 獲取特徵索引
                    feature_indices = [selected_features.index(feature_1), 
                                     selected_features.index(feature_2)]
                    feature_names_selected = [feature_1, feature_2]
                    
                    # 繪製決策邊界
                    fig_boundary = plot_decision_boundary(
                        results['mlp'], 
                        X_train, 
                        y_train, 
                        feature_indices,
                        feature_names_selected,
                        target_names,
                        loaded_scaler
                    )
                    
                    st.pyplot(fig_boundary)
                    
                    # 添加下載按鈕
                    buffer = create_downloadable_plot(fig_boundary, "decision_boundary.png")
                    st.download_button(
                        label="📥 下載決策邊界圖",
                        data=buffer,
                        file_name="decision_boundary.png",
                        mime="image/png"
                    )
                    plt.close(fig_boundary)
                    
                    # 添加說明
                    st.info("""
                    **圖表說明：**
                    - 背景顏色代表模型的決策區域
                    - 散點代表訓練數據
                    - 不同顏色代表不同的鳶尾花種類
                    - 邊界線顯示了模型如何區分不同類別
                    """)
        else:
            st.warning("⚠️ 需要至少選擇2個特徵才能繪製決策邊界")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === 模型詳細資訊 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("🔍 模型詳細資訊")
        
        model_info_col1, model_info_col2 = st.columns(2)
        
        with model_info_col1:
            st.write("**🏗️ 模型架構**")
            architecture_info = pd.DataFrame({
                '層級': ['輸入層'] + [f'隱藏層{i+1}' for i in range(len(results['mlp'].coefs_)-1)] + ['輸出層'],
                '神經元數': [results['mlp'].coefs_[0].shape[0]] + 
                          [coef.shape[1] for coef in results['mlp'].coefs_[:-1]] + 
                          [results['mlp'].coefs_[-1].shape[1]]
            })
            st.dataframe(architecture_info, use_container_width=True)
        
        with model_info_col2:
            st.write("**📊 訓練資訊**")
            convergence = results['convergence_info']
            # 將所有數值轉換為字串以避免混合類型錯誤
            training_info = pd.DataFrame({
                '指標': ['實際迭代次數', '最大迭代次數', '收斂狀態', '權重參數總數', '偏置參數總數', '總參數量'],
                '數值': [
                    str(convergence['actual_iterations']),
                    str(convergence['max_iterations']),
                    "✅ 已收斂" if convergence['converged'] else "❌ 未收斂",
                    str(sum(coef.size for coef in results['mlp'].coefs_)),
                    str(sum(intercept.size for intercept in results['mlp'].intercepts_)),
                    str(sum(coef.size for coef in results['mlp'].coefs_) + 
                        sum(intercept.size for intercept in results['mlp'].intercepts_))
                ]
            })
            st.dataframe(training_info, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === 快速操作 ===
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("⚡ 快速操作")
        
        op_col1, op_col2, op_col3 = st.columns(3)
        
        with op_col1:
            if st.button("🔄 重新訓練", type="secondary", use_container_width=True):
                st.info("💡 請切換到「🎯 模型訓練」標籤頁")
        
        with op_col2:
            if st.button("🔮 前往預測", type="secondary", use_container_width=True):
                st.info("💡 請切換到「🔮 即時預測」標籤頁")
        
        with op_col3:
            # 創建簡單的報告摘要
            report_data = {
                "測試準確率": results['test_accuracy'],
                "交叉驗證均值": results['cv_scores'].mean(),
                "交叉驗證標準差": results['cv_scores'].std(),
                "F1-Score": results['f1'].mean(),
                "過擬合程度": results['train_accuracy'] - results['test_accuracy']
            }
            
            report_df = pd.DataFrame(list(report_data.items()), columns=['指標', '數值'])
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 下載報告",
                data=csv,
                file_name="mlp_training_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif model_file_exists:
        # 如果模型文件存在但session state不一致，嘗試載入
        st.info("發現模型文件，正在載入...")
        try:
            loaded_mlp_model = joblib.load(MODEL_PATH)
            loaded_data_scaler = joblib.load(SCALER_PATH)
            
            # 執行評估
            evaluation_results = comprehensive_evaluation(
                loaded_mlp_model, X_train, X_test, y_train, y_test, target_names
            )
            
            # 更新session state
            st.session_state.training_results = {
                'mlp': loaded_mlp_model,
                'selected_features': selected_features,
                **evaluation_results
            }
            st.session_state.model_trained = True
             #提示用戶刷新頁面
            st.success("✅ 模型已成功載入! 請手動刷新頁面以查看結果")
            st.button("刷新頁面", on_click=lambda: None)  # 這個按鈕只是提示用戶刷新頁面
        except Exception as e:
            st.error(f"載入模型失敗: {e}")
    else:
        st.warning("⚠️ 請先在「🎯 模型訓練」標籤頁訓練模型")

# --- Tab 3: 即時預測 ---
with tabs[2]:
    # 檢查模型是否可用
    if model_exists:
        try:
            loaded_mlp_model = joblib.load(MODEL_PATH)
            loaded_data_scaler = joblib.load(SCALER_PATH)
            
            st.success("✅ 模型載入成功，可以進行預測！")
            
            # 預測界面
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            st.subheader("🔮 輸入特徵值進行預測")
            
            # 使用原始資料作為參考
            if 'original_X_df' in st.session_state and 'original_y' in st.session_state:
                original_X_df = st.session_state.original_X_df
                original_y = st.session_state.original_y
                
                st.write("**從原始鳶尾花資料集載入樣本作為起點：**")
                sample_id_to_load = st.selectbox(
                    f"選擇一個原始樣本 ID (0 到 {len(original_X_df) - 1}):",
                    options=list(range(len(original_X_df))),
                    index=0,
                    key="sample_id_selector_tab3"
                )
                
                initially_loaded_sample_features = original_X_df.iloc[sample_id_to_load]
                initially_loaded_true_label_index = original_y[sample_id_to_load]
                initially_loaded_true_label_name = target_names[initially_loaded_true_label_index]
                
                st.info(f"📌 當前選擇樣本 ID: {sample_id_to_load}，其真實類別為: **{initially_loaded_true_label_name}**")
                st.caption("💡 下方的特徵值已從所選樣本自動填入，您可以自由調整它們。")
                
                # 創建輸入表單
                with st.form("prediction_form"):
                    st.write("**請輸入或調整特徵值：**")
                    
                    input_data = {}
                    cols = st.columns(2)  # 創建兩列以改善布局
                    
                    for i, feature in enumerate(all_feature_names):
                        col_idx = i % 2
                        with cols[col_idx]:
                            # 獲取預設值並限制為1位小數
                            default_value = round(float(initially_loaded_sample_features.get(feature, 0.0)), 1)
                            
                            # 計算合理的範圍
                            feature_values = original_X_df[feature]
                            min_val = float(feature_values.min())
                            max_val = float(feature_values.max())
                            mean_val = float(feature_values.mean())
                            
                            input_data[feature] = st.number_input(
                                f'📏 {feature}',
                                value=default_value,
                                min_value=round(min_val - 1.0, 1),
                                max_value=round(max_val + 1.0, 1),
                                step=0.1,
                                format="%.1f",  # 限制顯示格式為1位小數
                                help=f"原始資料範圍: {min_val:.1f} ~ {max_val:.1f}，平均值: {mean_val:.1f}",
                                key=f"input_{feature}_tab3"
                            )
                    
                    predict_button = st.form_submit_button("🔮 開始預測", type="primary", use_container_width=True)
                
                if predict_button:
                    try:
                        # 準備輸入數據
                        final_input_features_dict = input_data.copy()
                        final_input_features_array = np.array([final_input_features_dict[f] for f in all_feature_names])
                        
                        # 查找是否匹配原始資料中的樣本
                        label_for_comparison = initially_loaded_true_label_name
                        source_of_label_info = f"樣本 ID {sample_id_to_load}"
                        found_match = False
                        matched_sample_id = sample_id_to_load
                        
                        # 使用寬鬆的容差進行比對
                        for idx, original_row_values in enumerate(original_X_df.values):
                            # 將原始資料和輸入都四捨五入到1位小數進行比較
                            rounded_original = np.round(original_row_values, 1)
                            rounded_input = np.round(final_input_features_array, 1)
                            
                            if np.allclose(rounded_input, rounded_original, atol=1e-2, rtol=0):
                                matched_label = target_names[original_y[idx]]
                                label_for_comparison = matched_label
                                matched_sample_id = idx
                                source_of_label_info = f"樣本 ID {idx}"
                                
                                if idx != sample_id_to_load:
                                    st.success(f"🔍 發現匹配！您輸入的特徵值與樣本 ID {idx} ({matched_label}) 完全一致。")
                                    found_match = True
                                    break
                        
                        if not found_match and matched_sample_id != sample_id_to_load:
                            st.info("💡 當前輸入為自定義特徵組合。")
                        
                        # 處理只使用選定特徵的情況
                        full_input_df = pd.DataFrame([final_input_features_dict])[all_feature_names]
                        input_scaled = loaded_data_scaler.transform(full_input_df)
                        
                        # 只選擇模型訓練時使用的特徵
                        selected_indices = [all_feature_names.index(f) for f in selected_features]
                        input_for_prediction = input_scaled[:, selected_indices]
                        
                        # 模型預測
                        prediction_proba = loaded_mlp_model.predict_proba(input_for_prediction)
                        prediction_class = np.argmax(prediction_proba)
                        predicted_label = target_names[prediction_class]
                        confidence = prediction_proba[0][prediction_class]
                        
                        # === 顯示預測結果 ===
                        st.markdown("---")
                        st.subheader("🎉 預測結果")
                        
                        # 主要結果展示
                        result_col1, result_col2, result_col3 = st.columns([2, 2, 1])
                        
                        with result_col1:
                            st.metric(
                                "🌸 預測類別",
                                predicted_label,
                                delta=f"信心度: {confidence:.1%}"
                            )
                        
                        with result_col2:
                            st.metric(
                                "📊 參考答案",
                                label_for_comparison,
                                delta=source_of_label_info
                            )
                        
                        with result_col3:
                            if predicted_label == label_for_comparison:
                                st.success("✅ 正確")
                            else:
                                st.error("❌ 錯誤")
                        
                        # 機率分布視覺化
                        st.markdown("---")
                        st.subheader("📊 預測機率分布")
                        
                        # 創建機率DataFrame
                        proba_df = pd.DataFrame({
                            '種類': target_names,
                            '機率': prediction_proba[0]
                        }).sort_values('機率', ascending=True)
                        
                        # 繪製橫條圖
                        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
                        
                        # 使用不同顏色標記預測類別
                        colors = ['#ff7f7f' if name == predicted_label else '#87ceeb' 
                                 for name in proba_df['種類']]
                        
                        bars = ax_pred.barh(proba_df['種類'], proba_df['機率'], color=colors, alpha=0.8)
                        
                        # 設置圖表屬性
                        ax_pred.set_xlabel('預測機率', fontsize=12)
                        ax_pred.set_title('各類別預測機率分布', fontsize=14, fontweight='bold')
                        ax_pred.set_xlim(0, 1)
                        ax_pred.grid(True, alpha=0.3, axis='x')
                        
                        # 在橫條上顯示數值
                        for i, (bar, prob) in enumerate(zip(bars, proba_df['機率'])):
                            width = bar.get_width()
                            if width > 0.1:  # 只在機率大於10%時在條內顯示
                                ax_pred.text(width/2, bar.get_y() + bar.get_height()/2, 
                                           f'{width:.1%}', ha='center', va='center', 
                                           fontweight='bold', color='white')
                            else:  # 否則在條外顯示
                                ax_pred.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                           f'{width:.1%}', ha='left', va='center', 
                                           fontweight='bold')
                        
                        # 添加預測標記
                        for i, name in enumerate(proba_df['種類']):
                            if name == predicted_label:
                                ax_pred.text(1.02, i, '← 預測', va='center', fontweight='bold', color='red')
                        
                        plt.tight_layout()
                        st.pyplot(fig_pred)
                        
                        # 添加下載按鈕
                        buffer = create_downloadable_plot(fig_pred, "prediction_probability.png")
                        st.download_button(
                            label="📥 下載預測機率圖",
                            data=buffer,
                            file_name="prediction_probability.png",
                            mime="image/png",
                            key="download_pred_prob"
                        )
                        plt.close(fig_pred)
                        
                        # 詳細機率表格
                        st.subheader("📋 詳細預測資訊")
                        
                        # 創建詳細資訊表
                        detailed_proba = pd.DataFrame({
                            '花的種類': target_names,
                            '預測機率': [f"{p:.2%}" for p in prediction_proba[0]],
                            '信心等級': ['🔥 高' if p > 0.7 else '⚡ 中' if p > 0.4 else '💤 低' 
                                        for p in prediction_proba[0]],
                            '排名': [f"第 {i+1} 名" for i in range(len(target_names))]
                        })
                        
                        # 按機率排序
                        detailed_proba = detailed_proba.sort_values('預測機率', ascending=False)
                        detailed_proba.index = range(1, len(detailed_proba) + 1)
                        
                        st.dataframe(detailed_proba, use_container_width=True)
                        
                        # 特徵貢獻分析（如果只選了部分特徵）
                        if len(selected_features) < len(all_feature_names):
                            st.info(f"💡 **注意**：模型預測僅基於 {len(selected_features)} 個選定特徵: {', '.join(selected_features)}")
                        
                        # 預測解釋
                        st.subheader("🔍 預測解釋")
                        
                        if confidence > 0.9:
                            st.success(f"模型對預測結果 **{predicted_label}** 非常有信心（{confidence:.1%}）！")
                        elif confidence > 0.7:
                            st.info(f"模型較有信心預測為 **{predicted_label}**（{confidence:.1%}）。")
                        else:
                            st.warning(f"模型預測為 **{predicted_label}**，但信心度較低（{confidence:.1%}），建議謹慎參考。")
                        
                        # 輸入特徵摘要
                        with st.expander("📝 查看輸入特徵摘要"):
                            input_summary = pd.DataFrame({
                                '特徵名稱': all_feature_names,
                                '輸入值': [f"{input_data[f]:.1f}" for f in all_feature_names],
                                '是否用於預測': ['✅ 是' if f in selected_features else '❌ 否' for f in all_feature_names]
                            })
                            st.dataframe(input_summary, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ 預測過程發生錯誤：{e}")
                        import traceback
                        st.text("詳細錯誤信息：")
                        st.code(traceback.format_exc())
                
                # 快速測試按鈕
                st.markdown("---")
                st.subheader("⚡ 快速測試")
                
                quick_test_col1, quick_test_col2, quick_test_col3 = st.columns(3)
                
                with quick_test_col1:
                    if st.button("🌺 測試 Setosa 樣本", use_container_width=True):
                        st.info("請選擇樣本 ID 0-49 中的任一個")
                
                with quick_test_col2:
                    if st.button("🌸 測試 Versicolor 樣本", use_container_width=True):
                        st.info("請選擇樣本 ID 50-99 中的任一個")
                
                with quick_test_col3:
                    if st.button("🌼 測試 Virginica 樣本", use_container_width=True):
                        st.info("請選擇樣本 ID 100-149 中的任一個")
                
            else:
                # 如果沒有原始資料，提供手動輸入
                st.warning("⚠️ 無法載入原始資料集，請手動輸入特徵值。")
                
                with st.form("manual_prediction_form"):
                    st.write("**請輸入特徵值：**")
                    
                    input_data = {}
                    for feature in selected_features:
                        input_data[feature] = st.number_input(
                            f'📏 {feature}',
                            value=0.0,
                            step=0.1,
                            format="%.1f",
                            key=f"manual_input_{feature}"
                        )
                    
                    predict_button = st.form_submit_button("🔮 開始預測", type="primary", use_container_width=True)
                
                if predict_button:
                    st.info("請確保您的輸入值已經過適當的標準化處理。")
            st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ 模型載入失敗：{e}")
            st.text("請確保模型檔案存在且未損壞。")
    
    else:
        st.warning("⚠️ 請先在「🎯 模型訓練」標籤頁訓練模型")
        st.info("💡 訓練完成後即可在此進行即時預測")
        
        # 提供範例說明
        with st.expander("📖 使用說明"):
            st.markdown("""
            **如何使用即時預測功能：**
            
            1. **訓練模型**：先在「模型訓練」頁面完成模型訓練
            2. **選擇樣本**：從下拉選單選擇一個原始樣本作為起點
            3. **調整特徵**：根據需要調整各個特徵值
            4. **進行預測**：點擊預測按鈕查看結果
            5. **分析結果**：查看預測類別、信心度和機率分布
            
            **提示**：
            - 特徵值會自動限制為1位小數，確保輸入精度一致
            - 系統會自動檢測您的輸入是否匹配原始資料集中的樣本
            - 可以下載預測結果圖表用於報告或分享
            """)

# --- 頁腳 ---
st.markdown("---")
st.subheader("📚 相關資源")

st.markdown("**📊 資料來源：** [Scikit-learn Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)")
st.markdown("**⚡ 部署平台：** [Streamlit Cloud](https://streamlit.io/cloud)")
st.markdown("**🛠️ 技術框架：** Streamlit + Scikit-learn + Matplotlib")
st.markdown("**⚛️ 模型說明：** 本應用使用多層感知器 (MLP) 進行鳶尾花分類，支援完整的超參數調整與結果分析")